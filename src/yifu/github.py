"""GitHub REST/GraphQL access with on-disk caching and rate-limit handling.

Everything that talks to the network lives here so the rest of the package can
stay pure and testable.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .progress import ProgressCallback, ProgressUpdate

DEFAULT_API_BASE = "https://api.github.com"
STAR_ACCEPT = "application/vnd.github.star+json"
USER_AGENT = "yifu/0.1 (+https://github.com/ADD-SP/yifu)"
USER_TTL_SECONDS = 7 * 24 * 3600
GRAPHQL_BATCH_SIZE = 100
# Only use GraphQL batching when the measured point cost per user is at least
# this much cheaper than the REST alternative (1 request per user).
GRAPHQL_MAX_COST_PER_USER = 0.2
# Same idea for the stargazers connection: it can return star timestamps and
# follower counts in one call, but only if GitHub bills it per connection page
# rather than per node. Verified by measurement, never assumed.
CONNECTION_MAX_COST_PER_USER = 0.2
CONNECTION_PAGE_SIZE = 100
# Below this many sampled users the per-user rate is too noisy to reject the
# connection path: a tiny repo costs a negligible absolute number of points.
CONNECTION_PROBE_MIN_USERS = 50


class GitHubError(RuntimeError):
    """Base class for expected failures; ``exit_code`` maps to the CLI."""

    exit_code = 2


class AuthError(GitHubError):
    exit_code = 3


class RepoNotFound(GitHubError):
    exit_code = 4


class StargazersRestricted(GitHubError):
    """GitHub restricts the stargazer list to repository admins/collaborators."""

    exit_code = 4


STARGAZERS_RESTRICTION_DOC = (
    "https://docs.github.com/en/rest/activity/starring#new-access-restrictions"
)
STARGAZERS_RESTRICTION_CHANGELOG = (
    "https://github.blog/changelog/2026-06-30-upcoming-access-restrictions-"
    "to-public-api-endpoints-and-ui-views/"
)


def collaborator_status(permissions: Mapping[str, Any] | None) -> bool | None:
    """Is the credential an admin/collaborator of the repository?

    ``True``/``False`` when GitHub tells us through the repository's
    ``permissions`` block, ``None`` when the information is unavailable.
    """

    if not isinstance(permissions, Mapping):
        return None
    if "push" not in permissions and "admin" not in permissions:
        return None
    return bool(permissions.get("push") or permissions.get("admin"))


def collaborator_required_message(
    owner: str, repo: str, meta: Mapping[str, Any]
) -> str:
    permissions = meta.get("permissions") or {}
    stars = meta.get("stargazers_count")
    return (
        f"无法分析 {owner}/{repo}：你的凭证对该仓库没有协作者权限。\n"
        "  · 该仓库 GitHub 报告 "
        f"{stars if stars is not None else '未知'} stars，但它不在你的协作范围内"
        f"（permissions: admin={permissions.get('admin')}, push={permissions.get('push')}）\n"
        "  · GitHub 自 2026 年 7 月起把 stargazers 名单限制为**仓库管理员与协作者**可见，"
        "非协作者的 token 会拿到 404、匿名请求拿到 401、GraphQL 连接返回空页\n"
        "  · 只能分析你有写权限的仓库：换成该仓库协作者/管理员账号的 token，"
        "或改分析你自己的仓库\n"
        f"  · 官方说明：{STARGAZERS_RESTRICTION_DOC}\n"
        f"  · 变更公告：{STARGAZERS_RESTRICTION_CHANGELOG}"
    )


class QuotaExhausted(GitHubError):
    exit_code = 5


class PartialData(GitHubError):
    exit_code = 6


@dataclass(slots=True)
class Response:
    status: int
    headers: dict[str, str]
    text: str

    def json(self) -> Any:
        return json.loads(self.text) if self.text else None


class Transport:
    """Minimal HTTP transport; tests inject a fake implementation."""

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        data: bytes | None,
        timeout: float,
    ) -> Response:  # pragma: no cover - protocol definition
        raise NotImplementedError


class UrllibTransport(Transport):
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        data: bytes | None,
        timeout: float,
    ) -> Response:
        request = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as raw:
                body = raw.read().decode("utf-8", "replace")
                return Response(raw.status, {k.lower(): v for k, v in raw.headers.items()}, body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            return Response(exc.code, {k.lower(): v for k, v in exc.headers.items()}, body)


@dataclass(slots=True)
class Stargazer:
    login: str
    starred_at: float
    starred_at_iso: str
    user_id: int | None = None
    node_id: str | None = None
    user_type: str | None = None
    html_url: str | None = None
    avatar_url: str | None = None


@dataclass(slots=True)
class UserProfile:
    login: str
    followers: int | None = None
    following: int | None = None
    public_repos: int | None = None
    user_type: str | None = None
    name: str | None = None
    html_url: str | None = None
    avatar_url: str | None = None
    source: str = "rest"
    fetched_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "login": self.login,
            "followers": self.followers,
            "following": self.following,
            "public_repos": self.public_repos,
            "user_type": self.user_type,
            "name": self.name,
            "html_url": self.html_url,
            "avatar_url": self.avatar_url,
            "source": self.source,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> UserProfile:
        return cls(
            login=str(payload.get("login") or ""),
            followers=payload.get("followers"),
            following=payload.get("following"),
            public_repos=payload.get("public_repos"),
            user_type=payload.get("user_type"),
            name=payload.get("name"),
            html_url=payload.get("html_url"),
            avatar_url=payload.get("avatar_url"),
            source=str(payload.get("source") or "cache"),
            fetched_at=float(payload.get("fetched_at") or 0.0),
        )


def parse_repo_url(value: str) -> tuple[str, str]:
    """Accept ``owner/repo``, ``https://github.com/owner/repo``, SSH remotes…"""

    raw = (value or "").strip()
    if not raw:
        raise GitHubError("仓库地址为空")
    if raw.startswith("git@") and ":" in raw:
        raw = raw.split(":", 1)[1]
    elif "://" in raw:
        parsed = urllib.parse.urlparse(raw)
        if parsed.netloc and "github" not in parsed.netloc.lower() and "." in parsed.netloc:
            raise GitHubError(f"暂不支持非 github.com 的仓库地址: {value}")
        raw = parsed.path
    else:
        parts = raw.split("/")
        if parts and parts[0].lower() in {"github.com", "www.github.com"}:
            raw = "/".join(parts[1:])
    raw = raw.strip("/")
    segments = [segment for segment in raw.split("/") if segment]
    if len(segments) < 2:
        raise GitHubError(f"无法从 {value!r} 解析出 owner/repo")
    owner, repo = segments[0], segments[1]
    repo = repo.removesuffix(".git")
    for part, label in ((owner, "owner"), (repo, "repo")):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", part):
            raise GitHubError(f"非法的 {label}: {part!r}")
    return owner, repo


def mask_token(token: str | None) -> str | None:
    if not token:
        return None
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}…{token[-4:]}"


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_utc(value: str) -> float:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


class CacheStore:
    """Filesystem layout shared by ``analyze`` and ``report``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def repo_dir(self, owner: str, repo: str) -> Path:
        return self.root / "repos" / f"{owner.lower()}__{repo.lower()}"

    def pages_dir(self, owner: str, repo: str) -> Path:
        return self.repo_dir(owner, repo) / "stargazers"

    def page_path(self, owner: str, repo: str, page: int) -> Path:
        return self.pages_dir(owner, repo) / f"p{page:05d}.json"

    def meta_path(self, owner: str, repo: str) -> Path:
        return self.repo_dir(owner, repo) / "meta.json"

    def analysis_path(self, owner: str, repo: str) -> Path:
        return self.repo_dir(owner, repo) / "analysis.json"

    def verification_path(self, owner: str, repo: str) -> Path:
        return self.repo_dir(owner, repo) / "verification.json"

    def stargazers_path(self, owner: str, repo: str) -> Path:
        return self.repo_dir(owner, repo) / "stargazers.json"

    def user_path(self, login: str) -> Path:
        return self.root / "users" / f"{login.lower()}.json"

    def asset_path(self, name: str) -> Path:
        return self.root / "assets" / name

    def read_json(self, path: Path) -> Any | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        tmp.replace(path)


class GitHubClient:
    """REST + GraphQL client with retries, quota pacing and request stats."""

    def __init__(
        self,
        token: str | None = None,
        api_base: str = DEFAULT_API_BASE,
        graphql_url: str | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
        log: Callable[[str], None] = lambda _msg: None,
        no_wait: bool = False,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self.token = token or None
        self.api_base = api_base.rstrip("/")
        self.graphql_url = graphql_url or f"{self.api_base}/graphql"
        self.transport = transport or UrllibTransport()
        self.sleep = sleep
        self.now = now
        self.log = log
        self.no_wait = no_wait
        self.max_retries = max_retries
        self.timeout = timeout
        self.requests = 0
        self.rate_remaining: int | None = None
        self.rate_limit: int | None = None
        self.rate_reset: float | None = None
        self.graphql_cost_per_user: float | None = None
        self._last_graphql_cost: float | None = None
        self._last_graphql_errors: list[str] = []

    # -- low level ---------------------------------------------------------
    def _headers(self, accept: str | None = None) -> dict[str, str]:
        headers = {
            "accept": accept or "application/vnd.github+json",
            "x-github-api-version": "2022-11-28",
            "user-agent": USER_AGENT,
        }
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return headers

    def _absolute(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        if not path_or_url.startswith("/"):
            path_or_url = "/" + path_or_url
        return f"{self.api_base}{path_or_url}"

    def _update_rate_state(self, headers: Mapping[str, str]) -> None:
        remaining = headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                self.rate_remaining = int(remaining)
            except ValueError:
                self.rate_remaining = None
        limit = headers.get("x-ratelimit-limit")
        if limit is not None:
            try:
                self.rate_limit = int(limit)
            except ValueError:
                self.rate_limit = None
        reset = headers.get("x-ratelimit-reset")
        if reset is not None:
            try:
                self.rate_reset = float(reset)
            except ValueError:
                self.rate_reset = None

    def _wait_for_reset(self, reason: str) -> None:
        reset = self.rate_reset
        wait = max(0.0, (reset - self.now())) if reset else 60.0
        if self.no_wait:
            raise QuotaExhausted(
                f"{reason}；配额将在 {wait / 60:.1f} 分钟后重置，进度已保存，可稍后续跑"
            )
        self.log(f"{reason}，等待 {wait / 60:.1f} 分钟后继续（Ctrl-C 可中断，进度已保存）")
        self.sleep(wait + 1.0)

    def ensure_quota(self, floor: int = 2) -> None:
        if self.rate_remaining is not None and self.rate_remaining <= floor:
            self._wait_for_reset(f"剩余配额仅 {self.rate_remaining} 次")

    def request(
        self,
        method: str,
        path_or_url: str,
        *,
        accept: str | None = None,
        payload: Mapping[str, Any] | None = None,
        etag: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        authenticated: bool = True,
        allow_auth_error: bool = False,
    ) -> Response:
        url = self._absolute(path_or_url)
        headers = self._headers(accept)
        if not authenticated:
            # "List stargazers" is readable without credentials for public
            # repositories, which lets us recover when a token is rejected.
            headers.pop("authorization", None)
        if extra_headers:
            headers.update({k.lower(): v for k, v in extra_headers.items()})
        if etag:
            headers["if-none-match"] = etag
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"

        attempt = 0
        while True:
            attempt += 1
            self.requests += 1
            try:
                response = self.transport(method, url, headers, data, self.timeout)
            except OSError as exc:
                if attempt > self.max_retries:
                    raise GitHubError(f"网络错误: {exc}") from exc
                self.sleep(min(2 ** (attempt - 1) * 0.5, 8.0))
                continue
            self._update_rate_state(response.headers)
            if response.status in (200, 201, 304):
                return response
            if response.status in (401,):
                if allow_auth_error:
                    # Callers such as the stargazer walk want to try an anonymous
                    # request before treating this as fatal.
                    return response
                raise AuthError(
                    f"GitHub 认证失败（401）：{method} {url}"
                    f"{'（凭证 ' + (mask_token(self.token) or '未设置') + '）' if self.token else '（未带凭证）'}\n"
                    "  · 先 `gh auth status` 确认凭证有效，必要时 `gh auth login` 重新登录\n"
                    "  · 注意：stargazers 接口现在要求认证，匿名访问同样会返回 401，"
                    "所以 `unset GITHUB_TOKEN` 并不能绕过"
                )
            if response.status == 404:
                return response
            if response.status in (403, 429):
                remaining = response.headers.get("x-ratelimit-remaining")
                exhausted = remaining == "0" or response.status == 429
                if exhausted:
                    self._wait_for_reset("GitHub 配额已耗尽")
                    if attempt > self.max_retries + 1:
                        raise QuotaExhausted("配额反复耗尽，已停止")
                    continue
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after else min(2 ** (attempt - 1) * 2.0, 60.0)
                if attempt > self.max_retries:
                    raise GitHubError(f"GitHub 返回 {response.status}: {response.text[:200]}")
                self.sleep(delay)
                continue
            if response.status >= 500:
                if attempt > self.max_retries:
                    raise GitHubError(f"GitHub 返回 {response.status}: {response.text[:200]}")
                self.sleep(min(2 ** (attempt - 1) * 0.5, 8.0))
                continue
            raise GitHubError(f"GitHub 返回 {response.status}: {response.text[:200]}")

    def graphql(self, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        self.ensure_quota()
        response = self.request("POST", self.graphql_url, payload={"query": query, "variables": dict(variables)})
        if response.status == 404:
            raise GitHubError("GraphQL 端点不可用（404），请用 --fetch-followers rest")
        payload = response.json() or {}
        data = payload.get("data") or {}
        errors = payload.get("errors") or []
        self._last_graphql_errors = [
            str(item.get("message")) for item in errors if isinstance(item, Mapping)
        ]
        if errors and not data:
            raise GitHubError("GraphQL 错误: " + "; ".join(self._last_graphql_errors[:3]))
        cost = (payload.get("extensions") or {}).get("cost") or {}
        actual = cost.get("actualQueryCost") or cost.get("requestedQueryCost")
        limits = data.get("rateLimit") or {}
        if actual is None:
            # ``rateLimit.cost`` reports the cost of the query we just ran, so it
            # is a usable fallback when ``extensions.cost`` is missing.
            actual = limits.get("cost")
        if actual is not None:
            self._last_graphql_cost = float(actual)
        if limits.get("remaining") is not None:
            self.rate_remaining = int(limits["remaining"])
        if limits.get("limit") is not None:
            self.rate_limit = int(limits["limit"])
        return data

    # -- endpoints ---------------------------------------------------------
    def rate_limit(self) -> dict[str, Any]:
        response = self.request("GET", "/rate_limit")
        if response.status != 200:
            return {}
        return response.json() or {}

    def whoami(self) -> dict[str, Any]:
        """Cheap credential check: who is this token, and what can it do?"""

        response = self.request("GET", "/user")
        payload = response.json() or {}
        return {
            "login": payload.get("login"),
            "scopes": response.headers.get("x-oauth-scopes", ""),
            "rate_limit": response.headers.get("x-ratelimit-limit"),
            "rate_remaining": response.headers.get("x-ratelimit-remaining"),
        }

    def repo_meta(self, owner: str, repo: str) -> dict[str, Any]:
        response = self.request("GET", f"/repos/{owner}/{repo}")
        if response.status == 404:
            raise RepoNotFound(f"仓库 {owner}/{repo} 不存在或无权访问（私有仓库请提供 token）")
        payload = response.json() or {}
        return payload


def fetch_repo_meta(
    client: GitHubClient,
    cache: CacheStore,
    owner: str,
    repo: str,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    path = cache.meta_path(owner, repo)
    cached = None if refresh else cache.read_json(path)
    if cached and cached.get("stargazers_count") is not None:
        merged = client.repo_meta(owner, repo)
        cached.update(
            {
                "stargazers_count": merged.get("stargazers_count", cached.get("stargazers_count")),
                "description": merged.get("description"),
                "full_name": merged.get("full_name"),
                "html_url": merged.get("html_url", cached.get("html_url")),
                "pushed_at": merged.get("pushed_at"),
                "permissions": merged.get("permissions"),
            }
        )
        cache.write_json(path, cached)
        return cached
    meta = client.repo_meta(owner, repo)
    slim = {
        "owner": owner,
        "name": repo,
        "full_name": meta.get("full_name") or f"{owner}/{repo}",
        "html_url": meta.get("html_url") or f"https://github.com/{owner}/{repo}",
        "description": meta.get("description"),
        "stargazers_count": meta.get("stargazers_count"),
        "permissions": meta.get("permissions"),
        "pushed_at": meta.get("pushed_at"),
        "fetched_at": time.time(),
    }
    cache.write_json(path, slim)
    return slim


def _stargazers_unavailable(
    client: GitHubClient, owner: str, repo: str
) -> GitHubError:
    """Explain a 404 from the stargazers endpoint.

    Since July 2026 GitHub limits the stargazer listing to repository admins and
    collaborators, so a readable public repository can still answer 404 here.
    """

    probe = client.request("GET", f"/repos/{owner}/{repo}")
    if probe.status != 200:
        return RepoNotFound(
            f"仓库 {owner}/{repo} 不存在或无权访问（私有仓库请提供 token）"
        )
    payload = probe.json() or {}
    stars = payload.get("stargazers_count")
    permissions = payload.get("permissions") or {}
    is_collaborator = bool(permissions.get("push") or permissions.get("admin"))
    if is_collaborator:
        hints = (
            "你的凭证对该仓库有协作者权限，按理应可读取名单，但仍被拒：\n"
            "  · 确认 token 属于你本人且未被组织策略限制（企业/组织可能有额外限制）\n"
            "  · 也可能是仓库刚改名，缓存里的名称失效，可加 --refresh 重试"
        )
    else:
        hints = (
            "你的凭证对该仓库没有协作者权限（permissions 里 push/admin 均为 false），"
            "这正是被拒的原因。\n"
            "  · 想分析自己的仓库：换成有该仓库写权限的账号的 token\n"
            "  · 想分析别人的公开仓库：API 层面已不可行（网页端 `/stargazers` 视图同样受限）"
        )
    return StargazersRestricted(
        f"仓库 {owner}/{repo} 本身可以读取（GitHub 报告 {stars} stars），"
        "但 stargazers 名单返回 404。\n"
        "原因：GitHub 自 2026 年 7 月起限制 star 相关接口，"
        "stargazers 名单**仅对仓库管理员与协作者开放**（为阻止抓取用户数据做垃圾推广）。\n"
        f"{hints}\n"
        f"官方说明：{STARGAZERS_RESTRICTION_DOC}\n"
        f"变更公告：{STARGAZERS_RESTRICTION_CHANGELOG}"
    )


def fetch_stargazers(
    client: GitHubClient,
    cache: CacheStore,
    owner: str,
    repo: str,
    *,
    refresh: bool = False,
    progress: ProgressCallback = lambda _update: None,
    per_page: int = 100,
    max_stargazers: int | None = None,
    total_hint: int | None = None,
) -> list[Stargazer]:
    """Walk every stargazers page, reusing cached pages via conditional GETs."""

    collected: list[Stargazer] = []
    page = 1
    while True:
        page_path = cache.page_path(owner, repo, page)
        cached = None if refresh else cache.read_json(page_path)
        etag = cached.get("etag") if cached else None
        url = f"/repos/{owner}/{repo}/stargazers?per_page={per_page}&page={page}"
        response = client.request(
            "GET",
            url,
            accept=STAR_ACCEPT,
            etag=etag,
            allow_auth_error=True,
        )
        if response.status == 404:
            raise _stargazers_unavailable(client, owner, repo)
        if response.status == 401:
            if not client.token:
                raise AuthError(
                    f"GitHub 认证失败（401）：GET {url}（未带凭证）\n"
                    "  · 该接口不接受匿名请求，且自 2026 年 7 月起"
                    "stargazers 名单仅对仓库管理员与协作者开放，请用对应账号的 token\n"
                    f"  · 官方说明：{STARGAZERS_RESTRICTION_DOC}"
                )
            raise AuthError(
                f"GitHub 认证失败（401）：GET {url}（凭证 {mask_token(client.token)}）\n"
                "  · `gh auth status` 确认凭证有效；若是组织开启了 SAML SSO，"
                "还需要在 token 设置页点 Authorize"
            )
        if response.status == 304 and cached is not None:
            items = cached.get("items", [])
        else:
            items = response.json() or []
            if not isinstance(items, list):
                raise GitHubError("stargazers 接口返回了非预期的数据结构")
            cache.write_json(
                page_path,
                {"etag": response.headers.get("etag"), "items": items, "fetched_at": time.time()},
            )
        if not items:
            break
        for item in items:
            collected.append(_stargazer_from_api(item))
            if max_stargazers and len(collected) >= max_stargazers:
                break
        progress(
            ProgressUpdate(
                "stars",
                done=min(len(collected), total_hint) if total_hint else len(collected),
                total=total_hint,
                detail=f"第 {page} 页",
            )
        )
        if max_stargazers and len(collected) >= max_stargazers:
            break
        if len(items) < per_page:
            break
        page += 1
        client.ensure_quota()
    collected.sort(key=lambda item: item.starred_at)
    save_stargazers(cache, owner, repo, collected)
    return collected


def save_stargazers(
    cache: CacheStore, owner: str, repo: str, stargazers: Sequence[Stargazer]
) -> None:
    """Persist the stargazer timeline shared by the REST and GraphQL paths."""

    cache.write_json(
        cache.stargazers_path(owner, repo),
        [
            {
                "login": item.login,
                "starred_at": item.starred_at,
                "starred_at_iso": item.starred_at_iso,
                "user_id": item.user_id,
                "node_id": item.node_id,
                "user_type": item.user_type,
                "html_url": item.html_url,
                "avatar_url": item.avatar_url,
            }
            for item in stargazers
        ],
    )


def load_cached_stargazers(cache: CacheStore, owner: str, repo: str) -> list[Stargazer]:
    payload = cache.read_json(cache.stargazers_path(owner, repo))
    if not payload:
        raise GitHubError("缓存中没有 stargazers 数据，请先运行 `yifu <仓库地址>`")
    return [
        Stargazer(
            login=str(item["login"]),
            starred_at=float(item["starred_at"]),
            starred_at_iso=str(item.get("starred_at_iso") or iso_utc(float(item["starred_at"]))),
            user_id=item.get("user_id"),
            node_id=item.get("node_id"),
            user_type=item.get("user_type"),
            html_url=item.get("html_url"),
            avatar_url=item.get("avatar_url"),
        )
        for item in payload
    ]


def _stargazer_from_api(item: Mapping[str, Any]) -> Stargazer:
    user = item.get("user") or {}
    starred_at = item.get("starred_at")
    epoch = parse_iso_utc(starred_at) if starred_at else 0.0
    return Stargazer(
        login=str(user.get("login") or ""),
        starred_at=epoch,
        starred_at_iso=starred_at or iso_utc(epoch),
        user_id=user.get("id"),
        node_id=user.get("node_id"),
        user_type=user.get("type"),
        html_url=user.get("html_url"),
        avatar_url=user.get("avatar_url"),
    )


def profile_from_rest(login: str, payload: Mapping[str, Any], now: float) -> UserProfile:
    return UserProfile(
        login=str(payload.get("login") or login),
        followers=payload.get("followers"),
        following=payload.get("following"),
        public_repos=payload.get("public_repos"),
        user_type=payload.get("type"),
        name=payload.get("name"),
        html_url=payload.get("html_url"),
        avatar_url=payload.get("avatar_url"),
        source="rest",
        fetched_at=now,
    )


def load_user_cache(
    cache: CacheStore,
    logins: Sequence[str],
    *,
    ttl_seconds: int = USER_TTL_SECONDS,
    now: float | None = None,
) -> tuple[dict[str, UserProfile], list[str]]:
    current = time.time() if now is None else now
    cached: dict[str, UserProfile] = {}
    missing: list[str] = []
    for login in logins:
        payload = cache.read_json(cache.user_path(login))
        if not payload:
            missing.append(login)
            continue
        profile = UserProfile.from_json(payload)
        age = current - profile.fetched_at
        if ttl_seconds >= 0 and age > ttl_seconds:
            missing.append(login)
            continue
        profile.source = "cache"
        cached[login] = profile
    return cached, missing


_FOLLOWERS_QUERY = """
query($ids: [ID!]!) {
  nodes(ids: $ids) {
    __typename
    ... on User { login name followers { totalCount } avatarUrl url }
    ... on Organization { login name avatarUrl url }
  }
  rateLimit { limit cost remaining resetAt }
}
"""


def _profile_from_graphql(node: Mapping[str, Any], now: float) -> UserProfile:
    # ``User.followers`` is a FollowerConnection: the count lives in ``totalCount``.
    # Organizations have no followers field in the GraphQL schema at all, so they
    # come back without a count and get routed to the REST fallback.
    followers = (node.get("followers") or {}).get("totalCount")
    return UserProfile(
        login=str(node.get("login") or ""),
        followers=followers,
        following=None,
        public_repos=None,
        user_type=node.get("__typename") or "User",
        name=node.get("name"),
        html_url=node.get("url"),
        avatar_url=node.get("avatarUrl"),
        source="graphql",
        fetched_at=now,
    )


_CONNECTION_STARGAZERS_QUERY = """
query($owner: String!, $name: String!, $cursor: String, $pageSize: Int!) {
  repository(owner: $owner, name: $name) {
    stargazers(first: $pageSize, after: $cursor, orderBy: {field: STARRED_AT, direction: ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      edges {
        starredAt
        node { __typename login name avatarUrl url followers { totalCount } }
      }
    }
  }
  rateLimit { limit cost remaining resetAt }
}
"""


@dataclass(slots=True)
class ConnectionPage:
    """One page of ``repository.stargazers`` with follower counts attached."""

    index: int
    cursor: str | None
    end_cursor: str | None
    has_next_page: bool
    total_count: int | None
    cost: float | None
    stargazers: list[Stargazer]
    followers: dict[str, int]
    profiles: dict[str, UserProfile]
    skipped: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cursor": self.cursor,
            "end_cursor": self.end_cursor,
            "has_next_page": self.has_next_page,
            "total_count": self.total_count,
            "cost": self.cost,
            "fetched_at": time.time(),
            "edges": [
                [item.login, item.starred_at_iso, self.followers.get(item.login)]
                for item in self.stargazers
            ],
            "skipped": self.skipped,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], index: int) -> ConnectionPage:
        stargazers: list[Stargazer] = []
        followers: dict[str, int] = {}
        profiles: dict[str, UserProfile] = {}
        fetched_at = float(payload.get("fetched_at") or 0.0)
        for edge in payload.get("edges") or []:
            login, starred_at_iso, count = (list(edge) + [None, None, None])[:3]
            if not login or not starred_at_iso:
                continue
            epoch = parse_iso_utc(str(starred_at_iso))
            stargazers.append(
                Stargazer(
                    login=str(login),
                    starred_at=epoch,
                    starred_at_iso=str(starred_at_iso),
                    user_type="User",
                )
            )
            if count is not None:
                followers[str(login)] = int(count)
                profiles[str(login)] = UserProfile(
                    login=str(login),
                    followers=int(count),
                    user_type="User",
                    source="graphql-connection",
                    fetched_at=fetched_at,
                )
        return cls(
            index=index,
            cursor=payload.get("cursor"),
            end_cursor=payload.get("end_cursor"),
            has_next_page=bool(payload.get("has_next_page")),
            total_count=payload.get("total_count"),
            cost=payload.get("cost"),
            stargazers=stargazers,
            followers=followers,
            profiles=profiles,
            skipped=int(payload.get("skipped") or 0),
        )


@dataclass(slots=True)
class ConnectionWalk:
    accepted: bool
    stargazers: list[Stargazer]
    profiles: dict[str, UserProfile]
    pages: int
    points: float
    cost_per_user: float | None
    exhausted: bool
    error: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "pages": self.pages,
            "stars": len(self.stargazers),
            "profiles": len(self.profiles),
            "points": round(self.points, 3),
            "cost_per_user": (
                None if self.cost_per_user is None else round(self.cost_per_user, 6)
            ),
            "exhausted": self.exhausted,
            "error": self.error,
            "reason": self.reason,
        }


def _parse_connection_page(
    data: Mapping[str, Any],
    index: int,
    cursor: str | None,
    cost: float | None,
) -> ConnectionPage:
    repository = data.get("repository") or {}
    connection = repository.get("stargazers") or {}
    page_info = connection.get("pageInfo") or {}
    stargazers: list[Stargazer] = []
    followers: dict[str, int] = {}
    profiles: dict[str, UserProfile] = {}
    skipped = 0
    now = time.time()
    for edge in connection.get("edges") or []:
        if not isinstance(edge, Mapping):
            skipped += 1
            continue
        node = edge.get("node") or {}
        login = str(node.get("login") or "")
        starred_at = edge.get("starredAt")
        if not login or not starred_at:
            skipped += 1
            continue
        try:
            epoch = parse_iso_utc(str(starred_at))
        except ValueError:
            skipped += 1
            continue
        user_type = str(node.get("__typename") or "User")
        stargazers.append(
            Stargazer(
                login=login,
                starred_at=epoch,
                starred_at_iso=str(starred_at),
                user_type=user_type,
                html_url=node.get("url"),
                avatar_url=node.get("avatarUrl"),
            )
        )
        count = (node.get("followers") or {}).get("totalCount")
        if count is None:
            continue  # organizations have no followers field in the GraphQL schema
        followers[login] = int(count)
        profiles[login] = UserProfile(
            login=login,
            followers=int(count),
            user_type=user_type,
            name=node.get("name"),
            html_url=node.get("url"),
            avatar_url=node.get("avatarUrl"),
            source="graphql-connection",
            fetched_at=now,
        )
    return ConnectionPage(
        index=index,
        cursor=cursor,
        end_cursor=page_info.get("endCursor"),
        has_next_page=bool(page_info.get("hasNextPage")),
        total_count=connection.get("totalCount"),
        cost=cost,
        stargazers=stargazers,
        followers=followers,
        profiles=profiles,
        skipped=skipped,
    )


def walk_stargazers_connection(
    client: GitHubClient,
    cache: CacheStore,
    owner: str,
    repo: str,
    *,
    page_size: int = CONNECTION_PAGE_SIZE,
    max_users: int | None = None,
    until: float | None = None,
    refresh: bool = False,
    max_cost_per_user: float = CONNECTION_MAX_COST_PER_USER,
    progress: ProgressCallback = lambda _update: None,
    log: Callable[[str], None] = lambda _msg: None,
) -> ConnectionWalk:
    """Walk ``repository.stargazers`` once, collecting star times and followers.

    The first page doubles as the cost probe: if GitHub bills this connection by
    node (1 point per stargazer) it is no cheaper than REST, and we bail out
    after that single page instead of burning quota. The page is cached either
    way, so nothing is wasted.
    """

    pages_dir = cache.repo_dir(owner, repo) / "graphql_stargazers"
    meta = cache.read_json(cache.meta_path(owner, repo)) or {}
    reported_stars = meta.get("stargazers_count")
    cached_pages: list[ConnectionPage] = []
    if not refresh:
        index = 0
        expected_cursor: str | None = None
        while True:
            payload = cache.read_json(pages_dir / f"p{index:05d}.json")
            if not payload or payload.get("cursor") != expected_cursor:
                break
            page = ConnectionPage.from_json(payload, index)
            cached_pages.append(page)
            if not page.has_next_page:
                break
            expected_cursor = page.end_cursor
            index += 1

    stargazers: list[Stargazer] = []
    profiles: dict[str, UserProfile] = {}
    points = 0.0
    measured_cost: float | None = None
    cursor: str | None = None
    if cached_pages and not any(page.stargazers for page in cached_pages):
        # A previous run cached an empty page (older versions did that when
        # GitHub answered with repository: null). Replaying it would look like a
        # finished, empty repository, so drop it and ask GitHub again.
        log("缓存里的 GraphQL 连接页不含任何 star，已丢弃并重新抓取")
        for page_file in sorted(pages_dir.glob("p*.json")):
            try:
                page_file.unlink()
            except OSError:
                pass
        cached_pages = []
    for page in cached_pages:
        stargazers.extend(page.stargazers)
        profiles.update(page.profiles)
        points += float(page.cost or 0.0)
        cursor = page.end_cursor
        for login, profile in page.profiles.items():
            cache.write_json(cache.user_path(login), profile.to_json())

    if cached_pages and not cached_pages[-1].has_next_page:
        stargazers.sort(key=lambda item: item.starred_at)
        save_stargazers(cache, owner, repo, stargazers)
        return ConnectionWalk(
            accepted=True,
            stargazers=stargazers,
            profiles=profiles,
            pages=len(cached_pages),
            points=points,
            cost_per_user=(points / len(stargazers)) if stargazers else None,
            exhausted=True,
            reason="cached",
        )

    index = len(cached_pages)
    exhausted = False
    while True:
        if client.rate_remaining is not None and client.rate_remaining <= 0:
            break
        variables = {
            "owner": owner,
            "name": repo,
            "cursor": cursor,
            "pageSize": page_size,
        }
        try:
            data = client.graphql(_CONNECTION_STARGAZERS_QUERY, variables)
        except GitHubError as exc:
            if not cached_pages and index == 0:
                return ConnectionWalk(
                    accepted=False,
                    stargazers=[],
                    profiles={},
                    pages=0,
                    points=0.0,
                    cost_per_user=None,
                    exhausted=False,
                    error=str(exc),
                    reason="query-failed",
                )
            log(f"GraphQL 连接在第 {index + 1} 页中断（{exc}），已收集的部分照常使用")
            break
        cost = client._last_graphql_cost
        repository = data.get("repository")
        if repository is None:
            # GitHub resolves the repository to null (with an errors entry) when
            # the credentials cannot see it. Treat that as a failure instead of
            # an empty stargazer list.
            detail = "; ".join(client._last_graphql_errors[:2]) or "repository 返回 null"
            if not cached_pages and index == 0:
                return ConnectionWalk(
                    accepted=False,
                    stargazers=[],
                    profiles={},
                    pages=0,
                    points=float(cost or 0.0),
                    cost_per_user=None,
                    exhausted=False,
                    error=f"GraphQL 无法解析该仓库：{detail}",
                    reason="repository-unresolved",
                )
            log(f"GraphQL 连接在第 {index + 1} 页无法解析仓库（{detail}），保留已收集的部分")
            break
        page = _parse_connection_page(data, index, cursor, cost)
        if not page.stargazers:
            # A page that claims to have a next page but carries no stars would
            # otherwise loop forever (and burn quota). Bail out instead, and do
            # not cache it: an empty page must never look like a finished walk.
            detail = (
                f"stargazers 连接返回 0 条（connection.totalCount="
                f"{page.total_count if page.total_count is not None else '未知'}，"
                f"仓库元数据 {reported_stars if reported_stars is not None else '未知'} stars）"
            )
            if reported_stars:
                detail += (
                    "：GitHub 自 2026 年 7 月起把 stargazers 名单限制为"
                    "仓库管理员与协作者可见（为阻止抓取用户数据做垃圾推广）"
                )
            log(f"{detail}，停止该路径并回退 REST 时间线 + 剪枝")
            return ConnectionWalk(
                accepted=False,
                stargazers=[],
                profiles={},
                pages=index + 1,
                points=points + float(cost or 0.0),
                cost_per_user=None,
                exhausted=False,
                error=detail,
                reason="empty",
            )
        pages_dir.mkdir(parents=True, exist_ok=True)
        cache.write_json(pages_dir / f"p{index:05d}.json", page.to_json())
        cached_pages.append(page)
        stargazers.extend(page.stargazers)
        profiles.update(page.profiles)
        points += float(cost or 0.0)
        for login, profile in page.profiles.items():
            cache.write_json(cache.user_path(login), profile.to_json())
        index += 1
        progress(
            ProgressUpdate(
                "stars",
                done=len(stargazers),
                total=page.total_count or reported_stars,
                detail=f"第 {index} 页 · {points:.1f} points",
            )
        )
        if measured_cost is None and page.stargazers:
            if cost is None:
                log(
                    "GitHub 没有返回本次查询的 point 成本，无法确认连接路径是否便宜，"
                    "保守起见回退 REST 时间线 + 剪枝"
                )
                return ConnectionWalk(
                    accepted=False,
                    stargazers=stargazers,
                    profiles=profiles,
                    pages=index,
                    points=points,
                    cost_per_user=None,
                    exhausted=False,
                    reason="cost-unknown",
                )
            if not stargazers:
                # An empty page costs points but yields nothing: never treat that
                # as "0 points per user" (which would look infinitely cheap).
                log("GraphQL 连接返回了 0 个 star，无法据此判断成本，回退 REST 时间线 + 剪枝")
                return ConnectionWalk(
                    accepted=False,
                    stargazers=[],
                    profiles={},
                    pages=index,
                    points=points,
                    cost_per_user=None,
                    exhausted=False,
                    error="stargazers 连接返回空页",
                    reason="empty",
                )
            measured_cost = points / len(stargazers)
            if measured_cost > max_cost_per_user and len(stargazers) >= CONNECTION_PROBE_MIN_USERS:
                log(
                    f"GraphQL 连接实测 {measured_cost:.4f} points 每人（> {max_cost_per_user}），"
                    "与 REST 相比不划算，改用 REST 时间线 + 剪枝"
                )
                return ConnectionWalk(
                    accepted=False,
                    stargazers=stargazers,
                    profiles=profiles,
                    pages=index,
                    points=points,
                    cost_per_user=measured_cost,
                    exhausted=False,
                    reason="too-expensive",
                )
        if not page.has_next_page:
            exhausted = True
            break
        if max_users is not None and len(stargazers) >= max_users:
            break
        if until is not None and page.stargazers and page.stargazers[-1].starred_at > until:
            break
        next_cursor = page.end_cursor
        if next_cursor is None:
            exhausted = True
            break
        if next_cursor == cursor:
            log("GraphQL 连接游标没有前进，停止以免重复抓取同一页")
            break
        cursor = next_cursor

    if not cached_pages:
        return ConnectionWalk(
            accepted=False,
            stargazers=[],
            profiles={},
            pages=0,
            points=0.0,
            cost_per_user=None,
            exhausted=False,
            error="没有取得任何 GraphQL 数据",
            reason="empty",
        )
    stargazers.sort(key=lambda item: item.starred_at)
    save_stargazers(cache, owner, repo, stargazers)
    return ConnectionWalk(
        accepted=True,
        stargazers=stargazers,
        profiles=profiles,
        pages=len(cached_pages),
        points=points,
        # Report the measured billing rate of a full page when we have it: the
        # final partial page would otherwise distort points/user.
        cost_per_user=(
            measured_cost if measured_cost is not None
            else ((points / len(stargazers)) if stargazers else None)
        ),
        exhausted=exhausted,
    )


class ProfileFetcher:
    """Fetch follower counts for a set of logins, REST or GraphQL batched."""

    def __init__(
        self,
        client: GitHubClient,
        cache: CacheStore,
        *,
        workers: int = 8,
        mode: str = "auto",
        log: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        self.client = client
        self.cache = cache
        self.workers = max(1, min(workers, 16))
        self.mode = mode
        self.log = log
        self.counts = {"cache": 0, "graphql": 0, "rest": 0, "unavailable": 0}
        self.misses = 0
        self.graphql_probe_error: str | None = None

    def fetch(
        self,
        logins: Sequence[str],
        node_ids: Mapping[str, str | None],
        *,
        user_types: Mapping[str, str | None] | None = None,
        refresh: bool = False,
        progress: ProgressCallback = lambda _update: None,
    ) -> dict[str, UserProfile]:
        profiles: dict[str, UserProfile] = {}
        types = user_types or {}
        pending = list(dict.fromkeys(logins))
        if not refresh:
            cached, pending = load_user_cache(self.cache, pending)
            profiles.update(cached)
            self.counts["cache"] += len(cached)
        if not pending:
            return profiles
        if not self.client.token or self.mode == "rest":
            mode = "rest"
        else:
            mode, probed = self._probe_graphql(pending, node_ids, types)
            if probed:
                profiles.update(probed)
                pending = [login for login in pending if login not in probed]
        self.misses = len(pending)
        if mode == "graphql":
            profiles.update(self._fetch_graphql(pending, node_ids, types, progress))
        else:
            profiles.update(self._fetch_rest(pending, progress, types))
        for profile in profiles.values():
            self.cache.write_json(self.cache.user_path(profile.login), profile.to_json())
        return profiles

    # -- GraphQL -----------------------------------------------------------
    def _probe_graphql(
        self,
        pending: Sequence[str],
        node_ids: Mapping[str, str | None],
        user_types: Mapping[str, str | None],
    ) -> tuple[str, dict[str, UserProfile]]:
        """Measure the real point cost of one batch before committing to it."""

        # Organizations have no follower count in the GraphQL schema, so they
        # always take the REST path; probing them would waste a batch slot.
        usable = [
            login
            for login in pending
            if node_ids.get(login) and user_types.get(login) != "Organization"
        ]
        if not usable:
            self.log("没有可用于 GraphQL 的用户（缺 node_id 或都是组织账号），改用 REST")
            return "rest", {}
        probe = usable[:GRAPHQL_BATCH_SIZE]
        ids = [node_ids[login] for login in probe]
        try:
            data = self.client.graphql(_FOLLOWERS_QUERY, {"ids": ids})
        except GitHubError as exc:
            self.graphql_probe_error = str(exc)
            self.log(f"GraphQL 探测失败（{exc}），改用 REST")
            return "rest", {}
        nodes = [node for node in (data.get("nodes") or []) if node]
        cost = self.client._last_graphql_cost
        if cost is None:
            cost = float(len(probe))
        per_user = cost / max(1, len(probe))
        self.client.graphql_cost_per_user = per_user
        self.log(
            f"GraphQL 实测成本 {cost} points / {len(probe)} 人 = {per_user:.4f} points 每人"
        )
        if per_user > GRAPHQL_MAX_COST_PER_USER:
            self.log("GraphQL 成本不划算，改用 REST 逐人查询")
            return "rest", {}
        now = time.time()
        profiles: dict[str, UserProfile] = {}
        for node in nodes:
            profile = _profile_from_graphql(node, now)
            if not profile.login:
                continue
            if profile.followers is None:
                continue  # e.g. an organization: leave it to the REST fallback
            profiles[profile.login] = profile
            self.counts["graphql"] += 1
            self.cache.write_json(self.cache.user_path(profile.login), profile.to_json())
        for login in probe:
            if login not in profiles:
                continue  # missing node: handled later by REST (or marked unavailable there)
        return "graphql", profiles

    def _fetch_graphql(
        self,
        pending: Sequence[str],
        node_ids: Mapping[str, str | None],
        user_types: Mapping[str, str | None],
        progress: ProgressCallback,
    ) -> dict[str, UserProfile]:
        profiles: dict[str, UserProfile] = {}
        batches: list[list[str]] = []
        current: list[str] = []
        for login in pending:
            if node_ids.get(login) and user_types.get(login) != "Organization":
                current.append(login)
            if len(current) >= GRAPHQL_BATCH_SIZE:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        without_ids = [
            login
            for login in pending
            if not node_ids.get(login) or user_types.get(login) == "Organization"
        ]

        def run_batch(batch: Sequence[str]) -> list[Mapping[str, Any]]:
            self.client.ensure_quota()
            ids = [node_ids[login] for login in batch]
            data = self.client.graphql(_FOLLOWERS_QUERY, {"ids": ids})
            return [node for node in (data.get("nodes") or []) if node]

        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(run_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    nodes = future.result()
                except GitHubError as exc:
                    self.log(f"GraphQL 批次失败（{exc}），该批 {len(batch)} 人改用 REST")
                    profiles.update(self._fetch_rest(batch, progress, user_types))
                    done += len(batch)
                    continue
                now = time.time()
                for node in nodes:
                    profile = _profile_from_graphql(node, now)
                    if not profile.login:
                        continue
                    if profile.followers is None:
                        self.log(f"{profile.login}: GraphQL 没有粉丝数（组织账号？），改用 REST")
                        profiles.update(self._fetch_rest([profile.login], progress, user_types))
                        continue
                    profiles[profile.login] = profile
                    self.counts["graphql"] += 1
                missing = [login for login in batch if login not in profiles]
                if missing:
                    for login in missing:
                        profiles[login] = UserProfile(
                            login=login, source="unavailable", fetched_at=now
                        )
                        self.counts["unavailable"] += 1
                done += len(batch)
                progress(
                    ProgressUpdate("profiles", done=done, total=len(pending), detail="GraphQL 批量")
                )
        if without_ids:
            profiles.update(self._fetch_rest(without_ids, progress, user_types))
        return profiles

    # -- REST --------------------------------------------------------------
    def _fetch_rest(
        self,
        pending: Sequence[str],
        progress: ProgressCallback,
        user_types: Mapping[str, str | None] | None = None,
    ) -> dict[str, UserProfile]:
        profiles: dict[str, UserProfile] = {}
        done = 0
        types = user_types or {}

        def fetch_one(login: str) -> UserProfile:
            self.client.ensure_quota()
            # Organizations only expose their follower count on /orgs/{login}.
            endpoint = "orgs" if types.get(login) == "Organization" else "users"
            quoted = urllib.parse.quote(login)
            response = self.client.request("GET", f"/{endpoint}/{quoted}")
            if response.status == 404 and endpoint == "users":
                # Stargazer metadata can be stale or missing: an account typed as
                # a user may actually be an organization, which some deployments
                # only expose under /orgs/.
                org_response = self.client.request("GET", f"/orgs/{quoted}")
                if org_response.status == 200:
                    return profile_from_rest(login, org_response.json() or {}, time.time())
            if response.status == 404:
                return UserProfile(login=login, source="unavailable", fetched_at=time.time())
            if response.status != 200:
                raise GitHubError(f"获取 {login} 资料失败: HTTP {response.status}")
            return profile_from_rest(login, response.json() or {}, time.time())

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(fetch_one, login): login for login in pending}
            for future in as_completed(futures):
                login = futures[future]
                try:
                    profile = future.result()
                except GitHubError as exc:
                    self.log(f"{login}: {exc}")
                    profile = UserProfile(login=login, source="unavailable", fetched_at=time.time())
                profiles[login] = profile
                self.counts["unavailable" if profile.source == "unavailable" else "rest"] += 1
                done += 1
                if done % 10 == 0 or done == len(pending):
                    progress(
                        ProgressUpdate("profiles", done=done, total=len(pending), detail="REST 逐人")
                    )
        return profiles
