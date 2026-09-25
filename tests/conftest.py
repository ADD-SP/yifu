"""Shared fixtures: an in-process fake GitHub API built on pytest-httpserver."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from werkzeug.wrappers import Response as WerkzeugResponse

from yifu.github import CacheStore, GitHubClient, UrllibTransport

PAGE_SIZE = 100


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FakeGitHub:
    """Minimal GitHub surface: repo meta, stargazers, users and GraphQL nodes."""

    server: Any
    owner: str = "owner"
    repo: str = "repo"
    stargazers: list[tuple[str, float]] = field(default_factory=list)
    followers: dict[str, int] = field(default_factory=dict)
    missing_users: set[str] = field(default_factory=set)
    organizations: set[str] = field(default_factory=set)
    points_per_user: float = 0.0005
    connection_points_per_page: float = 1.0
    connection_enabled: bool = True
    connection_cost_source: str = "extensions"
    connection_null_repository: bool = False
    connection_empty_page: bool = False
    stargazers_missing: bool = False
    stargazers_require_anonymous: bool = False
    stargazers_reject_token: bool = False
    auth_invalid: bool = False
    collaborator: bool = True
    report_permissions: bool = True
    graphql_enabled: bool = True
    users_endpoint_404_for_orgs: bool = False
    repo_missing: bool = False
    calls: list[str] = field(default_factory=list)
    pages_served: int = 0
    not_modified: int = 0

    def setup_method_like(self, stargazers: Sequence[tuple[str, float]], followers: dict[str, int]) -> None:
        self.stargazers = sorted(stargazers, key=lambda item: item[1])
        self.followers = dict(followers)

    # -- handlers ---------------------------------------------------------
    def _current_user(self, _request: Any) -> WerkzeugResponse:
        self.calls.append("user-self")
        if self.auth_invalid and _request.headers.get("Authorization"):
            return WerkzeugResponse(
                json.dumps({"message": "Bad credentials"}),
                status=401,
                content_type="application/json",
            )
        return WerkzeugResponse(
            json.dumps({"login": "yifu-tester"}),
            content_type="application/json",
            headers={"X-OAuth-Scopes": "public_repo, read:user", "X-RateLimit-Limit": "5000",
            "X-RateLimit-Remaining": "4990"},
        )

    def _repo_meta(self, _request: Any) -> WerkzeugResponse:
        self.calls.append("repo")
        if self.auth_invalid and _request.headers.get("Authorization"):
            return WerkzeugResponse(
                json.dumps({"message": "Bad credentials"}),
                status=401,
                content_type="application/json",
            )
        if self.repo_missing:
            return WerkzeugResponse("{}", status=404, content_type="application/json")
        return WerkzeugResponse(
            json.dumps(
                {
                    "full_name": f"{self.owner}/{self.repo}",
                    "html_url": f"https://github.com/{self.owner}/{self.repo}",
                    "description": "fake repo",
                    "stargazers_count": len(self.stargazers),
                    "permissions": (
                        {
                            "admin": self.collaborator,
                            "push": self.collaborator,
                            "pull": True,
                        }
                        if self.report_permissions
                        else None
                    ),
                }
            ),
            content_type="application/json",
        )

    def _stargazers(self, request: Any) -> WerkzeugResponse:
        page = int(request.args.get("page", "1"))
        authenticated = bool(request.headers.get("Authorization"))
        self.calls.append(("stargazers" if authenticated else "stargazers-anon") + f":{page}")
        if self.stargazers_require_anonymous and authenticated:
            # Mirrors GitHub: a token that cannot see the repository gets 404
            # here, while the same public endpoint works without credentials.
            return WerkzeugResponse("{}", status=404, content_type="application/json")
        if self.stargazers_reject_token and authenticated:
            return WerkzeugResponse(
                json.dumps({"message": "Bad credentials"}),
                status=401,
                content_type="application/json",
            )
        if self.stargazers_missing:
            return WerkzeugResponse("{}", status=404, content_type="application/json")
        chunk = self.stargazers[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        etag = f'W/"page-{page}"'
        if request.headers.get("If-None-Match") == etag and not chunk:
            self.not_modified += 1
            return WerkzeugResponse(status=304)
        if request.headers.get("If-None-Match") == etag and page <= self._page_count():
            self.not_modified += 1
            return WerkzeugResponse(status=304)
        self.pages_served += 1
        payload = [
            {
                "starred_at": iso(timestamp),
                "user": {
                    "login": login,
                    "id": index + 1,
                    "node_id": f"NODE{index + 1}",
                    "type": "Organization" if login in self.organizations else "User",
                    "html_url": f"https://github.com/{login}",
                    "avatar_url": "",
                },
            }
            for index, (login, timestamp) in enumerate(
                chunk, start=(page - 1) * PAGE_SIZE
            )
        ]
        return WerkzeugResponse(
            json.dumps(payload),
            content_type="application/json",
            headers={"ETag": etag},
        )

    def _page_count(self) -> int:
        return max(1, (len(self.stargazers) + PAGE_SIZE - 1) // PAGE_SIZE)

    def _user(self, request: Any) -> WerkzeugResponse:
        login = request.path.rsplit("/", 1)[-1]
        self.calls.append(f"user:{login}")
        if login in self.organizations and self.users_endpoint_404_for_orgs:
            return WerkzeugResponse("{}", status=404, content_type="application/json")
        if login in self.missing_users or login not in self.followers:
            return WerkzeugResponse("{}", status=404, content_type="application/json")
        user_type = "Organization" if login in self.organizations else "User"
        return WerkzeugResponse(
            json.dumps(
                {
                    "login": login,
                    "followers": self.followers[login],
                    "following": 12,
                    "public_repos": 4,
                    "type": user_type,
                    "name": login.title(),
                    "html_url": f"https://github.com/{login}",
                    "avatar_url": "",
                }
            ),
            content_type="application/json",
        )

    def _org(self, request: Any) -> WerkzeugResponse:
        login = request.path.rsplit("/", 1)[-1]
        self.calls.append(f"org:{login}")
        if login in self.missing_users or login not in self.followers:
            return WerkzeugResponse("{}", status=404, content_type="application/json")
        return WerkzeugResponse(
            json.dumps(
                {
                    "login": login,
                    "followers": self.followers[login],
                    "public_repos": 4,
                    "type": "Organization",
                    "name": login.title(),
                    "html_url": f"https://github.com/{login}",
                    "avatar_url": "",
                }
            ),
            content_type="application/json",
        )

    def _schema_errors(self, query: str) -> list[dict[str, str]]:
        """Validate the query the way the real GitHub schema would.

        The real API rejected our first attempt with "Field 'total' doesn't
        exist on type 'FollowerConnection'" and "Field 'followers' doesn't exist
        on type 'Organization'"; serving a query that those errors would catch
        keeps the tests honest.
        """

        errors: list[dict[str, str]] = []
        compact = re.sub(r"\s+", " ", query)
        if "followers { totalCount }" not in compact:
            errors.append(
                {"message": "Field 'total' doesn't exist on type 'FollowerConnection'"}
            )
        organization = re.search(r"\.\.\. on Organization \{ (.*?) \}", compact)
        if organization and "followers" in organization.group(1):
            errors.append({"message": "Field 'followers' doesn't exist on type 'Organization'"})
        if "stargazers(first:" in compact:
            if "starredAt" not in compact:
                errors.append({"message": "Field 'stargazers' requires a starredAt selection"})
            if "orderBy: {field: STARRED_AT, direction: ASC}" not in compact:
                errors.append({"message": "Field 'stargazers' requires a stable STARRED_AT order"})
        return errors

    def _connection_page(self, request: Any, payload: Mapping[str, Any]) -> Any:
        if not self.connection_enabled:
            return WerkzeugResponse(
                json.dumps({"errors": [{"message": "connection disabled"}]}),
                status=400,
                content_type="application/json",
            )
        variables = payload["variables"]
        window = self.stargazers[
            int(str(variables.get("cursor") or "cursor-0").split("-")[-1]) :
        ][: int(variables.get("pageSize") or PAGE_SIZE)]
        if self.connection_null_repository:
            # What GitHub returns when the credentials cannot see the repository.
            self.calls.append("connection:null-repository")
            return WerkzeugResponse(
                json.dumps(
                    {
                        "data": {"repository": None, "rateLimit": {"cost": 1, "remaining": 4900}},
                        "errors": [
                            {
                                "type": "NOT_FOUND",
                                "message": f"Could not resolve to a Repository with the name '{self.owner}/{self.repo}'.",
                            }
                        ],
                    }
                ),
                content_type="application/json",
            )
        cursor = variables.get("cursor") or "cursor-0"
        offset = int(str(cursor).split("-")[-1])
        if self.connection_empty_page:
            window = []  # a page that claims more pages while carrying no stars
        edges = []
        for index, (login, timestamp) in enumerate(window, start=offset):
            node: dict[str, Any] = {
                "__typename": "Organization" if login in self.organizations else "User",
                "login": login,
                "name": login.title(),
                "avatarUrl": "",
                "url": f"https://github.com/{login}",
            }
            if login not in self.organizations:
                node["followers"] = {"totalCount": self.followers.get(login, 0)}
            edges.append({"starredAt": iso(timestamp), "node": node})
        next_offset = offset + len(window)
        has_next = next_offset < len(self.stargazers) or self.connection_empty_page
        self.calls.append(f"connection:{len(window)}")
        body: dict[str, Any] = {
            "data": {
                        "repository": {
                            "stargazers": {
                                "totalCount": len(self.stargazers),
                                "pageInfo": {
                                    "hasNextPage": has_next,
                                    "endCursor": f"cursor-{next_offset}" if has_next else None,
                                },
                                "edges": edges,
                            }
                        },
                        "rateLimit": {
                            "limit": 5000,
                            "cost": self.connection_points_per_page,
                            "remaining": 4900,
                            "resetAt": iso(datetime.now(tz=UTC).timestamp() + 3600),
                        },
            },
        }
        if self.connection_cost_source == "extensions":
            body["extensions"] = {
                "cost": {
                    "requestedQueryCost": self.connection_points_per_page,
                    "actualQueryCost": self.connection_points_per_page,
                }
            }
        elif self.connection_cost_source == "none":
            body["data"]["rateLimit"].pop("cost", None)
        return WerkzeugResponse(
            json.dumps(body),
            content_type="application/json",
        )

    def _graphql(self, request: Any) -> WerkzeugResponse:
        if self.auth_invalid and request.headers.get("Authorization"):
            return WerkzeugResponse(
                json.dumps({"message": "Bad credentials"}),
                status=401,
                content_type="application/json",
            )
        if not self.graphql_enabled:
            return WerkzeugResponse("{}", status=404, content_type="application/json")
        payload = request.get_json()
        if "stargazers(first:" in payload.get("query", ""):
            errors = self._schema_errors(payload.get("query", ""))
            if errors:
                self.calls.append("graphql-schema-error")
                return WerkzeugResponse(
                    json.dumps({"errors": errors}), status=400, content_type="application/json"
                )
            return self._connection_page(request, payload)
        errors = self._schema_errors(payload.get("query", ""))
        if errors:
            self.calls.append("graphql-schema-error")
            return WerkzeugResponse(
                json.dumps({"errors": errors}), status=400, content_type="application/json"
            )
        ids = payload["variables"]["ids"]
        self.calls.append(f"graphql:{len(ids)}")
        node_map = {
            f"NODE{index + 1}": login for index, (login, _ts) in enumerate(self.stargazers)
        }
        nodes = []
        for node_id in ids:
            login = node_map.get(node_id)
            if login is None or login in self.missing_users:
                nodes.append(None)
                continue
            if login in self.organizations:
                # Organizations have no followers field in the GraphQL schema.
                nodes.append(
                    {
                        "__typename": "Organization",
                        "login": login,
                        "name": login.title(),
                        "avatarUrl": "",
                        "url": f"https://github.com/{login}",
                    }
                )
                continue
            nodes.append(
                {
                    "__typename": "User",
                    "login": login,
                    "name": login.title(),
                    "followers": {"totalCount": self.followers.get(login, 0)},
                    "avatarUrl": "",
                    "url": f"https://github.com/{login}",
                }
            )
        return WerkzeugResponse(
            json.dumps(
                {
                    "data": {
                        "nodes": nodes,
                        "rateLimit": {
                            "limit": 5000,
                            "cost": round(self.points_per_user * len(ids), 4),
                            "remaining": 4999,
                            "resetAt": iso(datetime.now(tz=UTC).timestamp() + 3600),
                        },
                    },
                    "extensions": {
                        "cost": {
                            "requestedQueryCost": round(self.points_per_user * len(ids), 4),
                            "actualQueryCost": round(self.points_per_user * len(ids), 4),
                        }
                    },
                }
            ),
            content_type="application/json",
        )

    def install(self, httpserver: Any) -> None:
        self.server = httpserver
        httpserver.expect_request(f"/repos/{self.owner}/{self.repo}", method="GET").respond_with_handler(
            self._repo_meta
        )
        httpserver.expect_request("/user", method="GET").respond_with_handler(self._current_user)
        httpserver.expect_request(
            f"/repos/{self.owner}/{self.repo}/stargazers", method="GET"
        ).respond_with_handler(self._stargazers)
        httpserver.expect_request("/graphql", method="POST").respond_with_handler(self._graphql)
        httpserver.expect_request(re.compile(r"^/users/.+$"), method="GET").respond_with_handler(
            self._user
        )
        httpserver.expect_request(re.compile(r"^/orgs/.+$"), method="GET").respond_with_handler(
            self._org
        )

    @property
    def api_base(self) -> str:
        return self.server.url_for("/").rstrip("/")

    def counts(self, prefix: str = "") -> int:
        return sum(1 for call in self.calls if call.startswith(prefix))


@pytest.fixture
def fake_github(httpserver: Any) -> FakeGitHub:
    fake = FakeGitHub(server=httpserver)
    fake.install(httpserver)
    return fake


@pytest.fixture
def cache_store(tmp_path: Path) -> CacheStore:
    return CacheStore(tmp_path / "cache")


@pytest.fixture
def make_client(fake_github: FakeGitHub):
    def factory(token: str | None = "ghp_test_token_1234", **kwargs: Any) -> GitHubClient:
        kwargs.setdefault("transport", UrllibTransport())
        kwargs.setdefault("log", lambda _msg: None)
        return GitHubClient(token=token, api_base=fake_github.api_base, **kwargs)

    return factory
