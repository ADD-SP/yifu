"""Orchestration: fetch → prune → attribute → assemble payloads."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import attribution as attr
from .github import (
    CacheStore,
    ConnectionWalk,
    GitHubClient,
    ProfileFetcher,
    Stargazer,
    StargazersRestricted,
    collaborator_required_message,
    collaborator_status,
    fetch_repo_meta,
    fetch_stargazers,
    iso_utc,
    load_user_cache,
    mask_token,
    walk_stargazers_connection,
)
from .progress import ProgressCallback, ProgressUpdate

SCHEMA_VERSION = 1
REST_HOURLY_LIMIT = 5000
GRAPHQL_HOURLY_POINTS = 5000


@dataclass(slots=True)
class AnalyzeOptions:
    window_hours: float = 24.0
    windows_hours: tuple[float, ...] = (6.0, 24.0, 72.0)
    candidate_pool: int = 3000
    prune: bool = True
    prune_verify: int = 300
    workers: int = 8
    fetch_mode: str = "auto"
    attribution_k: int = 20
    spike_top_k: int = 5
    spike_alpha: float = attr.DEFAULT_SPIKE_ALPHA
    max_users: int | None = None
    since: float | None = None
    until: float | None = None
    refresh: bool = False
    user_ttl_days: float = 7.0

    def window_seconds_set(self) -> tuple[float, ...]:
        windows = {float(self.window_hours)}
        windows.update(float(value) for value in self.windows_hours)
        return tuple(sorted(windows))

    def to_json(self) -> dict[str, Any]:
        return {
            "window_hours": self.window_hours,
            "windows_hours": list(self.window_seconds_set()),
            "candidate_pool": self.candidate_pool,
            "prune": self.prune,
            "prune_verify": self.prune_verify,
            "workers": self.workers,
            "fetch_mode": self.fetch_mode,
            "attribution_k": self.attribution_k,
            "spike_top_k": self.spike_top_k,
            "spike_alpha": self.spike_alpha,
            "max_users": self.max_users,
            "since": iso_utc(self.since) if self.since else None,
            "until": iso_utc(self.until) if self.until else None,
        }


@dataclass(slots=True)
class FreeSignals:
    stargazers: list[Stargazer]
    times: list[float]
    logins: list[str]
    node_ids: dict[str, str | None]
    user_types: dict[str, str | None]
    deltas: list[int]
    events: list[attr.SpikeEvent]
    plan: attr.PrunePlan


def filter_stargazers(
    stargazers: Sequence[Stargazer],
    since: float | None,
    until: float | None,
) -> list[Stargazer]:
    selected = [
        item
        for item in stargazers
        if (since is None or item.starred_at >= since) and (until is None or item.starred_at <= until)
    ]
    return selected


def derive_free_signals(
    stargazers: Sequence[Stargazer],
    options: AnalyzeOptions,
) -> FreeSignals:
    """Everything that only needs star timestamps: deltas, spikes, candidates."""

    ordered = sorted(stargazers, key=lambda item: item.starred_at)
    times = [item.starred_at for item in ordered]
    logins = [item.login for item in ordered]
    node_ids = {item.login: item.node_id for item in ordered}
    user_types = {item.login: item.user_type for item in ordered}
    window_seconds = options.window_hours * 3600.0
    deltas = attr.compute_deltas(times, window_seconds)
    events = attr.detect_spikes(
        times,
        window_seconds=window_seconds,
        alpha=options.spike_alpha,
    )
    pool_size = options.candidate_pool
    if options.max_users is not None:
        pool_size = min(pool_size, options.max_users)
    if options.prune:
        plan = attr.select_candidates(
            times, logins, deltas, events, window_seconds, pool_size=pool_size
        )
    else:
        plan = attr.PrunePlan(
            pool_size=pool_size,
            total=len(logins),
            candidates=list(dict.fromkeys(logins)),
            delta_candidates=len(logins),
            spike_candidates=0,
            pruned=False,
        )
    return FreeSignals(
        stargazers=list(ordered),
        times=times,
        logins=logins,
        node_ids=node_ids,
        user_types=user_types,
        deltas=deltas,
        events=events,
        plan=plan,
    )


def estimate_work(
    candidate_count: int,
    cache_hits: int,
    *,
    authenticated: bool,
    graphql_cost_per_user: float | None = None,
) -> dict[str, Any]:
    """Rough wall-clock estimate for the follower lookups that are still missing."""

    missing = max(0, candidate_count - cache_hits)
    cheap_cost = graphql_cost_per_user if graphql_cost_per_user is not None else 0.01
    rest_hours = missing / REST_HOURLY_LIMIT
    graphql_hours = missing * cheap_cost / GRAPHQL_HOURLY_POINTS
    if not authenticated:
        rest_hours = missing / 60.0
        graphql_hours = float("inf")
    return {
        "missing_profiles": missing,
        "cache_hits": cache_hits,
        "rest_requests": missing,
        "rest_seconds": round(rest_hours * 3600, 1),
        "graphql_points": round(missing * cheap_cost, 2),
        "graphql_seconds": None if math.isinf(graphql_hours) else round(graphql_hours * 3600, 1),
        "authenticated": authenticated,
    }


def _is_bot(login: str, user_type: str | None) -> bool:
    lowered = login.lower()
    if user_type and user_type.lower() == "bot":
        return True
    return lowered.endswith(("[bot]", "-bot", "_bot"))


def user_rows(
    signals: FreeSignals,
    profiles: Mapping[str, Any],
    candidates: Sequence[str],
    results: Mapping[float, attr.AttributionResult],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    candidate_set = set(candidates)
    primary = max(results)
    credit_primary = results[primary].credit
    ranked = sorted(credit_primary.items(), key=lambda item: (-item[1], item[0]))
    rows: list[dict[str, Any]] = []
    by_login = {item.login: item for item in signals.stargazers}
    for rank, (login, credit) in enumerate(ranked, start=1):
        if limit is not None and len(rows) >= limit:
            break
        profile = profiles.get(login)
        stargazer = by_login.get(login)
        entry: dict[str, Any] = {
            "rank": rank,
            "login": login,
            "followers": getattr(profile, "followers", None),
            "following": getattr(profile, "following", None),
            "public_repos": getattr(profile, "public_repos", None),
            "user_type": getattr(profile, "user_type", None)
            or (stargazer.user_type if stargazer else None),
            "name": getattr(profile, "name", None),
            "html_url": getattr(profile, "html_url", None)
            or (stargazer.html_url if stargazer else f"https://github.com/{login}"),
            "avatar_url": getattr(profile, "avatar_url", None)
            or (stargazer.avatar_url if stargazer else None),
            "is_bot": _is_bot(login, getattr(profile, "user_type", None)),
            "followers_source": getattr(profile, "source", "unknown") if profile else "pruned",
            "is_candidate": login in candidate_set,
            "starred_at": stargazer.starred_at if stargazer else None,
            "starred_at_iso": stargazer.starred_at_iso if stargazer else None,
            "credit": {str(int(window / 3600)): round(results[window].credit.get(login, 0.0), 4) for window in sorted(results)},
            "direct_lead": {
                str(int(window / 3600)): results[window].direct_leads.get(login, 0)
                for window in sorted(results)
            },
            "wins": {
                str(int(window / 3600)): results[window].wins.get(login, 0)
                for window in sorted(results)
            },
            "stable": False,
            "spike_share": 0.0,
        }
        rows.append(entry)
    return rows


def run_analysis(
    client: GitHubClient,
    cache: CacheStore,
    owner: str,
    repo: str,
    options: AnalyzeOptions,
    *,
    progress: ProgressCallback = lambda _update: None,
    log: Callable[[str], None] = lambda _msg: None,
) -> dict[str, Any]:
    started = time.time()
    identity: dict[str, Any] = {}
    if client.token:
        identity = client.whoami()
    meta = fetch_repo_meta(client, cache, owner, repo, refresh=options.refresh)
    status = collaborator_status(meta.get("permissions"))
    if status is False:
        raise StargazersRestricted(collaborator_required_message(owner, repo, meta))
    # One line for identity + permission; the rest is reported only on failure.
    brief: list[str] = []
    if identity:
        scopes = identity.get("scopes") or "fine-grained token"
        brief.append(f"认证 {identity.get('login') or '未知账号'}（{scopes}）")
    if status is True:
        brief.append("协作者权限 ✓")
    elif status is None:
        brief.append("权限未知（GitHub 未返回 permissions）")
    if meta.get("stargazers_count") is not None:
        brief.append(f"仓库 {meta.get('stargazers_count')} stars")
    log(" · ".join(brief))

    walk: ConnectionWalk | None = None
    if client.token and options.fetch_mode in ("auto", "connection"):
        walk = walk_stargazers_connection(
            client,
            cache,
            owner,
            repo,
            max_users=options.max_users,
            until=options.until,
            refresh=options.refresh,
            progress=progress,
            log=log,
        )
        if not walk.accepted:
            detail = walk.error or "未知原因"
            prefix = f"{walk.reason}：" if walk.reason else ""
            log(
                f"GraphQL 连接不可用（{prefix}{detail}），"
                "回退 REST 时间线 + 剪枝"
            )

    if walk is not None and walk.accepted and walk.stargazers:
        stargazers = filter_stargazers(walk.stargazers, options.since, options.until)
        if not stargazers:
            raise ValueError("筛选后没有可分析的 stargazers")
        signals = derive_free_signals(stargazers, options)
        known = dict(walk.profiles)
        candidate_logins = [login for login in dict.fromkeys(signals.logins) if login in known]
        plan = attr.PrunePlan(
            pool_size=signals.plan.pool_size,
            total=signals.plan.total,
            candidates=candidate_logins,
            delta_candidates=signals.plan.delta_candidates,
            spike_candidates=signals.plan.spike_candidates,
            pruned=False,
            inside_event=signals.plan.inside_event,
        )
        profiles: dict[str, Any] = {
            login: known[login] for login in candidate_logins
        }
        fetcher = ProfileFetcher(
            client, cache, workers=options.workers, mode="rest", log=log
        )

        org_logins = [
            item.login
            for item in stargazers
            if item.user_type == "Organization" and item.login not in profiles
        ]
        if org_logins:
            log(f"GraphQL 连接不返回组织账号粉丝数，改用 REST 补 {len(org_logins)} 个组织")
            extra = fetcher.fetch(
                org_logins,
                {login: None for login in org_logins},
                user_types={login: "Organization" for login in org_logins},
                progress=progress,
            )
            profiles.update(extra)
            candidate_logins = [
                login for login in dict.fromkeys(signals.logins) if login in profiles
            ]
            plan = attr.PrunePlan(
                pool_size=signals.plan.pool_size,
                total=signals.plan.total,
                candidates=candidate_logins,
                delta_candidates=signals.plan.delta_candidates,
                spike_candidates=signals.plan.spike_candidates,
                pruned=False,
                inside_event=signals.plan.inside_event,
            )
    else:
        stargazers = fetch_stargazers(
            client,
            cache,
            owner,
            repo,
            refresh=options.refresh,
            progress=progress,
            total_hint=meta.get("stargazers_count"),
        )
        stargazers = filter_stargazers(stargazers, options.since, options.until)
        if not stargazers:
            raise ValueError("筛选后没有可分析的 stargazers")
        signals = derive_free_signals(stargazers, options)
        plan = signals.plan
        fetcher = ProfileFetcher(
            client, cache, workers=options.workers, mode=options.fetch_mode, log=log
        )
        ttl = int(options.user_ttl_days * 86400) if options.user_ttl_days >= 0 else -1
        ttl = -1 if options.refresh else ttl
        cached_profiles, _missing = load_user_cache(cache, plan.candidates, ttl_seconds=ttl)
        work = estimate_work(
            len(plan.candidates),
            len(cached_profiles),
            authenticated=bool(client.token),
            graphql_cost_per_user=client.graphql_cost_per_user,
        )
        log(
            f"待查粉丝数：{work['missing_profiles']} 个账号"
            f"（缓存命中 {work['cache_hits']}，预计约 {work['rest_seconds'] / 60:.1f} 分钟）"
        )
        profiles = fetcher.fetch(
            plan.candidates,
            signals.node_ids,
            user_types=signals.user_types,
            refresh=options.refresh,
            progress=progress,
        )

    verification: dict[str, Any] | None = None
    candidate_set = set(plan.candidates)
    if plan.pruned and options.prune_verify > 0:
        pruned_logins = [login for login in dict.fromkeys(signals.logins) if login not in candidate_set]
        if pruned_logins:
            sample_size = min(options.prune_verify, len(pruned_logins))
            rng = random.Random(20240920)
            sample = rng.sample(pruned_logins, sample_size)
            log(f"抽样校验 {sample_size} 个未查询账号，用于估计剪枝遗漏")
            sampled = fetcher.fetch(
                sample,
                signals.node_ids,
                user_types=signals.user_types,
                progress=lambda update: progress(
                    ProgressUpdate("verify", update.done, update.total, update.detail)
                ),
            )
            verification = attr.summarize_verification(
                [(login, getattr(sampled.get(login), "followers", None)) for login in sample],
                len(pruned_logins),
            )
            cache.write_json(cache.verification_path(owner, repo), verification)

    followers = {
        login: getattr(profile, "followers", None)
        for login, profile in profiles.items()
        if login in candidate_set
    }
    results: dict[float, attr.AttributionResult] = {}
    for window_hours in options.window_seconds_set():
        window_seconds = window_hours * 3600.0
        deltas = signals.deltas if abs(window_hours - options.window_hours) < 1e-9 else None
        results[window_seconds] = attr.attribute_stars(
            signals.times,
            signals.logins,
            followers,
            window_seconds,
            top_k=options.attribution_k,
            deltas=deltas,
        )
    attr.attribute_spikes(
        signals.events,
        signals.times,
        signals.logins,
        followers,
        options.window_hours * 3600.0,
        top_k=options.spike_top_k,
    )

    payload = assemble_payload(
        owner=owner,
        repo=repo,
        meta=meta,
        options=options,
        signals=signals,
        plan=plan,
        profiles=profiles,
        results=results,
        verification=verification,
        fetch_info={
            "authenticated": bool(client.token),
            "token": mask_token(client.token),
            "requests": client.requests,
            "rate_remaining": client.rate_remaining,
            "rate_limit": client.rate_limit,
            "graphql_cost_per_user": client.graphql_cost_per_user,
            "graphql_probe_error": fetcher.graphql_probe_error,
            "connection": walk.to_json() if walk is not None else None,
            "profile_sources": fetcher.counts,
            "profile_misses": fetcher.misses,
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    cache.write_json(cache.analysis_path(owner, repo), payload)
    return payload


def assemble_payload(
    *,
    owner: str,
    repo: str,
    meta: Mapping[str, Any],
    options: AnalyzeOptions,
    signals: FreeSignals,
    plan: attr.PrunePlan,
    profiles: Mapping[str, Any],
    results: Mapping[float, attr.AttributionResult],
    verification: Mapping[str, Any] | None,
    fetch_info: Mapping[str, Any],
    generated_at: float | None = None,
) -> dict[str, Any]:
    rows = user_rows(signals, profiles, plan.candidates, results, limit=500)
    sensitivity = attr.build_sensitivity(results, top_n=10)
    stable = set(sensitivity["stable_users"])
    for row in rows:
        row["stable"] = row["login"] in stable
    spike_shares: dict[str, float] = {}
    for event in signals.events:
        for credit in event.attributed:
            spike_shares[credit["login"]] = spike_shares.get(credit["login"], 0.0) + float(
                credit["share"]
            )
    for row in rows:
        row["spike_share"] = round(spike_shares.get(row["login"], 0.0), 3)

    primary_seconds = options.window_hours * 3600.0
    primary_result = results[primary_seconds]
    top_credit = sum(credit for _login, credit in attr.rank_of(primary_result.credit)[:10])
    span_days = (signals.times[-1] - signals.times[0]) / 86400.0 if len(signals.times) > 1 else 0.0
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": iso_utc(time.time() if generated_at is None else generated_at),
        "repo": {
            "owner": owner,
            "name": repo,
            "full_name": meta.get("full_name") or f"{owner}/{repo}",
            "html_url": meta.get("html_url") or f"https://github.com/{owner}/{repo}",
            "description": meta.get("description"),
            "stargazers_count": meta.get("stargazers_count"),
            "permissions": meta.get("permissions"),
        },
        "params": options.to_json(),
        "fetch": dict(fetch_info),
        "stargazers": {
            "count": len(signals.times),
            "first_star": iso_utc(signals.times[0]),
            "last_star": iso_utc(signals.times[-1]),
            "first_star_epoch": signals.times[0],
            "last_star_epoch": signals.times[-1],
            "span_days": round(span_days, 2),
            "daily_average": round(len(signals.times) / span_days, 3) if span_days > 0 else None,
            "delta_stats": {
                str(int(options.window_hours)): {
                    "max": max(signals.deltas),
                    "p50": round(statistics_median(signals.deltas), 2),
                    "p95": round(percentile(signals.deltas, 0.95), 2),
                }
            },
        },
        "pruning": {
            **plan.to_json(),
            "unattributed_stars": round(primary_result.unattributed, 2),
            "attributed_ratio": round(primary_result.attributed_ratio, 6),
        },
        "verification": verification,
        "summary": {
            "total_stars": primary_result.total_stars,
            "unattributed_stars": round(primary_result.unattributed, 2),
            "top10_credit_share": round(top_credit / max(1.0, sum(primary_result.credit.values())), 6),
            "credit_total": round(sum(primary_result.credit.values()), 2),
            "spike_events": len(signals.events),
            "candidates": len(plan.candidates),
        },
        "users": {login: profile.to_json() for login, profile in profiles.items()},
        "candidates": list(plan.candidates),
        "influencers": rows,
        "events": [event.to_json() for event in signals.events],
        "sensitivity": sensitivity,
        "attribution": {
            str(int(window / 3600)): {
                "credit": {login: round(value, 4) for login, value in result.credit.items()},
                "wins": dict(result.wins),
                "direct_leads": result.direct_leads,
                "unattributed": round(result.unattributed, 2),
                "attributed_ratio": round(result.attributed_ratio, 6),
            }
            for window, result in results.items()
        },
    }
    return payload


def statistics_median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])
