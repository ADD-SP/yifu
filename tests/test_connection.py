from __future__ import annotations

import re

import pytest
from conftest import FakeGitHub
from test_pipeline import synthetic_history

from yifu import github as github_module
from yifu.analyze import AnalyzeOptions, run_analysis
from yifu.github import (
    CacheStore,
    fetch_repo_meta,
    fetch_stargazers,
    walk_stargazers_connection,
)


def test_connection_query_is_schema_shaped() -> None:
    query = re.sub(r"\s+", " ", github_module._CONNECTION_STARGAZERS_QUERY)
    assert "stargazers(first: $pageSize, after: $cursor" in query
    assert "orderBy: {field: STARRED_AT, direction: ASC}" in query
    assert "starredAt" in query
    assert "followers { totalCount }" in query


def test_walk_collects_timeline_and_followers_in_one_pass(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(250)]
    fake_github.setup_method_like(stars, {login: index for index, (login, _t) in enumerate(stars)})
    client = make_client()
    walk = walk_stargazers_connection(client, cache_store, "owner", "repo")

    assert walk.accepted and walk.exhausted
    assert len(walk.stargazers) == 250
    assert [item.login for item in walk.stargazers] == [login for login, _t in stars]
    assert len(walk.profiles) == 250
    assert walk.profiles["user007"].followers == 7
    assert walk.pages == 3
    # 1 point per 100-star page => 0.01 points per user, well under the threshold.
    assert walk.cost_per_user == pytest.approx(0.01, rel=0.05)
    # The REST stargazers listing is not touched at all on this path.
    assert fake_github.counts("stargazers:") == 0


def test_walk_bails_out_after_one_page_when_billed_per_node(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(250)]
    fake_github.setup_method_like(stars, {login: 1 for login, _t in stars})
    # 100 points per page of 100 users == 1 point/user, i.e. no better than REST.
    fake_github.connection_points_per_page = 100.0
    client = make_client()
    walk = walk_stargazers_connection(client, cache_store, "owner", "repo")

    assert walk.accepted is False
    assert walk.reason == "too-expensive"
    assert walk.cost_per_user == pytest.approx(1.0)
    assert fake_github.counts("connection:") == 1  # exactly one probe page
    # The probe page's data is kept, so the spend was not wasted.
    assert len(walk.stargazers) == 100
    assert len(walk.profiles) == 100


def test_walk_resumes_from_cache(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(150)]
    fake_github.setup_method_like(stars, {login: 5 for login, _t in stars})
    client = make_client()
    first = walk_stargazers_connection(client, cache_store, "owner", "repo")
    assert first.accepted and len(first.stargazers) == 150
    calls_before = fake_github.counts("connection:")
    second = walk_stargazers_connection(make_client(), cache_store, "owner", "repo")
    assert fake_github.counts("connection:") == calls_before, "cached pages must not refetch"
    assert [item.login for item in second.stargazers] == [item.login for item in first.stargazers]
    assert second.reason == "cached"


def test_walk_reports_query_errors(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(20)]
    fake_github.setup_method_like(stars, {login: 1 for login, _t in stars})
    fake_github.connection_enabled = False
    walk = walk_stargazers_connection(make_client(), cache_store, "owner", "repo")
    assert walk.accepted is False
    assert walk.reason == "query-failed"
    assert walk.error is not None


def test_null_repository_is_a_failure_not_an_empty_success(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """GitHub answers with repository: null + errors when it cannot see the repo."""

    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(30)]
    fake_github.setup_method_like(stars, {login: 1 for login, _t in stars})
    fake_github.connection_null_repository = True
    walk = walk_stargazers_connection(make_client(), cache_store, "owner", "repo")
    assert walk.accepted is False
    assert walk.reason == "repository-unresolved"
    assert "Could not resolve" in (walk.error or "")
    assert walk.cost_per_user is None  # never "0.0000 points per user"


def test_empty_connection_page_never_looks_infinitely_cheap(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(30)]
    fake_github.setup_method_like(stars, {login: 1 for login, _t in stars})
    fake_github.connection_empty_page = True
    client = make_client()
    # Real runs fetch the repository metadata first; the walk uses it to notice
    # that GitHub reports stars while the connection hands over none.
    fetch_repo_meta(client, cache_store, "owner", "repo")
    walk = walk_stargazers_connection(client, cache_store, "owner", "repo")
    assert walk.accepted is False
    assert walk.reason == "empty"
    assert walk.cost_per_user is None
    # The message names the real cause instead of blaming the credential.
    assert "2026 年 7 月" in (walk.error or "")


def test_stargazers_404_reports_the_documented_access_restriction(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """GitHub limits stargazer listings to admins/collaborators since 2026-07."""

    fake_github.setup_method_like([("someone", 1_700_000_000.0)], {"someone": 3})
    fake_github.collaborator = False
    fake_github.stargazers_missing = True
    with pytest.raises(github_module.StargazersRestricted) as excinfo:
        fetch_stargazers(make_client(), cache_store, "owner", "repo")
    message = str(excinfo.value)
    assert "本身可以读取" in message
    assert "2026 年 7 月" in message
    assert "仅对仓库管理员与协作者开放" in message
    assert "没有协作者权限" in message
    assert "rest/activity/starring#new-access-restrictions" in message
    assert "github.blog/changelog/2026-06-30" in message


def test_stargazers_404_for_a_collaborator_says_something_else(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    fake_github.setup_method_like([("someone", 1_700_000_000.0)], {"someone": 3})
    fake_github.stargazers_missing = True
    with pytest.raises(github_module.StargazersRestricted) as excinfo:
        fetch_stargazers(make_client(), cache_store, "owner", "repo")
    message = str(excinfo.value)
    assert "有协作者权限" in message
    assert "--refresh" in message


def test_stargazers_404_with_unreadable_repo_stays_repo_not_found(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    fake_github.setup_method_like([("someone", 1_700_000_000.0)], {"someone": 3})
    fake_github.stargazers_missing = True
    fake_github.repo_missing = True
    with pytest.raises(github_module.RepoNotFound):
        fetch_stargazers(make_client(), cache_store, "owner", "repo")


def test_invalid_token_fails_fast_with_actionable_hints(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    fake_github.setup_method_like([("someone", 1_700_000_000.0)], {"someone": 3})
    fake_github.auth_invalid = True
    client = make_client()
    with pytest.raises(github_module.AuthError) as excinfo:
        run_analysis(client, cache_store, "owner", "repo", AnalyzeOptions())
    message = str(excinfo.value)
    assert "/user" in message
    assert "gh auth status" in message
    assert "unset GITHUB_TOKEN" in message
    assert fake_github.counts("stargazers:") == 0  # no work attempted


def test_stale_empty_connection_cache_is_discarded_and_refetched(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """Older versions cached an empty page; it must not look like "finished"."""

    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(120)]
    fake_github.setup_method_like(stars, {login: 7 for login, _t in stars})
    pages_dir = cache_store.repo_dir("owner", "repo") / "graphql_stargazers"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_store.write_json(
        pages_dir / "p00000.json",
        {
            "index": 0,
            "cursor": None,
            "end_cursor": None,
            "has_next_page": False,
            "total_count": 0,
            "cost": 1.0,
            "fetched_at": 0,
            "edges": [],
            "skipped": 0,
        },
    )
    walk = walk_stargazers_connection(make_client(), cache_store, "owner", "repo")
    assert walk.accepted is True
    assert len(walk.stargazers) == 120
    assert walk.reason != "cached"


def test_rate_limit_cost_field_is_used_when_extensions_are_missing(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(120)]
    fake_github.setup_method_like(stars, {login: 2 for login, _t in stars})
    fake_github.connection_cost_source = "ratelimit"
    walk = walk_stargazers_connection(make_client(), cache_store, "owner", "repo")
    assert walk.accepted is True
    assert walk.cost_per_user == pytest.approx(1.0 / 100, rel=0.1)


def test_unknown_cost_is_conservatively_rejected(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """No cost information means we cannot prove the fast path is cheap."""

    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(120)]
    fake_github.setup_method_like(stars, {login: 2 for login, _t in stars})
    fake_github.connection_cost_source = "none"
    walk = walk_stargazers_connection(make_client(), cache_store, "owner", "repo")
    assert walk.accepted is False
    assert walk.reason == "cost-unknown"
    assert fake_github.counts("connection:") == 1


def test_run_analysis_prefers_the_connection_path_when_cheap(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    client = make_client()
    payload = run_analysis(
        client, cache_store, "owner", "repo", AnalyzeOptions(candidate_pool=50, prune_verify=0)
    )
    assert payload["fetch"]["connection"]["accepted"] is True
    assert payload["pruning"]["pruned"] is False
    assert payload["pruning"]["candidate_count"] == len(stars)
    assert payload["influencers"][0]["login"] == "influencer"
    assert fake_github.counts("stargazers:") == 0


def test_run_analysis_falls_back_to_pruned_rest_path_when_expensive(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    fake_github.connection_points_per_page = 100.0
    client = make_client()
    payload = run_analysis(
        client, cache_store, "owner", "repo", AnalyzeOptions(candidate_pool=50, prune_verify=0)
    )
    assert payload["fetch"]["connection"]["accepted"] is False
    assert payload["pruning"]["pruned"] is True
    assert payload["pruning"]["candidate_count"] == 50
    assert fake_github.counts("stargazers:") > 0  # REST timeline used instead
    assert payload["influencers"][0]["login"] == "influencer"


def test_organizations_in_the_connection_are_topped_up_via_rest(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [("acme", 1_700_000_000.0), ("human", 1_700_000_100.0), ("other", 1_700_000_200.0)]
    fake_github.setup_method_like(stars, {"acme": 9000, "human": 40, "other": 12})
    fake_github.organizations.add("acme")
    client = make_client()
    payload = run_analysis(
        client, cache_store, "owner", "repo", AnalyzeOptions(candidate_pool=10, prune_verify=0)
    )
    assert payload["fetch"]["connection"]["accepted"] is True
    assert payload["users"]["acme"]["followers"] == 9000
    assert "org:acme" in fake_github.calls
    assert payload["pruning"]["candidate_count"] == 3


def test_run_analysis_refuses_non_collaborator_before_any_stargazer_call(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """The permission check happens up front; nothing is fetched for nothing."""

    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(40)]
    fake_github.setup_method_like(stars, {login: 5 for login, _t in stars})
    fake_github.collaborator = False
    with pytest.raises(github_module.StargazersRestricted) as excinfo:
        run_analysis(make_client(), cache_store, "owner", "repo", AnalyzeOptions())
    message = str(excinfo.value)
    assert "没有协作者权限" in message
    assert "rest/activity/starring#new-access-restrictions" in message
    assert fake_github.counts("stargazers:") == 0
    assert fake_github.counts("connection:") == 0


def test_run_analysis_proceeds_when_permissions_are_not_reported(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """Enterprises may omit permissions; then the API call itself decides."""

    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(40)]
    fake_github.setup_method_like(stars, {login: 5 for login, _t in stars})
    fake_github.report_permissions = False
    payload = run_analysis(
        make_client(), cache_store, "owner", "repo", AnalyzeOptions(prune_verify=0)
    )
    assert payload["stargazers"]["count"] == 40
