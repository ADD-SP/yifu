from __future__ import annotations

import re

import pytest
from conftest import FakeGitHub, iso

from yifu import github as github_module
from yifu.github import (
    AuthError,
    CacheStore,
    GitHubClient,
    GitHubError,
    PartialData,
    ProfileFetcher,
    QuotaExhausted,
    RepoNotFound,
    Response,
    fetch_repo_meta,
    fetch_stargazers,
    load_user_cache,
    mask_token,
    parse_repo_url,
)

DAY = 86400.0


@pytest.mark.parametrize(
    "value,expected",
    [
        ("owner/repo", ("owner", "repo")),
        ("https://github.com/owner/repo", ("owner", "repo")),
        ("https://github.com/owner/repo.git", ("owner", "repo")),
        ("https://github.com/owner/repo/", ("owner", "repo")),
        ("https://github.com/owner/repo/tree/main/src", ("owner", "repo")),
        ("git@github.com:owner/repo.git", ("owner", "repo")),
        ("github.com/owner/repo", ("owner", "repo")),
    ],
)
def test_parse_repo_url(value: str, expected: tuple[str, str]) -> None:
    assert parse_repo_url(value) == expected


@pytest.mark.parametrize("value", ["", "just-a-name", "https://gitlab.com/owner/repo", "owner/re po"])
def test_parse_repo_url_rejects_bad_input(value: str) -> None:
    with pytest.raises(GitHubError):
        parse_repo_url(value)


def test_mask_token() -> None:
    assert mask_token(None) is None
    assert mask_token("short") == "*****"
    assert mask_token("ghp_abcdefghijkl") == "ghp_…ijkl"


def test_fetch_stargazers_paginates_and_reuses_etags(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(250)]
    fake_github.setup_method_like(stars, {login: index for index, (login, _t) in enumerate(stars)})
    client = make_client()
    first = fetch_stargazers(client, cache_store, "owner", "repo")
    assert [item.login for item in first] == [login for login, _t in stars]
    # A short final page ends the walk, so no empty sentinel request is needed.
    assert fake_github.counts("stargazers:") == 3

    client_again = make_client()
    second = fetch_stargazers(client_again, cache_store, "owner", "repo")
    assert [item.login for item in second] == [item.login for item in first]
    assert fake_github.not_modified >= 3
    cached = load_user_cache(cache_store, ["user000"])
    assert cached[1] == ["user000"]


def test_fetch_stargazers_probes_one_page_past_an_exact_multiple(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(200)]
    fake_github.setup_method_like(stars, {login: 3 for login, _t in stars})
    fetched = fetch_stargazers(make_client(), cache_store, "owner", "repo")
    assert len(fetched) == 200
    assert fake_github.counts("stargazers:") == 3  # two full pages plus an empty one


def test_fetch_repo_meta_reports_missing_repo(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    fake_github.repo_missing = True
    with pytest.raises(RepoNotFound):
        fetch_repo_meta(make_client(), cache_store, "owner", "repo")


def test_auth_failure_maps_to_auth_error() -> None:
    def transport(method, url, headers, data, timeout):
        return Response(401, {}, "{}")

    client = GitHubClient(token="bad", transport=transport, log=lambda _m: None)
    with pytest.raises(AuthError):
        client.request("GET", "/repos/owner/repo")


def test_quota_exhaustion_waits_then_raises_when_no_wait() -> None:
    slept: list[float] = []

    def transport(method, url, headers, data, timeout):
        return Response(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1000"}, "{}")

    client = GitHubClient(
        token="t",
        transport=transport,
        sleep=slept.append,
        now=lambda: 900.0,
        no_wait=True,
        log=lambda _m: None,
    )
    with pytest.raises(QuotaExhausted):
        client.request("GET", "/repos/owner/repo")
    assert slept == []


def test_quota_exhaustion_waits_for_reset_when_allowed() -> None:
    slept: list[float] = []
    state = {"calls": 0}

    def transport(method, url, headers, data, timeout):
        state["calls"] += 1
        if state["calls"] == 1:
            return Response(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1000"}, "{}")
        return Response(200, {}, '{"ok": true}')

    client = GitHubClient(
        token="t",
        transport=transport,
        sleep=slept.append,
        now=lambda: 900.0,
        log=lambda _m: None,
    )
    response = client.request("GET", "/repos/owner/repo")
    assert response.status == 200
    assert slept and slept[0] >= 100


def test_graphql_batches_profiles_and_caches_them(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(250)]
    fake_github.setup_method_like(stars, {login: 100 - index for index, (login, _t) in enumerate(stars)})
    client = make_client()
    fetch_stargazers(client, cache_store, "owner", "repo")
    from yifu.github import load_cached_stargazers

    stargazers = load_cached_stargazers(cache_store, "owner", "repo")
    node_ids = {item.login: item.node_id for item in stargazers}
    fetcher = ProfileFetcher(client, cache_store, workers=4, mode="auto", log=lambda _m: None)
    profiles = fetcher.fetch([item.login for item in stargazers], node_ids)
    assert len(profiles) == 250
    assert fetcher.counts["graphql"] == 250
    assert fetcher.counts["rest"] == 0
    assert client.graphql_cost_per_user == pytest.approx(0.0005)
    graphql_calls = [call for call in fake_github.calls if call.startswith("graphql:")]
    # Batches are fetched concurrently, so only the batch sizes are deterministic.
    assert sorted(graphql_calls) == ["graphql:100", "graphql:100", "graphql:50"]


def test_graphql_falls_back_to_rest_when_too_expensive(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(120)]
    fake_github.setup_method_like(stars, {login: 10 for login, _t in stars})
    fake_github.points_per_user = 1.0
    client = make_client()
    fetch_stargazers(client, cache_store, "owner", "repo")
    from yifu.github import load_cached_stargazers

    stargazers = load_cached_stargazers(cache_store, "owner", "repo")
    node_ids = {item.login: item.node_id for item in stargazers}
    fetcher = ProfileFetcher(client, cache_store, workers=4, mode="auto", log=lambda _m: None)
    profiles = fetcher.fetch([item.login for item in stargazers], node_ids)
    assert len(profiles) == 120
    assert fetcher.counts["rest"] == 120
    assert fake_github.counts("user:") == 120


def test_graphql_disabled_falls_back_to_rest(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(5)]
    fake_github.setup_method_like(stars, {login: 7 for login, _t in stars})
    fake_github.graphql_enabled = False
    client = make_client()
    fetch_stargazers(client, cache_store, "owner", "repo")
    from yifu.github import load_cached_stargazers

    stargazers = load_cached_stargazers(cache_store, "owner", "repo")
    node_ids = {item.login: item.node_id for item in stargazers}
    fetcher = ProfileFetcher(client, cache_store, workers=2, mode="auto", log=lambda _m: None)
    profiles = fetcher.fetch([item.login for item in stargazers], node_ids)
    assert len(profiles) == 5
    assert fetcher.counts["rest"] == 5


def test_missing_users_are_marked_unavailable(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [("ghost", 1_700_000_000), ("real", 1_700_000_100)]
    fake_github.setup_method_like(stars, {"real": 42})
    fake_github.missing_users.add("ghost")
    client = make_client()
    fetch_stargazers(client, cache_store, "owner", "repo")
    from yifu.github import load_cached_stargazers

    stargazers = load_cached_stargazers(cache_store, "owner", "repo")
    node_ids = {item.login: item.node_id for item in stargazers}
    fetcher = ProfileFetcher(client, cache_store, workers=2, mode="rest", log=lambda _m: None)
    profiles = fetcher.fetch(["ghost", "real"], node_ids)
    assert profiles["ghost"].source == "unavailable"
    assert profiles["real"].followers == 42
    assert fetcher.counts["unavailable"] == 1


def test_second_run_uses_the_user_cache(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [(f"user{i:03d}", 1_700_000_000 + i * 60) for i in range(20)]
    fake_github.setup_method_like(stars, {login: 5 for login, _t in stars})
    client = make_client()
    fetch_stargazers(client, cache_store, "owner", "repo")
    from yifu.github import load_cached_stargazers

    stargazers = load_cached_stargazers(cache_store, "owner", "repo")
    node_ids = {item.login: item.node_id for item in stargazers}
    logins = [item.login for item in stargazers]
    ProfileFetcher(client, cache_store, workers=2, mode="auto", log=lambda _m: None).fetch(
        logins, node_ids
    )
    before = fake_github.counts("graphql:") + fake_github.counts("user:")
    fetcher = ProfileFetcher(client, cache_store, workers=2, mode="auto", log=lambda _m: None)
    profiles = fetcher.fetch(logins, node_ids)
    after = fake_github.counts("graphql:") + fake_github.counts("user:")
    assert after == before
    assert fetcher.counts["cache"] == 20
    assert all(profile.followers == 5 for profile in profiles.values())


def test_graphql_query_matches_the_published_schema() -> None:
    """Guard against the schema mistakes that silently disabled batching."""

    query = github_module._FOLLOWERS_QUERY
    compact = re.sub(r"\s+", " ", query)
    # User.followers is a FollowerConnection: the count is totalCount, not total.
    assert "followers { totalCount }" in compact
    assert "followers { total }" not in compact
    organization = re.search(r"\.\.\. on Organization \{ (.*?) \}", compact)
    assert organization is not None
    # Organization has no followers field at all in the GraphQL schema.
    assert "followers" not in organization.group(1)
    assert "rateLimit { limit cost remaining resetAt }" in compact


def test_organizations_use_the_rest_org_endpoint(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars = [("acme", 1_700_000_000.0), ("human", 1_700_000_100.0)]
    fake_github.setup_method_like(stars, {"acme": 4200, "human": 30})
    fake_github.organizations.add("acme")
    client = make_client()
    fetch_stargazers(client, cache_store, "owner", "repo")
    from yifu.github import load_cached_stargazers

    stargazers = load_cached_stargazers(cache_store, "owner", "repo")
    node_ids = {item.login: item.node_id for item in stargazers}
    user_types = {item.login: item.user_type for item in stargazers}
    assert user_types["acme"] == "Organization"
    fetcher = ProfileFetcher(client, cache_store, workers=2, mode="auto", log=lambda _m: None)
    profiles = fetcher.fetch(
        [item.login for item in stargazers], node_ids, user_types=user_types
    )
    assert profiles["acme"].followers == 4200
    assert profiles["human"].followers == 30
    # Organizations never consume GraphQL batch slots and are read from /orgs/.
    assert "graphql:1" in fake_github.calls
    assert "org:acme" in fake_github.calls
    assert "user:acme" not in fake_github.calls


def test_mistyped_organization_falls_back_from_users_to_orgs(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """A stargazer typed as a user but only reachable under /orgs/ still resolves."""

    fake_github.setup_method_like([("acme", 1_700_000_000.0)], {"acme": 999})
    fake_github.organizations.add("acme")
    fake_github.users_endpoint_404_for_orgs = True
    client = make_client()
    fetch_stargazers(client, cache_store, "owner", "repo")
    from yifu.github import load_cached_stargazers

    stargazers = load_cached_stargazers(cache_store, "owner", "repo")
    node_ids = {item.login: item.node_id for item in stargazers}
    # Pretend the cached stargazer metadata lost the account type.
    user_types = {item.login: None for item in stargazers}
    fetcher = ProfileFetcher(client, cache_store, workers=1, mode="rest", log=lambda _m: None)
    profiles = fetcher.fetch(["acme"], node_ids, user_types=user_types)
    assert profiles["acme"].followers == 999
    assert profiles["acme"].source == "rest"
    assert "user:acme" in fake_github.calls and "org:acme" in fake_github.calls


def test_partial_data_exit_code_is_six() -> None:
    assert PartialData.exit_code == 6
    assert iso(0.0).endswith("Z")
