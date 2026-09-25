from __future__ import annotations

import random

import pytest
from conftest import FakeGitHub

from yifu.analyze import AnalyzeOptions, derive_free_signals, run_analysis
from yifu.github import CacheStore

HOUR = 3600.0
DAY = 24 * HOUR


def synthetic_history(seed: int = 4) -> tuple[list[tuple[str, float]], dict[str, int]]:
    rng = random.Random(seed)
    start = 1_700_000_000.0
    stars: list[tuple[str, float]] = []
    timestamp = start
    for index in range(400):
        timestamp += rng.uniform(1 * HOUR, 5 * HOUR)
        stars.append((f"user{index:04d}", timestamp))
    trigger = timestamp + 3 * HOUR
    stars.append(("influencer", trigger))
    for index in range(200):
        stars.append((f"burst{index:03d}", trigger + rng.uniform(60, 8 * HOUR)))
    timestamp = trigger + DAY
    for index in range(150):
        timestamp += rng.uniform(1 * HOUR, 6 * HOUR)
        stars.append((f"late{index:04d}", timestamp))
    stars.sort(key=lambda item: item[1])
    followers = {login: rng.randint(1, 30) for login, _ts in stars}
    followers["influencer"] = 120000
    for index in range(4):
        followers[f"burst{index:03d}"] = rng.randint(80, 300)
    return stars, followers


def test_derive_free_signals_prunes_without_any_profile_lookup(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    client = make_client()
    from yifu.github import fetch_stargazers

    stargazers = fetch_stargazers(client, cache_store, "owner", "repo")
    deltas_before = fake_github.counts("user:") + fake_github.counts("graphql:")
    signals = derive_free_signals(stargazers, AnalyzeOptions(candidate_pool=50, prune=True))
    assert signals.plan.pruned
    assert len(signals.plan.candidates) == 50
    assert signals.events, "the injected burst should be detected"
    trigger = next(item for item in stargazers if item.login == "influencer")
    assert trigger.login in signals.plan.candidates
    assert deltas_before == 0  # free pre-screening never touches the profile API


def test_run_analysis_ranks_the_influencer_first(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    client = make_client()
    options = AnalyzeOptions(
        candidate_pool=120,
        prune=True,
        prune_verify=20,
        workers=4,
        windows_hours=(6.0, 24.0, 72.0),
        # Pin the classic path: "auto" now prefers the GraphQL stargazers
        # connection, which is covered by tests/test_connection.py.
        fetch_mode="graphql",
    )
    payload = run_analysis(client, cache_store, "owner", "repo", options)
    assert payload["stargazers"]["count"] == len(stars)
    top = payload["influencers"][0]
    assert top["login"] == "influencer"
    assert top["followers"] == 120000
    assert payload["attribution"]["24"]["credit"]["influencer"] > 10
    assert payload["events"], "spike detection should fire"
    assert payload["events"][0]["attributed"][0]["login"] == "influencer"
    assert payload["verification"]["sample_size"] == 20
    assert payload["pruning"]["candidate_count"] == 120
    assert payload["sensitivity"]["stable_users"], "the top influencer should be stable"


def test_run_analysis_without_pruning_queries_every_stargazer(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    client = make_client()
    options = AnalyzeOptions(prune=False, prune_verify=0, workers=4, fetch_mode="graphql")
    payload = run_analysis(client, cache_store, "owner", "repo", options)
    assert payload["pruning"]["pruned"] is False
    assert payload["pruning"]["candidate_count"] == len(stars)
    assert payload["fetch"]["profile_sources"]["graphql"] == len(stars)


def test_payload_exposes_numeric_star_epochs(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """The report renders dates from numbers, not ISO strings.

    Mixing the two once crashed the report with "Invalid time value" and the
    failure was mislabelled as "cannot read report data".
    """

    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    client = make_client()
    payload = run_analysis(
        client, cache_store, "owner", "repo", AnalyzeOptions(candidate_pool=20, prune_verify=0)
    )
    stargazers = payload["stargazers"]
    assert isinstance(stargazers["first_star_epoch"], float)
    assert isinstance(stargazers["last_star_epoch"], float)
    assert stargazers["first_star"].endswith("Z")


def test_sensitivity_windows_are_labelled_in_hours(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    """Regression: seconds here left the slope chart permanently empty."""

    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    client = make_client()
    payload = run_analysis(
        client,
        cache_store,
        "owner",
        "repo",
        AnalyzeOptions(candidate_pool=40, prune_verify=0, windows_hours=(6.0, 24.0, 72.0)),
    )
    sensitivity = payload["sensitivity"]
    expected = [str(int(value)) for value in payload["params"]["windows_hours"]]
    assert [str(value) for value in sensitivity["windows"]] == expected
    assert str(sensitivity["primary_window"]) in expected
    assert sensitivity["users"], "expected at least one ranked user"
    for row in sensitivity["users"]:
        assert set(row["ranks"]) == set(expected)
        assert set(row["credit"]) == set(expected)
        assert any(isinstance(value, int) for value in row["ranks"].values())




def test_console_report_contains_the_essentials(
    fake_github: FakeGitHub, cache_store: CacheStore, make_client
) -> None:
    from yifu.console import render_report

    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    payload = run_analysis(
        make_client(), cache_store, "owner", "repo", AnalyzeOptions(candidate_pool=40, prune_verify=0)
    )
    text = render_report(payload, top=5, events=3)
    assert "概览" in text
    assert "可能带来更多 star 的人" in text
    assert "明显异常的时段" in text
    # The 说明 section is gone: the CLI has no switch for it any more.
    assert "说明" not in text
    assert "influencer" in text
    # The ranking table shows exactly `top` rows (boxed: header + body rows).
    table_block = text.split("可能带来更多 star 的人")[1].split("明显异常的时段")[0]
    rows = [line for line in table_block.splitlines() if line.startswith("│")]
    assert len(rows) - 1 == 5
    assert rows[1].startswith("│ 1 │ influencer")
    assert rows[0].startswith("│ # │")

    plain = render_report(payload, top=3, events=0, box=False)
    assert "│" not in plain and "┌" not in plain


def test_console_alignment_handles_wide_characters() -> None:
    from yifu.console import display_width, pad

    assert display_width("用户") == 4
    assert display_width("abc") == 3
    assert pad("用户", 10) == "用户" + " " * 6


def test_console_report_handles_missing_sections() -> None:
    from yifu.console import render_report

    payload = {
        "repo": {"full_name": "owner/repo"},
        "params": {"window_hours": 24, "windows_hours": [6, 24, 72]},
        "stargazers": {"count": 0, "first_star": "?", "last_star": "?"},
        "summary": {},
        "pruning": {},
        "fetch": {},
        "influencers": [],
        "events": [],
    }
    text = render_report(payload)
    assert "没有可归因的账号" in text
    assert "没有检测到明显异常的" in text


def _payload_with_organic(share: float) -> dict:
    return {
        "repo": {"full_name": "owner/repo"},
        "params": {"window_hours": 24, "windows_hours": [6, 24, 72]},
        "stargazers": {"count": 1000, "first_star": "2024-01-01T00:00:00Z", "last_star": "2024-06-01T00:00:00Z"},
        "summary": {"total_stars": 1000, "unattributed_stars": 1000 * share, "credit_total": 1000 * (1 - share)},
        "pruning": {},
        "fetch": {},
        "influencers": [],
        "events": [],
    }


@pytest.mark.parametrize(
    ("share", "phrase"),
    [
        (0.31, "自己一点点攒起来的呢"),
        (0.55, "全靠自己撑起来～"),
        (0.85, "自带光合作用"),
    ],
)
def test_verdict_praises_organic_growth(share: float, phrase: str) -> None:
    from yifu.console import render_report, render_verdict

    payload = _payload_with_organic(share)
    verdict = render_verdict(payload)
    assert verdict is not None
    assert phrase in verdict
    assert f"{share * 100:.1f}%" in verdict
    assert verdict in render_report(payload)
    # The praise has to be praise: never narrate the act of praising, and the
    # point is that hype is not needed (not that it was missed).
    assert "夸" not in verdict
    assert "没蹭到" not in verdict
    assert verdict.endswith(("(^_^)", "(*^_^*)", "(^o^)/"))


@pytest.mark.parametrize("share", [0.0, 0.1, 0.29])
def test_verdict_stays_quiet_below_the_threshold(share: float) -> None:
    from yifu.console import render_report, render_verdict

    payload = _payload_with_organic(share)
    assert render_verdict(payload) is None
    assert "评价" not in render_report(payload)


def test_verdict_handles_empty_payload() -> None:
    from yifu.console import render_verdict

    assert render_verdict({}) is None
    assert render_verdict({"summary": {"total_stars": 0}}) is None


def _payload_with_credits(credits: list[float]) -> dict:
    return {
        "repo": {"full_name": "owner/repo"},
        "params": {"window_hours": 24, "windows_hours": [6, 24, 72]},
        "stargazers": {"count": 100, "first_star": "2024-01-01T00:00:00Z", "last_star": "2024-06-01T00:00:00Z"},
        "summary": {"total_stars": 100, "unattributed_stars": 40.0, "credit_total": 60.0},
        "pruning": {},
        "fetch": {},
        "influencers": [
            {
                "login": f"user{index}",
                "followers": 100 - index,
                "credit": {"6": value, "24": value, "72": value},
                "wins": {"24": index},
                "starred_at": 1_700_000_000 + index,
            }
            for index, value in enumerate(credits)
        ],
        "events": [],
    }


def test_low_credit_rows_are_ignored_by_default() -> None:
    from yifu.console import render_report

    payload = _payload_with_credits([9.0, 4.0, 2.9, 1.0])
    text = render_report(payload)
    assert "user0" in text and "user1" in text
    assert "user2" not in text and "user3" not in text
    assert "只显示 ≥ 3 个的账号" in text
    assert "另有 2 个账号在 24 小时范围内不足 3 个，未列出" in text


def test_min_credit_zero_shows_everyone() -> None:
    from yifu.console import render_report

    payload = _payload_with_credits([9.0, 4.0, 2.9, 1.0])
    text = render_report(payload, min_credit=0)
    assert all(f"user{index}" in text for index in range(4))
    assert "未列出" not in text


def test_threshold_above_every_row_explains_how_to_see_more() -> None:
    from yifu.console import render_report

    payload = _payload_with_credits([2.0, 1.0])
    text = render_report(payload, min_credit=5)
    assert "没有账号在 24 小时范围内带来 5 个以上" in text
    assert "--min-credit 0" in text


def test_threshold_excludes_credit_from_the_ratios() -> None:
    """Sub-threshold accounts are not "attributed": their credit is dropped."""
    from yifu.console import render_overview, render_report, render_verdict

    payload = _payload_with_credits([9.0, 4.0, 2.9, 1.0])
    payload["summary"]["credit_total"] = 16.9
    payload["summary"]["unattributed_stars"] = 83.1

    # 9.0 + 4.0 survive the 3.0 threshold; the other 2.9 + 1.0 are dropped.
    overview = "\n".join(render_overview(payload))
    assert "地里长出来的" in overview and "87.0" in overview and "占 87.0%" in overview
    assert "被人安利来的" in overview and "13.0" in overview and "占 13.0%" in overview
    assert "2 个账号带动量 ≥ 3 个" in overview

    verdict = render_verdict(payload)
    assert verdict is not None
    assert "87.0% 的 star" in verdict
    assert "自带光合作用" in verdict  # 87% lands in the top tier

    # Lowering the threshold lets the small accounts back into the numbers.
    loose = render_verdict(payload, min_credit=0)
    assert "83.1%" in (loose or "")
    hidden_line = [line for line in render_report(payload).splitlines() if "另有" in line]
    assert hidden_line and "不计入上面的比例" in hidden_line[0]
