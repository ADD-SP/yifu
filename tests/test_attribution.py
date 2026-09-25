from __future__ import annotations

import random

import pytest

from yifu import attribution as attr

HOUR = 3600.0
DAY = 24 * HOUR


def test_compute_deltas_matches_bruteforce() -> None:
    rng = random.Random(11)
    times = sorted(rng.uniform(0, 100 * HOUR) for _ in range(200))
    deltas = attr.compute_deltas(times, DAY)
    expected = [
        sum(1 for other in times if timestamp < other <= timestamp + DAY) for timestamp in times
    ]
    assert deltas == expected


def test_compute_deltas_handles_simultaneous_stars() -> None:
    times = [0.0, 0.0, 0.0, 10.0]
    assert attr.compute_deltas(times, DAY) == [3, 2, 1, 0]


def test_window_topk_handles_sparse_indices() -> None:
    """Regression: only some stars become candidates, so indices are sparse."""

    window = attr.WindowTopK(2)
    window.add(1.0, 5)
    window.add(9.0, 11)
    window.add(4.0, 40)
    assert window.top() == [(9.0, 11), (4.0, 40)]
    window.expire(5)
    window.expire(11)
    assert window.top() == [(4.0, 40)]
    window.expire(40)
    assert window.top() == []


def test_attribute_stars_splits_and_counts_wins() -> None:
    times = [0.0, HOUR, 2 * HOUR]
    logins = ["small", "big", "later"]
    followers = {"small": 10, "big": 10000, "later": None}
    result = attr.attribute_stars(times, logins, followers, DAY, top_k=20)
    assert result.total_stars == 3
    assert result.unattributed == 1.0  # the first star has nothing before it
    assert pytest.approx(sum(result.credit.values())) == 2.0
    # The star at 1h only has "small" in front of it; the star at 2h is shared.
    assert result.wins == {"small": 1, "big": 1}
    assert result.credit["small"] > result.credit["big"]
    assert result.direct_leads["small"] == 2
    assert result.direct_leads["big"] == 1


def test_candidate_weights_prefers_followers_then_recency() -> None:
    times = [0.0, 10 * 60.0, 20 * 60.0]
    logins = ["big", "small", "target"]
    followers = {"big": 50000, "small": 10}
    weights = attr.candidate_weights(times, logins, followers, 2, DAY, top_k=5)
    assert [login for _weight, login in weights] == ["big", "small"]

    equal = attr.candidate_weights(times, logins, {"big": 100, "small": 100}, 2, DAY, top_k=5)
    # "small" is ten minutes closer to the target star.
    assert [login for _weight, login in equal] == ["small", "big"]


def test_reference_weights_ignore_unknown_and_zero_follower_accounts() -> None:
    times = [0.0, HOUR, 2 * HOUR]
    logins = ["pruned", "zero", "target"]
    followers = {"pruned": None, "zero": 0}
    assert attr.candidate_weights(times, logins, followers, 2, DAY) == []


def test_attribute_stars_matches_reference_weights() -> None:
    rng = random.Random(17)
    times = sorted(rng.uniform(0, 10 * DAY) for _ in range(200))
    logins = [f"u{index}" for index in range(len(times))]
    followers = {login: rng.choice([0, 3, 40, 900, 40000]) for login in logins}
    window = 12 * HOUR
    result = attr.attribute_stars(times, logins, followers, window, top_k=7)

    expected_credit: dict[str, float] = {}
    expected_wins: dict[str, int] = {}
    for index in range(len(times)):
        weights = attr.candidate_weights(times, logins, followers, index, window, top_k=7)
        if not weights:
            continue
        total = sum(weight for weight, _login in weights)
        expected_wins[weights[0][1]] = expected_wins.get(weights[0][1], 0) + 1
        for weight, login in weights:
            expected_credit[login] = expected_credit.get(login, 0.0) + weight / total
    assert result.wins == expected_wins
    assert set(result.credit) == set(expected_credit)
    for login, value in expected_credit.items():
        assert result.credit[login] == pytest.approx(value, rel=1e-9, abs=1e-9)


def test_attribute_stars_marks_zero_follower_seeds_unattributed() -> None:
    times = [0.0, HOUR, 2 * HOUR]
    logins = ["zero", "ghost", "third"]
    followers = {"zero": 0, "ghost": None, "third": 0}
    result = attr.attribute_stars(times, logins, followers, DAY)
    assert result.unattributed == 3.0
    assert result.credit == {}


def test_attribute_stars_topk_pruning_tracks_full_window() -> None:
    rng = random.Random(3)
    times = sorted(rng.uniform(0, 30 * DAY) for _ in range(400))
    logins = [f"u{i}" for i in range(400)]
    followers = {login: rng.choice([0, 5, 50, 500, 5000]) for login in logins}
    full = attr.attribute_stars(times, logins, followers, DAY, top_k=400)
    pruned = attr.attribute_stars(times, logins, followers, DAY, top_k=5)
    top_full = [login for login, _ in attr.rank_of(full.credit)[:5]]
    top_pruned = [login for login, _ in attr.rank_of(pruned.credit)[:5]]
    # The winner is stable under Top-K pruning; the tail of the ranking can move.
    assert top_full[0] == top_pruned[0]
    assert len(set(top_full) & set(top_pruned)) >= 4
    # The dominant candidate for each star never depends on K.
    assert full.wins == pruned.wins


def test_detect_spikes_finds_injected_burst() -> None:
    rng = random.Random(5)
    times: list[float] = []
    timestamp = 0.0
    for _ in range(600):
        timestamp += rng.uniform(0.5 * HOUR, 3 * HOUR)
        times.append(timestamp)
    burst_start = timestamp + HOUR
    times.extend(burst_start + rng.uniform(0, 5 * HOUR) for _ in range(400))
    times.sort()
    events = attr.detect_spikes(times, window_seconds=DAY)
    assert events, "expected the injected burst to be detected"
    top = max(events, key=lambda event: event.excess)
    assert top.observed >= 200
    assert top.excess > 100
    assert top.start <= burst_start + HOUR


def test_detect_spikes_ignores_uniform_growth() -> None:
    times = [index * 2 * HOUR + (index % 3) * 60 for index in range(500)]
    assert attr.detect_spikes(times, window_seconds=DAY) == []


def test_detect_spikes_needs_enough_history() -> None:
    assert attr.detect_spikes([0.0, HOUR, 2 * HOUR], window_seconds=DAY) == []


def test_attribute_spikes_reaches_trigger_inside_first_bin(make_event=None) -> None:
    trigger = 10 * DAY
    event = attr.SpikeEvent(
        start=trigger - 600,
        end=trigger + 6 * HOUR,
        observed=120,
        expected=5,
        excess=115,
        peak_hourly=40,
        p_value=1e-9,
    )
    times = [trigger - 30 * DAY, trigger, trigger + HOUR]
    logins = ["quiet", "influencer", "follower"]
    followers = {"quiet": 3, "influencer": 50000, "follower": 4}
    attr.attribute_spikes([event], times, logins, followers, DAY, top_k=3)
    assert event.attributed
    assert event.attributed[0]["login"] == "influencer"
    assert event.attributed[0]["share"] > 100


def test_select_candidates_protects_spike_window_and_skips_surge_members() -> None:
    trigger = 10 * DAY
    event = attr.SpikeEvent(
        start=trigger - 600,
        end=trigger + 6 * HOUR,
        observed=300,
        expected=10,
        excess=290,
        peak_hourly=100,
        p_value=1e-12,
    )
    times = [index * HOUR for index in range(240)] + [trigger] + [
        trigger + index * 60 for index in range(1, 200)
    ]
    times.sort()
    logins = [f"u{index}" for index in range(len(times))]
    deltas = attr.compute_deltas(times, DAY)
    plan = attr.select_candidates(times, logins, deltas, [event], DAY, pool_size=30)
    assert plan.pruned
    assert len(plan.candidates) == 30
    assert plan.spike_candidates >= 1
    assert plan.inside_event > 0
    # The trigger sits inside the first bin of the event and must stay protected.
    trigger_index = times.index(trigger)
    assert logins[trigger_index] in plan.candidates
    # Only the first bin (grace period) of the surge may claim pool slots; the
    # rest of the surge are followers of the burst, not its cause.
    grace_end = event.start + 3600
    late_surge = {
        logins[index]
        for index, timestamp in enumerate(times)
        if grace_end <= timestamp < event.end
    }
    assert late_surge, "the fixture should contain stars after the grace bin"
    assert not (late_surge & set(plan.candidates))


def test_select_candidates_without_pruning_keeps_everyone() -> None:
    times = [index * HOUR for index in range(50)]
    logins = [f"u{index}" for index in range(50)]
    deltas = attr.compute_deltas(times, DAY)
    plan = attr.select_candidates(times, logins, deltas, [], DAY, pool_size=10)
    assert plan.pruned
    assert len(plan.candidates) == 10
    assert plan.to_json()["pool_size"] == 10


def test_wilson_interval_and_verification_summary() -> None:
    low, high = attr.wilson_interval(5, 100)
    assert 0.0 < low < 0.05 < high < 0.2
    summary = attr.summarize_verification(
        [("a", 5000), ("b", 10), ("c", None), ("d", 2000)], pruned_total=1000
    )
    assert summary["sample_size"] == 4
    assert summary["high_follower_count"] == 2
    assert summary["estimated_missed"] == pytest.approx(500, abs=1)
    assert summary["examples"][0]["login"] == "a"


def test_build_sensitivity_marks_stable_users() -> None:
    times = [0.0, HOUR, 2 * HOUR, 30 * HOUR]
    logins = ["alpha", "beta", "gamma", "delta"]
    followers = {"alpha": 100000, "beta": 500, "gamma": 300, "delta": 100}
    results = {
        window: attr.attribute_stars(times, logins, followers, window, top_k=5)
        for window in (6 * HOUR, DAY, 72 * HOUR)
    }
    sensitivity = attr.build_sensitivity(results, top_n=2)
    # Windows are reported in hours, matching params.windows_hours and the
    # per-user credit/rank keys.
    assert sensitivity["windows"] == [6, 24, 72]
    assert "alpha" in sensitivity["stable_users"]
