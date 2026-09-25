"""Pure-python attribution, spike detection and pruning maths.

Nothing in this module touches the network or the filesystem, which keeps the
statistical core easy to reason about and easy to unit test.
"""

from __future__ import annotations

import bisect
import heapq
import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

HOUR = 3600.0
DEFAULT_BIN_SECONDS = HOUR
DEFAULT_BASELINE_DAYS = 30
DEFAULT_SPIKE_ALPHA = 0.001
MIN_BASELINE_RATE = 0.5


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def compute_deltas(times: Sequence[float], window_seconds: float) -> list[int]:
    """For each star, how many stars arrive within ``window_seconds`` after it."""

    n = len(times)
    deltas = [0] * n
    right = 0
    for i in range(n):
        right = max(right, i + 1)
        limit = times[i] + window_seconds
        while right < n and times[right] <= limit:
            right += 1
        deltas[i] = right - i - 1
    return deltas


def hourly_counts(
    times: Sequence[float], bin_seconds: float = DEFAULT_BIN_SECONDS
) -> tuple[float, list[int]]:
    if not times:
        return 0.0, []
    start = math.floor(times[0] / bin_seconds) * bin_seconds
    bins = int((times[-1] - start) // bin_seconds) + 1
    counts = [0] * bins
    for timestamp in times:
        index = int((timestamp - start) // bin_seconds)
        counts[min(index, bins - 1)] += 1
    return start, counts


def rolling_baseline(
    counts: Sequence[int],
    *,
    window_bins: int = int(DEFAULT_BASELINE_DAYS * 24),
    refresh_bins: int = 24,
    floor: float = MIN_BASELINE_RATE,
) -> list[float]:
    """Robust per-bin expectation: median of the preceding window, refreshed daily."""

    baseline = [floor] * len(counts)
    current = floor
    for index in range(1, len(counts)):
        if index % refresh_bins == 0:
            low = max(0, index - window_bins)
            sample = counts[low:index]
            if sample:
                current = max(floor, statistics.median(sample))
        baseline[index] = current
    return baseline


def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""

    if k <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0
    if k > 1000 or lam > 500:
        sigma = math.sqrt(lam)
        z = (k - 0.5 - lam) / sigma
        return max(0.0, min(1.0, 0.5 * math.erfc(z / math.sqrt(2.0))))
    log_lam = math.log(lam)
    cumulative = 0.0
    for i in range(k):
        cumulative += math.exp(-lam + i * log_lam - math.lgamma(i + 1))
        if cumulative >= 1.0:
            return 0.0
    return max(0.0, 1.0 - cumulative)


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------
class WindowTopK:
    """Sliding-window top-K over a static ranking key.

    The weight of a candidate decays as ``exp(-(t - t_j) / tau)``; because the
    decay factor is shared by every candidate at a given instant, the ordering
    of candidates never changes. That lets us rank with a fixed key and only
    apply the decay when computing the actual credit.
    """

    def __init__(self, k: int, max_heap: int | None = None) -> None:
        self.k = max(1, k)
        self.max_heap = max_heap or max(2048, 16 * self.k)
        self._heap: list[tuple[float, int]] = []
        self._members: deque[int] = deque()
        self._keys: list[float] = []
        self._active = bytearray()

    def _ensure(self, index: int) -> None:
        """Keep per-index arrays aligned: only some stars ever become candidates."""

        if len(self._active) <= index:
            padding = index + 1 - len(self._active)
            self._active.extend(b"\x00" * padding)
            self._keys.extend([0.0] * padding)

    def add(self, key: float, index: int) -> None:
        self._ensure(index)
        self._keys[index] = key
        self._active[index] = 1
        heapq.heappush(self._heap, (-key, index))
        self._members.append(index)

    def expire(self, index: int) -> None:
        if index < len(self._active):
            self._active[index] = 0
        while self._members and not self._active[self._members[0]]:
            self._members.popleft()

    def top(self) -> list[tuple[float, int]]:
        result: list[tuple[float, int]] = []
        while self._heap and len(result) < self.k:
            neg_key, index = heapq.heappop(self._heap)
            if index < len(self._active) and self._active[index]:
                result.append((-neg_key, index))
        for key, index in result:
            heapq.heappush(self._heap, (-key, index))
        if len(self._heap) > self.max_heap:
            self._rebuild()
        return result

    def _rebuild(self) -> None:
        self._heap = [(-self._keys[index], index) for index in self._members]
        heapq.heapify(self._heap)


@dataclass(slots=True)
class AttributionResult:
    window_seconds: float
    credit: dict[str, float]
    wins: dict[str, int]
    unattributed: float
    total_stars: int
    direct_leads: dict[str, int]
    top_k: int

    @property
    def attributed_ratio(self) -> float:
        if not self.total_stars:
            return 0.0
        return 1.0 - self.unattributed / self.total_stars


def attribute_stars(
    times: Sequence[float],
    logins: Sequence[str],
    followers: Mapping[str, int | None],
    window_seconds: float,
    *,
    top_k: int = 20,
    deltas: Sequence[int] | None = None,
) -> AttributionResult:
    """Spread every star over the plausible "seed" stargazers before it."""

    tau = window_seconds / 2.0
    credit: dict[str, float] = {}
    wins: dict[str, int] = {}
    unattributed = 0.0
    window = WindowTopK(top_k)
    window_start = 0
    inserted_upto = 0

    for i in range(len(times)):
        current = times[i]
        while inserted_upto < i:
            login = logins[inserted_upto]
            count = followers.get(login)
            weight = math.log10(1.0 + count) if count else 0.0
            if weight > 0.0:
                window.add(math.log(weight) + times[inserted_upto] / tau, inserted_upto)
            inserted_upto += 1
        while window_start < i and times[window_start] < current - window_seconds:
            window.expire(window_start)
            window_start += 1

        candidates = window.top()
        if not candidates:
            unattributed += 1.0
            continue
        weights: list[tuple[float, str]] = []
        for key, index in candidates:
            count = followers.get(logins[index])
            base = math.log10(1.0 + count) if count else 0.0
            if base <= 0.0:
                continue
            decay = math.exp(-(current - times[index]) / tau)
            weights.append((base * decay, logins[index]))
        total_weight = sum(weight for weight, _ in weights)
        if total_weight <= 0.0:
            unattributed += 1.0
            continue
        dominant = max(weights, key=lambda item: item[0])[1]
        wins[dominant] = wins.get(dominant, 0) + 1
        for weight, login in weights:
            credit[login] = credit.get(login, 0.0) + weight / total_weight

    if deltas is None:
        deltas = compute_deltas(times, window_seconds)
    direct_leads: dict[str, int] = {}
    for index, login in enumerate(logins):
        direct_leads[login] = direct_leads.get(login, 0) + deltas[index]

    return AttributionResult(
        window_seconds=window_seconds,
        credit=credit,
        wins=wins,
        unattributed=unattributed,
        total_stars=len(times),
        direct_leads=direct_leads,
        top_k=top_k,
    )


def candidate_weights(
    times: Sequence[float],
    logins: Sequence[str],
    followers: Mapping[str, int | None],
    index: int,
    window_seconds: float,
    *,
    top_k: int = 20,
) -> list[tuple[float, str]]:
    """Reference implementation of "who could have brought this star".

    ``attribute_stars`` keeps the same ordering with a sliding-window heap; this
    straightforward version exists so tests (and readers) can check it.
    """

    current = times[index]
    tau = window_seconds / 2.0
    scored: list[tuple[float, str]] = []
    for position in range(index):
        timestamp = times[position]
        if timestamp < current - window_seconds:
            continue
        count = followers.get(logins[position])
        base = math.log10(1.0 + count) if count else 0.0
        if base <= 0.0:
            continue
        scored.append((base * math.exp(-(current - timestamp) / tau), logins[position]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[:top_k]


# --------------------------------------------------------------------------
# spike detection
# --------------------------------------------------------------------------
@dataclass(slots=True)
class SpikeEvent:
    start: float
    end: float
    observed: int
    expected: float
    excess: float
    peak_hourly: int
    p_value: float
    attributed: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "observed": self.observed,
            "expected": round(self.expected, 3),
            "excess": round(self.excess, 3),
            "peak_hourly": self.peak_hourly,
            "p_value": self.p_value,
            "attributed": self.attributed,
        }


def detect_spikes(
    times: Sequence[float],
    *,
    window_seconds: float,
    alpha: float = DEFAULT_SPIKE_ALPHA,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
    baseline_days: int = DEFAULT_BASELINE_DAYS,
    bonferroni: bool = True,
    gap_bins: int = 1,
) -> list[SpikeEvent]:
    if len(times) < 4:
        return []
    start, counts = hourly_counts(times, bin_seconds)
    baseline = rolling_baseline(counts, window_bins=int(baseline_days * 24))
    threshold = alpha / len(counts) if bonferroni else alpha

    p_values: list[float] = [1.0] * len(counts)
    significant = [False] * len(counts)
    for index, count in enumerate(counts):
        lam = baseline[index]
        if count > lam and count >= 2:
            p_value = poisson_sf(count, lam)
            p_values[index] = p_value
            significant[index] = p_value < threshold

    events: list[SpikeEvent] = []
    index = 0
    while index < len(counts):
        if not significant[index]:
            index += 1
            continue
        begin = index
        end = index
        cursor = index + 1
        while cursor < len(counts):
            if significant[cursor]:
                end = cursor
                cursor += 1
                continue
            gap = 0
            probe = cursor
            while probe < len(counts) and not significant[probe]:
                gap += 1
                probe += 1
                if gap > gap_bins:
                    break
            if probe < len(counts) and significant[probe] and gap <= gap_bins:
                end = probe
                cursor = probe + 1
                continue
            break
        observed = sum(counts[begin : end + 1])
        expected = sum(baseline[begin : end + 1])
        events.append(
            SpikeEvent(
                start=start + begin * bin_seconds,
                end=start + (end + 1) * bin_seconds,
                observed=observed,
                expected=expected,
                excess=max(0.0, observed - expected),
                peak_hourly=max(counts[begin : end + 1]),
                p_value=min(p_values[begin : end + 1]),
            )
        )
        index = end + 1
    return events


def attribute_spikes(
    events: Sequence[SpikeEvent],
    times: Sequence[float],
    logins: Sequence[str],
    followers: Mapping[str, int | None],
    window_seconds: float,
    *,
    top_k: int = 5,
    grace_seconds: float = HOUR,
) -> None:
    """Credit each event's excess to the biggest stargazers right before it.

    ``grace_seconds`` covers the first bin of the event: burst detection is
    binned by the hour, so the account whose star triggered the surge often
    lands just inside the event window rather than strictly before it.
    """

    if not times:
        return
    for event in events:
        low = event.start - window_seconds
        high = event.start + grace_seconds
        candidates: list[tuple[float, str, float]] = []
        for index, timestamp in enumerate(times):
            if timestamp < low:
                continue
            if timestamp >= high:
                break
            count = followers.get(logins[index])
            weight = math.log10(1.0 + count) if count else 0.0
            if weight <= 0.0:
                continue
            decay = math.exp(-max(0.0, event.start - timestamp) / (window_seconds / 2.0))
            candidates.append((weight * decay, logins[index], timestamp))
        candidates.sort(key=lambda item: item[0], reverse=True)
        top = candidates[:top_k]
        total = sum(item[0] for item in top)
        if total <= 0.0 or event.excess <= 0.0:
            event.attributed = []
            continue
        event.attributed = [
            {
                "login": login,
                "share": round(weight / total * event.excess, 3),
                "followers": followers.get(login),
                "starred_at": timestamp,
            }
            for weight, login, timestamp in top
        ]


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------
@dataclass(slots=True)
class PrunePlan:
    pool_size: int
    total: int
    candidates: list[str]
    delta_candidates: int
    spike_candidates: int
    pruned: bool
    inside_event: int = 0

    @property
    def candidate_ratio(self) -> float:
        return len(self.candidates) / self.total if self.total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "total_stargazers": self.total,
            "candidate_count": len(self.candidates),
            "candidate_ratio": round(self.candidate_ratio, 6),
            "delta_candidates": self.delta_candidates,
            "spike_window_candidates": self.spike_candidates,
            "inside_event_stars": self.inside_event,
            "pruned": self.pruned,
        }


def select_candidates(
    times: Sequence[float],
    logins: Sequence[str],
    deltas: Sequence[int],
    events: Sequence[SpikeEvent],
    window_seconds: float,
    *,
    pool_size: int = 3000,
) -> PrunePlan:
    """Pick the logins whose follower counts are worth querying."""

    total = len(logins)
    if total == 0:
        return PrunePlan(pool_size, 0, [], 0, 0, False)
    if pool_size <= 0 or total <= pool_size:
        return PrunePlan(pool_size, total, list(dict.fromkeys(logins)), total, 0, False)

    # Surge detection bins by the hour, so the account that triggered a surge
    # usually lands in the first bin of the event. Give that bin a grace period
    # when collecting "who was in front of the surge" candidates, and treat the
    # rest of the surge as followers (their own delta is large only because the
    # burst was already running).
    grace = DEFAULT_BIN_SECONDS
    spike_indices: set[int] = set()
    inside_event: set[int] = set()
    for event in events:
        start_index = bisect.bisect_left(times, event.start - window_seconds)
        grace_index = bisect.bisect_left(times, event.start + grace)
        end_index = bisect.bisect_left(times, event.end)
        spike_indices.update(range(start_index, grace_index))
        inside_event.update(range(grace_index, end_index))
    outside = [index for index in range(total) if index not in inside_event]
    order = sorted(outside, key=lambda index: (-deltas[index], times[index]))
    order += sorted(inside_event, key=lambda index: (-deltas[index], times[index]))
    protected = sorted(spike_indices, key=lambda index: (-deltas[index], times[index]))
    if len(protected) >= pool_size:
        chosen = protected[:pool_size]
    else:
        chosen = list(protected)
        chosen_set = set(chosen)
        for index in order:
            if len(chosen) >= pool_size:
                break
            if index in chosen_set:
                continue
            chosen.append(index)
            chosen_set.add(index)
    chosen.sort()
    candidates = [logins[index] for index in chosen]
    return PrunePlan(
        pool_size=pool_size,
        total=total,
        candidates=list(dict.fromkeys(candidates)),
        delta_candidates=sum(
            1 for index in chosen if index not in spike_indices and index not in inside_event
        ),
        spike_candidates=len(spike_indices),
        pruned=len(chosen) < total,
        inside_event=len(inside_event),
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarize_verification(
    sample: Sequence[tuple[str, int | None]],
    pruned_total: int,
    *,
    threshold: int = 1000,
) -> dict[str, Any]:
    """Estimate how much influence pruning might have thrown away."""

    sized = [(login, followers or 0) for login, followers in sample]
    high = [(login, followers) for login, followers in sized if followers >= threshold]
    ratio = len(high) / len(sized) if sized else 0.0
    low, high_bound = wilson_interval(len(high), len(sized))
    return {
        "sample_size": len(sized),
        "pruned_total": pruned_total,
        "threshold": threshold,
        "high_follower_count": len(high),
        "high_follower_ratio": round(ratio, 6),
        "ratio_ci95": [round(low, 6), round(high_bound, 6)],
        "estimated_missed": round(ratio * pruned_total, 1),
        "estimated_missed_ci95": [
            round(low * pruned_total, 1),
            round(high_bound * pruned_total, 1),
        ],
        "examples": [
            {"login": login, "followers": followers}
            for login, followers in sorted(high, key=lambda item: -item[1])[:10]
        ],
    }


# --------------------------------------------------------------------------
# sensitivity + ranking
# --------------------------------------------------------------------------
def rank_of(credit: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(credit.items(), key=lambda item: (-item[1], item[0]))


def build_sensitivity(
    results: Mapping[int, AttributionResult],
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Rank stability across windows, keyed in hours like the rest of the payload."""

    windows = sorted(results)
    # ``results`` is keyed by seconds; every other payload field (credit, direct
    # lead, wins, params.windows_hours) uses whole hours. Mixing the two once
    # left the slope chart permanently empty.
    labels = {window: str(round(window / 3600.0)) for window in windows}
    ranks: dict[str, dict[int, int]] = {}
    for window in windows:
        for position, (login, _credit) in enumerate(rank_of(results[window].credit), start=1):
            ranks.setdefault(login, {})[window] = position
    primary = windows[len(windows) // 2] if windows else 0
    rows = []
    for login, per_window in ranks.items():
        if min(per_window.values()) > top_n:
            continue
        positions = [per_window.get(window, 10**9) for window in windows]
        rows.append(
            {
                "login": login,
                "ranks": {labels[window]: per_window.get(window) for window in windows},
                "credit": {
                    labels[window]: round(results[window].credit.get(login, 0.0), 3)
                    for window in windows
                },
                "spread": max(positions) - min(positions),
                "stable": all(position <= top_n for position in positions),
            }
        )
    rows.sort(key=lambda row: (row["ranks"].get(labels.get(primary, ""), 10**9), row["login"]))
    return {
        "windows": [int(labels[window]) for window in windows],
        "primary_window": int(labels.get(primary, 0)),
        "top_n": top_n,
        "users": rows,
        "stable_users": [row["login"] for row in rows if row["stable"]],
    }
