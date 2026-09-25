"""Plain-text report printed to the terminal.

The CLI is the only output surface now: no HTML, no CSV, no images. Everything
that matters is rendered as aligned text.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

HEADER = "可能带来更多 star 的人"
# Rows below this much credit in the primary window are noise for a human reader.
DEFAULT_MIN_CREDIT = 3.0
# Praise kicks in when this share of stars did *not* ride on a stargazer cascade.
ORGANIC_PRAISE_THRESHOLD = 0.30
# Kaomoji stay ASCII on purpose: terminals render them everywhere and their
# width is predictable, unlike emoji or exotic Unicode symbols.
VERDICT_LINES: dict[str, str] = {
    "low": "的 star 是自己一点点攒起来的呢，不需要谁来带的说 (^_^)",
    "mid": "的 star 全靠自己撑起来～背书什么的，才不需要呢！(*^_^*)",
    "high": "的 star 都是自己长出来的呀，热度根本不用蹭——毕竟自带光合作用 (^o^)/",
}


def display_width(text: str) -> int:
    """Width in terminal cells (CJK characters take two)."""

    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text
    )


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - display_width(text))
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def _num(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def _iso(epoch: Any) -> str:
    if not isinstance(epoch, (int, float)):
        return ""
    return datetime.fromtimestamp(float(epoch), tz=UTC).strftime("%Y-%m-%d %H:%MZ")


def _table(
    headers: Sequence[tuple[str, str]],
    rows: Sequence[Sequence[str]],
    *,
    box: bool = True,
) -> list[str]:
    widths = [
        max(display_width(header), *(display_width(row[index]) for row in rows))
        for index, (header, _align) in enumerate(headers)
    ]
    aligns = [align for _header, align in headers]
    if not box:
        lines = [
            "  ".join(
                pad(header, widths[index], align) for index, (header, align) in enumerate(headers)
            ).rstrip()
        ]
        lines.append("  ".join("-" * width for width in widths))
        for row in rows:
            lines.append(
                "  ".join(
                    pad(cell, widths[index], aligns[index]) for index, cell in enumerate(row)
                ).rstrip()
            )
        return lines

    def rule(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def line(cells: Sequence[str]) -> str:
        return "│" + "│".join(
            f" {pad(cell, widths[index], aligns[index])} " for index, cell in enumerate(cells)
        ) + "│"

    lines = [rule("┌", "┬", "┐"), line([header for header, _align in headers]), rule("├", "┼", "┤")]
    lines.extend(line(row) for row in rows)
    lines.append(rule("└", "┴", "┘"))
    return lines


def primary_window(payload: Mapping[str, Any]) -> str:
    return str(int(float((payload.get("params") or {}).get("window_hours") or 24)))


def meaningful_credit(payload: Mapping[str, Any], min_credit: float) -> float:
    """Credit that survives the threshold: sub-threshold accounts do not count.

    Returns ``None`` when the payload carries no ranking at all, so callers can
    fall back to the raw attribution summary.
    """

    rows = payload.get("influencers") or []
    if not rows:
        return None  # type: ignore[return-value]
    window = primary_window(payload)
    return sum(
        (row.get("credit") or {}).get(window, 0.0)
        for row in rows
        if (row.get("credit") or {}).get(window, 0.0) >= min_credit
    )


def render_overview(
    payload: Mapping[str, Any],
    *,
    box: bool = True,
    min_credit: float = DEFAULT_MIN_CREDIT,
) -> list[str]:
    summary = payload.get("summary") or {}
    total = summary.get("total_stars") or 0
    window = primary_window(payload)
    rows_all = payload.get("influencers") or []
    kept = [
        row
        for row in rows_all
        if (row.get("credit") or {}).get(window, 0.0) >= min_credit
    ]
    kept_credit = sum((row.get("credit") or {}).get(window, 0.0) for row in kept)
    rest = max(0.0, total - kept_credit)

    def pct(value: float) -> str:
        return f"占 {(value / total * 100) if total else 0:.1f}%"

    rows = [
        (
            "地里长出来的",
            _num(rest, 1),
            pct(rest),
        ),
        (
            "被人安利来的",
            _num(kept_credit, 1),
            (
                f"{pct(kept_credit)}（{len(kept):,} 个账号带动量 ≥ {min_credit:g} 个）"
                if min_credit > 0
                else pct(kept_credit)
            ),
        ),
    ]
    headers = [("指标", "left"), ("数值", "right"), ("备注", "left")]
    return _table(headers, rows, box=box)


def render_verdict(
    payload: Mapping[str, Any], *, min_credit: float = DEFAULT_MIN_CREDIT
) -> str | None:
    """Praise the project when enough stars did not come from a bringer.

    Sub-threshold accounts do not count, so their credit lands in the "grew on
    its own" bucket instead of being spread over the ranking.
    """

    summary = payload.get("summary") or {}
    total = summary.get("total_stars") or 0
    if not total:
        return None
    credited = meaningful_credit(payload, min_credit)
    if credited is None:
        credited = (total - (summary.get("unattributed_stars") or 0.0)) if min_credit > 0 else (
            summary.get("credit_total") or 0.0
        )
    share = max(0.0, (total - credited) / total)
    if share < ORGANIC_PRAISE_THRESHOLD:
        return None
    if share >= 0.8:
        tier = "high"
    elif share >= 0.5:
        tier = "mid"
    else:
        tier = "low"
    percent = f"{share * 100:.1f}%"
    return f"★ 评价：{percent} {VERDICT_LINES[tier]}"


def render_influencers(
    payload: Mapping[str, Any],
    top: int,
    *,
    min_credit: float = 0.0,
    box: bool = True,
) -> tuple[list[str], int]:
    """Render the ranking; returns the lines and how many rows were hidden."""

    window = str(int(float((payload.get("params") or {}).get("window_hours") or 24)))
    all_rows = payload.get("influencers") or []
    kept = [
        row
        for row in all_rows
        if (row.get("credit") or {}).get(window, 0.0) >= min_credit
    ]
    hidden = len(all_rows) - len(kept)
    rows = kept[: top if top > 0 else None]
    if not rows:
        if min_credit > 0 and all_rows:
            return (
                [
                    (
                        f"  （没有账号在 {window} 小时范围内带来 {min_credit:g} 个以上，"
                        "用 --min-credit 0 查看全部）"
                    )
                ],
                hidden,
            )
        return ["  （没有可归因的账号）"], hidden
    windows = [str(int(float(value))) for value in payload["params"]["windows_hours"]]
    headers: list[tuple[str, str]] = [("#", "right"), ("用户", "left"), ("粉丝数", "right")]
    headers += [(f"{window}h", "right") for window in windows]
    headers += [("star 时间", "left")]
    table_rows: list[list[str]] = []
    for index, row in enumerate(rows, start=1):
        credit = row.get("credit") or {}
        table_rows.append(
            [
                str(index),
                str(row.get("login")),
                _num(row.get("followers"), 0),
                *[_num(credit.get(window, 0.0), 1) for window in windows],
                _iso(row.get("starred_at")) or str(row.get("starred_at_iso") or ""),
            ]
        )
    return _table(headers, table_rows, box=box), hidden


def render_events(payload: Mapping[str, Any], limit: int = 10) -> list[str]:
    events = payload.get("events") or []
    if not events:
        return ["  （没有检测到明显异常的 star 增长时段）"]
    lines: list[str] = []
    ordered = sorted(events, key=lambda event: event.get("excess", 0), reverse=True)[:limit]
    for event in ordered:
        lines.append(
            f"  {_iso(event.get('start'))} → {_iso(event.get('end'))}"
            f"   比平时多 {_num(event.get('excess'), 1)} 个"
        )
        lines.append(
            f"      这一时段新增 {_num(event.get('observed'), 0)} 个"
            f"（平时同期约 {_num(event.get('expected'), 1)} 个），"
            f"峰值 {_num(event.get('peak_hourly'), 0)} 个/小时"
        )
        attributed = event.get("attributed") or []
        if attributed:
            who = " · ".join(
                f"{item.get('login')} +{_num(item.get('share'), 1)}" for item in attributed
            )
            lines.append(f"      可能的带动者：{who}")
        else:
            lines.append("      这一波没找到可能的带动者，可能来自站外（新闻、社交媒体、推荐）")
    if len(events) > len(ordered):
        lines.append(f"  … 另有 {len(events) - len(ordered)} 个较小的时段未列出")
    return lines


def render_report(
    payload: Mapping[str, Any],
    *,
    top: int = 20,
    events: int = 10,
    box: bool = True,
    min_credit: float = DEFAULT_MIN_CREDIT,
) -> str:
    repo = payload.get("repo") or {}
    stargazers = payload.get("stargazers") or {}
    window = int(float((payload.get("params") or {}).get("window_hours") or 24))
    lines: list[str] = []
    lines.append(
        f"{repo.get('full_name')} · {_num(stargazers.get('count'), 0)} 个 star"
        f" · {str(stargazers.get('first_star', '?'))[:10]} → {str(stargazers.get('last_star', '?'))[:10]}"
    )
    verdict = render_verdict(payload, min_credit=min_credit)
    if verdict:
        lines.append("")
        lines.append(verdict)
    lines.append("")
    lines.append("概览")
    lines.extend(render_overview(payload, box=box, min_credit=min_credit))
    lines.append("")
    scope = f"前 {top}" if top > 0 else "全部"
    threshold = f"，只显示 ≥ {min_credit:g} 个的账号" if min_credit > 0 else ""
    lines.append(f"{HEADER}（按 {window} 小时范围内的带动量排序{threshold}，{scope}）")
    lines.append("  带动量是分摊估算：一个 star 会按粉丝数与时间远近同时分给多个候选")
    table_lines, hidden = render_influencers(payload, top, min_credit=min_credit, box=box)
    lines.extend(table_lines)
    if hidden:
        lines.append(
            f"  … 另有 {hidden:,} 个账号在 {window} 小时范围内不足 {min_credit:g} 个，"
            "未列出且不计入上面的比例"
        )
    lines.append("")
    lines.append("明显异常的时段")
    lines.extend(render_events(payload, events))
    return "\n".join(lines)
