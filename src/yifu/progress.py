"""Progress reporting: structured updates plus a terminal progress bar."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

STAGE_LABELS = {
    "stars": "抓取 star 时间线",
    "profiles": "抓取用户资料",
    "verify": "抽样校验",
    "info": "",
}


@dataclass(slots=True)
class ProgressUpdate:
    """One progress event: a stage, optional counters and a short detail."""

    stage: str
    done: int | None = None
    total: int | None = None
    detail: str = ""

    @property
    def label(self) -> str:
        return STAGE_LABELS.get(self.stage, self.stage)


ProgressCallback = Callable[[ProgressUpdate], None]


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class ProgressBar:
    """Single-line progress bar that redraws itself on a TTY.

    When the stream is not a terminal (pipes, CI, tests) it degrades to one
    plain line per stage transition plus every ``plain_step`` percent, so logs
    stay readable and no ANSI escapes leak into files.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        width: int = 26,
        enabled: bool | None = None,
        plain_step: int = 10,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self.plain_step = plain_step
        self.now = now
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        # ``enabled`` only decides whether we report at all; a non-TTY stream
        # still gets plain lines instead of ANSI redraws.
        self.enabled = True if enabled is None else enabled
        self.stage: str | None = None
        self.started_at = now()
        self._last_plain_percent = -plain_step
        self._last_line_length = 0

    # -- rendering ---------------------------------------------------------
    def _bar(self, fraction: float | None) -> str:
        if fraction is None:
            return "·" * self.width
        filled = round(self.width * max(0.0, min(1.0, fraction)))
        return "█" * filled + "░" * (self.width - filled)

    def _line(self, update: ProgressUpdate) -> str:
        label = update.label or "处理中"
        parts = [label]
        fraction = None
        if update.done is not None and update.total:
            fraction = update.done / update.total
            parts.append(self._bar(fraction))
            parts.append(f"{fraction * 100:5.1f}%")
            parts.append(f"{update.done:,}/{update.total:,}")
        else:
            parts.append(self._bar(None))
            if update.done is not None:
                parts.append(f"{update.done:,}")
        elapsed = self.now() - self.started_at
        remaining = (elapsed / fraction - elapsed) if fraction else 0.0
        if fraction and elapsed >= 1.0 and remaining >= 1.0:
            parts.append(f"已用 {_format_seconds(elapsed)} / 约剩 {_format_seconds(remaining)}")
        elif elapsed >= 5.0:
            parts.append(f"已用 {_format_seconds(elapsed)}")
        if update.detail:
            parts.append(update.detail)
        return "  ".join(part for part in parts if part)

    def _clear(self) -> None:
        if self._last_line_length:
            self.stream.write("\r" + " " * self._last_line_length + "\r")

    # -- public API --------------------------------------------------------
    def update(self, update: ProgressUpdate) -> None:
        if not self.enabled:
            return
        if update.stage != self.stage:
            self.finish()
            self.stage = update.stage
            self.started_at = self.now()
            self._last_plain_percent = -self.plain_step
        line = self._line(update)
        if self.is_tty:
            self._clear()
            self.stream.write(line)
            self.stream.flush()
            self._last_line_length = len(line)
            return
        if update.done is None or not update.total:
            self.stream.write(line + "\n")
            self.stream.flush()
            return
        percent = int(update.done * 100 / update.total)
        if percent >= self._last_plain_percent + self.plain_step or update.done >= update.total:
            self._last_plain_percent = percent
            self.stream.write(line + "\n")
            self.stream.flush()

    def finish(self, detail: str = "") -> None:
        if not self.enabled or self.stage is None:
            self.stage = None
            return
        if self.is_tty:
            self._clear()
            if detail:
                self.stream.write(detail + "\n")
            self.stream.flush()
        self._last_line_length = 0
        self.stage = None


class Reporter:
    """Coordinates the progress bar with plain log lines.

    Text always goes through :meth:`log`, which clears the bar line first, so
    log output never lands in the middle of a redrawn bar.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        enabled: bool = True,
        width: int = 26,
        plain_step: int = 10,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        self.bar = ProgressBar(stream, width=width, enabled=enabled, plain_step=plain_step, now=now)

    def log(self, message: str) -> None:
        if not self.enabled or not message:
            return
        self.bar.finish()
        print(message, file=self.stream, flush=True)

    def progress(self, update: ProgressUpdate) -> None:
        if update.total or update.done is not None:
            self.bar.update(update)
            return
        self.log(update.detail)

    def finish(self, detail: str = "") -> None:
        self.bar.finish(detail)


def make_reporter(
    stream: TextIO | None = None,
    *,
    enabled: bool = True,
    verbose_log: Callable[[str], None] | None = None,
) -> Reporter:
    """Build a reporter; ``verbose_log`` is kept for callers that log elsewhere."""

    reporter = Reporter(stream, enabled=enabled)
    if verbose_log is not None:
        original = reporter.log

        def both(message: str) -> None:
            original(message)
            verbose_log(message)

        reporter.log = both  # type: ignore[method-assign]
    return reporter
