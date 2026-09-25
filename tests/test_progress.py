from __future__ import annotations

import io

from yifu.progress import ProgressBar, ProgressUpdate, make_reporter


class FakeStream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


def test_tty_bar_redraws_a_single_line() -> None:
    stream = FakeStream(tty=True)
    bar = ProgressBar(stream, width=10, now=FakeClock())
    bar.update(ProgressUpdate("stars", done=25, total=100, detail="第 1 页"))
    bar.update(ProgressUpdate("stars", done=50, total=100, detail="第 2 页"))
    bar.finish("抓取 star 时间线  完成")
    text = stream.getvalue()
    assert text.count("\r") >= 3  # each redraw clears the previous line
    assert "░░░" in text and "█" in text
    assert "50.0%" in text and "50/100" in text
    assert "抓取 star 时间线" in text
    assert text.endswith("抓取 star 时间线  完成\n")


def test_non_tty_stream_prints_percent_steps_only() -> None:
    stream = FakeStream(tty=False)
    bar = ProgressBar(stream, width=10, plain_step=25, now=FakeClock())
    for done in (1, 10, 25, 26, 49, 50, 75, 100):
        bar.update(ProgressUpdate("profiles", done=done, total=100))
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    # one line per 25% step plus the final one
    assert len(lines) == 4
    assert lines[0].strip().startswith("抓取用户资料")
    assert "100/100" in lines[-1]
    assert "\r" not in stream.getvalue()


def test_unknown_total_uses_an_indeterminate_bar() -> None:
    stream = FakeStream(tty=True)
    bar = ProgressBar(stream, width=8, now=FakeClock())
    bar.update(ProgressUpdate("stars", done=120, detail="第 2 页"))
    text = stream.getvalue()
    assert "·" * 8 in text
    assert "120" in text


def test_stage_change_starts_a_new_bar() -> None:
    stream = FakeStream(tty=True)
    bar = ProgressBar(stream, width=8, now=FakeClock())
    bar.update(ProgressUpdate("stars", done=100, total=100))
    bar.update(ProgressUpdate("profiles", done=1, total=50))
    text = stream.getvalue()
    assert "抓取 star 时间线" in text
    assert "抓取用户资料" in text


def test_make_reporter_routes_counters_to_the_bar_and_text_to_log() -> None:
    stream = FakeStream(tty=False)
    logged: list[str] = []
    reporter = make_reporter(stream, verbose_log=logged.append, enabled=True)
    reporter.progress(ProgressUpdate("info", detail="认证检查：someone"))
    reporter.progress(ProgressUpdate("stars", done=10, total=10, detail="第 1 页"))
    reporter.finish()
    assert logged == ["认证检查：someone"]
    assert "抓取 star 时间线" in stream.getvalue()


def test_disabled_reporter_stays_silent() -> None:
    stream = FakeStream(tty=True)
    reporter = make_reporter(stream, enabled=False)
    reporter.progress(ProgressUpdate("stars", done=5, total=10))
    reporter.log("认证检查：someone")
    reporter.finish()
    assert stream.getvalue() == ""


def test_log_clears_the_bar_line_first() -> None:
    stream = FakeStream(tty=True)
    reporter = make_reporter(stream, enabled=True)
    reporter.progress(ProgressUpdate("stars", done=5, total=10))
    reporter.log("证书检查完毕")
    text = stream.getvalue()
    assert text.index("\r") < text.index("证书检查完毕")
