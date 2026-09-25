from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeGitHub
from test_pipeline import synthetic_history

from yifu.cli import main


@pytest.fixture
def env(fake_github: FakeGitHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at the fake GitHub through environment variables only."""

    cache = tmp_path / "cache"
    monkeypatch.setenv("YIFU_API_BASE", fake_github.api_base)
    monkeypatch.setenv("YIFU_CACHE_DIR", str(cache))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token_1234")
    return cache


def test_analyze_prints_the_report(
    fake_github: FakeGitHub, env: Path, capsys: pytest.CaptureFixture
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    assert main(["owner/repo"]) == 0
    captured = capsys.readouterr()
    assert "概览" in captured.out
    assert "地里长出来的" in captured.out
    assert "被人安利来的" in captured.out
    assert "可能带来更多 star 的人" in captured.out
    assert "明显异常的时段" in captured.out
    assert first_line(captured.out).startswith("owner/repo · ")


def test_progress_and_logs_go_to_stderr(
    fake_github: FakeGitHub, env: Path, capsys: pytest.CaptureFixture
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    assert main(["https://github.com/owner/repo"]) == 0
    captured = capsys.readouterr()
    assert "抓取 star 时间线" in captured.err
    assert "抓取 star 时间线" not in captured.out


def test_missing_repository_returns_exit_code_four(
    fake_github: FakeGitHub, env: Path
) -> None:
    fake_github.repo_missing = True
    fake_github.setup_method_like([("a", 1_700_000_000.0)], {"a": 1})
    assert main(["owner/repo"]) == 4


def test_non_collaborator_is_refused_up_front(
    fake_github: FakeGitHub, env: Path, capsys: pytest.CaptureFixture
) -> None:
    stars, followers = synthetic_history()
    fake_github.setup_method_like(stars, followers)
    fake_github.collaborator = False
    assert main(["owner/repo"]) == 4
    err = capsys.readouterr().err
    assert "没有协作者权限" in err
    assert fake_github.counts("stargazers:") == 0
    assert fake_github.counts("connection:") == 0


def test_stargazers_404_explains_the_restriction(
    fake_github: FakeGitHub, env: Path, capsys: pytest.CaptureFixture
) -> None:
    fake_github.setup_method_like([("someone", 1_700_000_000.0)], {"someone": 3})
    fake_github.stargazers_missing = True
    fake_github.connection_enabled = False
    assert main(["owner/repo"]) == 4
    err = capsys.readouterr().err
    assert "本身可以读取" in err
    assert "仅对仓库管理员与协作者开放" in err


def test_bad_repository_argument_returns_exit_code_two(env: Path) -> None:
    assert main(["not-a-repo"]) == 2


def test_repo_argument_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_subcommands_are_gone_and_hinted(capsys: pytest.CaptureFixture) -> None:
    for legacy in ("analyze", "report"):
        assert main([legacy, "owner/repo"]) == 2
    err = capsys.readouterr().err
    assert "不再需要子命令" in err


@pytest.mark.parametrize(
    "flag",
    [
        "--token",
        "--window",
        "--windows",
        "--candidate-pool",
        "--no-prune",
        "--prune-verify",
        "--fetch-followers",
        "--workers",
        "--attribution-k",
        "--spike-top-k",
        "--spike-alpha",
        "--max-users",
        "--since",
        "--until",
        "--user-ttl-days",
        "--refresh",
        "--dry-run",
        "--cache-dir",
        "--top",
        "--events",
        "--min-credit",
        "--plain",
        "--notes",
        "--no-wait",
        "--quiet",
        "--from-cache",
        "--tone",
        "--out",
        "--format",
        "--inline-data",
        "--offline",
        "--echarts-asset",
        "--api-base",
    ],
)
def test_no_extra_flags_exist(flag: str, env: Path) -> None:
    """The CLI takes a repository and nothing else."""

    with pytest.raises(SystemExit) as excinfo:
        main(["owner/repo", flag, "value"])
    assert excinfo.value.code == 2


def first_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.strip())
