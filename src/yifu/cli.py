"""Command line interface: ``yifu <repo>``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .analyze import AnalyzeOptions, run_analysis
from .console import render_report
from .github import CacheStore, GitHubClient, GitHubError, parse_repo_url
from .progress import make_reporter

DEFAULT_API_BASE = "https://api.github.com"
TOP_N = 20
EVENT_LIMIT = 10
MIN_CREDIT = 3.0


def cache_dir() -> Path:
    override = os.environ.get("YIFU_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "yifu"


def api_base() -> str:
    return os.environ.get("YIFU_API_BASE") or DEFAULT_API_BASE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yifu",
        description="推断 GitHub 仓库里谁的 star 带来了更多 star",
        epilog=(
            "只需给出 owner/repo，没有任何额外开关。"
            "前置条件：GitHub 自 2026 年 7 月起只向仓库管理员与协作者提供 stargazers 名单，"
            "因此本工具只能分析你的 token 有写权限的仓库，开始前会自动检查。"
            "凭证读 GITHUB_TOKEN / GH_TOKEN；API 地址与缓存目录可用 YIFU_API_BASE / YIFU_CACHE_DIR 覆盖。"
        ),
    )
    parser.add_argument("repo", nargs="?", help="仓库地址，写 owner/repo 就行（也接受完整 URL 或 SSH 形式）")
    parser.add_argument("--version", action="version", version=f"yifu {__version__}")
    return parser


def command_analyze(args: argparse.Namespace) -> int:
    owner, repo = parse_repo_url(args.repo)
    cache = CacheStore(cache_dir())
    reporter = make_reporter()
    client = GitHubClient(
        token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None,
        api_base=api_base(),
        log=reporter.log,
    )
    payload = run_analysis(
        client, cache, owner, repo, AnalyzeOptions(), progress=reporter.progress, log=reporter.log
    )
    reporter.finish()
    print()
    print(render_report(payload, top=TOP_N, events=EVENT_LIMIT, min_credit=MIN_CREDIT))
    unavailable = [
        login
        for login, profile in (payload.get("users") or {}).items()
        if profile.get("source") == "unavailable"
    ]
    if unavailable:
        print(f"\n注意：{len(unavailable)} 个候选账号资料不可用（已注销或接口受限）")
    return 6 if unavailable else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    # The subcommand layer is gone; point that out instead of failing obscurely.
    first_positional = next((item for item in raw if not item.startswith("-")), None)
    if first_positional in {"analyze", "report"}:
        print("不再需要子命令：直接 `yifu <仓库地址>`。", file=sys.stderr)
        return 2
    args = parser.parse_args(raw)
    try:
        if not args.repo:
            parser.error("请提供仓库地址，例如：yifu owner/repo")
        return command_analyze(args)
    except GitHubError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("\n已中断，进度已保存在缓存中，可重新运行继续。", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
