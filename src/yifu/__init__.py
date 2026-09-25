"""yifu: infer which stargazers brought the most stars to a GitHub repository."""

from __future__ import annotations

from collections.abc import Sequence

__version__ = "0.1.0"
__all__ = ["__version__", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)
