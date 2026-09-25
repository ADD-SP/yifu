"""Validate the GraphQL query against GitHub's published schema.

Optional: skips unless ``YIFU_GH_SCHEMA`` points at a downloaded schema and
``graphql-core`` is importable. See the README for the two commands.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yifu.github import _CONNECTION_STARGAZERS_QUERY, _FOLLOWERS_QUERY


@pytest.mark.parametrize("query", [_FOLLOWERS_QUERY, _CONNECTION_STARGAZERS_QUERY])
def test_query_validates_against_official_schema(query: str) -> None:
    schema_path = os.environ.get("YIFU_GH_SCHEMA")
    if not schema_path:
        pytest.skip("设置 YIFU_GH_SCHEMA=<schema.graphql> 后运行该校验")
    graphql = pytest.importorskip(
        "graphql", reason="需要 graphql-core：uv run --with graphql-core pytest"
    )
    schema = graphql.build_schema(Path(schema_path).read_text(encoding="utf-8"))
    errors = [error.message for error in graphql.validate(schema, graphql.parse(query))]
    assert errors == []


def test_schema_documents_the_fields_we_rely_on() -> None:
    """Pin the two schema facts that broke batching, using the fake's own rules."""

    from conftest import FakeGitHub

    fake = FakeGitHub(server=None)
    assert fake._schema_errors(_FOLLOWERS_QUERY) == []
    broken = """
    query($ids: [ID!]!) {
      nodes(ids: $ids) {
        ... on User { login followers { total } }
        ... on Organization { login followers { total } }
      }
    }
    """
    messages = [error["message"] for error in fake._schema_errors(broken)]
    assert len(messages) == 2
