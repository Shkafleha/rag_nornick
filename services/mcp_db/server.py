import json
import os
from typing import Optional

import duckdb
from fastmcp import FastMCP

mcp = FastMCP("parquet-tools")

PARQUET_PATH = os.getenv("PARQUET_PATH", "data.parquet")
LAB_DICT_PATH = os.getenv("LAB_DICT_PATH", "lab_dict_search_clean.json")

con = duckdb.connect()

# Tag dictionary lives next to the data it describes. rag_api no longer needs it.
with open(LAB_DICT_PATH, "r", encoding="utf-8") as f:
    LAB_TAGS_DICT: dict[str, str] = json.load(f)


# ── Tag resolution (no LLM here — mcp_db stays dependency-free) ────────────────

def _find_tag_literal(query: str) -> Optional[str]:
    """Exact match: a tag id literally present in the query text."""
    for tag in LAB_TAGS_DICT:
        if tag in query:
            return tag
    return None


@mcp.tool()
def resolve_tag(query: str) -> str:
    """Resolve a lab-indicator tag from a natural-language query.

    Returns JSON: {"exact": "<tag>"|null, "candidates": [{"tag","desc"}, ...]}.
    - "exact" is set when a tag id appears verbatim in the query.
    - "candidates" is the full tag dictionary, for an LLM to pick from. The dict is
      small (~130 tags), so we hand it over wholesale instead of a lossy word-overlap
      pre-filter that misses Russian inflections (католите≠католит, никеля≠ni).
    """
    exact = _find_tag_literal(query)
    candidates = [{"tag": tag, "desc": desc} for tag, desc in LAB_TAGS_DICT.items()]
    return json.dumps({"exact": exact, "candidates": candidates}, ensure_ascii=False)


# ── Time-series access ────────────────────────────────────────────────────────

def _resolve_window(date_from: Optional[str], date_to: Optional[str]) -> tuple[str, str]:
    """Default to the full available range when bounds are omitted."""
    if date_from and date_to:
        return date_from, date_to
    bounds = con.execute(
        "SELECT min(DateTime), max(DateTime) FROM read_parquet(?)",
        [PARQUET_PATH],
    ).fetchone()
    lo, hi = (str(bounds[0]), str(bounds[1])) if bounds and bounds[0] else (
        "1970-01-01T00:00:00", "2100-01-01T00:00:00"
    )
    return (date_from or lo), (date_to or hi)


@mcp.tool()
def values_between(
    tag: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Get raw values for one TagName. Dates are optional (defaults to full range)."""
    lo, hi = _resolve_window(date_from, date_to)
    result = con.execute(
        """
        SELECT DateTime, TagName, Value
        FROM read_parquet(?)
        WHERE TagName = ?
          AND DateTime BETWEEN ? AND ?
        ORDER BY DateTime
        LIMIT ?
        """,
        [PARQUET_PATH, tag, lo, hi, min(limit, 500)],
    )
    cols = [d[0] for d in result.description]
    rows = [dict(zip(cols, row)) for row in result.fetchall()]
    return json.dumps(rows, ensure_ascii=False, default=str)


@mcp.tool()
def aggregate(
    tag: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Summary stats for one TagName over a window: count, min, max, avg, first, last.

    Use this instead of values_between when the question is about a level, a typical
    value, or a trend rather than the individual readings.
    """
    lo, hi = _resolve_window(date_from, date_to)
    row = con.execute(
        """
        WITH w AS (
            SELECT DateTime, Value
            FROM read_parquet(?)
            WHERE TagName = ?
              AND DateTime BETWEEN ? AND ?
        )
        SELECT
            count(*)                              AS n,
            min(Value)                            AS min_value,
            max(Value)                            AS max_value,
            avg(Value)                            AS avg_value,
            first(Value ORDER BY DateTime)        AS first_value,
            last(Value ORDER BY DateTime)         AS last_value,
            min(DateTime)                         AS first_dt,
            max(DateTime)                         AS last_dt
        FROM w
        """,
        [PARQUET_PATH, tag, lo, hi],
    ).fetchone()

    cols = ["n", "min_value", "max_value", "avg_value",
            "first_value", "last_value", "first_dt", "last_dt"]
    out = dict(zip(cols, row)) if row else {"n": 0}
    out["tag"] = tag
    out["desc"] = LAB_TAGS_DICT.get(tag, "")
    return json.dumps(out, ensure_ascii=False, default=str)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8090)
