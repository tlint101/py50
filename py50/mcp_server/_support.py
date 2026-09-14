"""
scripts to run py50 as an MCP server
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

logger = logging.getLogger("py50.mcp")

# py50 touches global pyplot state + stdout
# FastMCP threads tools, so serialize all calls.
_PY50_LOCK = threading.Lock()

_CACHE_SIZE = 8

# reduce image pixel count to save generated token cost
# if savepath given, then full image will be generated at 300 dpi
INLINE_DPI = 110
INLINE_MAX_PX = 1200
FILE_DPI = 300


# --------------------------------------------------------------------------- #
# py50 call session
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def py50_session() -> Iterator[io.StringIO]:
    """Serialize one py50 call, capture its stdout, and close the figures it opens.

    Yields the stdout buffer. Read it -- and render any figure -- *inside* the
    with-block, because every figure opened during the call is closed on exit.
    """
    buffer = io.StringIO()
    with _PY50_LOCK:
        before = set(plt.get_fignums())
        try:
            with contextlib.redirect_stdout(buffer):
                yield buffer
        finally:
            leaked = set(plt.get_fignums()) - before
            for num in leaked:
                plt.close(num)
            # This server owns pyplot outright, so nothing should survive a tool
            # call and contaminate the next one.
            plt.close("all")
            text = buffer.getvalue().strip()
            if text:
                logger.info("py50 stdout: %s", text.replace("\n", " | "))


def diagnostics(buffer: io.StringIO) -> list[str]:
    """py50's captured stdout, as clean lines suitable for returning to the model."""
    return [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Data loading and validation
# --------------------------------------------------------------------------- #


def resolve_path(filepath: str) -> Path:
    """Validate an input CSV path, raising ToolError with an actionable message."""
    path = Path(filepath).expanduser()
    if not path.is_absolute():
        raise ToolError(
            f"{filepath!r} is a relative path. This server is launched by an MCP "
            "client and has no meaningful working directory -- pass an absolute path."
        )
    if not path.exists():
        raise ToolError(f"No such file: {path}")
    if not path.is_file():
        raise ToolError(f"Not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise ToolError(f"Expected a .csv file, got {path.suffix!r}: {path}")
    return path


@lru_cache(maxsize=_CACHE_SIZE)
def _parse_csv(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    # mtime_ns and size participate in the cache key only: edit the file on disk
    # and the next call reparses it automatically.
    del mtime_ns, size
    return pd.read_csv(path_str)


def load_frame(filepath: str) -> pd.DataFrame:
    """Load a CSV, memoized on (path, mtime, size). Always returns a fresh copy."""
    path = resolve_path(filepath)
    stat = path.stat()
    try:
        frame = _parse_csv(str(path), stat.st_mtime_ns, stat.st_size)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not parse {path.name} as CSV: {exc}") from exc
    if frame.empty:
        raise ToolError(f"{path.name} parsed successfully but has no data rows.")
    # PlotCurve.multi_curve_plot mutates self.data in place (it adds an
    # 'inhibition_average' column), so never hand out the cached object itself.
    return frame.copy()


def require_columns(
    frame: pd.DataFrame,
    source: str,
    **requested: str | list[str] | None,
) -> None:
    """Check that named columns exist, naming the offending argument on failure."""
    problems: list[str] = []
    for arg, value in requested.items():
        if value is None:
            continue
        wanted = [value] if isinstance(value, str) else list(value)
        missing = [col for col in wanted if col not in frame.columns]
        if missing:
            problems.append(f"{arg}={missing}")
    if problems:
        raise ToolError(
            f"Column(s) not found in {source}: {'; '.join(problems)}. "
            f"Available columns: {list(frame.columns)}. "
            "Call read_csv on this file and use the exact column names it reports."
        )


def require_numeric(
    frame: pd.DataFrame,
    source: str,
    **requested: str | list[str] | None,
) -> None:
    """Check that named columns hold numeric data. Run after require_columns."""
    bad: list[str] = []
    for arg, value in requested.items():
        if value is None:
            continue
        for col in [value] if isinstance(value, str) else list(value):
            if not pd.api.types.is_numeric_dtype(frame[col]):
                bad.append(f"{arg}={col!r} (dtype {frame[col].dtype})")
    if bad:
        raise ToolError(
            f"Non-numeric column(s) in {source}: {'; '.join(bad)}. "
            "Concentration and response columns must contain numbers. If the "
            "column holds text such as '>100', clean the CSV first."
        )


def compound_names(frame: pd.DataFrame, name_col: str) -> list[str]:
    """Unique compound names in name_col, as strings, in first-seen order."""
    return frame[name_col].astype(str).drop_duplicates().tolist()


def select_compound(
    frame: pd.DataFrame,
    name_col: str,
    query: str | None,
    source: str,
) -> tuple[pd.DataFrame, str]:
    """Exact-match one compound and return (filtered_frame, resolved_name).

    Filtering here rather than in py50 is load-bearing twice over:

    * ``PlotCurve._filter_dataframe`` returns the frame *unfiltered* when the
      query is None, so py50 would silently pool every compound into one 4PL fit
      and label it with the first compound's name.
    * That same method substring-matches the query against every row joined into
      one string, so ``"Drug 1"`` would also select ``"Drug 10"``.

    Handing py50 a single-compound frame makes its internal match a no-op.
    """
    names = compound_names(frame, name_col)
    if query is None:
        if len(names) != 1:
            raise ToolError(
                f"{source} contains {len(names)} compounds in column {name_col!r}: "
                f"{names}. Pass query=<exact compound name> to plot one of them, "
                "or use multi_curve to plot them all on shared axes."
            )
        query = names[0]
    elif query not in names:
        near = [n for n in names if query.casefold() in n.casefold()]
        hint = f" Closest match(es): {near}." if near else ""
        raise ToolError(
            f"Compound {query!r} not found in column {name_col!r} of {source}. "
            f"Available compounds: {names}.{hint}"
        )
    selected = frame.loc[frame[name_col].astype(str) == query].copy()
    return selected, query


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #


def _inline_dpi(fig: Any) -> float:
    """Cap the inline render so a large figsize cannot flood the context."""
    width_in, height_in = fig.get_size_inches()
    return min(INLINE_DPI, INLINE_MAX_PX / max(width_in, height_in))


def _write_file(fig: Any, savepath: str) -> str:
    target = Path(savepath).expanduser()
    if not target.is_absolute():
        raise ToolError(
            f"savepath must be an absolute path, got {savepath!r}. This server has "
            "no meaningful working directory."
        )
    if not target.parent.is_dir():
        raise ToolError(f"Cannot write {target.name}: no such directory {target.parent}")
    try:
        fig.savefig(target, dpi=FILE_DPI, bbox_inches="tight")
    except OSError as exc:
        raise ToolError(f"Could not write {target}: {exc}") from exc
    return str(target)


def render_result(
    fig: Any,
    *,
    summary: str,
    savepath: str | None = None,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> ToolResult:
    """Return a figure as an inline PNG plus text and structured metadata.

    Must be called inside the ``py50_session`` block that produced ``fig``.
    """
    notes = notes or []
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=_inline_dpi(fig), bbox_inches="tight")
    png = buffer.getvalue()

    # Write the publication copy ourselves rather than letting py50 call
    # plt.savefig, which targets the *current* figure and validates nothing.
    written = _write_file(fig, savepath) if savepath else None

    text = summary
    if written:
        text += f"\nSaved a {FILE_DPI} dpi copy to {written}"
    if notes:
        text += "\nNotes from py50: " + "; ".join(notes)

    return ToolResult(
        content=[
            Image(data=png, format="png").to_image_content(),
            TextContent(type="text", text=text),
        ],
        structured_content={
            "summary": summary,
            "savepath": written,
            "inline_png_bytes": len(png),
            "notes": notes,
            **(extra or {}),
        },
    )
