"""
scripts to run py50 as an MCP server
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import threading
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

from py50 import Stats
from py50.utils import _normalize_units

logger = logging.getLogger("py50.mcp")

# py50 touches global pyplot state + stdout
# FastMCP threads tools, so serialize all calls.
_PY50_LOCK = threading.Lock()

_CACHE_SIZE = 8
_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx"}
_SCALE_TO_NM = {"pM": 1e-3, "nM": 1.0, "µM": 1e3}

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
    """Validate an input tabular file path (.csv, .tsv, or .xlsx)."""
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
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ToolError(
            f"Expected a .csv, .tsv, or .xlsx file, got {path.suffix!r}: {path}"
        )
    return path


def list_excel_sheets(filepath: str) -> list[str]:
    """Return worksheet names in an .xlsx file."""
    path = resolve_path(filepath)
    if path.suffix.lower() != ".xlsx":
        raise ToolError(f"Worksheet listing only applies to .xlsx files, got {path.name}.")
    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ToolError(
            "Reading .xlsx files requires 'openpyxl'. Install with: pip install 'py50[mcp]'"
        ) from exc
    try:
        with pd.ExcelFile(path, engine="openpyxl") as xl:
            return list(xl.sheet_names)
    except Exception as exc:
        raise ToolError(f"Could not open workbook {path.name}: {exc}") from exc


@lru_cache(maxsize=_CACHE_SIZE)
def _parse_table(
    path_str: str, suffix: str, mtime_ns: int, size: int, sheet: str | None
) -> pd.DataFrame:
    # mtime_ns and size participate in the cache key only: edit the file on disk
    # and the next call reparses it automatically.
    del mtime_ns, size
    if suffix == ".csv":
        return pd.read_csv(path_str)
    if suffix == ".tsv":
        return pd.read_csv(path_str, sep="\t")
    return pd.read_excel(path_str, sheet_name=sheet, engine="openpyxl")


def load_frame(filepath: str, sheet: str | None = None) -> pd.DataFrame:
    """Load a .csv, .tsv, or .xlsx table, memoized on (path, mtime, size, sheet)."""
    path = resolve_path(filepath)
    suffix = path.suffix.lower()
    resolved_sheet: str | None = None
    if suffix == ".xlsx":
        sheets = list_excel_sheets(str(path))
        if not sheets:
            raise ToolError(f"{path.name} contains no worksheets.")
        if sheet is None:
            resolved_sheet = sheets[0]
        elif sheet not in sheets:
            raise ToolError(
                f"Worksheet {sheet!r} not found in {path.name}. "
                f"Available worksheets: {sheets}."
            )
        else:
            resolved_sheet = sheet
    elif sheet is not None:
        raise ToolError(
            f"sheet={sheet!r} is only supported for .xlsx files, got {path.name}."
        )

    stat = path.stat()
    try:
        frame = _parse_table(
            str(path), suffix, stat.st_mtime_ns, stat.st_size, resolved_sheet
        )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not parse {path.name} ({suffix}): {exc}") from exc
    if frame.empty:
        raise ToolError(f"{path.name} parsed successfully but has no data rows.")
    # PlotCurve.multi_curve_plot mutates self.data in place, so never hand out the cached object.
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


def one_or_many(columns: list[str]) -> str | list[str]:
    """Unwrap a single-column list for py50.

    The tool schema always takes a list -- a union of ``str | list[str]`` made
    small local models collapse to the string branch. py50 itself still wants a
    bare string for one column: a 1-element list routes through its replicate
    averaging path instead, which renames the column and enables error bars.
    """
    return columns[0] if len(columns) == 1 else columns


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


def resolve_units(input_units: str, output_units: str | None = None) -> tuple[str, str]:
    """Normalize input and optional output units, defaulting output_units to input_units."""
    in_unit = _normalize_units(input_units)
    out_unit = in_unit if output_units is None else _normalize_units(output_units)
    return in_unit, out_unit


def scale_concentrations_to_nm(
    frame: pd.DataFrame, concentration_col: str, input_units: str
) -> pd.DataFrame:
    """Convert ``concentration_col`` to nM on a copy when ``input_units`` is not nM.

    ``PlotCurve`` assumes input concentrations are in nM and rescales from nM to
    ``conc_unit``. Normalizing the local frame copy to nM first lets the MCP
    layer separate ``input_units`` from ``output_units`` without mutating cached
    frames or breaking ``PlotCurve`` backwards compatibility.
    """
    in_unit = _normalize_units(input_units)
    scale = _SCALE_TO_NM[in_unit]
    if scale == 1.0:
        return frame
    scaled = frame.copy()
    scaled[concentration_col] = scaled[concentration_col].astype(float) * scale
    return scaled


def infer_xscale_ticks(
    concentrations: pd.Series, input_units: str, output_units: str
) -> tuple[int, int] | None:
    """Infer base-10 exponent bounds (lo, hi) from positive concentrations in output_units."""
    in_unit = _normalize_units(input_units)
    out_unit = _normalize_units(output_units)
    scale = _SCALE_TO_NM[in_unit] / _SCALE_TO_NM[out_unit]
    vals = pd.to_numeric(concentrations, errors="coerce").astype(float) * scale
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.empty:
        return None
    lo = math.floor(math.log10(float(pos.min())))
    hi = math.ceil(math.log10(float(pos.max())))
    if lo == hi:
        return (lo - 1, hi + 1)
    return (lo, hi)


def validate_bounds(
    bounds: tuple[float, float] | None, name: str
) -> tuple[float, float] | None:
    """Validate an optional 2-element [min, max] pair with finite min < max."""
    if bounds is None:
        return None
    lo, hi = bounds
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise ToolError(
            f"{name} must be [min, max] with finite min < max, got {list(bounds)}."
        )
    return (float(lo), float(hi))


def average_replicates(
    frame: pd.DataFrame, response_cols: list[str]
) -> tuple[pd.DataFrame, str]:
    """Average replicate columns into one column on a copy when len > 1."""
    if len(response_cols) == 1:
        return frame, response_cols[0]
    avg_df = frame.copy()
    avg_col = "_mcp_response_avg"
    while avg_col in avg_df.columns:
        avg_col = f"_{avg_col}"
    avg_df[avg_col] = avg_df[response_cols].mean(axis=1)
    return avg_df, avg_col


def clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records (np scalars -> Python, tuple -> list, NaN/Inf -> None)."""
    cleaned: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, tuple):
                value = [v.item() if isinstance(v, np.generic) else v for v in value]
            elif isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                value = None
            item[str(key)] = value
        cleaned.append(item)
    return cleaned


def new_figure(figsize: tuple[float, float] = (6.4, 4.8)) -> Any:
    """Create a fresh Matplotlib figure inside an active ``py50_session``."""
    return plt.figure(figsize=figsize)


def current_figure() -> Any:
    """Return the active Matplotlib figure inside an active ``py50_session``."""
    return plt.gcf()


def _require_param(name: str, value: str | None, test: str) -> str:
    if not value:
        raise ToolError(f"{name} is required when test={test!r}.")
    return value


def validate_stats_inputs(
    frame: pd.DataFrame,
    *,
    source: str,
    test: str,
    value_col: str,
    group_col: str | None = None,
    subgroup_col: str | None = None,
    subject_col: str | None = None,
    within_subject_col: str | None = None,
) -> None:
    """Validate required columns and group counts before calling Stats/Plots."""
    require_columns(
        frame,
        source,
        value_col=value_col,
        group_col=group_col,
        subgroup_col=subgroup_col,
        subject_col=subject_col,
        within_subject_col=within_subject_col,
    )
    require_numeric(frame, source, value_col=value_col)
    if subgroup_col and test not in ("mannu", "wilcoxon", "friedman"):
        raise ToolError(
            f"subgroup_col is only supported with test='mannu' or 'wilcoxon' "
            f"(or as subject for 'friedman'), not test={test!r}."
        )

    if test in (
        "homoscedasticity",
        "anova",
        "welch-anova",
        "tukey",
        "gameshowell",
        "kruskal",
        "wilcoxon",
        "mannu",
    ):
        g_col = _require_param("group_col", group_col, test)
        if test in ("wilcoxon", "mannu"):
            if subgroup_col:
                pairs_df = frame[[g_col, subgroup_col]].dropna().drop_duplicates()
                if len(pairs_df) < 2:
                    raise ToolError(
                        f"test={test!r} requires at least 2 distinct ({g_col!r}, {subgroup_col!r}) groups."
                    )
            elif frame[g_col].dropna().nunique() < 2:
                raise ToolError(
                    f"test={test!r} requires at least 2 distinct groups in {g_col!r}."
                )
    elif test in ("rm-anova", "pairwise-rm"):
        _require_param("within_subject_col", within_subject_col, test)
        _require_param("subject_col", subject_col, test)
    elif test in ("mixed-anova", "pairwise-mixed"):
        _require_param("group_col", group_col, test)
        _require_param("within_subject_col", within_subject_col, test)
        _require_param("subject_col", subject_col, test)
    elif test == "friedman":
        _require_param("group_col", group_col, test)
        if not (subject_col or subgroup_col):
            raise ToolError(
                "subject_col (or subgroup_col) is required when test='friedman'."
            )
    elif test in ("pairwise-parametric", "pairwise-nonparametric"):
        if not (group_col or within_subject_col):
            raise ToolError(
                f"At least one of group_col or within_subject_col is required when test={test!r}."
            )
        if within_subject_col:
            _require_param("subject_col", subject_col, test)


def run_stats_test(
    frame: pd.DataFrame,
    *,
    source: str,
    test: str,
    value_col: str,
    group_col: str | None = None,
    subgroup_col: str | None = None,
    subject_col: str | None = None,
    within_subject_col: str | None = None,
    alternative: str = "two-sided",
    padjust: str = "none",
) -> pd.DataFrame:
    """Dispatch a validated statistical test on ``frame`` via ``py50.Stats``."""
    validate_stats_inputs(
        frame,
        source=source,
        test=test,
        value_col=value_col,
        group_col=group_col,
        subgroup_col=subgroup_col,
        subject_col=subject_col,
        within_subject_col=within_subject_col,
    )
    stats = Stats(frame)
    if test == "normality":
        if group_col:
            result = stats.get_normality(value_col=value_col, group_col=group_col)
            return result.reset_index(names=result.index.name or group_col)
        return Stats(frame[[value_col]]).get_normality().reset_index(names="column")
    if test == "homoscedasticity":
        return stats.get_homoscedasticity(
            value_col=value_col, group_col=group_col
        ).reset_index(names="method")
    if test == "anova":
        return stats.get_anova(value_col=value_col, group_col=group_col)
    if test == "welch-anova":
        return stats.get_welch_anova(value_col=value_col, group_col=group_col)
    if test == "rm-anova":
        return stats.get_rm_anova(
            value_col=value_col,
            within_subject_col=within_subject_col,
            subject_col=subject_col,
        )
    if test == "mixed-anova":
        return stats.get_mixed_anova(
            value_col=value_col,
            group_col=group_col,
            within_subject_col=within_subject_col,
            subject_col=subject_col,
        )
    if test == "tukey":
        return stats.get_tukey(value_col=value_col, group_col=group_col)
    if test == "gameshowell":
        return stats.get_gameshowell(value_col=value_col, group_col=group_col)
    if test == "kruskal":
        return stats.get_kruskal(value_col=value_col, group_col=group_col).reset_index(
            drop=True
        )
    if test == "friedman":
        subj = subject_col or subgroup_col
        return stats.get_friedman(
            group_col=group_col, value_col=value_col, subgroup_col=subj
        ).reset_index(drop=True)
    if test == "wilcoxon":
        return stats.get_wilcoxon(
            value_col=value_col,
            group_col=group_col,
            subgroup_col=subgroup_col,
            alternative=alternative,
        )
    if test == "mannu":
        return stats.get_mannu(
            value_col=value_col,
            group_col=group_col,
            subgroup_col=subgroup_col,
            alternative=alternative,
        )
    if test in ("pairwise-parametric", "pairwise-nonparametric"):
        return stats.get_pairwise_tests(
            value_col=value_col,
            group_col=group_col,
            within_subject_col=within_subject_col,
            subject_col=subject_col,
            parametric=(test == "pairwise-parametric"),
            padjust=padjust,
        )
    if test == "pairwise-rm":
        return stats.get_pairwise_rm(
            value_col=value_col,
            group_col=group_col,
            within_subject_col=within_subject_col,
            subject_col=subject_col,
            padjust=padjust,
        )
    if test == "pairwise-mixed":
        return stats.get_pairwise_mixed(
            value_col=value_col,
            group_col=group_col,
            within_subject_col=within_subject_col,
            subject_col=subject_col,
            padjust=padjust,
        )
    raise ToolError(f"Unsupported test {test!r}.")


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
    if target.suffix.lower() != ".png":
        raise ToolError(
            f"savepath must end in .png, got {target.suffix!r}: {target}"
        )
    output_dir = os.environ.get("PY50_MCP_OUTPUT_DIR", "").strip()
    if output_dir:
        allowed_root = Path(output_dir).expanduser().resolve()
        resolved_target = target.resolve()
        if not resolved_target.is_relative_to(allowed_root):
            raise ToolError(
                f"savepath {target} is outside PY50_MCP_OUTPUT_DIR ({allowed_root}). "
                f"Pass a .png path inside {allowed_root}."
            )
        target = resolved_target
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
    table_markdown: str | None = None,
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
    if table_markdown:
        text += f"\n\n{table_markdown}"
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
