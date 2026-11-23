from __future__ import annotations
import logging
import os
import sys
from typing import Annotated, Any, Literal

# keep matplotlib image from displaying to avoid MCP server issues
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from fastmcp import Context, FastMCP
    from fastmcp.exceptions import ToolError
    from fastmcp.tools import ToolResult
    from pydantic import BeforeValidator, Field
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"The py50 MCP server needs {exc.name!r}, which is not installed. py50's MCP "
        "dependencies are optional -- install them with:\n\n"
        "    pip install 'py50[mcp]'\n\n"
        "or run the server without installing anything:\n\n"
        "    uvx --from 'py50[mcp]' py50-mcp\n"
    ) from exc

from py50 import Calculator, PlotCurve, Plots, Stats, __version__
from py50.mcp_server._support import (
    average_replicates,
    clean_records,
    compound_names,
    current_figure,
    diagnostics,
    infer_xscale_ticks,
    list_excel_sheets,
    load_frame,
    new_figure,
    one_or_many,
    py50_session,
    render_result,
    require_columns,
    require_numeric,
    resolve_path,
    resolve_units,
    run_stats_test,
    scale_concentrations_to_nm,
    select_compound,
    validate_bounds,
    validate_stats_inputs,
)
from py50.plot_settings import CBMARKERS, CBPALETTE

logger = logging.getLogger("py50.mcp")


def _normalize_mu(value: Any) -> Any:
    """Normalize Greek small mu (U+03BC) to micro sign (U+00B5) before Literal validation."""
    return value.replace("\u03bc", "\u00b5") if isinstance(value, str) else value


ConcUnit = Annotated[
    Literal["nM", "uM", "\u00b5M", "pM"],
    BeforeValidator(_normalize_mu),
]
XScale = Literal["log", "linear"]
ErrorBar = Literal["sd", "se", "ci", "pi"]
StatsTest = Literal[
    "normality",
    "homoscedasticity",
    "anova",
    "welch-anova",
    "rm-anova",
    "mixed-anova",
    "tukey",
    "gameshowell",
    "wilcoxon",
    "mannu",
    "kruskal",
    "friedman",
    "pairwise-parametric",
    "pairwise-nonparametric",
    "pairwise-rm",
    "pairwise-mixed",
]
PlotStatsTest = Literal[
    "tukey",
    "gameshowell",
    "pairwise-parametric",
    "pairwise-nonparametric",
    "pairwise-rm",
    "pairwise-mixed",
    "wilcoxon",
    "mannu",
]
StatsPlotType = Literal[
    "boxplot",
    "barplot",
    "violinplot",
    "swarmplot",
    "stripplot",
    "boxenplot",
    "pointplot",
    "ci_plot",
    "p_matrix",
    "histplot",
    "qqplot",
]
Alternative = Literal["two-sided", "greater", "less"]
PAdjust = Literal["none", "bonf", "sidak", "holm", "fdr_bh", "fdr_by"]
Orient = Literal["v", "h"]

# param descriptions
D_FILEPATH = "Absolute path to a .csv, .tsv, or .xlsx file."
D_SHEET = "Excel worksheet name (.xlsx only; defaults to the first sheet)."
D_CONC_COL = "Numeric concentration column name (interpreted in input_units)."
D_NAME_COL = "Compound-name column name."
D_RESPONSE_COL = (
    "Response column name(s) as a list of separate strings, one per column: "
    "['% Inhibition 1', '% Inhibition 2'] (NOT ['% Inhibition 1 and % Inhibition 2']). "
    "Multiple columns are averaged before fitting."
)
D_RESPONSE_COL_PLOT = (
    D_RESPONSE_COL + " Multiple columns are plotted with error bars."
)
D_INPUT_UNITS = (
    "Unit of values in concentration_col ('nM', 'uM'/'µM', 'pM'). Default 'nM'."
)
D_OUTPUT_UNITS = (
    "Target unit for reported IC50s or x-axis. Omit to keep input_units."
)
D_XSCALE = "X-axis scale ('log' or 'linear'). Default 'log'."
D_XSCALE_TICKS = (
    "X-axis [min_exp, max_exp] as base-10 exponents in the plotted unit "
    "(e.g. [-3, 2] for 0.001..100). Omit to infer from data."
)
D_YLIM = "Y-axis [min, max] limits. Omit to autoscale."
D_ERRORBAR = (
    "Replicate error bar type ('sd', 'se', 'ci', 'pi'). Ignored for one response column."
)
D_BOX_INTERCEPT = "Response % where IC50 guide lines meet the curve. Default 50."
D_SAVEPATH = (
    "Optional absolute .png path for a 300 dpi file copy "
    "(plot is always returned inline)."
)

# setup mcp
mcp = FastMCP(
    name="py50",
    version=__version__,
    instructions=(
        "py50 fits 4PL dose-response curves and runs group statistics on .csv, .tsv, "
        "and .xlsx plate-assay files, returning tables and publication-quality PNG plots.\n"
        "\n"
        "Workflow -- follow it in order:\n"
        "1. Call read_csv FIRST on every new file (and pass sheet for .xlsx tabs) to "
        "inspect exact column names, dtypes, and group/compound names. Never guess names.\n"
        "2. For dose-response curves: calculate_ic50 (IC50/pIC50/Hill table), "
        "single_curve (one compound PNG + metrics), multi_curve (overlaid curves), "
        "or grid_curve (one subplot per compound).\n"
        "3. For group statistics: run_stats (normality, ANOVA, post-hoc, and "
        "non-parametric tables) or stats_plot (categorical plots with significance "
        "brackets, ci_plot, p_matrix, histplot, qqplot).\n"
        "\n"
        "Rules:\n"
        "- Every path must be absolute.\n"
        "- Specify the file's concentration unit via input_units (default 'nM'); pass "
        "output_units only to convert results or the x-axis to a different unit.\n"
        "- Compound and column names must match exactly (case and spacing).\n"
        "- Errors list valid columns, sheets, or compounds; read them and retry."
    ),
)

_READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
# plot tools for when savepath is given
_WRITES = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


@mcp.tool(annotations=_READ_ONLY, tags={"py50"}, timeout=60)
def read_csv(
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    preview_rows: Annotated[
        int,
        Field(
            description="Number of leading data rows to include as a markdown preview.",
            ge=0,
            le=50,
        ),
    ] = 5,
    sheet: Annotated[str | None, Field(description=D_SHEET)] = None,
) -> dict[str, Any]:
    """Inspect a .csv, .tsv, or .xlsx dataset and report its columns and preview.

    Call this before any other py50 tool. It returns the exact column names,
    dtypes, missing-value counts, compound/group candidates, and (for .xlsx)
    available worksheet names, plus a short preview.
    """
    path = resolve_path(filepath)
    frame = load_frame(filepath, sheet=sheet)

    numeric_columns = [c for c in frame.columns if frame[c].dtype.kind in "iufc"]
    name_candidates: dict[str, list[str]] = {}
    for column in frame.columns:
        if column in numeric_columns:
            continue
        distinct = frame[column].astype(str).drop_duplicates().tolist()
        # A compound-name column has few distinct values relative to the row count.
        if 1 < len(distinct) <= 50 or len(distinct) == 1:
            name_candidates[column] = distinct

    result: dict[str, Any] = {
        "filepath": str(path),
        "rows": int(len(frame)),
        "columns": [
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "nulls": int(frame[column].isna().sum()),
            }
            for column in frame.columns
        ],
        "numeric_columns": numeric_columns,
        "name_col_candidates": name_candidates,
        "preview_markdown": frame.head(preview_rows).to_markdown(index=False)
        if preview_rows
        else "",
    }
    if path.suffix.lower() == ".xlsx":
        sheets = list_excel_sheets(filepath)
        result["sheets"] = sheets
        result["sheet"] = sheet or sheets[0]
    return result


@mcp.tool(annotations=_READ_ONLY, tags={"py50"}, timeout=120)
async def calculate_ic50(
    ctx: Context,
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    concentration_col: Annotated[str, Field(description=D_CONC_COL)],
    response_col: Annotated[
        list[str],
        Field(description=D_RESPONSE_COL, min_length=1),
    ],
    name_col: Annotated[str, Field(description=D_NAME_COL)],
    sheet: Annotated[str | None, Field(description=D_SHEET)] = None,
    input_units: Annotated[
        ConcUnit,
        Field(description=D_INPUT_UNITS),
    ] = "nM",
    output_units: Annotated[
        ConcUnit | None,
        Field(description=D_OUTPUT_UNITS),
    ] = None,
) -> dict[str, Any]:
    """Fit a 4PL dose-response curve per compound and report IC50 and pIC50 values.

    Returns one row per compound in name_col. Call read_csv first for the exact
    column names; do not guess them.
    """
    frame = load_frame(filepath, sheet=sheet)
    source = resolve_path(filepath).name
    require_columns(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
        name_col=name_col,
    )
    require_numeric(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
    )
    in_unit, out_unit = resolve_units(input_units, output_units)

    with py50_session() as buffer:
        table = Calculator(frame).calculate_pic50(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=one_or_many(response_col),
            input_units=in_unit,
            output_units=out_unit,
        )
        notes = diagnostics(buffer)
        records = clean_records(table)
        markdown = table.to_markdown(index=False)

    if notes:
        await ctx.info("py50: " + "; ".join(notes))

    return {
        "input_units": in_unit,
        "output_units": out_unit,
        "compounds": compound_names(frame, name_col),
        "records": records,
        "table_markdown": markdown,
        "notes": notes,
    }


@mcp.tool(annotations=_WRITES, tags={"py50", "plot"}, timeout=120)
async def single_curve(
    ctx: Context,
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    concentration_col: Annotated[str, Field(description=D_CONC_COL)],
    response_col: Annotated[
        list[str],
        Field(description=D_RESPONSE_COL_PLOT, min_length=1),
    ],
    name_col: Annotated[str, Field(description=D_NAME_COL)],
    query: Annotated[
        str | None,
        Field(
            description=(
                "Exact compound name in name_col. Optional only when the file has one compound."
            )
        ),
    ] = None,
    sheet: Annotated[str | None, Field(description=D_SHEET)] = None,
    input_units: Annotated[ConcUnit, Field(description=D_INPUT_UNITS)] = "nM",
    output_units: Annotated[
        ConcUnit | None, Field(description=D_OUTPUT_UNITS)
    ] = None,
    title: Annotated[str | None, Field(description="Figure title.")] = None,
    xlabel: Annotated[
        str | None,
        Field(description="X-axis label. Omit for 'Concentration (<unit>)'."),
    ] = None,
    ylabel: Annotated[
        str | None,
        Field(description="Y-axis label, e.g. 'Inhibition %'."),
    ] = None,
    xscale: Annotated[XScale, Field(description=D_XSCALE)] = "log",
    xscale_ticks: Annotated[
        tuple[float, float] | None, Field(description=D_XSCALE_TICKS)
    ] = None,
    ylim: Annotated[
        tuple[float, float] | None, Field(description=D_YLIM)
    ] = None,
    errorbar: Annotated[ErrorBar, Field(description=D_ERRORBAR)] = "sd",
    box: Annotated[
        bool,
        Field(description="Draw dashed IC50 guide lines at box_intercept."),
    ] = False,
    box_intercept: Annotated[float, Field(description=D_BOX_INTERCEPT)] = 50,
    savepath: Annotated[str | None, Field(description=D_SAVEPATH)] = None,
) -> ToolResult:
    """Plot the 4PL dose-response curve for one compound and return fitted metrics.

    Call read_csv first to get exact column and compound names. Returns an inline
    PNG plot plus a fitted IC50, absolute IC50, Hill slope, and pIC50 summary.

    Args:
        ctx: FastMCP request context, injected automatically.
        filepath: Absolute path to the CSV.
        concentration_col: Numeric dose column, interpreted in input_units.
        response_col: List of response column names; multiple columns are averaged.
        name_col: Column of compound names, used to select and label the compound.
        query: Exact compound name. Optional only for a single-compound file.
        input_units: Concentration unit of the input data in the CSV.
        output_units: Target concentration unit for the plot x-axis and metrics.
        box: Marks the IC50 on the plot when box_intercept is left at 50.
        savepath: Optional absolute .png path for an additional 300 dpi copy.

    Returns:
        Inline PNG image, text summary with the fitted metrics table, and
        structured metadata (including fit_records and fit_table_markdown).
    """
    frame = load_frame(filepath, sheet=sheet)
    source = resolve_path(filepath).name
    require_columns(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
        name_col=name_col,
    )
    require_numeric(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
    )
    in_unit, out_unit = resolve_units(input_units, output_units)
    explicit_ticks = validate_bounds(xscale_ticks, "xscale_ticks")
    resolved_ylim = validate_bounds(ylim, "ylim")

    # Filter here rather than in py50: it pools every compound when the query is
    # None, and substring-matches names otherwise.
    selected, resolved = select_compound(frame, name_col, query, source)
    resolved_ticks = (
        explicit_ticks
        if explicit_ticks is not None
        else infer_xscale_ticks(selected[concentration_col], in_unit, out_unit)
    )
    resolved_xlabel = (
        xlabel if xlabel is not None else f"Concentration ({out_unit})"
    )
    resolved_ylabel = (
        ylabel
        if ylabel is not None
        else (response_col[0] if len(response_col) == 1 else "Response")
    )
    plot_frame = scale_concentrations_to_nm(selected, concentration_col, in_unit)
    resp_arg = one_or_many(response_col)

    with py50_session() as buffer:
        fit_table = Calculator(selected).calculate_pic50(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=resp_arg,
            input_units=in_unit,
            output_units=out_unit,
        )
        fit_records = clean_records(fit_table)
        fit_markdown = fit_table.to_markdown(index=False)

        figure = PlotCurve(plot_frame).curve_plot(
            concentration_col=concentration_col,
            response_col=resp_arg,
            name_col=name_col,
            query=resolved,
            title=title,
            xlabel=resolved_xlabel,
            ylabel=resolved_ylabel,
            conc_unit=out_unit,
            xscale=xscale,
            xscale_ticks=resolved_ticks,
            ymin=resolved_ylim[0] if resolved_ylim else None,
            ymax=resolved_ylim[1] if resolved_ylim else None,
            line_color="black",
            errorbar=errorbar,
            legend=True,
            legend_loc="best",
            box=box,
            box_intercept=box_intercept,
            savepath=None,
        )
        notes = diagnostics(buffer)
        result = render_result(
            figure,
            summary=(
                f"Dose-response curve for {resolved!r} from {source} "
                f"({len(selected)} points, x-axis in {out_unit})."
            ),
            table_markdown=fit_markdown,
            savepath=savepath,
            notes=notes,
            extra={
                "compound": resolved,
                "points": int(len(selected)),
                "input_units": in_unit,
                "output_units": out_unit,
                "xscale_ticks": list(resolved_ticks)
                if resolved_ticks is not None
                else None,
                "fit_records": fit_records,
                "fit_table_markdown": fit_markdown,
            },
        )

    if notes:
        await ctx.info("py50: " + "; ".join(notes))
    return result


@mcp.tool(annotations=_WRITES, tags={"py50", "plot"}, timeout=120)
async def multi_curve(
    ctx: Context,
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    concentration_col: Annotated[str, Field(description=D_CONC_COL)],
    response_col: Annotated[
        list[str],
        Field(description=D_RESPONSE_COL_PLOT, min_length=1),
    ],
    name_col: Annotated[str, Field(description=D_NAME_COL)],
    sheet: Annotated[str | None, Field(description=D_SHEET)] = None,
    input_units: Annotated[ConcUnit, Field(description=D_INPUT_UNITS)] = "nM",
    output_units: Annotated[
        ConcUnit | None, Field(description=D_OUTPUT_UNITS)
    ] = None,
    title: Annotated[str | None, Field(description="Figure title.")] = None,
    xlabel: Annotated[
        str | None,
        Field(description="X-axis label. Omit for 'Concentration (<unit>)'."),
    ] = None,
    ylabel: Annotated[
        str | None,
        Field(description="Y-axis label, e.g. 'Inhibition %'."),
    ] = None,
    xscale: Annotated[XScale, Field(description=D_XSCALE)] = "log",
    xscale_ticks: Annotated[
        tuple[float, float] | None, Field(description=D_XSCALE_TICKS)
    ] = None,
    ylim: Annotated[
        tuple[float, float] | None, Field(description=D_YLIM)
    ] = None,
    errorbar: Annotated[ErrorBar, Field(description=D_ERRORBAR)] = "sd",
    box_target: Annotated[
        str | None,
        Field(
            description=(
                "Exact compound name in name_col to mark with dashed IC50 guide lines."
            )
        ),
    ] = None,
    box_intercept: Annotated[float, Field(description=D_BOX_INTERCEPT)] = 50,
    savepath: Annotated[str | None, Field(description=D_SAVEPATH)] = None,
) -> ToolResult:
    """Plot all compounds' 4PL dose-response curves on shared axes and return metrics.

    Call read_csv first to get exact column names. Returns an inline PNG plot
    plus a fitted IC50, absolute IC50, Hill slope, and pIC50 table per compound.

    Args:
        ctx: FastMCP request context, injected automatically.
        filepath: Absolute path to the CSV.
        concentration_col: Numeric dose column, interpreted in input_units.
        response_col: List of response column names; multiple columns are averaged.
        name_col: Column of compound names; one curve is drawn per distinct value.
        input_units: Concentration unit of the input data in the CSV.
        output_units: Target concentration unit for the plot x-axis and metrics.
        box_target: Annotates a single named compound with IC50 guide lines.
        savepath: Optional absolute .png path for an additional 300 dpi copy.

    Returns:
        Inline PNG image, text summary with the fitted metrics table, and
        structured metadata (including fit_records and fit_table_markdown).
    """
    frame = load_frame(filepath, sheet=sheet)
    source = resolve_path(filepath).name
    require_columns(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
        name_col=name_col,
    )
    require_numeric(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
    )
    in_unit, out_unit = resolve_units(input_units, output_units)
    explicit_ticks = validate_bounds(xscale_ticks, "xscale_ticks")
    resolved_ylim = validate_bounds(ylim, "ylim")

    names = compound_names(frame, name_col)
    if box_target is not None and box_target not in names:
        raise ToolError(
            f"box_target={box_target!r} is not a compound in column {name_col!r} of "
            f"{source}. Available compounds: {names}."
        )
    resolved_ticks = (
        explicit_ticks
        if explicit_ticks is not None
        else infer_xscale_ticks(frame[concentration_col], in_unit, out_unit)
    )
    resolved_xlabel = (
        xlabel if xlabel is not None else f"Concentration ({out_unit})"
    )
    resolved_ylabel = (
        ylabel
        if ylabel is not None
        else (response_col[0] if len(response_col) == 1 else "Response")
    )
    plot_frame = scale_concentrations_to_nm(frame, concentration_col, in_unit)
    resp_arg = one_or_many(response_col)

    with py50_session() as buffer:
        fit_table = Calculator(frame).calculate_pic50(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=resp_arg,
            input_units=in_unit,
            output_units=out_unit,
        )
        fit_records = clean_records(fit_table)
        fit_markdown = fit_table.to_markdown(index=False)

        figure = PlotCurve(plot_frame).multi_curve_plot(
            concentration_col=concentration_col,
            response_col=resp_arg,
            name_col=name_col,
            title=title,
            xlabel=resolved_xlabel,
            ylabel=resolved_ylabel,
            conc_unit=out_unit,
            xscale=xscale,
            xscale_ticks=resolved_ticks,
            ymin=resolved_ylim[0] if resolved_ylim else None,
            ymax=resolved_ylim[1] if resolved_ylim else None,
            line_color=CBPALETTE,
            marker=CBMARKERS,
            errorbar=errorbar,
            legend=True,
            legend_loc="best",
            box_target=box_target,
            box_intercept=box_intercept,
            savepath=None,
        )
        notes = diagnostics(buffer)
        result = render_result(
            figure,
            summary=(
                f"Dose-response curves for {len(names)} compounds from {source} "
                f"({', '.join(names)}; x-axis in {out_unit})."
            ),
            table_markdown=fit_markdown,
            savepath=savepath,
            notes=notes,
            extra={
                "compounds": names,
                "input_units": in_unit,
                "output_units": out_unit,
                "xscale_ticks": list(resolved_ticks)
                if resolved_ticks is not None
                else None,
                "fit_records": fit_records,
                "fit_table_markdown": fit_markdown,
            },
        )

    if notes:
        await ctx.info("py50: " + "; ".join(notes))
    return result


@mcp.tool(annotations=_WRITES, tags={"py50", "plot"}, timeout=120)
async def grid_curve(
    ctx: Context,
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    concentration_col: Annotated[str, Field(description=D_CONC_COL)],
    response_col: Annotated[
        list[str],
        Field(description=D_RESPONSE_COL, min_length=1),
    ],
    name_col: Annotated[str, Field(description=D_NAME_COL)],
    sheet: Annotated[str | None, Field(description=D_SHEET)] = None,
    column_num: Annotated[
        int, Field(description="Subplot columns per row.", ge=1, le=6)
    ] = 2,
    input_units: Annotated[ConcUnit, Field(description=D_INPUT_UNITS)] = "nM",
    output_units: Annotated[
        ConcUnit | None, Field(description=D_OUTPUT_UNITS)
    ] = None,
    title: Annotated[str | None, Field(description="Figure title.")] = None,
    xlabel: Annotated[
        str | None,
        Field(description="X-axis label. Omit for 'Concentration (<unit>)'."),
    ] = None,
    ylabel: Annotated[
        str | None,
        Field(description="Y-axis label, e.g. 'Inhibition %'."),
    ] = None,
    xscale: Annotated[XScale, Field(description=D_XSCALE)] = "log",
    xscale_ticks: Annotated[
        tuple[float, float] | None, Field(description=D_XSCALE_TICKS)
    ] = None,
    ylim: Annotated[
        tuple[float, float] | None, Field(description=D_YLIM)
    ] = None,
    box: Annotated[
        bool,
        Field(description="Draw dashed IC50 guide lines at box_intercept."),
    ] = False,
    box_intercept: Annotated[float, Field(description=D_BOX_INTERCEPT)] = 50,
    savepath: Annotated[str | None, Field(description=D_SAVEPATH)] = None,
) -> ToolResult:
    """Plot each compound's 4PL dose-response curve in its own grid subplot and return metrics.

    Call read_csv first to get exact column names. Returns an inline PNG grid
    plus a fitted IC50, absolute IC50, Hill slope, and pIC50 table per compound.
    """
    frame = load_frame(filepath, sheet=sheet)
    source = resolve_path(filepath).name
    require_columns(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
        name_col=name_col,
    )
    require_numeric(
        frame,
        source,
        concentration_col=concentration_col,
        response_col=response_col,
    )
    in_unit, out_unit = resolve_units(input_units, output_units)
    explicit_ticks = validate_bounds(xscale_ticks, "xscale_ticks")
    resolved_ylim = validate_bounds(ylim, "ylim")

    names = compound_names(frame, name_col)
    resolved_ticks = (
        explicit_ticks
        if explicit_ticks is not None
        else infer_xscale_ticks(frame[concentration_col], in_unit, out_unit)
    )
    resolved_xlabel = (
        xlabel if xlabel is not None else f"Concentration ({out_unit})"
    )
    resolved_ylabel = (
        ylabel
        if ylabel is not None
        else (response_col[0] if len(response_col) == 1 else "Response")
    )
    grid_frame, grid_response_col = average_replicates(
        scale_concentrations_to_nm(frame, concentration_col, in_unit),
        response_col,
    )
    cols = min(column_num, len(names))
    rows = -(-len(names) // column_num)
    figsize = (max(4.2 * cols, 6.4), max(3.4 * rows, 4.8))

    with py50_session() as buffer:
        fit_table = Calculator(frame).calculate_pic50(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=one_or_many(response_col),
            input_units=in_unit,
            output_units=out_unit,
        )
        fit_records = clean_records(fit_table)
        fit_markdown = fit_table.to_markdown(index=False)

        figure = PlotCurve(grid_frame).grid_curve_plot(
            concentration_col=concentration_col,
            response_col=grid_response_col,
            name_col=name_col,
            column_num=column_num,
            title=title,
            xlabel=resolved_xlabel,
            ylabel=resolved_ylabel,
            conc_unit=out_unit,
            xscale=xscale,
            xscale_ticks=resolved_ticks,
            ymin=resolved_ylim[0] if resolved_ylim else None,
            ymax=resolved_ylim[1] if resolved_ylim else None,
            line_color=CBPALETTE,
            box=box,
            box_intercept=box_intercept,
            figsize=figsize,
            savepath=None,
        )
        notes = diagnostics(buffer)
        result = render_result(
            figure,
            summary=(
                f"Grid of {len(names)} dose-response curves from {source} "
                f"({', '.join(names)}; x-axis in {out_unit})."
            ),
            table_markdown=fit_markdown,
            savepath=savepath,
            notes=notes,
            extra={
                "compounds": names,
                "column_num": column_num,
                "input_units": in_unit,
                "output_units": out_unit,
                "xscale_ticks": list(resolved_ticks)
                if resolved_ticks is not None
                else None,
                "fit_records": fit_records,
                "fit_table_markdown": fit_markdown,
            },
        )

    if notes:
        await ctx.info("py50: " + "; ".join(notes))
    return result


@mcp.tool(annotations=_READ_ONLY, tags={"py50", "stats"}, timeout=120)
async def run_stats(
    ctx: Context,
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    test: Annotated[StatsTest, Field(description="Statistical test to run.")],
    value_col: Annotated[str, Field(description="Numeric dependent-variable column.")],
    group_col: Annotated[
        str | None,
        Field(description="Between-group column (required for most group tests)."),
    ] = None,
    subgroup_col: Annotated[
        str | None,
        Field(description="Subgroup/hue column for wilcoxon/mannu (or friedman subject)."),
    ] = None,
    subject_col: Annotated[
        str | None,
        Field(description="Subject ID column for rm-anova, mixed-anova, friedman, or pairwise tests."),
    ] = None,
    within_subject_col: Annotated[
        str | None,
        Field(description="Within-subject factor column for rm/mixed/pairwise tests."),
    ] = None,
    alternative: Annotated[
        Alternative,
        Field(description="Alternative hypothesis for wilcoxon/mannu."),
    ] = "two-sided",
    padjust: Annotated[
        PAdjust,
        Field(description="Multiple-comparison correction for pairwise-* tests."),
    ] = "none",
    sheet: Annotated[str | None, Field(description=D_SHEET)] = None,
) -> dict[str, Any]:
    """Run a parametric, non-parametric, or assumption statistical test via py50.Stats.

    Call read_csv first to get exact column names. Returns structured rows and a
    markdown summary table.
    """
    frame = load_frame(filepath, sheet=sheet)
    source = resolve_path(filepath).name

    with py50_session() as buffer:
        table = run_stats_test(
            frame,
            source=source,
            test=test,
            value_col=value_col,
            group_col=group_col,
            subgroup_col=subgroup_col,
            subject_col=subject_col,
            within_subject_col=within_subject_col,
            alternative=alternative,
            padjust=padjust,
        )
        notes = diagnostics(buffer)
        records = clean_records(table)
        markdown = table.to_markdown(index=False)

    if notes:
        await ctx.info("py50: " + "; ".join(notes))

    return {
        "test": test,
        "records": records,
        "table_markdown": markdown,
        "notes": notes,
    }


@mcp.tool(annotations=_WRITES, tags={"py50", "stats", "plot"}, timeout=120)
async def stats_plot(
    ctx: Context,
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    plot_type: Annotated[
        StatsPlotType, Field(description="Statistical plot type to draw.")
    ],
    value_col: Annotated[str, Field(description="Numeric dependent-variable column.")],
    group_col: Annotated[
        str | None,
        Field(description="Grouping column (required except for histplot/qqplot)."),
    ] = None,
    test: Annotated[
        PlotStatsTest | None,
        Field(
            description=(
                "Post-hoc/pairwise test for significance brackets or p_matrix "
                "(required for categorical plots and p_matrix)."
            )
        ),
    ] = None,
    subgroup_col: Annotated[
        str | None,
        Field(
            description="Subgroup/hue column for grouped plots (only supported with test='mannu' or 'wilcoxon')."
        ),
    ] = None,
    subject_col: Annotated[
        str | None,
        Field(description="Subject ID column for pairwise-rm or pairwise-mixed."),
    ] = None,
    within_subject_col: Annotated[
        str | None,
        Field(description="Within-subject factor column for pairwise-rm or pairwise-mixed."),
    ] = None,
    group_order: Annotated[
        list[str] | None,
        Field(description="Explicit display order for groups in group_col."),
    ] = None,
    hide_ns: Annotated[
        bool,
        Field(description="Hide non-significant ('n.s.') brackets on categorical plots."),
    ] = False,
    orient: Annotated[
        Orient,
        Field(description="Categorical plot orientation ('v' vertical or 'h' horizontal)."),
    ] = "v",
    title: Annotated[str | None, Field(description="Figure title.")] = None,
    sheet: Annotated[str | None, Field(description=D_SHEET)] = None,
    savepath: Annotated[str | None, Field(description=D_SAVEPATH)] = None,
) -> ToolResult:
    """Draw a statistical plot (categorical with significance brackets, CI, p-matrix, or distribution).

    Call read_csv first to get exact column and group names. Returns an inline PNG
    plot plus the underlying statistical table when applicable.
    """
    frame = load_frame(filepath, sheet=sheet)
    source = resolve_path(filepath).name
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

    if group_order is not None:
        if not group_col:
            raise ToolError("group_col is required when group_order is provided.")
        present = set(frame[group_col].dropna().astype(str))
        unknown = [g for g in group_order if str(g) not in present]
        if unknown:
            raise ToolError(
                f"group_order contains unknown value(s) {unknown} for {group_col!r}. "
                f"Available groups: {sorted(present)}."
            )

    if plot_type in ("histplot", "qqplot"):
        pass
    elif plot_type == "ci_plot":
        if not group_col:
            raise ToolError("group_col is required when plot_type='ci_plot'.")
    elif plot_type == "p_matrix":
        if not group_col:
            raise ToolError("group_col is required when plot_type='p_matrix'.")
        if test is None:
            raise ToolError("test is required when plot_type='p_matrix'.")
    else:
        if not group_col:
            raise ToolError(f"group_col is required when plot_type={plot_type!r}.")
        if test is None:
            raise ToolError(f"test is required when plot_type={plot_type!r}.")
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

    with py50_session() as buffer:
        stat_records: list[dict[str, Any]] | None = None
        stat_markdown: str | None = None
        plots = Plots(frame)

        if plot_type in ("histplot", "qqplot"):
            new_figure()
            ax = plots.distribution(val_col=value_col, type=plot_type)
            if title:
                ax.set_title(title)
            figure = ax.figure
            figure.tight_layout()
        elif plot_type == "ci_plot":
            tukey_df, ax = plots.ci_plot(
                data=frame,
                value_col=value_col,
                group_col=group_col,
                title=title or "Tukey HSD Confidence Intervals",
                return_stats=True,
            )
            figure = ax.figure
            stat_records = clean_records(tukey_df)
            stat_markdown = tukey_df.to_markdown(index=False)
        elif plot_type == "p_matrix":
            stat_df = run_stats_test(
                frame,
                source=source,
                test=test,
                value_col=value_col,
                group_col=group_col,
                subgroup_col=subgroup_col,
                subject_col=subject_col,
                within_subject_col=within_subject_col,
            )
            matrix_df = Stats.get_p_matrix(stat_df, test=test, order=group_order)
            new_figure()
            plots.p_matrix(data=matrix_df)
            figure = current_figure()
            if title and figure.axes:
                figure.axes[0].set_title(title)
            figure.tight_layout()
            stat_records = clean_records(stat_df)
            stat_markdown = matrix_df.to_markdown()
        else:
            new_figure()
            plot_fn = getattr(plots, plot_type)
            extra_kwargs: dict[str, Any] = {"title": title} if title else {}
            stat_df, annotator = plot_fn(
                test=test,
                group_col=group_col,
                value_col=value_col,
                group_order=group_order,
                subgroup_col=subgroup_col,
                subject_col=subject_col,
                within_subject_col=within_subject_col,
                hide_ns=hide_ns,
                orient=orient,
                return_df=True,
                **extra_kwargs,
            )
            figure = annotator.ax.figure
            stat_records = clean_records(stat_df)
            stat_markdown = stat_df.to_markdown(index=False)

        notes = diagnostics(buffer)
        test_label = f", test={test}" if test else ""
        result = render_result(
            figure,
            summary=f"{plot_type} for {value_col!r} from {source} ({len(frame)} rows{test_label}).",
            table_markdown=stat_markdown,
            savepath=savepath,
            notes=notes,
            extra={
                "plot_type": plot_type,
                "test": test,
                "stat_records": stat_records,
                "stat_table_markdown": stat_markdown,
            },
        )

    if notes:
        await ctx.info("py50: " + "; ".join(notes))
    return result


def main() -> None:
    """Console-script entry point: serve py50 over MCP stdio transport."""
    logging.basicConfig(
        level=os.environ.get("PY50_MCP_LOG_LEVEL", "INFO"),
        # Never stdout: on stdio transport that is the JSON-RPC channel.
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
