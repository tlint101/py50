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
    from pydantic import Field
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"The py50 MCP server needs {exc.name!r}, which is not installed. py50's MCP "
        "dependencies are optional -- install them with:\n\n"
        "    pip install 'py50[mcp]'\n\n"
        "or run the server without installing anything:\n\n"
        "    uvx --from 'py50[mcp]' py50-mcp\n"
    ) from exc

from py50 import Calculator, PlotCurve
from py50.mcp_server._support import (
    compound_names,
    diagnostics,
    load_frame,
    py50_session,
    render_result,
    require_columns,
    require_numeric,
    resolve_path,
    select_compound,
)
from py50.plot_settings import CBMARKERS, CBPALETTE

logger = logging.getLogger("py50.mcp")

ConcUnit = Literal["nM", "uM", "µM", "pM"]
CalcUnit = Literal["nM", "µM", "pM"]
XScale = Literal["log", "linear"]
ErrorBar = Literal["sd", "se", "ci", "pi"]
LegendLoc = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

# param descriptions
D_FILEPATH = "Absolute path to the CSV file holding the assay data."
D_CONC_COL = (
    "Name of the numeric column holding dose/concentration values. Interpreted in "
    "conc_unit."
)
D_NAME_COL = "Name of the column holding compound names."
D_CONC_UNIT = (
    "Concentration unit of the data in concentration_col. Sets the x-axis unit and "
    "rescales values internally, so a wrong value silently shifts every reported "
    "potency. 'uM' is accepted as an alias for 'µM'."
)
D_XSCALE = "X-axis scale. Dose-response curves are conventionally logarithmic."
D_XSCALE_TICKS = (
    "X-axis range as base-10 exponents, e.g. [0, 5] spans 1 to 100000 conc_unit. "
    "Omit to let py50 span 1 to 100000. Also controls how far the fitted line is drawn."
)
D_ERRORBAR = (
    "Error bar statistic drawn when response_col is a list of replicates: 'sd' "
    "standard deviation, 'se' standard error, 'ci' 95% confidence interval, 'pi' "
    "percentile interval. Ignored for a single response column."
)
D_BOX_INTERCEPT = (
    "Response percentage at which the guide lines meet the curve. 50 marks the IC50."
)
D_SAVEPATH = (
    "Absolute path ending in .png to also write a 300 dpi copy to. The plot is "
    "always returned inline as an image regardless; pass this only when the user "
    "asked for a file on disk."
)
D_FIGSIZE = "Figure size in inches as [width, height]."

# setup mcp
mcp = FastMCP(
    name="py50",
    version="1.1.1",
    instructions=(
        "py50 fits four-parameter-logistic (4PL) dose-response curves to plate-assay "
        "data in CSV files and renders publication-quality plots.\n"
        "\n"
        "Workflow -- follow it in order:\n"
        "1. Call read_csv FIRST on every new file. It reports the exact column names, "
        "dtypes and compound names. Never guess column names such as 'Compound Conc' "
        "or '% Inhibition 1'; they differ between files, and a wrong name is an error.\n"
        "2. Pick arguments from what read_csv reported: a numeric concentration "
        "column, a numeric response column (or several replicate columns to average), "
        "and the compound-name column.\n"
        "3. Call calculate_pic50 for numbers -- IC50, absolute IC50 and pIC50 per "
        "compound.\n"
        "4. Call a plot tool for figures: single_curve for one compound, multi_curve "
        "to overlay every compound on shared axes. Both return the plot inline as a "
        "PNG image.\n"
        "\n"
        "Rules:\n"
        "- Every path must be absolute. This server has no working directory.\n"
        "- Always state the concentration unit of the data via conc_unit or "
        "input_units. The default is nM; getting it wrong rescales every result.\n"
        "- Compound names must match exactly, including case and spacing.\n"
        "- Errors list the valid columns or compounds. Read them and retry with a "
        "corrected argument instead of guessing again."
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
) -> dict[str, Any]:
    """Inspect a CSV of dose-response data and report its structure.

    Call this before any other py50 tool. It returns the exact column names,
    dtypes, missing-value counts and compound names needed to fill in the other
    tools' arguments, plus a short preview of the data. It deliberately does not
    return the whole file.

    Args:
        filepath: Absolute path to the CSV.
        preview_rows: How many leading rows to show. Keep it small.

    Returns:
        Shape, column metadata, the numeric columns (candidates for concentration
        and response arguments), the low-cardinality text columns (candidates for
        name_col) with their distinct values, and a markdown preview.

    Raises:
        ToolError: If the path is relative, missing, not a .csv, or unparseable.
    """
    path = resolve_path(filepath)
    frame = load_frame(filepath)

    numeric_columns = [c for c in frame.columns if frame[c].dtype.kind in "iufc"]
    name_candidates: dict[str, list[str]] = {}
    for column in frame.columns:
        if column in numeric_columns:
            continue
        distinct = frame[column].astype(str).drop_duplicates().tolist()
        # A compound-name column has few distinct values relative to the row count.
        if 1 < len(distinct) <= 50 or len(distinct) == 1:
            name_candidates[column] = distinct

    return {
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


@mcp.tool(annotations=_READ_ONLY, tags={"py50"}, timeout=120)
async def calculate_ic50(
    ctx: Context,
    filepath: Annotated[str, Field(description=D_FILEPATH)],
    concentration_col: Annotated[str, Field(description=D_CONC_COL)],
    response_col: Annotated[
        str | list[str],
        Field(
            description=(
                "Name of the numeric response column, or a list of replicate column "
                "names to average before fitting."
            )
        ),
    ],
    name_col: Annotated[str, Field(description=D_NAME_COL)],
    input_units: Annotated[
        CalcUnit,
        Field(
            description=(
                "Concentration unit of concentration_col. This changes the names of "
                "the returned columns, e.g. 'ic50 (nM)' versus 'ic50 (µM)', and "
                "rescales the values, so a wrong value invalidates every result."
            )
        ),
    ] = "nM",
) -> dict[str, Any]:
    """Fit a 4PL dose-response curve per compound and report IC50 and pIC50 values.

    Returns one row per compound in name_col. Call read_csv first for the exact
    column names; do not guess them.

    Args:
        ctx: FastMCP request context, injected automatically.
        filepath: Absolute path to the CSV.
        concentration_col: Numeric dose column.
        response_col: One response column, or several replicate columns to average.
        name_col: Column of compound names, used to group the fits.
        input_units: Concentration unit of the input data.

    Returns:
        A records list and an equivalent markdown table with columns
        compound_name, maximum, minimum, relative ic50, absolute ic50, hill_slope,
        relative pIC50 and absolute pIC50. The two ic50 columns carry the unit in
        their name.

    Raises:
        ToolError: If a named column is missing or non-numeric, or the fit fails.
    """
    frame = load_frame(filepath)
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

    with py50_session() as buffer:
        table = Calculator(frame).calculate_pic50(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=response_col,
            input_units=input_units,
        )
        notes = diagnostics(buffer)
        records = table.to_dict(orient="records")
        markdown = table.to_markdown(index=False)

    if notes:
        await ctx.info("py50: " + "; ".join(notes))

    return {
        "input_units": input_units,
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
        str | list[str],
        Field(
            description=(
                "Name of the numeric response column, or a list of replicate column "
                "names. A list is averaged for the fit and drawn with error bars; a "
                "single column produces no error bars."
            )
        ),
    ],
    name_col: Annotated[
        str,
        Field(
            description=(
                "Name of the column holding compound names. Required: it is used to "
                "select and label the compound."
            )
        ),
    ],
    query: Annotated[
        str | None,
        Field(
            description=(
                "Exact compound name from name_col to plot. Matched exactly, "
                "including case and spacing. Required unless the file contains "
                "exactly one compound."
            )
        ),
    ] = None,
    title: Annotated[str | None, Field(description="Figure title.")] = None,
    titlesize: Annotated[
        float, Field(description="Title font size in points.", gt=0)
    ] = 16,
    xlabel: Annotated[
        str | None,
        Field(
            description=(
                "X-axis label. Omit to let py50 generate one from conc_unit."
            )
        ),
    ] = None,
    ylabel: Annotated[
        str | None, Field(description="Y-axis label, e.g. 'Inhibition %'.")
    ] = None,
    axis_fontsize: Annotated[
        float, Field(description="Axis label font size in points.", gt=0)
    ] = 14,
    conc_unit: Annotated[ConcUnit, Field(description=D_CONC_UNIT)] = "nM",
    xscale: Annotated[XScale, Field(description=D_XSCALE)] = "log",
    xscale_ticks: Annotated[
        tuple[float, float] | None, Field(description=D_XSCALE_TICKS)
    ] = None,
    ymax: Annotated[
        float | None, Field(description="Y-axis upper limit. Omit to autoscale.")
    ] = None,
    ymin: Annotated[
        float | None,
        Field(
            description=(
                "Y-axis lower limit. Omit to autoscale; -10 is a common choice so "
                "points near 0% are not clipped."
            )
        ),
    ] = None,
    line_color: Annotated[
        str,
        Field(
            description=(
                "Matplotlib color for the fitted line and data points, e.g. 'black', "
                "'#E69F00'."
            )
        ),
    ] = "black",
    line_width: Annotated[
        float, Field(description="Fitted line width in points.", gt=0)
    ] = 1.5,
    errorbar: Annotated[ErrorBar, Field(description=D_ERRORBAR)] = "sd",
    marker: Annotated[
        str | None,
        Field(
            description=(
                "Matplotlib marker code for the data points, e.g. 'o', '^', 's', 'D'. "
                "Defaults to 'o'."
            )
        ),
    ] = None,
    markersize: Annotated[
        float, Field(description="Data point marker size in points.", gt=0)
    ] = 8,
    legend: Annotated[bool, Field(description="Show the legend.")] = True,
    legend_loc: Annotated[
        LegendLoc, Field(description="Legend placement.")
    ] = "best",
    box: Annotated[
        bool,
        Field(
            description=(
                "Draw dashed guide lines from the curve to both axes, marking the "
                "concentration at box_intercept percent response."
            )
        ),
    ] = False,
    box_color: Annotated[
        str, Field(description="Matplotlib color for the guide lines.")
    ] = "gray",
    box_intercept: Annotated[float, Field(description=D_BOX_INTERCEPT)] = 50,
    conc_target: Annotated[
        float | None,
        Field(
            description=(
                "Place the guide lines at this concentration (in conc_unit) instead "
                "of at box_intercept. Requires box=True."
            )
        ),
    ] = None,
    hline: Annotated[
        float | None,
        Field(
            description=(
                "Draw a horizontal reference line at this response percentage."
            )
        ),
    ] = None,
    hline_color: Annotated[
        str, Field(description="Matplotlib color for the horizontal line.")
    ] = "gray",
    vline: Annotated[
        float | None,
        Field(
            description=(
                "Draw a vertical reference line at this concentration, in conc_unit."
            )
        ),
    ] = None,
    vline_color: Annotated[
        str, Field(description="Matplotlib color for the vertical line.")
    ] = "gray",
    figsize: Annotated[
        tuple[float, float], Field(description=D_FIGSIZE)
    ] = (6.4, 4.8),
    savepath: Annotated[str | None, Field(description=D_SAVEPATH)] = None,
) -> ToolResult:
    """Plot the 4PL dose-response curve for a single compound as an inline image.

    Fits a four-parameter-logistic model to one compound's data and returns the
    plot as a PNG rendered directly in the conversation. Call read_csv first to
    get the exact column names and compound names; do not guess them.

    Args:
        ctx: FastMCP request context, injected automatically.
        filepath: Absolute path to the CSV.
        concentration_col: Numeric dose column, interpreted in conc_unit.
        response_col: One response column, or replicate columns to average.
        name_col: Column of compound names, used to select and label the compound.
        query: Exact compound name. Optional only for a single-compound file.
        box: Marks the IC50 on the plot when box_intercept is left at 50.
        savepath: Optional absolute .png path for an additional 300 dpi copy.

    Returns:
        A PNG image of the curve, a text summary, and structured metadata holding
        the resolved compound name, the point count, any file written, and any
        diagnostics py50 emitted.

    Raises:
        ToolError: If a column is missing or non-numeric, or the compound is not
            found. The message lists the available columns or compounds.
    """
    frame = load_frame(filepath)
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
    # Filter here rather than in py50: it pools every compound when the query is
    # None, and substring-matches names otherwise.
    selected, resolved = select_compound(frame, name_col, query, source)

    with py50_session() as buffer:
        figure = PlotCurve(selected).curve_plot(
            concentration_col=concentration_col,
            response_col=response_col,
            name_col=name_col,
            query=resolved,
            title=title,
            titlesize=titlesize,
            xlabel=xlabel,
            ylabel=ylabel,
            axis_fontsize=axis_fontsize,
            conc_unit=conc_unit,
            xscale=xscale,
            xscale_ticks=tuple(xscale_ticks) if xscale_ticks else None,
            ymax=ymax,
            ymin=ymin,
            line_color=line_color,
            line_width=line_width,
            errorbar=errorbar,
            marker=marker,
            markersize=markersize,
            legend=legend,
            legend_loc=legend_loc,
            box=box,
            box_color=box_color,
            box_intercept=box_intercept,
            conc_target=conc_target,
            hline=hline,
            hline_color=hline_color,
            vline=vline,
            vline_color=vline_color,
            figsize=tuple(figsize),
            savepath=None,
        )
        notes = diagnostics(buffer)
        result = render_result(
            figure,
            summary=(
                f"Dose-response curve for {resolved!r} from {source} "
                f"({len(selected)} points, concentrations in {conc_unit})."
            ),
            savepath=savepath,
            notes=notes,
            extra={"compound": resolved, "points": int(len(selected))},
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
        str | list[str],
        Field(
            description=(
                "Name of the numeric response column, or a list of replicate column "
                "names to average before fitting."
            )
        ),
    ],
    name_col: Annotated[
        str,
        Field(
            description=(
                "Name of the column holding compound names. One curve is drawn per "
                "distinct value."
            )
        ),
    ],
    title: Annotated[str | None, Field(description="Figure title.")] = None,
    titlesize: Annotated[
        float, Field(description="Title font size in points.", gt=0)
    ] = 12,
    xlabel: Annotated[
        str | None,
        Field(description="X-axis label. Omit to let py50 generate one."),
    ] = None,
    ylabel: Annotated[
        str | None, Field(description="Y-axis label, e.g. 'Inhibition %'.")
    ] = None,
    conc_unit: Annotated[ConcUnit, Field(description=D_CONC_UNIT)] = "nM",
    xscale: Annotated[XScale, Field(description=D_XSCALE)] = "log",
    xscale_ticks: Annotated[
        tuple[float, float] | None, Field(description=D_XSCALE_TICKS)
    ] = None,
    ymax: Annotated[
        float | None, Field(description="Y-axis upper limit. Omit to autoscale.")
    ] = None,
    ymin: Annotated[
        float | None,
        Field(
            description=(
                "Y-axis lower limit. Omit to autoscale; -10 is a common choice so "
                "points near 0% are not clipped."
            )
        ),
    ] = None,
    axis_fontsize: Annotated[
        float, Field(description="Axis label font size in points.", gt=0)
    ] = 10,
    line_color: Annotated[
        list[str] | None,
        Field(
            description=(
                "Matplotlib colors, one per compound, cycled in order. Defaults to "
                "py50's colorblind-safe palette."
            )
        ),
    ] = None,
    marker: Annotated[
        list[str] | None,
        Field(
            description=(
                "Matplotlib marker codes, one per compound, cycled in order, e.g. "
                "['o', '^', 's', 'D']. Defaults to py50's marker set."
            )
        ),
    ] = None,
    markersize: Annotated[
        float, Field(description="Data point marker size in points.", gt=0)
    ] = 8,
    line_width: Annotated[
        float, Field(description="Fitted line width in points.", gt=0)
    ] = 1.5,
    errorbar: Annotated[ErrorBar, Field(description=D_ERRORBAR)] = "sd",
    legend: Annotated[
        bool,
        Field(
            description=(
                "Show the legend. Strongly recommended here, since the plot holds "
                "several compounds."
            )
        ),
    ] = True,
    legend_loc: Annotated[
        LegendLoc, Field(description="Legend placement.")
    ] = "best",
    box_target: Annotated[
        str | None,
        Field(
            description=(
                "Exact compound name to annotate with dashed guide lines at "
                "box_intercept. Must match a value in name_col."
            )
        ),
    ] = None,
    box_color: Annotated[
        str, Field(description="Matplotlib color for the guide lines.")
    ] = "gray",
    box_intercept: Annotated[float, Field(description=D_BOX_INTERCEPT)] = 50,
    hline: Annotated[
        float | None,
        Field(
            description="Draw a horizontal reference line at this response percentage."
        ),
    ] = None,
    hline_color: Annotated[
        str, Field(description="Matplotlib color for the horizontal line.")
    ] = "gray",
    vline: Annotated[
        float | None,
        Field(
            description=(
                "Draw a vertical reference line at this concentration, in conc_unit."
            )
        ),
    ] = None,
    vline_color: Annotated[
        str, Field(description="Matplotlib color for the vertical line.")
    ] = "gray",
    figsize: Annotated[
        tuple[float, float], Field(description=D_FIGSIZE)
    ] = (6.4, 4.8),
    savepath: Annotated[str | None, Field(description=D_SAVEPATH)] = None,
) -> ToolResult:
    """Plot every compound's 4PL dose-response curve on shared axes, as one image.

    Draws one curve per distinct value in name_col and returns the plot as a PNG
    rendered directly in the conversation. Call read_csv first to get the exact
    column names; do not guess them.

    Args:
        ctx: FastMCP request context, injected automatically.
        filepath: Absolute path to the CSV.
        concentration_col: Numeric dose column, interpreted in conc_unit.
        response_col: One response column, or replicate columns to average.
        name_col: Column of compound names; one curve is drawn per distinct value.
        box_target: Annotates a single named compound with IC50 guide lines.
        savepath: Optional absolute .png path for an additional 300 dpi copy.

    Returns:
        A PNG image of the overlaid curves, a text summary, and structured
        metadata holding the compounds plotted, any file written, and any
        diagnostics py50 emitted.

    Raises:
        ToolError: If a column is missing or non-numeric, or box_target names a
            compound that is not present. The message lists the valid options.
    """
    frame = load_frame(filepath)
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

    names = compound_names(frame, name_col)
    if box_target is not None and box_target not in names:
        raise ToolError(
            f"box_target={box_target!r} is not a compound in column {name_col!r} of "
            f"{source}. Available compounds: {names}."
        )

    with py50_session() as buffer:
        figure = PlotCurve(frame).multi_curve_plot(
            concentration_col=concentration_col,
            response_col=response_col,
            name_col=name_col,
            title=title,
            titlesize=titlesize,
            xlabel=xlabel,
            ylabel=ylabel,
            conc_unit=conc_unit,
            xscale=xscale,
            xscale_ticks=tuple(xscale_ticks) if xscale_ticks else None,
            ymax=ymax,
            ymin=ymin,
            axis_fontsize=axis_fontsize,
            line_color=list(line_color) if line_color else CBPALETTE,
            marker=list(marker) if marker else CBMARKERS,
            markersize=markersize,
            line_width=line_width,
            errorbar=errorbar,
            legend=legend,
            legend_loc=legend_loc,
            box_target=box_target,
            box_color=box_color,
            box_intercept=box_intercept,
            hline=hline,
            hline_color=hline_color,
            vline=vline,
            vline_color=vline_color,
            figsize=tuple(figsize),
            savepath=None,
        )
        notes = diagnostics(buffer)
        result = render_result(
            figure,
            summary=(
                f"Dose-response curves for {len(names)} compounds from {source} "
                f"({', '.join(names)}; concentrations in {conc_unit})."
            ),
            savepath=savepath,
            notes=notes,
            extra={"compounds": names},
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
