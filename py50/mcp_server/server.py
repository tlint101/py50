import io
from fastmcp import FastMCP
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from py50 import Calculator, PlotCurve
from py50.plot_settings import CBMARKERS, CBPALETTE, CurveSettings
from typing import Union
import matplotlib
import matplotlib.pyplot as plt

# prevent matplotlib pop-ups
matplotlib.use('Agg')

mcp = FastMCP("DataAnalysis",
              instructions="Provide tools to read a csv file and calculate IC50. Start with calculating the ic50 values. If the user requests images, then use the single_curve or multi_curve tools."
              )


@mcp.tool
def read_csv(filepath: str) -> TextFileReader | DataFrame | str:
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        return f"Error reading CSV: {e}"


@mcp.tool
def calculate_ic50(filepath: str, name_col: str = 'Compound Name', concentration_col: str = 'Compound Conc',
                   response_col: list[str] = ['% Inhibition 1', '% Inhibition 2']):
    try:
        df = pd.read_csv(filepath)
        calc_data = Calculator(df)
        calculation = calc_data.calculate_pic50(name_col=name_col, concentration_col=concentration_col,
                                                response_col=response_col)
        return calculation.to_markdown(index=False)
    except Exception as e:
        return {"status": "error", "message": f"Calculation failed: {str(e)}"}


@mcp.tool
def single_curve(filepath: str, concentration_col: str = None, response_col: list[str] = None,
                 name_col: str = None, query: str = None, title: str = None, titlesize: int = 16,
                 xlabel: str = 'Logarithmic Concentration (nM)', ylabel: str = 'Inhibition %', axis_fontsize: int = 14,
                 conc_unit: str = "nM", xscale: str = "log", xscale_ticks: tuple = None, ymax: int = None,
                 ymin: int = None, line_color: str = "black", line_width: int = 1.5, errorbar: str = "sd",
                 marker: bool = None, markersize: int = 8, legend: bool = False, legend_loc: str = "best",
                 box: bool = False, box_color: str = "gray", box_intercept: int = 50, conc_target: int = None,
                 hline: int = None, hline_color: str = "gray", vline: int = None, vline_color: str = "gray",
                 figsize: tuple = (6.4, 4.8), savepath: str = None, verbose: bool = None):
    try:
        df = pd.read_csv(filepath)
        plot_data = PlotCurve(df)
        plot_data.curve_plot(concentration_col=concentration_col, response_col=response_col, name_col=name_col,
                             query=query, title=title, titlesize=titlesize, xlabel=xlabel, ylabel=ylabel,
                             axis_fontsize=axis_fontsize, conc_unit=conc_unit, xscale=xscale,
                             xscale_ticks=xscale_ticks, ymax=ymax, ymin=ymin, line_color=line_color,
                             line_width=line_width, errorbar=errorbar, marker=marker, markersize=markersize,
                             legend=legend, legend_loc=legend_loc, box=box, box_color=box_color,
                             box_intercept=box_intercept, conc_target=conc_target, hline=hline,
                             hline_color=hline_color, vline=vline, vline_color=vline_color, figsize=figsize,
                             savepath=savepath, verbose=verbose)
        # plt.close(fig)
        # must return as a JSON-serializable script
        return {"savepath": savepath, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": f"Plotting failed: {str(e)}"}


@mcp.tool
def multi_curve(filepath: str, concentration_col: str = None, response_col: str = None, name_col: str = None,
                title: str = None, titlesize: int = 12, xlabel: str = None, ylabel: str = None, conc_unit: str = "nM",
                xscale: str = "log", xscale_ticks: tuple = None, ymax: int = None, ymin: int = None,
                axis_fontsize: int = 10, line_color: list = CBPALETTE, marker: list = CBMARKERS, markersize: int = 8,
                line_width: int = 1.5, errorbar: str = "sd", legend: bool = False, legend_loc: str = "best",
                box_target: str = None, box_color: str = "gray", box_intercept: int = 50, hline: int = None,
                hline_color: str = "gray", vline: int = None, vline_color: str = "gray", figsize: tuple = (6.4, 4.8),
                savepath: str = None, verbose: bool = None):
    try:
        df = pd.read_csv(filepath)
        plot_data = PlotCurve(df)
        fig = plot_data.multi_curve_plot(concentration_col=concentration_col, response_col=response_col,
                                         name_col=name_col, title=title, titlesize=titlesize, xlabel=xlabel,
                                         ylabel=ylabel, conc_unit=conc_unit, xscale=xscale, xscale_ticks=xscale_ticks,
                                         ymax=ymax, ymin=ymin, axis_fontsize=axis_fontsize, line_color=line_color,
                                         marker=marker, markersize=markersize, line_width=line_width, errorbar=errorbar,
                                         legend=legend, legend_loc=legend_loc, box_target=box_target,
                                         box_color=box_color, box_intercept=box_intercept, hline=hline,
                                         hline_color=hline_color, vline=vline, vline_color=vline_color, figsize=figsize,
                                         savepath=savepath, verbose=verbose)
        # must return as a JSON-serializable script
        return {"savepath": savepath, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": f"Plotting failed: {str(e)}"}


if __name__ == "__main__":
    mcp.run()
    # calculate_ic50(filepath='/Users/tonyelin/Coding/py50/dataset/single_example.csv')
    # single_curve(filepath='/Users/tonyelin/Coding/py50/dataset/single_example.csv',
    #              concentration_col='Compound Conc',
    #              response_col='% Inhibition Avg',
    #              title='Default Plot Single Example (Positive)',
    #              name_col='Compound Name',
    #              xlabel='Logarithmic Concentration (nM)',
    #              ylabel='Inhibition %',
    #              legend=True,
    #              savepath='here.png')
    # multi_curve(filepath='/Users/tonyelin/Coding/py50/dataset/multiple_example.csv',
    #             name_col='Compound Name',
    #             concentration_col='Compound Conc',
    #             response_col='% Inhibition Avg',
    #             title='Multi-Curve Plot',
    #             xlabel='Logarithmic Concentration (nM)',
    #             ylabel='Inhibition %',
    #             legend=True,
    #             ymin=-10,
    #             markersize=10)
