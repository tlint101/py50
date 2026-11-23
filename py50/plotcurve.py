import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import itertools
from scipy.interpolate import interp1d
from py50.plot_settings import CBMARKERS, CBPALETTE, CurveSettings
from py50.calculator import Calculator
from typing import Union, Optional

__all__ = ["PlotCurve"]


class PlotCurve:
    # Will accept input DataFrame and output said DataFrame for double checking.
    def __init__(
            self,
            data: pd.DataFrame,
            name_col: Optional[str] = None,
            concentration_col: Optional[str] = None,
            response_col: Optional[Union[str, list]] = None,
    ):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        else:
            self.data = data
        self.name_col = name_col
        self.concentration_col = concentration_col
        self.response_col = response_col
        self.calculation = None

    def show(self, rows: Optional[int] = None):
        """
        show DataFrame

        :param rows: int
            Indicates the number of rows to display. If none, automatically show 5.
        :return: DataFrame
        """

        if rows is None:
            return self.data.head()
        elif isinstance(rows, int):
            return self.data.head(rows)

    def to_csv(self, path: Optional[str] = None, index: bool = False, **kwargs):
        """
        Save table to csv file.
        :param path: str
            Designate save path for table.
        :param index: bool
            Write row names.
        **kwargs
            Keyword arguments for pandas.DataFrame.to_csv.
        :return:
        """

        self.data.to_csv(path, index=index, **kwargs)

    # Filter input data based on Compound Name to generate single plot
    def _filter_dataframe(
        self,
        drug_name: Optional[Union[str, int, float]] = None,
        name_col: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Filter input DataFrame by exact compound name match in ``name_col``.

        :param drug_name: Optional[Union[str, int, float]]
            Compound name to select. When ``None``, returns all rows if ``name_col``
            is not provided or contains a single unique compound; raises ``ValueError``
            if ``name_col`` contains multiple compounds.
        :param name_col: Optional[str]
            Column name containing compound names. Defaults to ``self.name_col``.
        :param data: Optional[pd.DataFrame]
            DataFrame to filter. Defaults to ``self.data``.
        :return: DataFrame
        """
        source_df = self.data if data is None else data

        if name_col is None:
            name_col = self.name_col

        if name_col is None:
            if drug_name is not None:
                raise ValueError(
                    "name_col must be provided when filtering by a compound query."
                )
            if source_df.empty:
                raise ValueError("Input DataFrame is empty.")
            return source_df

        if name_col not in source_df.columns:
            raise ValueError(
                f"Column {name_col!r} not found in DataFrame. "
                f"Available columns: {list(source_df.columns)}"
            )

        name_series = source_df[name_col]
        unique_names = name_series.dropna().astype(str).drop_duplicates().tolist()

        if drug_name is None:
            if len(unique_names) == 0:
                raise ValueError(
                    f"Column {name_col!r} contains no compound names."
                )
            if len(unique_names) > 1:
                raise ValueError(
                    f"Multiple compounds found in column {name_col!r}: {unique_names}. "
                    "Pass `query` with an exact compound name to plot a single curve, "
                    "or use `multi_curve_plot` / `grid_curve_plot`."
                )
            target = unique_names[0]
        else:
            target = str(drug_name)

        filtered_df = source_df.loc[
            name_series.notna() & (name_series.astype(str) == target)
        ]

        if filtered_df.empty:
            raise ValueError(
                f"Compound {drug_name!r} not found in column {name_col!r}. "
                f"Available compounds: {unique_names}"
            )

        return filtered_df

    @staticmethod
    def _cycle_style(
        values: Union[str, list, tuple],
        count: int,
        param_name: str,
    ) -> tuple:
        """Normalize a single style string or sequence to a tuple of length ``count`` by cycling."""
        if isinstance(values, str):
            return tuple([values] * count)
        seq = tuple(values)
        if not seq:
            raise ValueError(f"{param_name} must not be empty.")
        return tuple(itertools.islice(itertools.cycle(seq), count))

    def curve_plot(
            self,
            concentration_col: Optional[str] = None,
            response_col: Optional[Union[str, list[str]]] = None,
            name_col: Optional[str] = None,
            query: Optional[str] = None,
            title: Optional[str] = None,
            titlesize: int = 16,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            axis_fontsize: int = 14,
            conc_unit: Optional[str] = "nM",
            xscale: str = "log",
            xscale_ticks: Optional[tuple] = None,
            ymax: Optional[Union[int, float]] = None,
            ymin: Optional[Union[int, float]] = None,
            line_color: str = "black",
            line_width: float = 1.5,
            errorbar: Optional[str] = "sd",
            marker: Optional[str] = None,
            markersize: int = 8,
            legend: bool = False,
            legend_loc: str = "best",
            box: bool = False,
            box_color: str = "gray",
            box_intercept: Optional[Union[int, float]] = 50,
            conc_target: Optional[Union[int, float]] = None,
            hline: Optional[Union[int, float]] = None,
            hline_color: str = "gray",
            vline: Optional[Union[int, float]] = None,
            vline_color: str = "gray",
            figsize: tuple = (6.4, 4.8),
            savepath: Optional[str] = None,
            verbose: Optional[bool] = None,
            **kwargs,
    ):
        """
        Generate a dose-response curve for a single drug target. Because a data table can contain multiple drugs, user
        must specify specific target.

        :param concentration_col: str
            Concentration column from DataFrame.
        :param response_col: Union[str, list]
            Response column from DataFrame.
        :param name_col: str
            Column containing drug name for plotting.
        :param query: str
            Exact compound name to plot from ``name_col``. Required when the dataset contains multiple compounds; optional for single-compound datasets.
        :param title: str
            Title of the figure.
        :param titlesize: tuple
            Modify plot title font size.
        :param xlabel: str
            Title of the X-axis.
        :param ylabel: str
            Title of the Y-axis.
        :param axis_fontsize: int
            Modify axis label font size.
        :param conc_unit: str
            Input unit of concentration. Can accept nanomolar (nM) and micromolar (uM or µM). If the units are
            different, for example in the DataFrame units are in nM, but the units for the graph are µM, the units from
            the DataFrame will be converted to match the conc_unit input. The final plot will scale based on the
            conc_unit input. By default, it will assume input concentration will be in nM.
        :param xscale: int
            Set the scale of the X-axis as logarithmic or linear. It is logarithmic by default.
        :param xscale_ticks: tuple
            Set the scale of the X-axis
        :param ymax: int
            Give a set maximum limit for the Y-Axis
        :param ymin: int
            Give a set minimum limit for the Y-Axis
        :param line_color: str.
            Takes a list of colors. By default, it uses the CBPALETTE. List can contain name of colors or colors in hex
            code.
        :param line_width: int
            Set width of lines in plot.
        :param errorbar: str
             Set the type of seaborn errorbar to use. Defaults to 'sd'.
        :param marker: Optional, list
            Takes a list of for point markers.
        :param markersize: int
            Set the marker size.
        :param legend: Optional, bool
            Denotes a figure legend.
        :param legend_loc: str
            Determine legend location. Default is best. Matplotlib options can be found here
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        :param box: Optional bool.
            Draw a box to highlight a specific location. Uses box_color and box_intercept (default 50%) or
            conc_target when provided.
        :param box_color: str
            Set color of box. Default is gray.
        :param box_intercept: int
            Set horizontal location of box. By default, it is set at 50% of the Y-axis.
        :param conc_target: int
            Set vertical location of the box. By default, this is set to None. For example, if the box_intercept is set
            to 50%, then the conc_target must be the Absolute IC50 value. If there is an input to conc_target,
            it will override the box_intercept and the response data will move accordingly. Finally, the number must be
            in the same unit as the X-axis. i.e., if the axis is in µM, then the number for the conc_target should
            be in µM and vice versa.
        :param hline: Int or float
            Draw a horizontal line across the graph. This line will stretch across the length of the plot. This is
            optional and set to 0 by default.
        :param hline_color: str
            Set color of horizontal line. Default color is gray.
        :param vline: int or float
            This line will stretch across the height of the plot. This is  optional and set to 0 by default.
        :param vline_color: str
            Set color of vertical line. Default color is gray.
        :param figsize: tuple
            Set figure size.
        :param savepath: str
            File path for save location.
        :param verbose: bool
            Output information about the plot.


        :return: Figure
        """
        # set instance variables
        if name_col is None:
            name_col = self.name_col
        if concentration_col is None:
            concentration_col = self.concentration_col
        if response_col is None:
            response_col = self.response_col

        query_df = self._filter_dataframe(drug_name=query, name_col=name_col).copy()
        if name_col is not None:
            drug_name = str(query_df[name_col].iloc[0])
        else:
            drug_name = "Unknown Drug"

        # if response_col is a list, table will be reformated to produce a column with average values
        if isinstance(response_col, list):
            response_col_is_list = True  # bool to indicate sns usage
            response_col_list = response_col  # set response_col input for reshaping data
            avg_col = "_inhibition_average"
            while avg_col in query_df.columns:
                avg_col = f"_{avg_col}"
            query_df[avg_col] = query_df[response_col_list].mean(axis=1)
            response_col = avg_col  # reset response_col input
        else:
            response_col_is_list = False

        concentration = query_df[concentration_col]
        response = query_df[response_col]

        # Initialize in Calculator class for calculating results for plotting
        calculator = Calculator(query_df)

        # set a new copy of the DataFrame to avoid warnings
        calc_query = query_df.copy()
        calc_query.sort_values(by=concentration_col, ascending=True, inplace=True)

        """
        x_fit and concentration adjustments mut come before the calc_logic or curve and datapoints will not bre aligned
        """

        # Obtain x_fit. Because calculator does not require xscale_ticks, it is set to None
        x_fit, xscale_unit = CurveSettings().scale_units(
            drug_name, conc_unit, xscale_ticks, verbose
        )

        # Function to scale the concentration by nM or µM
        concentration = CurveSettings().conc_scale(
            xscale_unit, concentration, verbose=verbose
        )

        reverse, params, covariance = calculator._calc_logic(
            data=calc_query,
            concentration=concentration,
            response_col=response_col,
            response=response,
        )
        # Extract parameter values
        maximum, minimum, ic50, hill_slope = params
        # print(drug_name, ' IC50: ', ic50, 'µM') # For checking

        hill_slope, ic50, final_unit, x_intersection, y_fit = (
            calculator._reverse_absolute_calculation(hill_slope, ic50, conc_unit, maximum, minimum, params, reverse,
                                                     x_fit))

        # Boolean check for marker
        if marker is not None:
            marker = marker
        else:
            marker = "o"

        # Check for xscale label:
        if xscale == "log" or xscale == "linear":
            pass
        else:
            raise ValueError("xscale must be 'log' or 'linear'")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim(top=100)  # Set maximum y axis limit

        # for error bars
        if response_col_is_list:
            id_list = [col for col in query_df.columns if col not in response_col_list]

            # melt table format, with drugs and replicate in a single column
            melted_query = query_df.assign(**{concentration_col: concentration}).melt(
                id_vars=id_list,
                value_vars=response_col_list,
                var_name="Replicate",
                value_name="Inhibition",
            )
            # todo the id_vars must match input, find a way to automatically do this
            sns.lineplot(data=melted_query, x=concentration_col, y="Inhibition", errorbar=errorbar, marker=marker,
                         markersize=markersize, err_style="bars", linestyle="", label=None, color=line_color)
            ax.plot(x_fit, y_fit, color=line_color, linewidth=line_width, label="fit_line")
        else:
            ax.scatter(concentration, response, marker=marker, color=line_color, label="data_points")
            ax.plot(x_fit, y_fit, color=line_color, linewidth=line_width, label="fit_line")

        # set scales
        ax.set_xscale(xscale)  # Use a logarithmic scale for the x-axis
        ax.set_xlabel("" if xlabel is None else xlabel, fontsize=axis_fontsize)
        ax.set_ylabel("" if ylabel is None else ylabel, fontsize=axis_fontsize)
        ax.set_title("" if title is None else title, fontsize=titlesize)

        # Set grid lines to False by default
        plt.grid(kwargs.get("grid", False))

        # Set tick for axis
        ax.tick_params(
            axis="both", which="both", bottom=True, top=False, left=True, right=False
        )

        # Set y-axis limit
        # Y-axis limit uses replicate extremes when response_col is a list so error bars do not clip
        y_values = melted_query["Inhibition"] if response_col_is_list else response
        if ymax is None:
            max_value = float(y_values.max()) + 10
        else:
            max_value = ymax

        if ymin is None:
            min_y = float(y_values.min())
            min_value = min_y - 10 if min_y < 0 else 0
        else:
            min_value = ymin
        ax.set_ylim(min_value, max_value)

        # Plot box to IC50 on curve
        # Interpolate to find the x-value (Concentration) at the intersection point
        y_intersection = None
        if conc_target is not None:
            x_intersection = conc_target
            y_intersection = np.interp(x_intersection, x_fit, y_fit)
            if verbose is True:
                print(
                    "Box X intersection: ", np.round(x_intersection, 3), f"{final_unit}"
                )
                print("Box Y intersection: ", np.round(y_intersection, 3), "%")
        elif box_intercept is None:
            if box is True:
                print("Input Inhibition % target")
        elif reverse == 1:
            y_intersection = box_intercept
            interpretation = interp1d(
                y_fit, x_fit, kind="linear", fill_value="extrapolate"
            )
            x_intersection = interpretation(y_intersection)
            if verbose is True:
                print(
                    "Box X intersection: ", np.round(x_intersection, 3), f"{final_unit}"
                )
                print("Box Y intersection: ", np.round(y_intersection, 3), "%")
        elif reverse == 0:
            y_intersection = box_intercept
            x_intersection = np.interp(y_intersection, y_fit, x_fit)
            if verbose is True:
                print(
                    "Box X intersection: ", np.round(x_intersection, 3), f"{final_unit}"
                )
                print("Box Y intersection: ", np.round(y_intersection, 3), "%")

        # Calculate yaxis scale for box highlight
        if box is True and x_intersection is not None and y_intersection is not None:
            CurveSettings().yaxis_scale(
                box=box,
                reverse=reverse,
                y_intersection=y_intersection,
                x_intersection=x_intersection,
                box_color=box_color,
            )

        # Arguments for hline and vline
        if hline is not None:
            plt.axhline(y=hline, color=hline_color, linestyle="--")
        if vline is not None:
            plt.axvline(x=vline, color=vline_color, linestyle="--")

        # Figure legend
        if legend:
            ax.legend(
                handles=[
                    mlines.Line2D([0], [0], color=line_color, marker=marker, label=drug_name)], loc=legend_loc)

        plt.tight_layout()
        # Save the plot to a file
        if savepath is not None:
            plt.savefig(savepath, dpi=300)

        return fig

    def multi_curve_plot(
            self,
            concentration_col: Optional[str] = None,
            response_col: Optional[Union[str, list[str]]] = None,
            name_col: Optional[str] = None,
            title: Optional[str] = None,
            titlesize: int = 12,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            conc_unit: Optional[str] = "nM",
            xscale: str = "log",
            xscale_ticks: Optional[tuple] = None,
            ymax: Optional[Union[int, float]] = None,
            ymin: Optional[Union[int, float]] = None,
            axis_fontsize: int = 10,
            line_color: Union[str, list, tuple] = CBPALETTE,
            marker: Union[str, list, tuple] = CBMARKERS,
            markersize: int = 8,
            line_width: float = 1.5,
            errorbar: Optional[str] = "sd",
            legend: Optional[bool] = False,
            legend_loc: str = "best",
            box_target: Optional[str] = None,
            box_color: str = "gray",
            box_intercept: Optional[Union[int, float]] = 50,
            hline: Optional[Union[int, float]] = None,
            hline_color: str = "gray",
            vline: Optional[Union[int, float]] = None,
            vline_color: str = "gray",
            figsize: tuple = (6.4, 4.8),
            savepath: Optional[str] = None,
            verbose: Optional[bool] = None,
            **kwargs,
    ):
        """
        Generate a dose-response plot for multiple drug targets. Curves will be placed into a single plot.

        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: str
            Response column from DataFrame
        :param name_col:
            Column containing name of drug from DataFrame
        :param title: str
            Title of the figure
        :param titlesize: int
            Modify plot title font size
        :param xlabel: str
            Title of the X-axis
        :param ylabel: str
            Title of the Y-axis
        :param conc_unit: str
            Input will assume that the concentration will be in nM. Thus, it will be automatically converted into µM. If
            xscale_unit is given as nM, no conversion will be performed.
        :param xscale: str
            Set the scale of the X-axis as logarithmic or linear. It is logarithmic by default.
        :param xscale_ticks: tuple
            Set the scale of the X-axis
        :param ymax: int
            Give a set maximum limit for the Y-Axis
        :param ymin: int
            Give a set minimum limit for the Y-Axis
        :param axis_fontsize:int
            Modify axis label font size
        :param line_color: str
            Takes a list of colors. By default, it uses the CBPALETTE. List can contain name of colors or colors in hex
            code.
        :param markersize:
            Set the marker size.
        :param line_width: int
            Set width of lines in plot.
        :param errorbar:
            Set the type of seaborn errorbar to use. Defaults to 'sd'.
        :param marker: list
            Takes a list for point markers. Marker options can be found here: https://matplotlib.org/stable/api/markers_api.html
        :param legend: bool
            Denotes a figure legend.
        :param legend_loc: str
            Determine legend location. Matplotlib options can be found here
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        :param box_target: str
            Draw a box to highlight a specific drug curve. Must use specific drug name.
        :param box_color: str
            Set color of box. Default color is gray.
        :param box_intercept: int
            Set horizontal location of box. By default, it is set at Absolute IC50.
        :param hline: int or float
            Draw a horizontal line that will stretch across the length of the plot. This is optional and set to 0 by
            default.
        :param hline_color: str
            Set color of horizontal line. Default color is gray.
        :param vline: int or float
            Draw a line that will stretch across the height of the plot. This is  optional and set to 0 by default.
        :param vline_color: str
            Set color of vertical line. Default color is gray.
        :param figsize: tuple
            Set figure size.
        :param savepath: str
            File path for save location.
        :param verbose: bool
            Output information about the plot.

        :return: Figure
        """
        # Resolve instance defaults before checking response_col type
        if name_col is None:
            name_col = self.name_col
        if concentration_col is None:
            concentration_col = self.concentration_col
        if response_col is None:
            response_col = self.response_col

        plot_df = self.data
        name_list = np.unique(plot_df[name_col])
        if len(name_list) == 0:
            raise ValueError(f"Column {name_col!r} contains no compound names.")
        melted_df = None

        concentration_list = []
        response_list = []
        y_fit_list = []
        reverse_list = []

        # if response_col is a list, compute averages on a local copy without mutating self.data
        if isinstance(response_col, list):
            response_col_is_list = True  # bool to indicate sns usage
            response_col_list = (
                response_col  # set response_col input for reshaping data
            )
            plot_df = self.data.copy()
            avg_col = "_inhibition_average"
            while avg_col in plot_df.columns:
                avg_col = f"_{avg_col}"
            reshape_data = pd.melt(
                plot_df,
                id_vars=[name_col, concentration_col],
                value_vars=response_col_list,
                value_name=avg_col,
            )
            # drop the variable column
            # set dataframe for use in drawing error bars
            melted_df = reshape_data.drop(
                columns=["variable"]
            )  # reset table to reshaped table
            response_col = avg_col  # reset response_col input

            # add column with averages on the local copy only
            plot_df[avg_col] = plot_df[response_col_list].mean(axis=1)
        else:
            response_col_is_list = False

        for drug in name_list:
            drug_query = self._filter_dataframe(drug, name_col=name_col, data=plot_df)

            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Initialize in Calculator class for calculating results for plotting
            calculator = Calculator(drug_query)

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, ascending=True, inplace=True)

            """
            x_fit and concentration adjustments must come before the calc_logic or curve and datapoints will 
            not be aligned
            """

            # Obtain x_fit. Because calculator does not require xscale_ticks, it is set to None
            x_fit, xscale_unit = CurveSettings().scale_units(
                drug, conc_unit, xscale_ticks, verbose
            )

            # Function to scale the concentration by nM or µM
            concentration = CurveSettings().conc_scale(
                xscale_unit, concentration, verbose=verbose
            )

            # Append values for each drug into list for multi curve plotting
            concentration_for_list = concentration.values  # Convert into np.array
            concentration_list.append(concentration_for_list)
            response_for_list = response.values
            response_list.append(response_for_list)

            reverse, params, covariance = calculator._calc_logic(
                data=query,
                concentration=concentration,
                response_col=response_col,
                response=response,
            )
            reverse_list.append(reverse)
            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, 'nM') # For checking

            # Calculate from parameters 4PL equation
            if reverse == 1:
                y_fit = calculator._reverse_fourpl(
                    x_fit, maximum, minimum, ic50, hill_slope
                )
                y_intersection = 50
                interpretation = interp1d(
                    y_fit, x_fit, kind="linear", fill_value="extrapolate"
                )
                x_intersection = np.round(
                    interpretation(y_intersection), 3
                )  # give results and round to 3 sig figs
                hill_slope = (
                        -1 * hill_slope
                )  # ensure hill_slope is negative # may not be needed if fixed
                y_fit_list.append(y_fit)

            else:
                y_fit = calculator._fourpl(x_fit, *params)
                y_intersection = 50
                x_intersection = np.interp(y_intersection, y_fit, x_fit)
                y_fit_list.append(y_fit)

            # This script is from single_curve_plot. It is not needed for multi-curve
            # Confirm ic50 unit output
            # ic50, x_intersection = calculator.unit_convert(ic50, x_intersection, conc_unit)

        # Generate plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim(top=100)  # Set maximum y axis limit

        line_color = self._cycle_style(line_color, len(name_list), "line_color")
        marker = self._cycle_style(marker, len(name_list), "marker")

        # Check for xscale label:
        if xscale == "log" or xscale == "linear":
            pass
        else:
            raise ValueError("xscale must be 'log' or 'linear'")
        # Plotting the data for each line
        legend_handles = (
            []
        )  # Store data for each line as a dictionary inside a list for the legend
        for i, (
                y_fit_point,
                concentration_point,
                response_point,
                name,
                color,
                mark,
        ) in enumerate(
            zip(
                y_fit_list,
                concentration_list,
                response_list,
                name_list,
                line_color,
                marker,
            )
        ):
            # Convert Seaborn sizes (diameters) to Matplotlib sizes (area)
            if response_col_is_list is True:
                # use info from melted data for errorbars
                data_to_split = melted_df[melted_df[name_col] == name]
                concentration_data = CurveSettings().conc_scale(
                    xscale_unit, data_to_split[concentration_col]
                )
                response_data = data_to_split[response_col]

                sns.lineplot(
                    data=data_to_split,
                    x=concentration_data,
                    y=response_data,
                    errorbar=errorbar,
                    marker=mark,
                    markersize=markersize,
                    err_style="bars",
                    linestyle="",
                    label="data_points",
                    color=color,
                )
                plt.plot(
                    x_fit, y_fit_point, color=color, label=name, linewidth=line_width
                )
            else:
                plt.plot(
                    x_fit, y_fit_point, color=color, label=name, linewidth=line_width
                )
                ax.scatter(
                    concentration_point, response_point, color=color, marker=mark
                )

            ax.set_title("" if title is None else title)
            ax.set_xscale(xscale)  # Use a logarithmic scale for the x-axis
            # Set ticks for axis
            ax.tick_params(
                axis="both",
                which="both",
                bottom=True,
                top=False,
                left=True,
                right=False,
            )
            ax.set_xlabel("" if xlabel is None else xlabel, fontsize=axis_fontsize)
            ax.set_ylabel("" if ylabel is None else ylabel, fontsize=axis_fontsize)

            # Set grid lines to False by default
            plt.grid(kwargs.get("grid", False))

            # Append scatter plot handle to the legend
            legend_piece = {"marker": mark, "name": name, "line_color": color}
            legend_handles.append(legend_piece)

        # Set y-axis limit
        # Y-axis limit uses all compounds (and replicate extremes when response_col is a list)
        y_values = (
            melted_df[response_col] if response_col_is_list else plot_df[response_col]
        )
        if ymax is None:
            max_value = float(y_values.max()) + 10
        else:
            max_value = ymax

        if ymin is None:
            min_y = float(y_values.min())
            min_value = min_y - 10 if min_y < 0 else 0
        else:
            min_value = ymin

        ax.set_ylim(min_value, max_value)

        # Plot box to IC50 on curve
        # Interpolate to find the x-value (Concentration) at the intersection point
        if box_intercept is None:
            y_intersection = 50
        else:
            y_intersection = box_intercept

        # Specify box target
        if box_target is not None:
            matches = np.where(name_list.astype(str) == str(box_target))[0]
            if matches.size > 0:
                name_index = matches[0]
                if reverse_list[name_index] == 1:
                    # match data to drug using y_fit_list[name_index]
                    interpretation = interp1d(
                        y_fit_list[name_index],
                        x_fit,
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    x_intersection = interpretation(y_intersection)
                else:
                    x_intersection = np.interp(
                        y_intersection, y_fit_list[name_index], x_fit
                    )
                box_ymin = 0  # Starts at the bottom of the plot
                box_ymax = (y_intersection - plt.gca().get_ylim()[0]) / (
                    plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
                )
                if verbose is True:
                    print(f"Box will target {box_target}")
                    print(
                        "Box X intersection: ",
                        np.round(x_intersection, 3),
                        f"{conc_unit}",
                    )
                    print("Box Y intersection: ", np.round(y_intersection, 3), "%")
                # Converted x_intersection from a numpy array into a float
                plt.axvline(
                    x=x_intersection,
                    ymin=box_ymin,
                    ymax=box_ymax,
                    color=box_color,
                    linestyle="--",
                )
                plt.hlines(
                    y=y_intersection,
                    xmin=0,
                    xmax=x_intersection,
                    colors=box_color,
                    linestyles="--",
                )
            else:
                print("Drug name does not match box target!")

        # Arguments for hline and vline
        if hline is not None:
            plt.axhline(y=hline, color=hline_color, linestyle="--")
        if vline is not None:
            plt.axvline(x=vline, color=vline_color, linestyle="--")

        # Figure legend
        # Extract elements (Line, Scatterplot, and color) from figures and append into a list for generating legend
        if legend:
            legend_elements = []
            for data in legend_handles:
                legend_element = mlines.Line2D(
                    [0],
                    [0],
                    color=data["line_color"],
                    marker=data["marker"],
                    label=data["name"],
                )
                legend_elements.append(legend_element)
            ax.legend(handles=legend_elements, loc=legend_loc)

        plt.title("" if title is None else title, fontsize=titlesize)
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, dpi=300)

        return fig

    def grid_curve_plot(
            self,
            concentration_col: Optional[str] = None,
            response_col: Optional[str] = None,
            name_col: Optional[str] = None,
            column_num: int = 2,
            title: Optional[str] = None,
            titlesize: int = 20,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            conc_unit: Optional[str] = "nM",
            xscale: str = "log",
            xscale_ticks: Optional[tuple] = None,
            ymax: Optional[Union[int, float]] = None,
            ymin: Optional[Union[int, float]] = None,
            line_color: Union[str, list, tuple] = CBPALETTE,
            line_width: float = 1.5,
            box: bool = False,
            box_color: str = "gray",
            box_intercept: Optional[Union[int, float]] = 50,
            hline: Optional[Union[int, float]] = None,
            hline_color: str = "gray",
            vline: Optional[Union[int, float]] = None,
            vline_color: str = "gray",
            figsize: tuple = (8.4, 4.8),
            savepath: Optional[str] = None,
            verbose: Optional[bool] = None,
            **kwargs,
    ):
        """
        Generate a dose-response curve for mutliple drugs. Each curve will be placed in its own plot which is then
        placed in a grid.

        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: str
            Response column from DataFrame
        :param name_col: str
            Name column from DataFrame
        :param column_num: int
            Set number of column grid
        :param title: str
            Title of the figure
        :param titlesize: int
            Modify plot title font size
        :param xlabel:str
            Title of the X-axis
        :param ylabel: str
            Title of the Y-axis
        :param ymax: int
            Give a set maximum limit for the Y-Axis
        :param ymin: int
            Give a set minimum limit for the Y-Axis
        :param conc_unit: str
            Input will assume that the concentration will be in nM. Thus, it will be automatically converted into µM. If
            xscale_unit is given as nM, no conversion will be performed.
        :param xscale: str
            Set the scale of the X-axis as logarithmic or linear. It is logarithmic by default.
        :param xscale_ticks: tuple
            Set the scale of the X-axis
        :param line_color: list
            Takes a list of colors. By default, it uses the CBPALETTE. List can contain name of
            colors or colors in hex code.
        :param line_width: int
            Set width of lines in plot.
        :param errorbar:
            Set the type of seaborn errorbar to use. Defaults to 'sd'.
        :param box: bool
            Draw a box to highlight a specific location. If box = True, then the box_color, and box_intercept MUST ALSO
            BE GIVEN.
        :param box_color: str
            Set color of box. Default color is gray.
        :param box_intercept: int
            Set horizontal location of box. By default, it is set at Absolute IC50.
        :param hline: int or float
            Draw horizontal line that will stretch across the length of the plot. This is optional and set to 0 by
            default.
        :param hline_color: str
            Set color of horizontal line. Default color is gray.
        :param vline: int or float
            Draw a line that will stretch across the height of the plot. This is  optional and set to 0 by default.
        :param vline_color: str
            Set color of vertical line. Default color is gray.
        :param figsize: tuple
            Set figure size for subplot.
        :param savepath: str
            File path for save location.
        :param verbose: bool
            Output information about the plot.

        :return: Figure
        """

        if name_col is None:
            name_col = self.name_col
        name_list = np.unique(self.data[name_col])
        if len(name_list) == 0:
            raise ValueError(f"Column {name_col!r} contains no compound names.")

        # Generate lists for modifying plots (vline, box, etc)
        concentration_list = []
        response_list = []
        y_fit_list = []
        reverse_list = []
        x_fit_list = []

        for drug in name_list:
            drug_query = self._filter_dataframe(drug, name_col=name_col)

            # Create variables for inputs. Extract column from Dataframe
            if concentration_col is None:
                concentration_col = self.concentration_col
            if response_col is None:
                response_col = self.response_col

            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Initialize in Calculator class for calculating results for plotting
            calculator = Calculator(drug_query)

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, ascending=True, inplace=True)

            """
            x_fit and concentration adjustments must come before the calc_logic or curve and datapoints will 
            not be aligned
            """

            # Obtain x_fit. Because calculator does not require xscale_ticks, it is set to None
            x_fit, xscale_unit = CurveSettings().scale_units(
                drug, conc_unit, xscale_ticks, verbose
            )
            x_fit_list.append(x_fit)

            # Function to scale the concentration by nM or µM
            concentration = CurveSettings().conc_scale(
                xscale_unit, concentration, verbose=verbose
            )

            reverse, params, covariance = calculator._calc_logic(
                data=query,
                concentration=concentration,
                response_col=response_col,
                response=response,
            )
            reverse_list.append(reverse)
            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, 'nM') # For checking

            # Calculate from parameters 4PL equation
            if reverse == 1:
                y_fit = calculator._reverse_fourpl(
                    x_fit, maximum, minimum, ic50, hill_slope
                )
                y_intersection = 50
                interpretation = interp1d(
                    y_fit, x_fit, kind="linear", fill_value="extrapolate"
                )
                x_intersection = np.round(
                    interpretation(y_intersection), 3
                )  # give results and round to 3 sig figs
                hill_slope = (
                        -1 * hill_slope
                )  # ensure hill_slope is negative # may not be needed if fixed
                y_fit_list.append(y_fit)
            else:
                y_fit = calculator._fourpl(x_fit, *params)
                y_intersection = 50
                x_intersection = np.interp(y_intersection, y_fit, x_fit)
                y_fit_list.append(y_fit)

            """Appended list will contain concentrations in nM or µM depending on user input"""
            # Append values for each drug into list
            concentration_for_list = concentration.values  # Convert into np.array
            concentration_list.append(concentration_for_list)
            response_for_list = response.values
            response_list.append(response_for_list)

        # Set up color options for line colors (cycles defaults and custom sequences to match len(name_list))
        line_color = self._cycle_style(line_color, len(name_list), "line_color")

        # Check for xscale label:
        if xscale == "log" or xscale == "linear":
            pass
        else:
            raise ValueError("xscale must be 'log' or 'linear'")

        # Generate figure in grid layout
        # Calculate the number of rows needed
        if not isinstance(column_num, int) or isinstance(column_num, bool) or column_num < 1:
            raise ValueError("column_num must be a positive integer.")
        num_plots = len(name_list)  # determine total plots to make
        row_num = -(-num_plots // column_num)  # Round up to the nearest integer
        # Squeeze to handle possible 1D array
        fig, axes = plt.subplots(row_num, column_num, figsize=figsize, squeeze=False)
        fig.suptitle("" if title is None else title, fontsize=titlesize)

        # Loop through the data and plot scatter and line plots on each subplot
        for i in range(row_num):
            for j in range(column_num):
                idx = i * column_num + j
                if idx >= num_plots:
                    axes[i, j].set_visible(False)
                    continue

                # Line plot
                axes[i, j].plot(
                    x_fit_list[idx],
                    y_fit_list[idx],
                    label="Line Plot",
                    color=line_color[idx],
                    linewidth=line_width,
                )

                # Scatter plot
                axes[i, j].scatter(
                    concentration_list[idx],
                    response_list[idx],
                    label="Scatter Plot",
                    color=line_color[idx],
                )

                # Set x-axis scale
                axes[i, j].set_xscale(xscale)

                # Set y-axis limit
                # Y-axis limit will be limited to the largest response number and add 10 for spacing
                if ymax is None:
                    max_value = (
                            np.amax([np.amax(max_value) for max_value in response_list])
                            + 10
                    )
                else:
                    max_value = ymax
                # Y-axis minimum to the lowest response - 10 for better plotting
                if ymin is None:
                    ymin = (
                            np.amin([np.amin(max_value) for max_value in response_list])
                            - 10
                    )
                else:
                    ymin = ymin
                axes[i, j].set_ylim(ymin, max_value)

                # Set subplot title
                axes[i, j].set_title(name_list[idx])

                if box is True:
                    if isinstance(box_intercept, (int, float)) and not isinstance(box_intercept, bool):
                        y_intersection = box_intercept
                        if reverse_list[idx] == 1:
                            interpretation = interp1d(
                                y_fit_list[idx],
                                x_fit_list[idx],
                                kind="linear",
                                fill_value="extrapolate",
                            )
                            x_concentration = interpretation(y_intersection)
                        else:
                            x_concentration = np.interp(
                                y_intersection,
                                y_fit_list[idx],
                                x_fit_list[idx],
                            )

                        # Constrain box to target drug response
                        ymax_box = (y_intersection - axes[i, j].get_ylim()[0]) / (
                            axes[i, j].get_ylim()[1] - axes[i, j].get_ylim()[0]
                        )

                        axes[i, j].axvline(
                            x=x_concentration,
                            ymin=0,
                            ymax=ymax_box,
                            color=box_color,
                            linestyle="--",
                        )
                        axes[i, j].hlines(
                            y=y_intersection,
                            xmin=0,
                            xmax=x_concentration,
                            colors=box_color,
                            linestyles="--",
                        )
                        if verbose is True:
                            print(
                                f"Box X intersection ({name_list[idx]}): ",
                                np.round(x_concentration, 3),
                                f"{conc_unit}",
                            )
                            print(
                                f"Box Y intersection ({name_list[idx]}): ",
                                np.round(y_intersection, 3),
                                "%",
                            )
                    elif box_intercept is None:
                        pass

                # Arguments for hline and vline
                if hline is not None:
                    axes[i, j].axhline(y=hline, color=hline_color, linestyle="--")

                if vline is not None:
                    axes[i, j].axvline(x=vline, color=vline_color, linestyle="--")

                # Set axis labels
                axes[i, j].set_xlabel("" if xlabel is None else xlabel)
                axes[i, j].set_ylabel("" if ylabel is None else ylabel)

        # Remove grid lines and plot ticks for each subplot
        for ax_row in axes:
            for ax in ax_row:
                if not ax.get_visible():
                    continue
                ax.grid(kwargs.get("grid", False))
                # Set tick for axis
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=True,
                    top=False,
                    left=True,
                    right=False,
                )

        plt.tight_layout()

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        if savepath is not None:
            plt.savefig(savepath, dpi=300)

        return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
