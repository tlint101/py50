import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import itertools
from scipy.interpolate import interp1d
from py50.plot_settings import CBMARKERS, CBPALETTE, CurveSettings
from py50.calculator import Calculator
from typing import Union

__all__ = ["PlotCurve"]


class PlotCurve:
    # Will accept input DataFrame and output said DataFrame for double checking.
    def __init__(
        self,
        data: pd.DataFrame,
        name_col: str = None,
        concentration_col: str = None,
        response_col: Union[str, list] = None,
    ):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        else:
            self.data = data
        self.name_col = name_col
        self.concentration_col = concentration_col
        self.response_col = response_col
        self.calculation = None

    def show(self, rows: int = None):
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

    def to_csv(self, path: str = None, index: bool = False, **kwargs):
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
    def _filter_dataframe(self, drug_name: str = None):
        """
        Filter input DataFrame by query drug name.

        :param drug_name: str
            Column name containing drug name.
        :return: DataFrame
        """
        # Filter row based on drug name input. Row must match drug name somewhere
        filtered_df = self.data[
            self.data.apply(lambda row: drug_name in str(row), axis=1)
        ]
        return filtered_df

    def curve_plot(
        self,
        concentration_col: str = None,
        response_col: str = None,
        name_col: str = None,
        plot_title: str = None,
        plot_title_size: int = 16,
        xlabel: str = None,
        ylabel: str = None,
        axis_fontsize: int = 14,
        conc_unit: str = "nM",
        xscale: str = "log",
        xscale_ticks: tuple = None,
        ymax: int = None,
        ymin: int = None,
        line_color: str = "black",
        line_width: int = 1.5,
        errorbar: str = "sd",
        marker: bool = None,
        markersize: int = 8,
        legend: bool = False,
        legend_loc: str = "best",
        box: bool = False,
        box_color: str = "gray",
        box_intercept: int = 50,
        conc_target: int = None,
        hline: int = None,
        hline_color: str = "gray",
        vline: int = None,
        vline_color: str = "gray",
        figsize: tuple = (6.4, 4.8),
        savefig: str = None,
        verbose: bool = None,
        **kwargs,
    ):
        """
        Generate a dose-response curve for a single drug target. Because a data table can contain multiple drugs, user
        must specify specific target.

        :param concentration_col: str
            Concentration column from DataFrame.
        :param response_col: str
            Response column from DataFrame.
        :param name_col: str
            Column containing drug name for plotting.
        :param plot_title: str
            Title of the figure.
        :param plot_title_size: tuple
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
            Draw a box to highlight a specific location. If box = True, then the box_color, box_intercept, and
            x_concentration MUST ALSO BE GIVEN.
        :param box_color: str
            Set color of box. Default is gray.
        :param box_intercept: int
            Set horizontal location of box. By default, it is set at 50% of the Y-axis.
        :param conc_target: int
            Set vertical location of the box. By default, this is set to None. For example, if the box_intercept is set
            to 50%, then the x_concentration must be the Absolute IC50 value. If there is an input to x_concentration,
            it will override the box_intercept and the response data will move accordingly. Finally, the number must be
            in the same unit as the X-axis. i.e., if the axis is in µM, then the number for the x_concentration should
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
        :param savefig: str
            File path for save location.
        :param verbose: bool
            Output information about the plot.


        :return: Figure
        """
        global x_fit, drug_query, y_intersection, x_intersection, reverse, data, response_col_is_list
        # set instance variables
        if name_col is None:
            name_col = self.name_col
        if concentration_col is None:
            concentration_col = self.concentration_col
        if response_col is None:
            response_col = self.response_col

        if name_col is not None:
            drug_query = self._filter_dataframe(drug_name=name_col)
            if len(drug_query) > 0:
                pass
            elif len(drug_query) == 0:
                print("Drug not found!")
        else:
            print("Drug not found!")

        # if response_col is a list, table will be reformated to produce a column with average values
        if isinstance(response_col, list):
            response_col_is_list = True  # bool to indicate sns usage
            response_col_list = (
                response_col  # set response_col input for reshaping data
            )
            reshape_data = pd.melt(
                self.data,
                id_vars=[name_col, concentration_col],
                value_vars=response_col,
                value_name="inhibition_average",
            )
            # drop the variable column
            drug_query = reshape_data.drop(
                columns=["variable"]
            )  # reset table to reshaped table
            response_col = "inhibition_average"  # reset response_col input

            # reset input data for the reshaped data and add column with averages
            self.data["inhibition_average"] = self.data[response_col_list].mean(axis=1)
        else:
            response_col_is_list = False

        concentration = drug_query[concentration_col]
        response = drug_query[response_col]

        # Set initial guess for 4PL equation
        initial_guess = [
            max(response),
            min(response),
            0.5 * (max(response) + min(response)),
            1.0,
        ]  # Max, Min, ic50, and hill_slope

        # Initialize in Calculator class for calculating results for plotting
        calculator = Calculator(drug_query)

        # set a new coy of the DataFrame to avoid warnings
        query = drug_query.copy()
        query.sort_values(by=concentration_col, ascending=True, inplace=True)

        """
        x_fit and concentration adjustments mut come before the calc_logic or curve and datapoints will not bre aligned
        """

        # Obtain x_fit. Because calculator does not require xscale_ticks, it is set to None
        x_fit, xscale_unit = CurveSettings().scale_units(
            name_col, conc_unit, xscale_ticks, verbose
        )

        # Function to scale the concentration by nM or µM
        concentration = CurveSettings().conc_scale(
            xscale_unit, concentration, verbose=verbose
        )

        reverse, params, covariance = calculator._calc_logic(
            data=query,
            concentration=concentration,
            response_col=response_col,
            initial_guess=initial_guess,
            response=response,
        )
        # Extract parameter values
        maximum, minimum, ic50, hill_slope = params
        # print(drug_name, ' IC50: ', ic50, 'µM') # For checking

        hill_slope, ic50, final_unit, x_intersection, y_fit = (
            calculator._reverse_absolute_calculation(
                hill_slope,
                ic50,
                conc_unit,
                maximum,
                minimum,
                params,
                reverse,
                x_fit,
            )
        )

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
        if response_col_is_list is True:  # for error bars
            sns.lineplot(
                data=drug_query,
                x=concentration,
                y=response,
                errorbar=errorbar,
                marker=marker,
                markersize=markersize,
                err_style="bars",
                linestyle="",
                label="data_points",
                color=line_color,
            )
            ax.plot(
                x_fit, y_fit, color=line_color, linewidth=line_width, label="fit_line"
            )
        else:
            ax.scatter(
                concentration,
                response,
                marker=marker,
                color=line_color,
                label="data_points",
            )
            ax.plot(
                x_fit, y_fit, color=line_color, linewidth=line_width, label="fit_line"
            )

        # set scales
        ax.set_xscale(xscale)  # Use a logarithmic scale for the x-axis
        ax.set_xlabel(xlabel, fontsize=axis_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
        ax.set_title(plot_title, fontsize=plot_title_size)

        # Set grid lines to False by default
        plt.grid(kwargs.get("grid", False))

        # Set tick for axis
        ax.tick_params(
            axis="both", which="both", bottom=True, top=False, left=True, right=False
        )

        # Set y-axis limit
        # Y-axis limit will be limited to the largest response number and add 10 for spacing
        if ymax is None:
            max_value = max(response) + 10
        else:
            max_value = ymax

        if ymin is None:
            min_value = min(response)
            if min_value < 0:
                min_value = min_value - 5
            else:
                min_value = 0
        else:
            min_value = ymin
        ax.set_ylim(min_value, max_value)

        # Plot box to IC50 on curve
        # Interpolate to find the x-value (Concentration) at the intersection point
        if box_intercept is None:
            print("Input Inhibition % target")
        elif box_intercept and conc_target is None and reverse == 1:
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

        elif box_intercept and conc_target is None and reverse == 0:
            y_intersection = box_intercept
            x_intersection = np.interp(y_intersection, y_fit, x_fit)
            if verbose is True:
                print(
                    "Box X intersection: ", np.round(x_intersection, 3), f"{final_unit}"
                )
                print("Box Y intersection: ", np.round(y_intersection, 3), "%")

        if conc_target is not None:
            x_intersection = conc_target
            y_intersection = np.interp(x_intersection, x_fit, y_fit)
            if verbose is True:
                print(
                    "Box X intersection: ", np.round(x_intersection, 3), f"{final_unit}"
                )
                print("Box Y intersection: ", np.round(y_intersection, 3), "%")

        # Calculate yaxis scale for box highlight
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
                    plt.Line2D(
                        [0], [0], color=line_color, marker=marker, label=name_col
                    ),
                ],
                loc=legend_loc,
            )

        # Save the plot to a file
        if savefig == None:
            pass
        else:
            plt.savefig(
                savefig, dpi=300
            )  # Save the plot to a file with the specified filename

        return fig

    def multi_curve_plot(
        self,
        concentration_col: str = None,
        response_col: str = None,
        name_col: str = None,
        plot_title: str = None,
        plot_title_size: int = 12,
        xlabel: str = None,
        ylabel: str = None,
        conc_unit: str = "nM",
        xscale: str = "log",
        xscale_ticks: tuple = None,
        ymax: int = None,
        ymin: int = None,
        axis_fontsize: int = 10,
        line_color: list = CBPALETTE,
        marker: list = CBMARKERS,
        markersize: int = 8,
        line_width: int = 1.5,
        errorbar: str = "sd",
        legend: bool = False,
        legend_loc: str = "best",
        box_target: str = None,
        box_color: str = "gray",
        box_intercept: int = 50,
        hline: int = None,
        hline_color: str = "gray",
        vline: int = None,
        vline_color: str = "gray",
        figsize: tuple = (6.4, 4.8),
        savefig: str = None,
        verbose: bool = None,
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
        :param plot_title: str
            Title of the figure
        :param plot_title_size: int
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
        :param savefig: str
            File path for save location.
        :param verbose: bool
            Output information about the plot.

        :return: Figure
        """
        global response, x_fit, y_fit, y_intersection, x_intersection, reverse, params, response_col_is_list, query, concentration, list_data, melted_df
        name_list = np.unique(self.data[name_col])

        concentration_list = []
        response_list = []
        y_fit_list = []

        # if response_col is a list, table will be reformated to produce a column with average values
        if isinstance(response_col, list):
            response_col_is_list = True  # bool to indicate sns usage
            response_col_list = (
                response_col  # set response_col input for reshaping data
            )
            reshape_data = pd.melt(
                self.data,
                id_vars=[name_col, concentration_col],
                value_vars=response_col,
                value_name="inhibition_average",
            )
            # drop the variable column
            # set dataframe for use in drawing error bars
            melted_df = reshape_data.drop(
                columns=["variable"]
            )  # reset table to reshaped table
            response_col = "inhibition_average"  # reset response_col input

            # reset input data for the reshaped data and add column with averages
            self.data["inhibition_average"] = self.data[response_col_list].mean(axis=1)
        else:
            response_col_is_list = False

        # Create variables for inputs. Extract column from Dataframe
        if concentration_col is None:
            concentration_col = self.concentration_col
        if response_col is None:
            response_col = self.response_col

        for drug in name_list:
            drug_query = self._filter_dataframe(drug)

            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Append values for each drug into list for multi curve plotting
            concentration_for_list = concentration.values  # Convert into np.array
            concentration_list.append(concentration_for_list)
            response_for_list = response.values
            response_list.append(response_for_list)

            # Set initial guess for 4PL equation
            initial_guess = [
                max(response),
                min(response),
                1.0,
                1.0,
            ]  # Max, Min, ic50, and hill_slope

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

            reverse, params, covariance = calculator._calc_logic(
                data=query,
                concentration=concentration,
                response_col=response_col,
                initial_guess=initial_guess,
                response=response,
            )
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

        if line_color is CBPALETTE:
            pass
        else:
            line_color = line_color

        if marker is CBMARKERS:
            pass
        else:
            marker = marker

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
                concentration_data = data_to_split[concentration_col]
                response_data = data_to_split[response_col]

                sns.lineplot(
                    data=melted_df,
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

            ax.set_title(plot_title)
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
            ax.set_xlabel(xlabel, fontsize=axis_fontsize)
            ax.set_ylabel(ylabel, fontsize=axis_fontsize)

            # Set grid lines to False by default
            plt.grid(kwargs.get("grid", False))

            # Append scatter plot handle to the legend
            legend_piece = {"marker": mark, "name": name, "line_color": color}
            legend_handles.append(legend_piece)

        # Set y-axis limit
        # Y-axis limit will be limited to the largest response number and add 10 for spacing
        if ymax is None:
            max_y = self.data[response_col].max()
            max_value = max_y + 10
        else:
            max_value = ymax

        if ymin is None:
            min_value = min(response)
            if min_value < 0:
                min_value = min_value
            else:
                ax.set_ylim(0, max_value)
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
        if box_target in name_list:
            if isinstance(box_target, str) and reverse == 1:
                if box_target in name_list:
                    name_index = np.where(name_list == box_target)[0]
                    if name_index.size > 0:
                        name_index = name_index[0]
                        # match data to drug using y_fit_list[name_index]
                        interpretation = interp1d(
                            y_fit_list[name_index],
                            x_fit,
                            kind="linear",
                            fill_value="extrapolate",
                        )
                        x_intersection = interpretation(y_intersection)
                    ymin = 0  # Starts at the bottom of the plot
                    ymax = (y_intersection - plt.gca().get_ylim()[0]) / (
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
                        ymin=ymin,
                        ymax=ymax,
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
            elif isinstance(box_target, str) and reverse == 0:
                if box_target in name_list:
                    name_index = np.where(name_list == box_target)[0]
                    if name_index.size > 0:
                        name_index = name_index[0]
                        x_intersection = np.interp(
                            y_intersection, y_fit_list[name_index], x_fit
                        )
                    ymin = 0  # Starts at the bottom of the plot
                    ymax = (y_intersection - plt.gca().get_ylim()[0]) / (
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
                        ymin=ymin,
                        ymax=ymax,
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

        plt.title(plot_title, fontsize=plot_title_size)

        if savefig is None:
            pass
        else:
            plt.savefig(savefig, dpi=300)

        return fig

    def grid_curve_plot(
        self,
        concentration_col: str = None,
        response_col: str = None,
        name_col: str = None,
        column_num: int = 2,
        plot_title: str = None,
        plot_title_size: int = 20,
        xlabel: str = None,
        ylabel: str = None,
        conc_unit: str = "nM",
        xscale: str = "log",
        xscale_ticks: tuple = None,
        ymax: int = None,
        ymin: int = None,
        line_color: list = CBPALETTE,
        line_width: int = 1.5,
        box: bool = False,
        box_color: str = "gray",
        box_intercept: int = 50,
        hline: int = None,
        hline_color: str = "gray",
        vline: int = None,
        vline_color: str = "gray",
        figsize: tuple = (8.4, 4.8),
        savefig: str = None,
        verbose: bool = None,
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
        :param plot_title: str
            Title of the figure
        :param plot_title_size: int
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
        :param savefig: str
            File path for save location.
        :param verbose: bool
            Output information about the plot.

        :return: Figure
        """

        global x_fit, reverse, params, response, concentration
        name_list = np.unique(self.data[name_col])

        # Generate lists for modifying plots (vline, box, etc)
        concentration_list = []
        response_list = []
        y_fit_list = []
        x_fit_list = []

        for drug in name_list:
            drug_query = self._filter_dataframe(drug)

            # Create variables for inputs. Extract column from Dataframe
            if concentration_col is None:
                concentration_col = self.concentration_col
            if response_col is None:
                response_col = self.response_col

            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Set initial guess for 4PL equation
            initial_guess = [
                max(response),
                min(response),
                1.0,
                1.0,
            ]  # Max, Min, ic50, and hill_slope

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
                initial_guess=initial_guess,
                response=response,
            )
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

        # Set up color options for line colors
        if line_color is not CBPALETTE:
            # if user uses list and it does not match name of drug names
            if isinstance(line_color, list) and len(line_color) == len(name_list):
                cycle_color = itertools.cycle(line_color)
                line_color = tuple([next(cycle_color) for _ in range(len(name_list))])

            # if user uses list and it does not match the length of the drug names
            elif isinstance(line_color, list) and len(line_color) != len(name_list):
                # Extend the list of colors to match the length of names
                extended_colors = line_color * ((len(name_list) // len(line_color)) + 1)
                line_color = extended_colors[
                    : len(name_list)
                ]  # Trim the extended list to the same length as names

            # If user only gives a string of 1 color, duplicate color name to match length of drug names
            elif len(line_color) is not len(name_list):
                line_color_list = tuple([line_color] * len(name_list))
                line_color = line_color_list
        else:
            pass

        # Check for xscale label:
        if xscale == "log" or xscale == "linear":
            pass
        else:
            raise ValueError("xscale must be 'log' or 'linear'")

        # Generate figure in grid layout
        # Calculate the number of rows needed
        num_plots = len(name_list)  # determine total plots to make
        row_num = -(-num_plots // column_num)  # Round up to the nearest integer
        # Squeeze to handle possible 1D array
        fig, axes = plt.subplots(row_num, column_num, figsize=figsize, squeeze=False)
        fig.suptitle(plot_title, fontsize=plot_title_size)

        # Loop through the data and plot scatter and line plots on each subplot
        for i in range(row_num):
            for j in range(column_num):
                # Line plot
                axes[i, j].plot(
                    x_fit_list[i * column_num + j],
                    y_fit_list[i * column_num + j],
                    label="Line Plot",
                    color=line_color[i * column_num + j],
                    linewidth=line_width,
                )

                # Scatter plot
                axes[i, j].scatter(
                    concentration_list[i * column_num + j],
                    response_list[i * column_num + j],
                    label="Scatter Plot",
                    color=line_color[i * column_num + j],
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
                axes[i, j].set_title(name_list[i * column_num + j])

                if box is True:
                    if isinstance(box_intercept, (int, float)) and reverse == 1:
                        y_intersection = box_intercept
                        interpretation = interp1d(
                            y_fit_list[i * column_num + j],
                            x_fit_list[i * column_num + j],
                            kind="linear",
                            fill_value="extrapolate",
                        )

                        x_concentration = interpretation(y_intersection)
                        # Constrain box to 50% drug response
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
                                f"Box X intersection ({name_list[j]}): ",
                                np.round(x_concentration, 3),
                                f"{conc_unit}",
                            )
                            print(
                                f"Box Y intersection ({name_list[j]}): ",
                                np.round(y_intersection, 3),
                                "%",
                            )

                    elif (
                        box_intercept is not None
                        and isinstance(box_intercept, (int, float))
                        and reverse == 0
                    ):
                        y_intersection = box_intercept

                        x_concentration = np.interp(
                            y_intersection,
                            y_fit_list[i * column_num + j],
                            x_fit_list[i * column_num + j],
                        )

                        # Constrain box to 50% drug response
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
                                f"Box X intersection ({name_list[j]}): ",
                                np.round(x_concentration, 3),
                                f"{conc_unit}",
                            )
                            print(
                                f"Box Y intersection ({name_list[j]}): ",
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
                axes[i, j].set_xlabel(xlabel)
                axes[i, j].set_ylabel(ylabel)

        # Remove grid lines and plot ticks for each subplot
        for ax_row in axes:
            for ax in ax_row:
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

        if savefig is None:
            pass
        else:
            plt.savefig(savefig, dpi=300)

        return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
