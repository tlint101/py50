import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import itertools
from scipy.optimize import curve_fit
from py50.plot_settings import CBMARKERS, CBPALETTE
from py50.calculate import Calculate


# todo Generate composite functions

class PlotCurve:
    # Will accept input DataFrame and output said DataFrame for double checking.
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self.df = df

    def show(self):
        return self.df

    def show_column(self, key):
        if key not in self.df.columns:
            raise ValueError('Column not found')
        return self.df[key]

    # Filter input data based on Compound Name to generate single plot
    def filter_dataframe(self, drug_name):
        filtered_df = self.df[self.df['Compound Name'] == drug_name]
        return filtered_df

    def single_curve_plot(self,
                          concentration_col,
                          response_col,
                          drug_name=None,
                          plot_title=None,
                          xlabel=None,
                          ylabel=None,
                          xscale='log',
                          xscale_unit=None,
                          xscale_ticks=None,
                          line_color='black',
                          marker=None,
                          legend=False,
                          box=False,
                          box_color='gray',
                          box_intercept=None,
                          x_concentration=None,
                          output_filename=None):
        """
        Generate a plot for one drug target.
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param drug_name: Identify name of drug for plotting
        :param plot_title: Title of the figure
        :param xlabel: Title of the X-axis
        :param ylabel: Title of the Y-axis
        :param xscale: Set the scale of the X-axis as logarithmic or linear. It is logarithmic by default.
        :param xscale_unit: Input will assume that the concentration will be in nM.
        Thus, it will be automatically converted into µM.
        If xscale_unit is given as nM, no conversion will be performed.
        :param xscale_ticks: Set the scale of the X-axis
        :param line_color: Optional. Takes a list of colors. By default, it uses the CBPALETTE. List can contain name of
        colors or colors in hex code.
        :param marker: Optional. Takes a list of for point markers.
        :param legend: Optional. Denotes a figure legend.
        :param box: Optional. Draw a box to highlight a specific location. If box = True, then the box_color,
        box_intercept, and x_concentration MUST ALSO BE GIVEN.
        :param box_color: Set color of box.
        :param box_intercept: Set horizontal location of box. By default, it is set at Absolute IC50.
        :param x_concentration: Set vertical location of the box. By default, this is set to None. For example, if the
        box_intercept is set to 50%, then the x_concentration must be the Absolute IC50 value. If there is an input, it
        will override the box_intercept and the response data will move accordingly. Finally, the number must be in the
        same unit as the X-axis. i.e., if the axis is in µM, then the number for hte x_concentration should be in µM and
        vice versa.
        :param output_filename: File path for save location.
        :return: Figure
        """

        global x_fit, df, y_intersection, x_intersection
        if drug_name is not None:
            df = self.filter_dataframe(drug_name=drug_name)
            if len(df) > 0:
                pass
            elif len(df) == 0:
                print('Drug not found!')
        else:
            print('Drug not found!')

        # Create variables for inputs. Extract column from Dataframe
        concentration = df[concentration_col]
        response = df[response_col]

        # Convert concentration for scaling
        if xscale_unit == 'nM':
            print('Concentration on X-axis is in nM')
        elif xscale_unit == 'µM':
            print('Concentration on X-axis converted to µM')
            concentration = concentration / 1000  # convert drug concentration to µM
        else:
            print(f'Assume {drug_name} Concentration is in nM')

        # Perform constrained nonlinear regression to estimate the parameters
        initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope
        params, covariance, *_ = curve_fit(Calculate.fourpl,  # Static method from Calculate class
                                           concentration,
                                           response,
                                           p0=initial_guess,
                                           maxfev=10000)  # params = maximum, minimum, ic50, and hill_slope

        # Create constraints for the concentration values. Only specificy xscale_unit will use default ticks
        # Modifying both will increase or decrease the x tick scale
        if xscale_unit == None and xscale_ticks == None:
            x_fit = np.logspace(0, 5, 100)
        elif xscale_unit == 'nM' and xscale_ticks == None:
            x_fit = np.logspace(0, 5, 100)
        elif xscale_unit == 'µM' and xscale_ticks == None:
            x_fit = np.logspace(-3, 2, 100)
        elif xscale_unit == 'nM' and xscale_ticks is not None:
            print('nM with ticks constraints!')
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
        elif xscale_unit == 'µM' and xscale_ticks is not None:
            print('µM with ticks constraints!')
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
        else:
            print('Insufficient input for xscale_unit and xscale_ticks')

        # Compute the corresponding response values using the 4PL equation and fitted parameters
        y_fit = Calculate.fourpl(x_fit, *params)

        # Boolean check for marker
        if marker is not None:
            marker = marker
        else:
            marker = 'o'

        # Create the plot
        fig, ax = plt.subplots()
        ax.set_ylim(top=100)  # Set maximum y axis limit
        ax.scatter(concentration, response, marker=marker, color=line_color)
        ax.plot(x_fit, y_fit, color=line_color)
        ax.set_title(plot_title) # todo add hidden functions to increase font of title and labels.
        ax.set_xscale(xscale)  # Use a logarithmic scale for the x-axis
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Set y-axis limit
        # Y-axis limit will be limited to the largest response number and add 10 for spacing
        max_value = max(response) + 10
        min_value = min(response)
        if min_value < 0:
            ax.set_ylim(min_value-5, max_value)
        else:
            ax.set_ylim(0, max_value)

        # Plot box to IC50 on curve
        # Interpolate to find the x-value (Concentration) at the intersection point
        if box_intercept:
            y_intersection = box_intercept
            x_intersection = np.interp(y_intersection, y_fit, x_fit)

        if x_concentration is not None and box_intercept is not None:
            x_intersection = x_concentration
            y_intersection = np.interp(x_intersection, x_fit, y_fit)
            print(x_intersection)
            print(y_intersection)

        # Calculate ymin and ymax for box
        if box:
            ymin = 0  # Starts at the bottom of the plot
            ymax = (y_intersection - plt.gca().get_ylim()[0]) / (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
            # Converted x_intersection from a numpy array into a float
            plt.axvline(x=x_intersection, ymin=ymin, ymax=ymax, color=box_color, linestyle='--')
            plt.hlines(y=y_intersection, xmin=0, xmax=x_intersection, colors=box_color, linestyles='--')

        # Figure legend
        if legend:
            ax.legend(handles=[plt.Line2D([0], [0], color=line_color, marker=marker, label=drug_name), ], loc='best')

        # Save the plot to a file
        if output_filename == None:
            pass
        else:
            plt.savefig(output_filename, dpi=300)  # Save the plot to a file with the specified filename

        return fig

    def multi_curve_plot(self,
                         concentration_col,
                         response_col,
                         name_col,
                         plot_title='Dose-Response',
                         xlabel='Logrithmic Concentration (nM)',
                         ylabel='Inhibition %',
                         xscale='log',
                         xscale_unit=None,
                         xscale_ticks=None,
                         line_color=CBPALETTE,
                         marker=CBMARKERS,
                         legend=False,
                         box_target=False,
                         box_color='gray',
                         box_intercept=None,
                         output_filename=None):
        """
        Genereate a plot with multiple curves.
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param name_col: Name column from DataFrame
        :param plot_title: Title of the figure
        :param xlabel: Title of the X-axis
        :param ylabel: Title of the Y-axis
        :param xscale: Set the scale of the X-axis as logarithmic or linear. It is logarithmic by default.
        :param xscale_unit: Input will assume that the concentration will be in nM.
        Thus, it will be automatically converted into µM.
        If xscale_unit is given as nM, no conversion will be performed.
        :param xscale_ticks: Set the scale of the X-axis
        :param line_color: Optional. Takes a list of colors. By default, it uses the CBPALETTE. List can contain name of
        colors or colors in hex code.
        :param marker: Optional. Takes a list of for point markers.
        :param legend: Optional. Denotes a figure legend.
        :param box_target: Optional. Draw a box to highlight a specific location. If box = True, then the box_color,
        box_intercept, and x_concentration MUST ALSO BE GIVEN.
        :param box_color: Set color of box.
        :param box_intercept: Set horizontal location of box. By default, it is set at Absolute IC50.
        :param output_filename: File path for save location.
        :return: Figure
        """
        global response, x_fit, y_fit, y_intersection, x_intersection, ymin, ymax
        name_list = np.unique(self.df[name_col])

        concentration_list = []
        response_list = []
        y_fit_list = []
        x_fit_list = []

        for drug in name_list:
            df = self.filter_dataframe(drug)

            # Create variables for inputs. Extract column from Dataframe
            concentration = df[concentration_col]
            response = df[response_col]

            # Convert concentration for scaling
            if xscale_unit == 'nM':
                print('Concentration on X-axis is in nM')
            elif xscale_unit == 'µM':
                print('Concentration on X-axis converted to µM')
                concentration = concentration / 1000  # convert drug concentration to µM
            else:
                print(f'Assume {drug} Concentration is in nM')

            concentration_for_list = concentration.values  # Convert into np.array
            concentration_list.append(concentration_for_list)
            response_for_list = response.values
            response_list.append(response_for_list)

            # Perform constrained nonlinear regression to estimate the parameters
            initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope
            params, covariance, *_ = curve_fit(Calculate.fourpl,  # Static method from Calculate class
                                               concentration,
                                               response,
                                               p0=initial_guess,
                                               maxfev=10000)  # params = maximum, minimum, ic50, and hill_slope

            # Generate script to calculate the covariance and plot them
            # todo Calculate standard deviations from the covariance matrix
            std_dev = np.sqrt(np.diag(covariance))

            # Create constraints for the concentration values. Only specificy xscale_unit will use default ticks
            # Modifying both will increase or decrease the x tick scale
            if xscale_unit == None and xscale_ticks == None:
                x_fit = np.logspace(0, 5, 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'nM' and xscale_ticks == None:
                x_fit = np.logspace(0, 5, 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'µM' and xscale_ticks == None:
                x_fit = np.logspace(-3, 2, 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'nM' and xscale_ticks is not None:
                print('nM with ticks constraints!')
                x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'µM' and xscale_ticks is not None:
                print('µM with ticks constraints!')
                x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
                x_fit_list.append(x_fit)
            else:
                print('Insufficient input for xscale_unit and xscale_ticks')

            # Compute the corresponding response values using the 4PL equation and fitted parameters
            y_fit = Calculate.fourpl(x_fit, *params)
            y_fit_list.append(y_fit)

        # Generate figure
        fig, ax = plt.subplots()
        ax.set_ylim(top=100)  # Set maximum y axis limit

        if line_color is CBPALETTE:
            pass
        else:
            line_color = None

        if marker is CBMARKERS:
            pass
        else:
            marker = None

        # Plotting the data for each line
        legend_handles = []  # Store data for each line as a dictionary inside a list for the legend
        for i, (y_fit_point, concentration_point, response_point, name, color, mark) in enumerate(
                zip(y_fit_list, concentration_list, response_list, name_list, line_color, marker)):
            line = plt.plot(x_fit, y_fit_point, color=color, label=name)
            scatter = ax.scatter(concentration_point, response_point, color=color, marker=mark)

            ax.set_title(plot_title)
            ax.set_xscale(xscale)  # Use a logarithmic scale for the x-axis
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Append scatter plot handle to the legend
            legend_piece = {}
            legend_piece['marker'] = mark
            legend_piece['name'] = name
            legend_piece['line_color'] = color
            legend_handles.append(legend_piece)

        # Set y-axis limit
        # Y-axis limit will be limited to the largest response number and add 10 for spacing
        max_value = max(response) + 10
        min_value = min(response)
        if min_value < 0:
            ax.set_ylim(min_value, max_value)
        else:
            ax.set_ylim(0, max_value)

        # Plot box to IC50 on curve
        # Interpolate to find the x-value (Concentration) at the intersection point
        if box_intercept:
            y_intersection = box_intercept
            x_intersection = np.interp(y_intersection, y_fit, x_fit)

        # Figure legend
        # Extract elements (Line, Scatterplot, and color) from figures and append into a list for generating legend
        if legend:
            legend_elements = []
            for data in legend_handles:
                legend_element = mlines.Line2D([0], [0], color=data['line_color'], marker=data['marker'],
                                               label=data['name'])
                legend_elements.append(legend_element)
            ax.legend(handles=legend_elements, loc='best')

            # Calculate ymin and ymax for box
            if box_target is True:
                ymin = 0  # Starts at the bottom of the plot
                ymax = (y_intersection - plt.gca().get_ylim()[0]) / (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
                # Converted x_intersection from a numpy array into a float
                plt.axvline(x=x_intersection.item(), ymin=ymin, ymax=ymax, color=box_color, linestyle='--')
                plt.hlines(y=y_intersection, xmin=0, xmax=x_intersection, colors=box_color, linestyles='--')
            elif isinstance(box_target, str):
                if box_target in name_list:
                    indices = np.where(name_list == box_target)[0]
                    if indices.size > 0:
                        first_index = indices[0]
                        x_intersection = np.interp(y_intersection, y_fit_list[first_index], x_fit)
                        ymin = 0  # Starts at the bottom of the plot
                        ymax = (y_intersection - plt.gca().get_ylim()[0]) / (
                                plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
                    # Converted x_intersection from a numpy array into a float
                    plt.axvline(x=x_intersection.item(), ymin=ymin, ymax=ymax, color=box_color, linestyle='--')
                    plt.hlines(y=y_intersection, xmin=0, xmax=x_intersection, colors=box_color, linestyles='--')
                else:
                    print('Drug name does not match box target!')
            else:
                print('Something wrong with box inputs!')

        if output_filename is None:
            pass
        else:
            plt.savefig(output_filename, dpi=300)

        return fig

    def grid_curve_plot(self,
                        concentration_col,
                        response_col,
                        name_col,
                        column_num=2,
                        # row_num=2,
                        plot_title=None,
                        xlabel='Logrithmic Concentration (nM)',
                        ylabel='Inhibition %',
                        xscale='log',
                        xscale_unit=None,
                        xscale_ticks=None,
                        line_color=CBPALETTE,
                        box=True,
                        box_color='gray',
                        box_intercept=None,
                        output_filename=None):
        """
        Generate multiple curves in a grid.
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param name_col: Name column from DataFrame
        :param column_num: Set number of column grid
        :param plot_title: Title of the figure
        :param xlabel: Title of the X-axis
        :param ylabel: Title of the Y-axis
        :param xscale: Set the scale of the X-axis as logarithmic or linear. It is logarithmic by default.
        :param xscale_unit: Input will assume that the concentration will be in nM.
        Thus, it will be automatically converted into µM.
        If xscale_unit is given as nM, no conversion will be performed.
        :param xscale_ticks: Set the scale of the X-axis
        :param line_color: Optional. Takes a list of colors. By default, it uses the CBPALETTE. List can contain name of
        colors or colors in hex code.
        :param box: Optional. Draw a box to highlight a specific location. If box = True, then the box_color,
        box_intercept, and x_concentration MUST ALSO BE GIVEN.
        :param box_color: Set color of box.
        :param box_intercept: Set horizontal location of box. By default, it is set at Absolute IC50.
        :param output_filename: File path for save location.
        :return: Figure
        """

        global x_fit
        name_list = np.unique(self.df[name_col])

        concentration_list = []
        response_list = []
        y_fit_list = []
        x_fit_list = []

        for drug in name_list:
            df = self.filter_dataframe(drug)

            # Create variables for inputs. Extract column from Dataframe
            concentration = df[concentration_col]
            response = df[response_col]

            # Convert concentration
            if xscale_unit == 'nM':
                print('Concentration on X-axis is in nM')
            elif xscale_unit == 'µM':
                print('Concentration on X-axis converted to µM')
                concentration = concentration / 1000  # convert drug concentration to µM
            else:
                print(f'Assume {drug} Concentration is in nM')

            concentration_for_list = concentration.values  # Convert into np.array
            concentration_list.append(concentration_for_list)
            response_for_list = response.values
            response_list.append(response_for_list)

            # Perform constrained nonlinear regression to estimate the parameters
            initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope
            params, covariance, *_ = curve_fit(Calculate.fourpl,  # Static method from Calculate class
                                               concentration,
                                               response,
                                               p0=initial_guess,
                                               maxfev=10000)  # params = maximum, minimum, ic50, and hill_slope

            # Create constraints for the concentration values. Only specificy xscale_unit will use default ticks
            # Modifying both will increase or decrease the x tick scale
            if xscale_unit == None and xscale_ticks == None:
                x_fit = np.logspace(0, 5, 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'nM' and xscale_ticks == None:
                x_fit = np.logspace(0, 5, 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'µM' and xscale_ticks == None:
                x_fit = np.logspace(-3, 2, 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'nM' and xscale_ticks is not None:
                print('nM with ticks constraints!')
                x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
                x_fit_list.append(x_fit)
            elif xscale_unit == 'µM' and xscale_ticks is not None:
                print('µM with ticks constraints!')
                x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
                x_fit_list.append(x_fit)
            else:
                print('Insufficient input for xscale_unit and xscale_ticks')

            # Compute the corresponding response values using the 4PL equation and fitted parameters
            y_fit = Calculate.fourpl(x_fit, *params)
            y_fit_list.append(y_fit)

        # Set up color options for line colors
        if line_color is not CBPALETTE:
            # if user uses list and it does not match name of drug names
            if isinstance(line_color, list) and len(line_color) == len(name_list):
                cycle_color = itertools.cycle(line_color)
                line_color = tuple([next(cycle_color) for _ in range(len(name_list))])

            # if user uses list and it doesn ot match the length of the drug names
            elif isinstance(line_color, list) and len(line_color) != len(name_list):
                # Extend the list of colors to match the length of names
                extended_colors = line_color * ((len(name_list) // len(line_color)) + 1)
                line_color = extended_colors[:len(name_list)]  # Trim the extended list to the same length as names

            # If user only gives a string of 1 color, duplicate color name to match length of drug names
            elif len(line_color) is not len(name_list):
                line_color_list = tuple([line_color] * len(name_list))
                line_color = line_color_list
        else:
            pass

        # Generate figure in grid layout
        # Calculate the number of rows needed
        num_plots = len(name_list)  # determine total plots to make
        row_num = -(-num_plots // column_num)  # Round up to the nearest integer
        fig, axes = plt.subplots(row_num, column_num, figsize=(10, 8))
        fig.suptitle(plot_title, fontsize=20)

        # Loop through the data and plot scatter and line plots on each subplot
        for i in range(row_num):
            for j in range(column_num):
                # Line plot
                axes[i, j].plot(x_fit_list[i * column_num + j], y_fit_list[i * column_num + j], label='Line Plot',
                                color=line_color[i * column_num + j])

                # Scatter plot
                axes[i, j].scatter(concentration_list[i * column_num + j], response_list[i * column_num + j],
                                   label='Scatter Plot',
                                   color=line_color[i * column_num + j])

                # Set x-axis scale
                axes[i, j].set_xscale(xscale)

                # Set y-axis limit
                # Y-axis limit will be limited to the largest response number and add 10 for spacing
                max_value = np.amax([np.amax(max_value) for max_value in response_list]) + 10
                ymin = -10  # Y-axis allowed to go -10 for better curve viewing
                axes[i, j].set_ylim(ymin, max_value)

                # Set subplot title
                axes[i, j].set_title(name_list[i * column_num + j])

                # todo add exception
                if box is True:
                    if box_intercept is not None and isinstance(box_intercept, (int, float)):
                        y_intersection = box_intercept

                        x_concentration = np.interp(y_intersection, y_fit_list[i * column_num + j],
                                                x_fit_list[i * column_num + j])

                        # Constrain box to 50% drug response
                        ymax = (y_intersection - axes[i, j].get_ylim()[0]) / (
                                axes[i, j].get_ylim()[1] - axes[i, j].get_ylim()[0])

                        axes[i, j].axvline(x=x_concentration, ymin=0, ymax=ymax, color=box_color, linestyle='--')
                        axes[i, j].hlines(y=y_intersection, xmin=0, xmax=x_concentration, colors=box_color, linestyles='--')

                    elif box_intercept is not None and isinstance(box_intercept, str):
                        print('box_intercept is not an int or float')
                    elif box_intercept is None:
                        print('no box_intercept given')

                # Set axis labels
                axes[i, j].set_xlabel(xlabel)
                axes[i, j].set_ylabel(ylabel)

        plt.tight_layout()

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        if output_filename is None:
            pass
        else:
            plt.savefig(output_filename, dpi=300)

        return fig


if __name__ == '__main__':
    import doctest

    doctest.testmod()
