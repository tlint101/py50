import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

"""
Color and Marker schemes and functions for plotting
"""

# Color-blind safe palette from
# http://bconnelly.net/2013/10/creating-colorblind-friendly-figures
CBPALETTE = ('#000000', '#E69F00', '#56B4E9', '#009E73',
             '#F0E442', '#0072B2', '#D55E00', '#CC79A7')

# Matplotlib markers. Default is same length as CBPALETTE. More marker info can be found here:
# https://matplotlib.org/api/markers_api.html
CBMARKERS = ('o', '^', 's', 'D', 'v', '<', '>', 'p')
assert len(CBPALETTE) == len(CBMARKERS)


class CurveSettings:
    def scale_units(self, drug_name, xscale_unit, xscale_ticks, verbose=None):
        """
        Logic funtion for a curve plot. This will scale units depending on input (nanomolar (nM) or
        micromolar (uM or µM)). The results will also determine the drawing position for the line curve.

        :param drug_name: Name of the query drug
        :param xscale_unit: Unit concentration for query drug
        :param xscale_ticks: Range to scale the line curve. This will also influence the X-axis length
        :param verbose: Ouput information about drug query and its concentration
        """
        if xscale_unit == 'nM' and xscale_ticks is None:
            x_fit = np.logspace(0, 5, 1000)
            if verbose is True:
                print(f'{drug_name} concentration will be in {xscale_unit}!')
            return x_fit, xscale_unit
        elif xscale_unit == 'nM' and xscale_ticks is not None:
            # uses np.logspace tuple to control line length
            # If xscale_ticks is none, this format will break. See first 2 if/else above
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 1000)
            if verbose is True:
                print(f'{drug_name} concentration will be in {xscale_unit}!')
            return x_fit, xscale_unit
        elif (xscale_unit == 'µM' or xscale_unit == 'uM') and xscale_ticks is None:
            x_fit = np.logspace(0, 5, 1000)
            # convert uM to µM
            if xscale_unit == 'uM':
                xscale_unit = 'µM'
            if verbose is True:
                print(f'{drug_name} concentration will be in {xscale_unit}!')
            return x_fit, xscale_unit
        elif (xscale_unit == 'µM' or xscale_unit == 'uM') and xscale_ticks is not None:
            # uses np.logspace tuple to control line length
            # If xscale_ticks is none, this format will break. See first 2 if/else above
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 1000)
            # convert uM to µM
            if xscale_unit == 'uM':
                xscale_unit = 'µM'
            if verbose is True:
                print(f'{drug_name} concentration will be in {xscale_unit}!')
            return x_fit, xscale_unit
        elif xscale_unit is None and xscale_ticks is None:
            x_fit = np.logspace(0, 5, 1000)
            if verbose is True:
                print(f'Assuming {drug_name} concentration are in nM!')
            return x_fit, xscale_unit

    def conc_scale(self, xscale_unit, concentration, verbose=None):
        """
        Logic function for curve plot. This will help set the concentration to the correct units for plotting. By
        default, program will assume that input concentration is in nM and the output for graph will be in µM.
        :param xscale_unit: The unit of drug concentration
        :param concentration: The concentration unit of interest. This can be nanomolar (nM) or micromolar (uM or µM)
        :param verbose: Output unit information for the x-axis
        """

        if xscale_unit == 'nM':
            if verbose is True:
                print('Concentration on X-axis will be in nM')
            return concentration
        elif xscale_unit == 'uM' or xscale_unit == 'µM':
            if verbose is True:
                print('Concentration on X-axis will be in µM')
            concentration = concentration / 1000  # convert drug concentration to µM
            return concentration
        else:
            print(f'Assume concentration will be in nM')

    def yaxis_scale(self, box=None, reverse=None, y_intersection=None, x_intersection=None,
                    box_color=None):
        """Logic function for scaling box highlight"""
        if box == True and reverse == 0:
            ymin = 0  # Starts at the bottom of the plot
            ymax = (y_intersection - plt.gca().get_ylim()[0]) / (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
            # Converted x_intersection from a numpy array into a float
            plt.axvline(x=x_intersection, ymin=ymin, ymax=ymax, color=box_color, linestyle='--')
            plt.hlines(y=y_intersection, xmin=0, xmax=x_intersection, colors=box_color, linestyles='--')
        elif box == True and reverse == 1:
            ymin = 0  # Starts at the bottom of the plot
            ymax = (y_intersection - plt.gca().get_ylim()[0]) / (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
            # Converted x_intersection from a numpy array into a float
            plt.axvline(x=x_intersection, ymin=ymin, ymax=ymax, color=box_color, linestyle='--')
            plt.hlines(y=y_intersection, xmin=0, xmax=x_intersection, colors=box_color, linestyles='--')

    def multi_curve_box_highlight(self, box_target=None, box_color=None, box_intercept=None,
                                  y_intersection=None, x_intersection=None, name_list=None, y_fit=None, y_fit_list=None,
                                  x_fit=None, reverse=None, ymin=0, ymax_vline=50):
        if box_target is True and reverse == 1:
            y_intersection = box_intercept
            interpretation = interp1d(y_fit, x_fit, kind='linear', fill_value="extrapolate")
            x_intersection = interpretation(y_intersection)
        elif box_target is True and reverse == 0:
            ymin = 0  # Starts at the bottom of the plot
            ymax = (y_intersection - plt.gca().get_ylim()[0]) / (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
            # Converted x_intersection from a numpy array into a float
            plt.axvline(x=x_intersection.item(), ymin=ymin, ymax=ymax, color=box_color, linestyle='--')
            plt.hlines(y=y_intersection, xmin=0, xmax=x_intersection, colors=box_color, linestyles='--')
        if isinstance(box_target, str) and reverse == 1:
            if box_target in name_list:
                indices = np.where(name_list == box_target)[0]
                if indices.size > 0:
                    first_index = indices[0]
                    y_intersection = box_intercept
                    interpretation = interp1d(y_fit, x_fit, kind='linear', fill_value="extrapolate")
                    x_intersection = interpretation(y_intersection)
                else:
                    x_intersection = np.interp(y_intersection, y_fit_list[0], x_fit)
                    ymin = 0  # Starts at the bottom of the plot
                    ymax_vline = (y_intersection - plt.gca().get_ylim()[0]) / (
                            plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
                # Converted x_intersection from a numpy array into a float
                plt.axvline(x=x_intersection.item(), ymin=ymin, ymax=ymax_vline, color=box_color, linestyle='--')
                plt.hlines(y=y_intersection, xmin=0, xmax=x_intersection, colors=box_color, linestyles='--')
            else:
                print('Drug name does not match box target!')
        else:
            pass


if __name__ == '__main__':
    import doctest

    doctest.testmod()
