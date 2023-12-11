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
    def scale_units(self, xscale_unit, xscale_ticks): # todo have xscale_unit be automatic based on concentration?
        """Logic function for curve plot"""

        if xscale_unit == None and xscale_ticks == None:
            x_fit = np.logspace(0, 5, 100)
            return x_fit
        elif xscale_unit == 'nM' and xscale_ticks == None:
            x_fit = np.logspace(0, 5, 100)
            print('ticks constrained to nM!')
            return x_fit
        elif xscale_unit == 'µM' and xscale_ticks == None:
            print('ticks constrained to µM!')
            x_fit = np.logspace(-3, 3, 100)
            return x_fit
        elif xscale_unit == 'µM' and xscale_ticks is not None:
            print('ticks constrained to µM!')
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
            return x_fit
        elif xscale_unit == 'nM' and xscale_ticks is not None:
            print('ticks constrained to nM!')
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
            return x_fit
        elif xscale_unit == 'uM' and xscale_ticks is not None:
            print('ticks constrained to µM!')
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 100)
            return x_fit
        else:
            print('Insufficient input for xscale_unit and xscale_ticks')

    def xscale(self, xscale_unit, concentration):
        """Logic function for curve plot"""

        if xscale_unit == 'nM':
            print('Concentration on X-axis will be nM')
            return concentration
        elif xscale_unit == 'uM':
            print('Concentration on X-axis will be µM')
            concentration = concentration / 1000  # convert drug concentration to µM
            return concentration
        elif xscale_unit == 'µM':
            print('Concentration on X-axis will be µM')
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
        if box_target is True and reverse is 1:
            y_intersection = box_intercept
            interpretation = interp1d(y_fit, x_fit, kind='linear', fill_value="extrapolate")
            x_intersection = interpretation(y_intersection)
        elif box_target is True and reverse is 0:
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
