from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from py50.utils import _normalize_units

__all__ = ["CurveSettings", "CBPALETTE", "CBMARKERS"]

"""
Color and Marker schemes and functions for plotting
"""

# Color-blind safe palette from
# http://bconnelly.net/2013/10/creating-colorblind-friendly-figures
CBPALETTE = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)

# Matplotlib markers. Default is same length as CBPALETTE. More marker info can be found here:
# https://matplotlib.org/api/markers_api.html
CBMARKERS = ("o", "^", "s", "D", "v", "<", ">", "p")
assert len(CBPALETTE) == len(CBMARKERS)


class CurveSettings:
    @staticmethod
    def _normalize_unit(xscale_unit: Optional[str] = None) -> str:
        """Normalize concentration unit aliases to 'nM', 'µM', or 'pM'."""
        return _normalize_units(xscale_unit)

    def scale_units(self, drug_name, xscale_unit, xscale_ticks, verbose=None):
        """
        Logic function for a curve plot. Scales units depending on input (nanomolar (nM),
        micromolar (uM or µM), or picomolar (pM)) and determines the drawing range for the fitted curve.

        :param drug_name: Name of the query drug
        :param xscale_unit: Unit concentration for query drug (defaults to 'nM' when None)
        :param xscale_ticks: Optional 2-tuple of log10 exponents controlling the fitted x-range
        :param verbose: Output information about drug query and its concentration
        """
        normalized_unit = self._normalize_unit(xscale_unit)
        if xscale_ticks is None:
            x_fit = np.logspace(0, 5, 1000)
        else:
            x_fit = np.logspace(xscale_ticks[0], xscale_ticks[1], 1000)

        if verbose is True:
            if xscale_unit is None:
                print(f"Assuming {drug_name} concentration are in {normalized_unit}!")
            else:
                print(f"{drug_name} concentration will be in {normalized_unit}!")

        return x_fit, normalized_unit

    def conc_scale(self, xscale_unit, concentration, verbose=None):
        """
        Logic function for curve plot. Rescales concentration values from nM to the target plot unit
        ('nM', 'µM', or 'pM').

        :param xscale_unit: The target unit of drug concentration (defaults to 'nM' when None)
        :param concentration: Concentration Series/array in nM
        :param verbose: Output unit information for the x-axis
        """
        normalized_unit = self._normalize_unit(xscale_unit)

        if normalized_unit == "nM":
            if verbose is True:
                if xscale_unit is None:
                    print("Assume concentration will be in nM")
                else:
                    print("Concentration on X-axis will be in nM")
            return concentration

        if normalized_unit == "µM":
            if verbose is True:
                print("Concentration on X-axis will be in µM")
            return concentration / 1000

        if verbose is True:
            print("Concentration on X-axis will be in pM")
        return concentration * 1000

    def yaxis_scale(
        self,
        box=None,
        reverse=None,
        y_intersection=None,
        x_intersection=None,
        box_color=None,
    ):
        """Logic function for scaling box highlight"""
        if box == True and reverse == 0:
            ymin = 0  # Starts at the bottom of the plot
            ymax = (y_intersection - plt.gca().get_ylim()[0]) / (
                plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
            )
            # Converted x_intersection from a numpy array into a float
            plt.axvline(
                x=x_intersection, ymin=ymin, ymax=ymax, color=box_color, linestyle="--"
            )
            plt.hlines(
                y=y_intersection,
                xmin=0,
                xmax=x_intersection,
                colors=box_color,
                linestyles="--",
            )
        elif box == True and reverse == 1:
            ymin = 0  # Starts at the bottom of the plot
            ymax = (y_intersection - plt.gca().get_ylim()[0]) / (
                plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
            )
            # Converted x_intersection from a numpy array into a float
            plt.axvline(
                x=x_intersection, ymin=ymin, ymax=ymax, color=box_color, linestyle="--"
            )
            plt.hlines(
                y=y_intersection,
                xmin=0,
                xmax=x_intersection,
                colors=box_color,
                linestyles="--",
            )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
