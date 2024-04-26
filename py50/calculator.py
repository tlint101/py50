import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from py50.plot_settings import CurveSettings


class Calculator:
    # Will accept input DataFrame and output said DataFrame for double checking.
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self.data = data
        self.calculation = None

    def show(self, rows: int = None):
        """
        show DataFrame

        :param rows: int
            Indicate the number of rows to display. If none, automatically show 5.
        :return: DataFrame
        """

        if rows is None:
            return self.data.head()
        elif isinstance(rows, int):
            return self.data.head(rows)

    def show_column(self, key: str = None):
        """
        View specific column from DataFrame

        :param  key: String
            Input column header. Must be a column header found in class input DataFrame.

        :return: DataFrame
        """
        if key not in self.data.columns:
            raise ValueError("Column not found")
        return self.data[key]

    """Functions for calculations below"""

    def calculate_ic50(
        self,
        name_col: str = None,
        concentration_col: str = None,
        response_col: str = None,
        input_units: str = None,
        verbose: bool = None,
    ):
        """
        Calculations previously performed in relative_calculation(). The dictionary results are converted into into a
        pandas DataFrame

        :param name_col: str
            Name column from DataFrame
        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: str
            Response column from DataFrame
        :param input_units: str
            Units of input dataset. Default is nM.
        :param verbose: bool
            Output drug concentration units.

        :return: DataFrame generated from the list from the relative_calculation method
        """

        # Set variables from function and convert name_col to np array
        values = self._relative_calculation(
            name_col, concentration_col, response_col, input_units, verbose
        )

        result_df = pd.DataFrame(values)

        self.calculation = result_df
        return self.calculation

    def calculate_absolute_ic50(
        self,
        name_col: str = None,
        concentration_col: str = None,
        response_col: str = None,
        input_units: str = None,
        verbose: bool = None,
    ):
        """
        Calculations previously performed in absolute_calculation(). The dictionary results are converted into a
        pandas DataFrame

        :param name_col: str
            Name column from DataFrame
        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: str
            Response column from DataFrame
        :param input_units: str
            Units of input dataset. Default is nM.
        :param verbose: bool
            Output drug concentration units.

        :return: DataFrame generated from the list from the absolute_calculation method
        """

        values = self._absolute_calculation(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=response_col,
            input_units=input_units,
            verbose=verbose,
        )
        result_df = pd.DataFrame(values)

        self.calculation = result_df
        return self.calculation

    def calculate_pic50(
        self,
        name_col: str = None,
        concentration_col: str = None,
        response_col: str = None,
        input_units: str = None,
        verbose: bool = None,
    ):
        """
        Convert IC50 into pIC50 values. Calculation is performed using the absolute_calculation. As such, two columns
        will be appended - relative pIC50 and absolute pIC50. Conversion is performed by convert the IC50 values from nM
        to M levels and then taking the negative log value of said number.

        :param name_col: str
            Name column from DataFrame
        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: str
            Response column from DataFrame
        :param input_units: str
            Units of input dataset. Default is nM.
        :param verbose: bool
            Output drug concentration units.

        :return: DataFrame from calculate_absolute_ic50 along with the pIC50 values
        """
        values = self._absolute_calculation(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=response_col,
            input_units=input_units,
            verbose=verbose,
        )
        result_df = pd.DataFrame(values)

        if input_units is None or input_units == "nM":
            result_df["relative pIC50"] = -np.log10(
                result_df["relative ic50 (nM)"] * 0.000000001
            )
            result_df["absolute pIC50"] = -np.log10(
                result_df["absolute ic50 (nM)"] * 0.000000001
            )
        elif input_units == "µM":
            result_df["relative pIC50"] = -np.log10(
                result_df["relative ic50 (µM)"] * 0.000001
            )
            result_df["absolute pIC50"] = -np.log10(
                result_df["absolute ic50 (µM)"] * 0.000001
            )

        self.calculation = result_df
        return self.calculation

    """Support functions below"""

    """Define the 4-parameter logistic (4PL) equation"""

    @staticmethod
    def _fourpl(concentration, minimum, maximum, ic50, hill_slope):
        """
        Four-Parameter Logistic (4PL) Equation for calculating curve fit:


        :param concentration: concentration
        :param minimum: minimum concentration in drug query (bottom plateau)
        :param maximum: maximum concentration for drug query (top plateau)
        :param ic50: Concentration at inflection point (where curve shifts from up or down)
        :param hill_slope: Steepness of hte curve (Hill Slope)

        :return: equation
        """
        return minimum + (maximum - minimum) / (
            1 + (concentration / ic50) ** hill_slope
        )

    @staticmethod
    def _reverse_fourpl(concentration, minimum, maximum, ic50, hill_slope):
        """
        Four-Parameter Logistic (4PL) Equation. This reverse function will graph the sigmoid curve from 100% to 0%


        :param concentration: concentration
        :param minimum: minimum concentration in drug query (bottom plateau)
        :param maximum: maximum concentration for drug query (top plateau)
        :param ic50: Concentration at inflection point (where curve shifts from up or down)
        :param hill_slope: Steepness of hte curve (Hill Slope)

        :return: equation
        """
        return minimum + (maximum - minimum) / (
            1 + (concentration / ic50) ** -hill_slope
        )

    def _verbose_calculation(
        self, drug: str = None, input_units: str = None, verbose: bool = True
    ):
        """
        Logic function to calculate unit concentration for Relative and Absolute IC50 calculation. Information will
        detail drug name and concentration unit. Units available a
        re nanomolar (nM) or micromolar (µM or uM).

        :param drug: str
            Input drug name.
        :param input_units: str
            Input drug concentration. Units available are nanomolar (nM) or micromolar (uM or µM)
        :param verbose: bool
            Print out information regarding the concentration unit.

        :return: input_unit concentration
        """

        # Verbose conditions
        if verbose is True:
            # Logic to append concentration units to output DataFrame
            if input_units is None:
                conc_unit = "nM"
            elif input_units == "uM" or input_units == "µM":
                conc_unit = "µM"
            else:
                conc_unit = input_units
            print(f"{drug} concentration is in {conc_unit}!")

    # This method will be used to reduce the functions in the calculating methods below.
    # This will loop through each drug item.
    def _relative_calculation(
        self,
        name_col: str = None,
        concentration_col: str = None,
        response_col: str = None,
        input_units: str = None,
        verbose: bool = None,
    ):
        """
        Calculate relative IC50 values for a given drug. Output will be a dictionary that will be converted into a
        pandas dataframe using the calculate_ic50() function.

        :param name_col: str
            Name column from DataFrame.
        :param concentration_col: str
            Concentration column from DataFrame.
        :param response_col: str
            Response column from DataFrame.
        :param input_units: str
            Concentration units for tested drug. By default, units given will be in nM.
        :param verbose: bool
            Output drug concentration units.

        :return: A dictionary containing drug name, maximum response, minimum response, IC50 (relative) and hill slope.
        """
        # Set variables from function and convert name_col to np array
        global params, conc_unit
        name_col = name_col
        name = self.data[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.data[self.data[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Set initial guess for 4PL equation
            initial_guess = [
                max(response),
                min(response),
                0.5 * (max(response) + min(response)),
                1.0,
            ]  # Max, Min, ic50, and hill_slope

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, inplace=True)

            # todo Calculate standard deviations from the covariance matrix
            # std_dev = np.sqrt(np.diag(covariance))

            # tag response col to determine direction of fourpl equation and fit to 4PL equation
            reverse, params, covariance = self._calc_logic(
                data=query,
                concentration=concentration,
                response_col=response_col,
                initial_guess=initial_guess,
                response=response,
            )

            # If verbose, output info
            self._verbose_calculation(drug, input_units, verbose)

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, f'{input_units}') # For checking

            # Confirm ic50 unit output
            # x_intersection is not needed for relative ic50
            ic50, x_intersection, input_units = self._unit_convert(
                ic50, x_intersection=None, input_units=input_units
            )

            # Logic to append concentration units to output DataFrame
            if input_units is None:
                conc_unit = "nM"
            elif input_units == "nM":
                conc_unit = "nM"
            elif input_units == "uM" or input_units == "µM":
                conc_unit = "µM"

            # Generate DataFrame from parameters
            values.append(
                {
                    "compound_name": drug,
                    "maximum": maximum,
                    "minimum": minimum,
                    f"ic50 ({conc_unit})": ic50,
                    "hill_slope": hill_slope,
                }
            )
        return values

    def _absolute_calculation(
        self,
        name_col: str = None,
        concentration_col: str = None,
        response_col: str = None,
        input_units: str = None,
        verbose: bool = None,
    ):
        """
        Calculate relative IC50 values for a given drug. Output will be a dictionary that will be converted into a
        pandas dataframe using the calculate_absolute_ic50() function.

        :param name_col: str
            Name column from DataFrame.
        :param concentration_col: str
            Concentration column from DataFrame.
        :param response_col: str
            Response column from DataFrame.
        :param input_units: str
            Concentration units for tested drug. By default, units given will be in nM.
        :param verbose: bool
            Output drug concentration units.

        :return: A dictionary containing drug name, maximum response, minimum response, relative IC50,
         absolute IC50, and hill slope.
        """

        # Set variables from function and convert name_col to np array
        global params, conc_unit
        name_col = name_col
        name = self.data[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.data[self.data[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Set initial guess for 4PL equation
            initial_guess = [
                max(response),
                min(response),
                0.5 * (max(response) + min(response)),
                1.0,
            ]  # Max, Min, ic50, and hill_slope

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, ascending=True, inplace=True)

            reverse, params, covariance = self._calc_logic(
                data=query,
                concentration=concentration,
                response_col=response_col,
                initial_guess=initial_guess,
                response=response,
            )

            # If verbose, output info
            # self.verbose_calculation(drug, input_units, verbose)

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, 'nM') # For checking

            # Obtain x_fit. Because calculator does not require xscale_ticks, it is set to None
            # If verbose, more info will be printed
            x_fit, input_units = CurveSettings().scale_units(
                drug_name=drug,
                xscale_unit=input_units,
                xscale_ticks=None,
                verbose=verbose,
            )

            hill_slope, ic50, input_units, x_intersection, y_fit = (
                self._reverse_absolute_calculation(
                    hill_slope,
                    ic50,
                    input_units,
                    maximum,
                    minimum,
                    params,
                    reverse,
                    x_fit,
                )
            )

            # Logic to append concentration units to output DataFrame
            if input_units is None:
                conc_unit = "nM"
            elif input_units == "nM":
                conc_unit = "nM"
            elif input_units == "uM" or input_units == "µM":
                conc_unit = "µM"

            # Generate DataFrame from parameters
            values.append(
                {
                    "compound_name": drug,
                    "maximum": maximum,
                    "minimum": minimum,
                    f"relative ic50 ({conc_unit})": ic50,
                    f"absolute ic50 ({conc_unit})": x_intersection,
                    "hill_slope": hill_slope,
                }
            )
        return values

    def _reverse_absolute_calculation(
        self, hill_slope, ic50, input_units, maximum, minimum, params, reverse, x_fit
    ):
        """
        Support function to condense code. Script will allow the generation of reverse curves.
        """
        # Calculate from parameters 4PL equation
        if reverse == 1:
            y_fit = self._reverse_fourpl(x_fit, maximum, minimum, ic50, hill_slope)
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
        else:
            y_fit = self._fourpl(x_fit, *params)
            y_intersection = 50
            x_intersection = np.interp(y_intersection, y_fit, x_fit)
        # Confirm ic50 unit output
        ic50, x_intersection, input_units = self._unit_convert(
            ic50, x_intersection, input_units
        )
        return hill_slope, ic50, input_units, x_intersection, y_fit

    # When data is reversed, program is not obtaining correct column.
    def _calc_logic(
        self,
        data: pd.DataFrame,
        concentration: pd.Series = None,
        initial_guess: list = None,
        response: pd.Series = None,
        response_col: str = None,
    ):
        """
        Set logic to determine positive or negative sigmoid curve. This method is called by internally by the
        absolute_calculation() method.

        :param data: pd.DataFrame
            Input DataFrame. Must columns with drug name, tested concentration, and Response
        :param concentration: pd. Series
            The concentration column from input DataFrame.
        :param initial_guess: list
            The initial guesses for the 4PL equation.
        :param response: pd.Series
            The response column from the input Dataframe.
        :param response_col: str
            Name of the response column.

        :return: variables for further calculation. This includes: reverse, params, covariance
        """
        global reverse, params, covariance
        if (
            data[response_col].iloc[0] > data[response_col].iloc[-1]
        ):  # Sigmoid curve 100% to 0%
            params, covariance, *_ = curve_fit(
                self._reverse_fourpl,
                concentration,
                response,
                p0=[initial_guess],
                maxfev=100000,
            )
            reverse = 1  # Tag direction of sigmoid curve

        elif (
            data[response_col].iloc[0] < data[response_col].iloc[-1]
        ):  # sigmoid curve 0% to 100%
            params, covariance, *_ = curve_fit(
                self._fourpl, concentration, response, p0=[initial_guess], maxfev=100000
            )
            reverse = 0  # Tag direction of sigmoid curve
        return reverse, params, covariance

    def _unit_convert(
        self, ic50: int = None, x_intersection: int = None, input_units: str = None
    ):
        """
        Converts ic50 to desired input units for the plot_curve class

        :param ic50: int
            IC50 value for conversion. Obtained from the curve parameter values.
        :param x_intersection: int
            This value will correspond to the absolute ic50 value. This is calculated from the curve_fit.
        :param input_units: str
            Unites for the converted IC50 value. Only "nM" or "µM" are supported.
        :return:
        """
        # convert ic50 by input units
        if input_units == "nM":
            return ic50, x_intersection, input_units
        elif input_units == "µM" or input_units == "uM":
            ic50 = ic50 / 1000
            if x_intersection is not None:
                x_intersection = x_intersection / 1000
            return ic50, x_intersection, input_units
        elif input_units is None:
            input_units = "nM"
            return ic50, x_intersection, input_units
        else:
            print("Need to be in 'nM' (Nanomolar) or 'µM' (Micromolar) concentrations!")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
