import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from py50.plot_settings import CurveSettings


class Calculator:
    # Will accept input DataFrame and output said DataFrame for double checking.
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self.df = df

    def show(self):
        """
        show DataFrame

        :return: DataFrame
        """
        return self.df

    def show_column(self, key):
        """
        View specific column from DataFrame

        :param key: column header name.

        :return: DataFrame
        """
        if key not in self.df.columns:
            raise ValueError('Column not found')
        return self.df[key]

    def unit_convert(self, ic50, x_intersection=None, input_units=None):
        """
        Converts ic50 to desired input units for the plot_curve class
        :param ic50:
        :param x_intersection: Corresponds to the absolute ic50 value. This is calculated from the curve_fit
        :param input_units:
        :return:
        """
        # convert ic50 by input units
        if input_units == 'nM':
            return ic50, x_intersection, input_units
        elif input_units == 'µM' or input_units == 'uM':
            ic50 = ic50 / 1000
            if x_intersection is not None:
                x_intersection = x_intersection / 1000
            return ic50, x_intersection, input_units
        elif input_units is None:
            input_units = 'nM'
            return ic50, x_intersection, input_units
        else:
            print('Need nM (Nanomolar) or µM (Micromolar) concentrations!')

    # Define the 4-parameter logistic (4PL) equation
    @staticmethod
    def fourpl(concentration, minimum, maximum, ic50, hill_slope):
        """
        Four-Parameter Logistic (4PL) Equation:


        :param concentration: concentration
        :param minimum: minimum concentration in drug query (bottom plateau)
        :param maximum: maximum concentration for drug query (top plateau)
        :param ic50: Concentration at inflection point (where curve shifts from up or down)
        :param hill_slope: Steepness of hte curve (Hill Slope)

        :return: equation
        """
        return minimum + (maximum - minimum) / (1 + (concentration / ic50) ** hill_slope)

    @staticmethod
    def reverse_fourpl(concentration, minimum, maximum, ic50, hill_slope):
        """
        Four-Parameter Logistic (4PL) Equation. This reverse function will graph the sigmoid curve from 100% to 0%


        :param concentration: concentration
        :param minimum: minimum concentration in drug query (bottom plateau)
        :param maximum: maximum concentration for drug query (top plateau)
        :param ic50: Concentration at inflection point (where curve shifts from up or down)
        :param hill_slope: Steepness of hte curve (Hill Slope)

        :return: equation
        """
        return minimum + (maximum - minimum) / (1 + (concentration / ic50) ** -hill_slope)

    def verbose_calculation(self, drug, input_units, verbose):
        """
        Logic function to calculate unit concentration for Relative and Absolute IC50 calculation. Information will
        detail drug name and concentration unit. Units available are nanomolar (nM) or micromolar (µM or uM).
        :param drug: Input drug name.
        :param input_units: Input drug concentration. Units available are nanomolar (nM) or micromolar (uM or µM)
        :param verbose: Print out information regarding the concentration unit.

        :return: input_unit concentration
        """
        # Verbose conditions
        if verbose is True:
            # Logic to append concentration units to output DataFrame
            if input_units is None:
                conc_unit = 'nM'
            elif input_units == 'uM' or input_units == 'µM':
                conc_unit = 'µM'
            else:
                conc_unit = input_units
            print(f'{drug} concentration is in {conc_unit}!')

    # This method will be used to reduce the functions in the calculating methods below.
    # This will loop through each drug item.
    def relative_calculation(self, name_col, concentration_col, response_col, input_units, verbose=None):
        """
        Calculate relative IC50 values for a given drug. Output will be a dictionary that will be converted into a
        pandas dataframe using the calculate_ic50() function.

        :param name_col: Name column from DataFrame.
        :param concentration_col: Concentration column from DataFrame.
        :param response_col: Response column from DataFrame.
        :param input_units: Concentration units for tested drug. By default, units given will be in nM.
        :param verbose: Output drug concentration units.

        :return: A dictionary containing drug name, maximum response, minimum response, IC50 (relative) and hill slope.
        """
        # Set variables from funtion and convert name_col to np array
        global params, conc_unit
        name_col = name_col
        name = self.df[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.df[self.df[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Set initial guess for 4PL equation
            initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, inplace=True)

            # todo Calculate standard deviations from the covariance matrix
            # std_dev = np.sqrt(np.diag(covariance))

            # tag response col to determine direction of fourpl equation and fit to 4PL equation
            reverse, params, covariance = self.calc_logic(df=query, concentration=concentration,
                                                          response_col=response_col, initial_guess=initial_guess,
                                                          response=response)

            # If verbose, output info
            self.verbose_calculation(drug, input_units, verbose)

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, f'{input_units}') # For checking

            # Confirm ic50 unit output
            # x_intersection is not needed for relative ic50
            ic50, x_intersection, input_units = self.unit_convert(ic50, x_intersection=None, input_units=input_units)

            # Logic to append concentration units to output DataFrame
            if input_units is None:
                conc_unit = 'nM'
            elif input_units == 'nM':
                conc_unit = 'nM'
            elif input_units == 'uM' or input_units == 'µM':
                conc_unit = 'µM'

            # Generate DataFrame from parameters
            values.append({
                'compound_name': drug,
                'maximum': maximum,
                'minimum': minimum,
                f'ic50 ({conc_unit})': ic50,
                'hill_slope': hill_slope
            })
        return values

    def absolute_calculation(self, name_col, concentration_col, response_col, input_units, verbose=None):
        """
        Calculate relative IC50 values for a given drug. Output will be a dictionary that will be converted into a
        pandas dataframe using the calculate_absolute_ic50() function.

        :param name_col: Name column from DataFrame
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param input_units: Concentration units for tested drug. By default, units given will be in nM.
        :param verbose:  Output drug concentration units.

        :return: A dictionary containing drug name, maximum response, minimum response, relative IC50,
         absolute IC50, and hill slope.
        """

        # Set variables from funtion and convert name_col to np array
        global params, conc_unit
        name_col = name_col
        name = self.df[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.df[self.df[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Set initial guess for 4PL equation
            initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, ascending=True, inplace=True)

            reverse, params, covariance = self.calc_logic(df=query, concentration=concentration,
                                                          response_col=response_col, initial_guess=initial_guess,
                                                          response=response)

            # If verbose, output info
            # self.verbose_calculation(drug, input_units, verbose)

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, 'nM') # For checking

            # Obtain x_fit. Because calculator does not require xscale_ticks, it is set to None
            # If verbose, more info will be printed
            x_fit, input_units = CurveSettings().scale_units(drug_name=drug, xscale_unit=input_units,
                                                             xscale_ticks=None, verbose=verbose)

            # Calculate from parameters 4PL equation
            if reverse == 1:
                y_fit = self.reverse_fourpl(x_fit, maximum, minimum, ic50, hill_slope)
                y_intersection = 50
                interpretation = interp1d(y_fit, x_fit, kind='linear', fill_value="extrapolate")
                x_intersection = np.round(interpretation(y_intersection), 3)  # give results and round to 3 sig figs
                hill_slope = -1 * hill_slope  # ensure hill_slope is negative # may not be needed if fixed
            else:
                y_fit = self.fourpl(x_fit, *params)
                y_intersection = 50
                x_intersection = np.interp(y_intersection, y_fit, x_fit)

            # Confirm ic50 unit output
            ic50, x_intersection, input_units = self.unit_convert(ic50, x_intersection, input_units)

            # Logic to append concentration units to output DataFrame
            if input_units is None:
                conc_unit = 'nM'
            elif input_units == 'nM':
                conc_unit = 'nM'
            elif input_units == 'uM' or input_units == 'µM':
                conc_unit = 'µM'

            # Generate DataFrame from parameters
            values.append({
                'compound_name': drug,
                'maximum': maximum,
                'minimum': minimum,
                f'relative ic50 ({conc_unit})': ic50,
                f'absolute ic50 ({conc_unit})': x_intersection,
                'hill_slope': hill_slope
            })
        return values

    # When data is reversed, program is not obtaining correct column.
    def calc_logic(self, df, concentration=None, initial_guess=None, response=None, response_col=None):
        """
        Set logic to determine positive or negative sigmoid curve. This method is called by internally by the
        absolute_calculation() method.

        :param df: Input DataFrame. Must columns with drug name, tested concentration, and Response
        :param concentration: The concentration column from input DataFrame.
        :param initial_guess: The initial guesses for the 4PL equation.
        :param response: The response column from the input Dataframe.
        :param response_col: Name of the response column.

        :return variables for further calculation. this includes: reverse, params, covariance
        """
        global reverse, params, covariance
        if df[response_col].iloc[0] > df[response_col].iloc[-1]:  # Sigmoid curve 100% to 0%
            params, covariance, *_ = curve_fit(self.reverse_fourpl, concentration, response, p0=[initial_guess],
                                               maxfev=10000)
            reverse = 1  # Tag direction of sigmoid curve

        elif df[response_col].iloc[0] < df[response_col].iloc[-1]:  # sigmoid curve 0% to 100%
            params, covariance, *_ = curve_fit(self.fourpl, concentration, response, p0=[initial_guess],
                                               maxfev=10000)
            reverse = 0  # Tag direction of sigmoid curve
        return reverse, params, covariance

    def calculate_ic50(self, name_col, concentration_col, response_col, input_units=None, verbose=None):
        """
        Calculations previously performed in relative_calculation(). The dictionary results are converted into into a
        pandas DataFrame

        :param name_col: Name column from DataFrame
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param input_units:
        :param verbose: Output drug concentration units.

        :return: DataFrame generated from the list from the relative_calculation method
        """

        # Set variables from funtion and convert name_col to np array
        values = self.relative_calculation(name_col, concentration_col, response_col, input_units, verbose)

        df = pd.DataFrame(values)
        return df

    def calculate_absolute_ic50(self, name_col, concentration_col, response_col, input_units=None, verbose=None):
        """
        Calculations previously performed in absolute_calculation(). The dictionary results are converted into into a
        pandas DataFrame

        :param name_col: Name column from DataFrame
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param input_units: Concentration units for tested drug. By default, units given will be in nM.
        :param verbose:  Output drug concentration units.

        :return: DataFrame generated from the list from the absolute_calculation method
        """

        values = self.absolute_calculation(name_col=name_col, concentration_col=concentration_col,
                                           response_col=response_col, input_units=input_units, verbose=verbose)
        df = pd.DataFrame(values)

        return df

    def calculate_pic50(self, name_col, concentration_col, response_col, input_units=None, verbose=None):
        """
        Convert IC50 into pIC50 values. Calculation is performed using the absolute_calculation. As such, two columns
        will be appended - relative pIC50 and absolute pIC50. Conversion is performed by convert the IC50 values from nM
        to M levels and then taking the negative log value of said number.

        :param name_col: Name column from DataFrame
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param input_units: Concentration units for tested drug. By default, units given will be in nM.
        :param verbose:  Output drug concentration units.

        :return: DataFrame from calculate_absolute_ic50 along with the pIC50 values
        """
        values = self.absolute_calculation(name_col=name_col, concentration_col=concentration_col,
                                           response_col=response_col, input_units=input_units, verbose=verbose)
        df = pd.DataFrame(values)

        if input_units is None or input_units == 'nM':
            df['relative pIC50'] = -np.log10(df['relative ic50 (nM)'] * 0.000000001)
            df['absolute pIC50'] = -np.log10(df['absolute ic50 (nM)'] * 0.000000001)
        elif input_units == 'µM':
            df['relative pIC50'] = -np.log10(df['relative ic50 (µM)'] * 0.000001)
            df['absolute pIC50'] = -np.log10(df['absolute ic50 (µM)'] * 0.000001)

        return df


if __name__ == '__main__':
    import doctest

    doctest.testmod()
