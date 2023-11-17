import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class Calculate:
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

    # Calculate X and Y_fit to  coordinates for line curve
    def fit_curve(self, name_col, concentration_col, response_col, initial_guess=None):  # Not sure if I need this
        """
        Calculate curve fit and return as DataFrame
        :param name_col: Name column from DataFrame.
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param initial_guess: Initial guesses for calculation. Must be given as a list in the following format:
        [Max, Min, ic50, and hill_slope]. By default, initial guesses are [max(response), min(response), 1.0, 1.0]
        :return: DataFrame containing calculated X and Y coordinates for curve.
        """
        # Set variables from funtion and convert name_col to np array
        name_col = name_col
        name = self.df[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.df[self.df[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            if initial_guess is None:
                initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope
            else:
                pass

            # Perform the curve fitting
            params, covariance, *_ = curve_fit(Calculate.fourpl,  # Static method from Calculate class
                                               concentration,
                                               response,
                                               p0=[initial_guess],
                                               maxfev=10000)

            # Extract the fitted parameters
            maximum, minimum, ic50, hill_slope = params

            # Create an array of x coordinates (concentration values)
            x_coordinates = np.linspace(min(concentration), max(concentration), 100)

            # Calculate the corresponding y coordinates (response values) using the fitted parameters
            y_fit_coordinates = self.fourpl(x_coordinates, maximum, minimum, ic50, hill_slope)

            # Generate DataFrame from parameters
            values.append({
                'compound_name': drug,
                'X': x_coordinates,
                'Y': y_fit_coordinates,
            })

        # place coordinates into DataFrame
        df = pd.DataFrame(values)
        df = df.explode(list('XY')).reset_index(drop=True)

        return df

    def data_points(self, name_col, concentration_col, response_col):
        """This will output a dataframe containing datapoints. This will correspond to the concentration datapoints"""
        name_col = name_col
        name = self.df[name_col].values
        concentration_col = concentration_col
        response_col = response_col

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.df[self.df[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # Generate DataFrame from parameters
            values.append({
                'compound_name': drug,
                'X': concentration,
                'Y': response,
            })

        # place datapoints into DataFrame
        df = pd.DataFrame(values)
        df = df.explode(list('XY')).reset_index(drop=True)

        return df

    # Function to calculate curve fits.
    # This method will be used to reduce the functions in the calculating methods below.
    # This will loop through each group.
    # todo rename function. Separate from final function calculate_ic50
    def relative_calculation(self, name_col, concentration_col, response_col, input_units='nM'):
        name_col = name_col
        name = self.df[name_col].values

        values = []

        drug_name = np.unique(name)
        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.df[self.df[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope

            # Fit the data to the 4PL equation using non-linear regression
            params, covariance, *_ = curve_fit(self.fourpl, concentration, response, p0=[initial_guess],
                                               maxfev=10000)

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, f'{input_units}') # For checking

            # Generate DataFrame from parameters
            values.append({
                'compound_name': drug,
                'maximum': maximum,
                'minimum': minimum,
                'ic50 (nM)': ic50,
                'hill_slope': hill_slope
            })
        return values
    # todo rename function. Separate from final function calculate_absolute_ic50
    def absolute_calculation(self, name_col, concentration_col, response_col, input_units='nM'):
        """
        This function will output a DataFrame containing Compound_name, maximum, minimum, ic50 (nM) (relative), and
        hill_slope. this will require a DataFrame with the following input:
        :param name_col: Name column from DataFrame
        :param concentration_col: Concentration column from DataFrame
        :param response_col: Response column from DataFrame
        :param input_units: Units of results. By default, the units given will be in nM. Results can be reformated by
        using the 'µM' argument.
        :return: params
        """

        # Set variables from funtion and convert name_col to np array
        global params
        name_col = name_col
        name = self.df[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = self.df[self.df[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            initial_guess = [max(response), min(response), 1.0, 1.0]  # Max, Min, ic50, and hill_slope

            # Fit the data to the 4PL equation using non-linear regression
            params, covariance, *_ = curve_fit(self.fourpl, concentration, response, p0=[initial_guess], maxfev=10000)

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            print(drug, ' IC50: ', ic50, 'nM')

            # Create constraints for the concentration values. This would be for extracting absolute IC50 value
            if input_units == 'nM':
                x_fit = np.logspace(0, 5, 100)
            elif input_units == 'µM':
                x_fit = np.logspace(-3, 2, 100)
            else:
                print('Assuming that =input concentrations is in nM!')
                x_fit = np.logspace(0, 5, 100)

            # Calculate from parameters 4PL equation
            y_fit = self.fourpl(x_fit, *params)
            y_intersection = 50
            x_intersection = np.interp(y_intersection, y_fit, x_fit)
            # todo conversion may be an issue. Need to double check.
            # print(f'{drug} ABSOLUTE IC50:', x_intersection * 1000, 'nM')

            # Generate DataFrame from parameters
            values.append({
                'compound_name': drug,
                'maximum': maximum,
                'minimum': minimum,
                'relative ic50 (nM)': ic50,
                'absolute ic50 (nM)': x_intersection,
                'hill_slope': hill_slope
            })
        return values

    def calculate_ic50(self, name_col, concentration_col, response_col):
        """
        This will outuput a DataFrame containing Compound_name, maximum, minimum, ic50 (nM) (relative), and hill_slope
        Requires DataFrame as input
        - data:
        - concentration_col:
        - response_col:
        return DataFrame
        """
        # Set variables from funtion and convert name_col to np array
        values = self.relative_calculation(name_col, concentration_col, response_col)

        df = pd.DataFrame(values)
        return df

    def calculate_absolute_ic50(self, name_col, concentration_col, response_col, input_units='nM'):

        values = self.absolute_calculation(name_col, concentration_col, response_col, input_units=input_units)
        df = pd.DataFrame(values)

        return df


if __name__ == '__main__':
    import doctest

    doctest.testmod()
