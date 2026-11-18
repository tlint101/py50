from typing import Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

__all__ = ["Calculator"]


class Calculator:
    _MOLAR_SCALE = {"pM": 1e-12, "nM": 1e-9, "µM": 1e-6}

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
            Indicate the number of rows to display. If none, automatically show 5.
        :return: DataFrame
        """

        if rows is None:
            return self.data.head()
        elif isinstance(rows, int):
            return self.data.head(rows)

    def to_csv(self, path: Optional[str] = None, index: bool = False, **kwargs):
        """
        Save the tabel to csv file.

        :param path: str
            Designate the save path for the table.
        :param index: bool
            Write the row names.
        :param kwargs:
            Keyword arguments for pandas.DataFrame.to_csv.
        :return: csv file
        """

        self.data.to_csv(path, index=index, **kwargs)

    def show_column(self, key: Optional[str] = None):
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
            name_col: Optional[str] = None,
            concentration_col: Optional[str] = None,
            response_col: Optional[Union[str, list]] = None,
            input_units: Optional[str] = "nM",
            output_units: Optional[str] = None,
            verbose: Optional[bool] = None,
    ):
        """
        Calculations previously performed in relative_calculation(). The dictionary results are converted into a
        pandas DataFrame

        :param name_col: str
            Name column from DataFrame
        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: Union[str, list]
            Response column from DataFrame. Can be a single column (i.e. already a calculated average) or a list of
            columns to be averaged. The columns will be averaged internally within the function.
        :param input_units: str
            Concentration units of the input dataset ("pM", "nM", or "µM"). Default is "nM".
        :param output_units: str
            Target concentration units for reported IC50 values ("pM", "nM", or "µM"). Defaults to input_units.
        :param verbose: bool
            Output drug concentration units.

        :return: DataFrame generated from the list from the relative_calculation method
        """

        # set instance variables
        if name_col is None:
            name_col = self.name_col
        if concentration_col is None:
            concentration_col = self.concentration_col
        if response_col is None:
            response_col = self.response_col

        calc_data, calc_response_col = self._prepare_response_data(
            name_col, concentration_col, response_col
        )

        # Set variables from function and convert name_col to np array
        values = self._relative_calculation(
            name_col,
            concentration_col,
            calc_response_col,
            input_units,
            output_units,
            verbose,
            data=calc_data,
        )

        result_df = pd.DataFrame(values)

        self.calculation = result_df
        return self.calculation

    def calculate_absolute_ic50(
            self,
            name_col: Optional[str] = None,
            concentration_col: Optional[str] = None,
            response_col: Optional[Union[str, list]] = None,
            input_units: Optional[str] = "nM",
            output_units: Optional[str] = None,
            verbose: Optional[bool] = None,
    ):
        """
        Calculations previously performed in absolute_calculation(). The dictionary results are converted into a
        pandas DataFrame

        :param name_col: str
            Name column from DataFrame
        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: Union[str, list]
            Response column from DataFrame. Can be a single column (i.e. already a calculated average) or a list of
            columns to be averaged. The columns will be averaged internally within the function.
        :param input_units: str
            Concentration units of the input dataset ("pM", "nM", or "µM"). Default is "nM".
        :param output_units: str
            Target concentration units for reported IC50 values ("pM", "nM", or "µM"). Defaults to input_units.
        :param verbose: bool
            Output drug concentration units.

        :return: DataFrame generated from the list from the absolute_calculation method
        """
        # set instance variables
        if name_col is None:
            name_col = self.name_col
        if concentration_col is None:
            concentration_col = self.concentration_col
        if response_col is None:
            response_col = self.response_col

        calc_data, calc_response_col = self._prepare_response_data(
            name_col, concentration_col, response_col
        )

        values = self._absolute_calculation(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=calc_response_col,
            input_units=input_units,
            output_units=output_units,
            verbose=verbose,
            data=calc_data,
        )
        result_df = pd.DataFrame(values)

        self.calculation = result_df
        return self.calculation

    def calculate_pic50(
            self,
            name_col: Optional[str] = None,
            concentration_col: Optional[str] = None,
            response_col: Optional[Union[str, list]] = None,
            input_units: Optional[str] = "nM",
            output_units: Optional[str] = None,
            verbose: Optional[bool] = None,
    ):
        """
        Convert IC50 into pIC50 values. Calculation is performed using the absolute_calculation. As such, two columns
        will be appended - relative pIC50 and absolute pIC50. Conversion is performed by converting the IC50 values to
        molar (M) using the output units and taking the negative base-10 logarithm.

        :param name_col: str
            Name column from DataFrame
        :param concentration_col: str
            Concentration column from DataFrame
        :param response_col: Union[str, list]
            Response column from DataFrame. Can be a single column (i.e. already a calculated average) or a list of
            columns to be averaged. The columns will be averaged internally within the function.
        :param input_units: str
            Concentration units of the input dataset ("pM", "nM", or "µM"). Default is "nM".
        :param output_units: str
            Target concentration units for reported IC50 values ("pM", "nM", or "µM"). Defaults to input_units.
        :param verbose: bool
            Output drug concentration units.

        :return: DataFrame from calculate_absolute_ic50 along with the pIC50 values
        """

        # set instance variables
        if name_col is None:
            name_col = self.name_col
        if concentration_col is None:
            concentration_col = self.concentration_col
        if response_col is None:
            response_col = self.response_col

        # normalize units for query
        in_unit, out_unit = self._resolve_units(input_units, output_units)

        calc_data, calc_response_col = self._prepare_response_data(
            name_col, concentration_col, response_col
        )

        values = self._absolute_calculation(
            name_col=name_col,
            concentration_col=concentration_col,
            response_col=calc_response_col,
            input_units=in_unit,
            output_units=out_unit,
            verbose=verbose,
            data=calc_data,
        )
        result_df = pd.DataFrame(values)

        unit_scale = self._MOLAR_SCALE[out_unit]
        result_df["relative pIC50"] = -np.log10(
            result_df[f"relative ic50 ({out_unit})"] * unit_scale
        )
        result_df["absolute pIC50"] = -np.log10(
            result_df[f"absolute ic50 ({out_unit})"] * unit_scale
        )

        self.calculation = result_df
        return self.calculation

    """Support functions below"""

    """Define the 4-parameter logistic (4PL) equation"""

    @staticmethod
    def _fourpl(concentration, maximum, minimum, ic50, hill_slope):
        """
        Four-Parameter Logistic (4PL) Equation for calculating curve fit:


        :param concentration: concentration
        :param maximum: maximum response for drug query (top plateau)
        :param minimum: minimum response in drug query (bottom plateau)
        :param ic50: Concentration at inflection point (where curve shifts from up or down)
        :param hill_slope: Steepness of the curve (Hill Slope)

        :return: equation
        """
        return maximum + (minimum - maximum) / (
                1 + (concentration / ic50) ** hill_slope
        )

    @staticmethod
    def _reverse_fourpl(concentration, maximum, minimum, ic50, hill_slope):
        """
        Four-Parameter Logistic (4PL) Equation. This reverse function will graph the sigmoid curve from 100% to 0%


        :param concentration: concentration
        :param maximum: maximum response for drug query (top plateau)
        :param minimum: minimum response in drug query (bottom plateau)
        :param ic50: Concentration at inflection point (where curve shifts from up or down)
        :param hill_slope: Steepness of the curve (Hill Slope)

        :return: equation
        """
        return maximum + (minimum - maximum) / (
                1 + (concentration / ic50) ** -hill_slope
        )

    def _prepare_response_data(
            self,
            name_col: Optional[str],
            concentration_col: Optional[str],
            response_col: Optional[Union[str, list]],
    ) -> tuple[pd.DataFrame, Optional[str]]:
        """Reshape replicate response columns on a local frame without mutating self.data."""
        if isinstance(response_col, list):
            subset = self.data[[name_col, concentration_col, *response_col]]
            melted_col = "_melted_response"
            while melted_col in subset.columns:
                melted_col = f"_{melted_col}"
            reshape_data = pd.melt(
                subset,
                id_vars=[name_col, concentration_col],
                value_vars=response_col,
                value_name=melted_col,
            )
            return reshape_data.drop(columns=["variable"]), melted_col
        return self.data, response_col

    @staticmethod
    def _normalize_units(input_units: Optional[str] = None) -> str:
        """Normalize concentration unit aliases to 'nM', 'µM', or 'pM'."""
        if input_units is None or input_units in ("nM", "nm"):
            return "nM"
        if input_units in ("µM", "uM", "um", "μM", "μm", "µm"):
            return "µM"
        if input_units in ("pM", "pm"):
            return "pM"
        raise ValueError(
            f"Unsupported concentration unit {input_units!r}. "
            "Supported units are 'nM', 'µM' ('uM'), or 'pM'."
        )

    @classmethod
    def _resolve_units(
            cls,
            input_units: Optional[str] = "nM",
            output_units: Optional[str] = None,
    ) -> tuple[str, str]:
        """Normalize input and optional output units, defaulting output_units to input_units."""
        in_unit = cls._normalize_units(input_units)
        out_unit = (
            in_unit if output_units is None else cls._normalize_units(output_units)
        )
        return in_unit, out_unit

    def _verbose_calculation(
            self,
            drug: Optional[str] = None,
            input_units: Optional[str] = "nM",
            verbose: Optional[bool] = True,
            output_units: Optional[str] = None,
    ):
        """
        Logic function to calculate unit concentration for Relative and Absolute IC50 calculation. Information will
        detail drug name and concentration unit. Units available are nanomolar (nM), micromolar (µM or uM), or
        picomolar (pM).

        :param drug: str
            Input drug name.
        :param input_units: str
            Input drug concentration units ("nM", "µM" or "uM", or "pM"). Default is "nM".
        :param verbose: bool
            Print out information regarding the concentration unit.
        :param output_units: str
            Target concentration units ("nM", "µM" or "uM", or "pM"). Defaults to input_units.

        :return: None
        """

        # Verbose conditions
        if verbose is True:
            in_unit, out_unit = self._resolve_units(input_units, output_units)
            if in_unit == out_unit:
                print(f"{drug} concentration is in {out_unit}!")
            else:
                print(f"{drug} concentration converted from {in_unit} to {out_unit}!")

    # This will loop through each drug item.
    def _relative_calculation(
            self,
            name_col: Optional[str] = None,
            concentration_col: Optional[str] = None,
            response_col: Optional[str] = None,
            input_units: Optional[str] = "nM",
            output_units: Optional[str] = None,
            verbose: Optional[bool] = None,
            data: Optional[pd.DataFrame] = None,
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
            Concentration units for tested drug. Default is "nM".
        :param output_units: str
            Target concentration units for reported IC50 values. Defaults to input_units.
        :param verbose: bool
            Output drug concentration units.

        :return: A dictionary containing drug name, maximum response, minimum response, IC50 (relative) and hill slope.
        """
        # normalize units for query
        in_unit, out_unit = self._resolve_units(input_units, output_units)
        calc_data = self.data if data is None else data
        name_col = name_col
        name = calc_data[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = calc_data[calc_data[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, inplace=True)

            # tag response col to determine direction of fourpl equation and fit to 4PL equation
            reverse, params, covariance = self._calc_logic(
                data=query,
                concentration=concentration,
                response_col=response_col,
                response=response,
            )

            # If verbose, output info
            self._verbose_calculation(
                drug, input_units=in_unit, verbose=verbose, output_units=out_unit
            )

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            if reverse == 1:
                hill_slope = -1 * hill_slope
            # print(drug, ' IC50: ', ic50, f'{out_unit}') # For checking

            # Confirm ic50 unit output
            # x_intersection is not needed for relative ic50
            ic50, x_intersection, scaled_unit = self._unit_convert(
                ic50, x_intersection=None, input_units=in_unit, output_units=out_unit
            )

            # Generate DataFrame from parameters
            values.append(
                {
                    "compound_name": drug,
                    "maximum": maximum,
                    "minimum": minimum,
                    f"ic50 ({scaled_unit})": ic50,
                    "hill_slope": hill_slope,
                }
            )
        return values

    def _absolute_calculation(
            self,
            name_col: Optional[str] = None,
            concentration_col: Optional[str] = None,
            response_col: Optional[str] = None,
            input_units: Optional[str] = "nM",
            output_units: Optional[str] = None,
            verbose: Optional[bool] = None,
            data: Optional[pd.DataFrame] = None,
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
            Concentration units for tested drug. Default is "nM".
        :param output_units: str
            Target concentration units for reported IC50 values. Defaults to input_units.
        :param verbose: bool
            Output drug concentration units.

        :return: A dictionary containing drug name, maximum response, minimum response, relative IC50,
         absolute IC50, and hill slope.
        """

        # normalize units for query
        in_unit, out_unit = self._resolve_units(input_units, output_units)
        calc_data = self.data if data is None else data
        name_col = name_col
        name = calc_data[name_col].values

        drug_name = np.unique(name)

        values = []

        # Loop through each drug name and perform calculation
        for drug in drug_name:
            drug_query = calc_data[calc_data[name_col] == drug]
            concentration = drug_query[concentration_col]
            response = drug_query[response_col]

            # set a new coy of the DataFrame to avoid warnings
            query = drug_query.copy()
            query.sort_values(by=concentration_col, ascending=True, inplace=True)

            reverse, params, covariance = self._calc_logic(
                data=query,
                concentration=concentration,
                response_col=response_col,
                response=response,
            )

            # If verbose, output info
            self._verbose_calculation(
                drug, input_units=in_unit, verbose=verbose, output_units=out_unit
            )

            # Extract parameter values
            maximum, minimum, ic50, hill_slope = params
            # print(drug, ' IC50: ', ic50, f'{in_unit}') # For checking

            hill_slope, ic50, scaled_unit, x_intersection, _ = (
                self._reverse_absolute_calculation(
                    hill_slope,
                    ic50,
                    in_unit,
                    maximum,
                    minimum,
                    params,
                    reverse,
                    x_fit=None,
                    output_units=out_unit,
                )
            )

            # Generate DataFrame from parameters
            values.append(
                {
                    "compound_name": drug,
                    "maximum": maximum,
                    "minimum": minimum,
                    f"relative ic50 ({scaled_unit})": ic50,
                    f"absolute ic50 ({scaled_unit})": x_intersection,
                    "hill_slope": hill_slope,
                }
            )
        return values

    def _reverse_absolute_calculation(
            self,
            hill_slope,
            ic50,
            input_units,
            maximum,
            minimum,
            params,
            reverse,
            x_fit: Optional[np.ndarray] = None,
            output_units: Optional[str] = None,
    ):
        """
        Support function to compute optional y_fit and solve the exact 4PL concentration at 50% response.
        """
        if reverse == 1:
            y_fit = (
                self._reverse_fourpl(x_fit, maximum, minimum, ic50, hill_slope)
                if x_fit is not None
                else None
            )
            hill_slope = (
                    -1 * hill_slope
            )  # ensure hill_slope is negative # may not be needed if fixed
        else:
            y_fit = self._fourpl(x_fit, *params) if x_fit is not None else None

        low_plateau = min(minimum, maximum)
        high_plateau = max(minimum, maximum)
        if low_plateau < 50 < high_plateau and hill_slope != 0 and ic50 > 0:
            x_intersection = float(
                ic50 * ((50.0 - minimum) / (maximum - 50.0)) ** (1.0 / hill_slope)
            )
        else:
            x_intersection = np.nan

        # Confirm ic50 unit output
        ic50, x_intersection, scaled_unit = self._unit_convert(
            ic50, x_intersection, input_units, output_units
        )
        return hill_slope, ic50, scaled_unit, x_intersection, y_fit

    # When data is reversed, program is not obtaining correct column.
    def _calc_logic(
            self,
            data: pd.DataFrame,
            concentration: Optional[pd.Series] = None,
            initial_guess: Optional[list] = None,
            response: Optional[pd.Series] = None,
            response_col: Optional[str] = None,
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
        if initial_guess is None:
            p0 = [
                max(response),
                min(response),
                max(float(np.median(concentration)), 1e-12),
                1.0,
            ]
        else:
            p0 = list(initial_guess)
            if p0[2] <= 1e-30:
                p0[2] = max(float(np.median(concentration)), 1e-12)
        bounds = (
            [-np.inf, -np.inf, 1e-30, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        )

        ordered = data[response_col]
        low, high = ordered.iloc[0], ordered.iloc[-1]
        if low == high and len(ordered) > 1:
            mid = len(ordered) // 2
            low = ordered.iloc[:mid].mean()
            high = ordered.iloc[mid:].mean()

        if low > high:  # Sigmoid curve 100% to 0%
            params, covariance, *_ = curve_fit(
                self._reverse_fourpl,
                concentration,
                response,
                p0=p0,
                bounds=bounds,
                maxfev=100000,
            )
            reverse = 1  # Tag direction of sigmoid curve
        else:  # Sigmoid curve 0% to 100% (including flat ties)
            params, covariance, *_ = curve_fit(
                self._fourpl,
                concentration,
                response,
                p0=p0,
                bounds=bounds,
                maxfev=100000,
            )
            reverse = 0  # Tag direction of sigmoid curve
        return reverse, params, covariance

    def _unit_convert(
            self,
            ic50: Optional[float] = None,
            x_intersection: Optional[float] = None,
            input_units: Optional[str] = "nM",
            output_units: Optional[str] = None,
    ):
        """
        Converts IC50 values between supported concentration units ("pM", "nM", or "µM").

        :param ic50: float
            IC50 value for conversion. Obtained from the curve parameter values.
        :param x_intersection: float
            This value will correspond to the absolute ic50 value. This is calculated from the curve_fit.
        :param input_units: str
            Concentration units of the input IC50 values ("pM", "nM", or "µM"). Default is "nM".
        :param output_units: str
            Target concentration units for converted IC50 values. Defaults to input_units.
        :return: Tuple of (ic50, x_intersection, output_unit)
        """
        in_unit, out_unit = self._resolve_units(input_units, output_units)
        scale = self._MOLAR_SCALE[in_unit] / self._MOLAR_SCALE[out_unit]
        if ic50 is not None and scale != 1.0:
            ic50 = ic50 * scale
        if x_intersection is not None and scale != 1.0:
            x_intersection = x_intersection * scale
        return ic50, x_intersection, out_unit


if __name__ == "__main__":
    import doctest

    doctest.testmod()
