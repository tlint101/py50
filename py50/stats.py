"""
Script to calculate statistics.
"""

import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import pingouin as pg
from statannotations.Annotator import Annotator
from py50 import utils
import warnings

sns.set_style("ticks")


class Stats:
    """
    Class contains wrappers for pingouin module. The functions output data as a Pandas DataFrame. This is in a format
    needed for plotting with functions in class Plots(), however they can also be used individually to output single
    DataFrame for output as a csv or xlsx file using pandas.
    """

    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self.data = data

    def show(self, rows=None):
        """
        show DataFrame

        :param rows: Int
            Indicate the number of rows to display. If none, automatically show 5.
        :return: DataFrame
        """

        returned_df = self.data

        if rows is None:
            # print("rows is none") # for troubleshooting
            return returned_df.head()
        elif isinstance(rows, int):
            # print("rows are given!") # for troubleshooting
            return returned_df.head(rows)

    def get_normality(self, value_col=None, group_col=None, method="shapiro", **kwargs):
        """
        Test data normality of dataset.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the grouping variable.
        :param method: String
            Normality test. ‘shapiro’ (default). Additional tests can be found with
            [pingouin.normality()](https://pingouin-stats.org/build/html/generated/pingouin.normality.html)
        :param kwargs: optional
            Other options available with pingouin.normality()
        :return: Pandas.DataFrame
        """

        result_df = pg.normality(
            data=self.data, dv=value_col, group=group_col, method=method, **kwargs
        )
        return result_df

    def get_homoscedasticity(
        self, value_col=None, group_col=None, method="levene", **kwargs
    ):
        """
        Test for data variance.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the grouping variable.
        :param method: String
            Statistical test. ‘levene’ (default). Additional tests can be found with
            [pingouin.homoscedasticity()](https://pingouin-stats.org/build/html/generated/pingouin.homoscedasticity.html#pingouin.homoscedasticity)
        :param kwargs: optional
            Other options available with pingouin.homoscedasticity()
        :return: Pandas.DataFrame
        """

        result_df = pg.homoscedasticity(
            data=self.data, dv=value_col, group=group_col, method=method, **kwargs
        )
        return result_df

    """
    Parametric posts below
    """

    def get_anova(self, value_col=None, group_col=None, **kwargs):
        """
        One-way and N-way ANOVA.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String or list of strings
            Name of columnName of column containing the grouping variable.
        :param kwargs: optional
            Other options available with
            [pingouin.anova()](https://pingouin-stats.org/build/html/generated/pingouin.anova.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.anova(data=self.data, dv=value_col, between=group_col, **kwargs)

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_welch_anova(self, value_col=None, group_col=None):
        """
        One-way Welch ANOVA

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of column containing the grouping variable.
        :return: Pandas.DataFrame
        """

        result_df = pg.welch_anova(data=self.data, dv=value_col, between=group_col)

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_rm_anova(
        self,
        value_col=None,
        within_subject_col=None,
        subject_col=None,
        correction="auto",
        detailed=False,
        effsize="ng2",
    ):
        """
        One-way and two-way repeated measures ANOVA.

        :param value_col: String
            Name of column containing the dependent variable.
        :param within_subject_col: String
            Name of column containing the within factor.
        :param subject_col: String
            Name of column containing the subject identifier.
        :param correction: String or Boolean
            If True, also return the Greenhouse-Geisser corrected p-value.
        :param detailed: Boolean
            If True, return full ANOVA table.
        :param effsize: String
            Effect size.

        :return: Pandas.DataFrame
        """

        result_df = pg.rm_anova(
            data=self.data,
            dv=value_col,
            within=within_subject_col,
            subject=subject_col,
            correction=correction,
            detailed=detailed,
            effsize=effsize,
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_mixed_anova(
        self,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        **kwargs,
    ):
        """
        Mixed-design ANOVA.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of column containing the between factor.
        :param within_subject_col: String
            Name of column containing the within-subject factor (repeated measurements).
        :param subject_col:
            Name of column containing the between-subject identifier.
        :param kwargs: optional
            Other options available with
            [pingouin.mixed_anova()](https://pingouin-stats.org/build/html/generated/pingouin.mixed_anova.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.mixed_anova(
            data=self.data,
            dv=value_col,
            between=group_col,
            within=within_subject_col,
            subject=subject_col,
            **kwargs,
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_tukey(self, value_col=None, group_col=None, effsize="hedges"):
        """
        Pairwise Tukey post-hoc test.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the between factor.
        :param effsize: String or None
            Effect size. Additional methods can be found with
            [pingouin.pairwise_tukey()](https://pingouin-stats.org/build/html/generated/pingouin.pairwise_tukey.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.pairwise_tukey(
            data=self.data, dv=value_col, between=group_col, effsize=effsize
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-tukey"]]
        result_df["significance"] = pvalue

        return result_df

    def get_gameshowell(self, value_col=None, group_col=None, effsize="hedges"):
        """
        Pairwise Games-Howell post-hoc test

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the between factor.
        :param effsize: String or None
            Effect size. Additional methods can be found with
            [pingouin.pairwise_gameshowell()](https://pingouin-stats.org/build/html/generated/pingouin.pairwise_gameshowell.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.pairwise_gameshowell(
            data=self.data, dv=value_col, between=group_col, effsize=effsize
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["pval"]]
        result_df["significance"] = pvalue

        return result_df

    """
    non-parametric tests below
    """

    def get_wilcoxon(
        self,
        value_col=None,
        group_col=None,
        subgroup_col=None,
        alternative="two-sided",
        **kwargs,
    ):
        """
        Calculate wilcoxon tests. This is non-parametric version of paired T-test. Data number must be uniform to work.

        :param value_col: String
            Columns containing values for testing.
        :param group_col: String
            Column containing group name.
        :param subgroup_col: String
            Column containing subgroup name.
        :param alternative: String
            Defines the alternative hypothesis, or tail of the test. Must be one of “two-sided”. Must be one of
            “two-sided” (default), “greater” or “less”.
        :param kwargs: Optional
            Other options available with
            [pingouin.wilcoxon()](https://pingouin-stats.org/build/html/generated/pingouin.wilcoxon.html)
        :return: Pandas.DataFrame
        """

        # ignore Wilcoxon warnings
        warnings.filterwarnings(
            "ignore",
            message="Exact p-value calculation does not work if there are zeros.*",
        )

        if subgroup_col:
            # Convert 'Name' and 'Status' columns to string
            self.data[group_col] = self.data[group_col].astype(str)
            self.data[subgroup_col] = self.data[subgroup_col].astype(str)
            self.data["subgroup"] = self.data[group_col] + "-" + self.data[subgroup_col]

            subgroup_list = self.data["subgroup"].unique().tolist()
            subgroup_df = self.data[self.data["subgroup"].isin(subgroup_list)].copy()

            # Get unique pairs between group and subgroup
            group = subgroup_df["subgroup"].unique()

            # From unique items in group list, generate pairs
            pairs = list(combinations(group, 2))

            results_list = []
            for pair in pairs:
                # Get items from pair list and split by hyphen
                group1, subgroup1 = pair[0].split("-", 1)
                group2, subgroup2 = pair[1].split("-", 1)

                # # For troubleshooting
                # print("first:", data[(data[group_col] == group1)][value_col].shape)
                # print("second:", data[(data[group_col] == group2)][value_col].shape)

                # Check length of groups
                group1_length = self.data[self.data[group_col] == group1][value_col]
                group2_length = self.data[self.data[group_col] == group2][value_col]

                # print(len(group1_length), len(group2_length)) # For troubleshooting

                if len(group1_length) != len(group2_length):
                    raise ValueError(
                        "The lengths of the groups in group_col are not equal!"
                    )

                # Perform Wilcoxon signed-rank test
                result = pg.wilcoxon(
                    self.data[
                        (self.data[group_col] == group1)
                        & (self.data[subgroup_col] == subgroup1)
                    ][value_col],
                    self.data[
                        (self.data[group_col] == group2)
                        & (self.data[subgroup_col] == subgroup2)
                    ][value_col],
                    alternative=alternative,
                    **kwargs,
                )

                # Convert significance by pvalue
                pvalue = [utils.star_value(value) for value in result["p-val"]]

                # Store the results in the list
                results_list.append(
                    {
                        "A": f"{group1}-{subgroup1}",
                        "B": f"{group2}-{subgroup2}",
                        "W-val": result["W-val"].iloc[0],
                        "p-val": result["p-val"].iloc[0],
                        "significance": pvalue[0],
                        "RBC": result["RBC"].iloc[0],
                        "CLES": result["CLES"].iloc[0],
                    }
                )

            # Convert the list of dictionaries to a DataFrame
            result_df = pd.DataFrame(results_list)

            # Split values into and separate by comma
            result_df["A"] = result_df["A"].apply(lambda x: tuple(x.split("-", 1)))
            result_df["B"] = result_df["B"].apply(lambda x: tuple(x.split("-", 1)))

            return result_df
        else:
            """
            No subgroups found. Tests single group and values.
            """
            # Get unique pairs from group
            group = self.data[group_col].unique()

            # From unique items in group list, generate pairs
            pairs = list(combinations(group, 2))

            results_list = []
            for pair in pairs:
                # Get items from pair list and split by hyphen
                group1 = pair[0]
                group2 = pair[1]

                # # For troubleshooting
                # print("first:", data[(data[group_col] == group1)][value_col].shape)
                # print("second:", data[(data[group_col] == group2)][value_col].shape)

                # Check length of groups
                group1_length = self.data[self.data[group_col] == group1][value_col]
                group2_length = self.data[self.data[group_col] == group2][value_col]

                # print(len(group1_length), len(group2_length)) # For troubleshooting

                if len(group1_length) != len(group2_length):
                    raise ValueError(
                        "The lengths of the groups in group_col are not equal!"
                    )

                # Perform wilcoxon
                result = pg.wilcoxon(
                    self.data[(self.data[group_col] == group1)][value_col],
                    self.data[(self.data[group_col] == group2)][value_col],
                    alternative=alternative,
                    **kwargs,
                )
                pvalue = [utils.star_value(value) for value in result["p-val"]]
                results_list.append(
                    {
                        "A": group1,
                        "B": group2,
                        "W-val": result["W-val"].iloc[0],
                        "p-val": result["p-val"].iloc[0],
                        "significance": pvalue[0],
                        "RBC": result["RBC"].iloc[0],
                        "CLES": result["CLES"].iloc[0],
                    }
                )

            # Convert the list of dictionaries to a DataFrame
            result_df = pd.DataFrame(results_list)

            # Add significance asterisk
            pvalue = [utils.star_value(value) for value in result_df["p-val"]]
            result_df["significance"] = pvalue

            return result_df

    def get_mannu(
        self,
        value_col=None,
        group_col=None,
        subgroup_col=None,
        alternative="two-sided",
        **kwargs,
    ):
        """
        Calculate Mann-Whitney U Test. This is a non-parametric version of the independent T-test.

        :param self: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Columns containing values for testing.
        :param group_col: String
            Column containing group name.
        :param subgroup_col: String
            Column containing subgroup name.
        :param alternative: String
            Defines the alternative hypothesis, or tail of the test. Must be one of “two-sided”. Must be one of
            “two-sided” (default), “greater” or “less”.
        :param kwargs: Optional
            Other options available with [pingouin.mwu()](https://pingouin-stats.org/build/html/generated/pingouin.mwu.html)
        :return: Pandas.DataFrame
        """

        if subgroup_col:
            # Convert 'Name' and 'Status' columns to string
            self.data[group_col] = self.data[group_col].astype(str)
            self.data[subgroup_col] = self.data[subgroup_col].astype(str)
            self.data["subgroup"] = self.data[group_col] + "-" + self.data[subgroup_col]

            subgroup_list = self.data["subgroup"].unique().tolist()
            subgroup_df = self.data[self.data["subgroup"].isin(subgroup_list)].copy()

            # Get unique pairs between group and subgroup
            group = subgroup_df["subgroup"].unique()

            # From unique items in group list, generate pairs
            pairs = list(combinations(group, 2))

            # Check to ensure right columns selected
            if self.data[group_col].dtype != "object":
                raise ValueError(f"Is group_col: '{group_col}' strings?")
            elif self.data[subgroup_col].dtype != "object":
                raise ValueError(f"Is subgroup_col: '{subgroup_col}' strings?")
            elif self.data[value_col].dtype == "object":
                raise ValueError(f"Is value_col: '{value_col}' should be numerical?")

            results_list = []
            for pair in pairs:
                # print('this is the pair:', pair)  # for troubleshooting
                # print('this is the pairs:', pairs)
                # Get items from pair list and split by hyphen
                group1, subgroup1 = pair[0].split("-", 1)
                group2, subgroup2 = pair[1].split("-", 1)

                # Perform mwu
                result = pg.mwu(
                    self.data[
                        (self.data[group_col] == group1)
                        & (self.data[subgroup_col] == subgroup1)
                    ][value_col],
                    self.data[
                        (self.data[group_col] == group2)
                        & (self.data[subgroup_col] == subgroup2)
                    ][value_col],
                    alternative=alternative,
                    **kwargs,
                )

                # Convert significance by pvalue
                pvalue = [utils.star_value(value) for value in result["p-val"]]

                # Store the results in the list
                results_list.append(
                    {
                        "A": f"{group1}-{subgroup1}",
                        "B": f"{group2}-{subgroup2}",
                        "U-val": result["U-val"].iloc[0],
                        "p-val": result["p-val"].iloc[0],
                        "significance": pvalue[0],
                        "RBC": result["RBC"].iloc[0],
                        "CLES": result["CLES"].iloc[0],
                    }
                )

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)

            # Split values into and separate by comma
            df["A"] = df["A"].apply(lambda x: tuple(x.split("-", 1)))
            df["B"] = df["B"].apply(lambda x: tuple(x.split("-", 1)))

            return df
        else:
            """
            No subgroups found. Tests single group and values.
            """
            # Get unique pairs from group
            group = self.data[group_col].unique()

            # From unique items in group list, generate pairs
            pairs = list(combinations(group, 2))

            results_list = []
            for pair in pairs:
                # Get items from pair list and split by hyphen
                group1 = pair[0]
                group2 = pair[1]
                # Perform mwu
                result = pg.mwu(
                    self.data[(self.data[group_col] == group1)][value_col],
                    self.data[(self.data[group_col] == group2)][value_col],
                    alternative=alternative,
                    **kwargs,
                )
                pvalue = [utils.star_value(value) for value in result["p-val"]]
                results_list.append(
                    {
                        "A": group1,
                        "B": group2,
                        "U-val": result["U-val"].iloc[0],
                        "p-val": result["p-val"].iloc[0],
                        "significance": pvalue[0],
                        "RBC": result["RBC"].iloc[0],
                        "CLES": result["CLES"].iloc[0],
                    }
                )

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(results_list)

            return df

    def get_kruskal(self, value_col=None, group_col=None, detailed=False):
        """
        Calculate Kruskal-Wallis H-test for independent samples.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of column containing the between factor.
        :param detailed: Boolean
            Output additional details from Kruskal-Wallis H-test.
        :return: Pandas.DataFrame
        """

        result_df = pg.kruskal(
            data=self.data, dv=value_col, between=group_col, detailed=detailed
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_cochran(self, value_col=None, group_col=None, subgroup_col=None):
        """
        Calculate Cochran Q Test. This is used when the dependent variable, or value_col, is binary. For details between
        groups, posthoc test will be needed.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of column containing the within factor.
        :param subgroup_col: String
            Name of column containing the subject identifier.
        :return: Pandas.DataFrame
        """

        if subgroup_col:
            result_df = pg.cochran(
                data=self.data, dv=value_col, within=subgroup_col, subject=group_col
            )
        else:
            result_df = pg.cochran(data=self.data, dv=value_col, within=group_col)

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_friedman(
        self, group_col=None, value_col=None, subgroup_col=None, method="chisq"
    ):
        """
        Calculate Friedman Test. Determines if distributions of two or more paired samples are equal. For details between
        groups, posthoc test (get_pairwise_tests(parametric=False)) will be needed.

        :param value_col: String
            Name of column containing the dependent variable
        :param group_col: String
            Name of column containing the between-subject factor.
        :param subgroup_col: String
            Name of column containing the subject/rater identifier
        :param method: String
            Statistical test to perform. Must be 'chisq' (chi-square test) or 'f' (F test). See Pingouin
            documentation for further details
        :return: Pandas.DataFrame
        """

        # Raise error if subgroup_col not given
        if subgroup_col is None:
            raise ValueError(
                "Friedman test must be in long format and requires a subgroup_col as subject"
            )

        result_df = pg.friedman(
            data=self.data,
            dv=value_col,
            within=group_col,
            subject=subgroup_col,
            method=method,
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    """
    pairwise t-tests below
    """

    def get_pairwise_tests(
        self,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        parametric=True,
        **kwargs,
    ):
        """
        Posthoc test for parametric or nonparametric statistics. By default, the parametric parameter is set as True.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String or list with 2 elements
            Name of column containing the between-subject factors.
        :param within_subject_col: String or list with 2 elements
            Name of column containing the within-subject identifier.
        :param subject_col: String
            Name of column containing the subject identifier. This is mandatory if subgroup_col is used.
        :param parametric: Boolean
            If True (default), use the parametric ttest() function. If False, use [pingouin.wilcoxon()](https://pingouin-stats.org/build/html/generated/pingouin.wilcoxon.html#pingouin.wilcoxon) or [pingouin.mwu()](https://pingouin-stats.org/build/html/generated/pingouin.mwu.html#pingouin.mwu)
            for paired or unpaired samples, respectively.
        :param kwargs: dict
            Additional keywords arguments that are passed to [pingouin.pairwise_tests()](https://pingouin-stats.org/build/html/generated/pingouin.pairwise_tests.html#pingouin.pairwise_tests).
        :return: pandas.DataFrame
        """

        result_df = pg.pairwise_tests(
            data=self.data,
            dv=value_col,
            between=group_col,
            within=within_subject_col,
            subject=subject_col,
            parametric=parametric,
            **kwargs,
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_pairwise_rm(
        self,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        parametric=True,
        **kwargs,
    ):
        """
        Posthoc test for repeated measures.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String or list with 2 elements
            Name of column containing the between-subject factors.
        :param within_subject_col: String or list with 2 elements
            Name of column containing the within-subject identifier.
        :param subject_col: String
            Name of column containing the subject identifier. This is mandatory if subgroup_col is used.
        :param parametric: Boolean
            If True (default), use the parametric ttest() function. If False, use [pingouin.wilcoxon()](https://pingouin-stats.org/build/html/generated/pingouin.wilcoxon.html#pingouin.wilcoxon) or [pingouin.mwu()](https://pingouin-stats.org/build/html/generated/pingouin.mwu.html#pingouin.mwu)
            for paired or unpaired samples, respectively.
        :param kwargs: dict
            Additional keywords arguments that are passed to [pingouin.pairwise_tests()](https://pingouin-stats.org/build/html/generated/pingouin.pairwise_tests.html#pingouin.pairwise_tests).
        :return: pandas.DataFrame
        """

        result_df = pg.pairwise_tests(
            data=self.data,
            dv=value_col,
            between=group_col,
            within=within_subject_col,
            subject=subject_col,
            parametric=parametric,
            **kwargs,
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    def get_pairwise_mixed(
        self,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        parametric=True,
        **kwargs,
    ):
        """
        Posthoc test for mixed ANOVA.

        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String or list with 2 elements
            Name of column containing the between-subject factors.
        :param within_subject_col: String or list with 2 elements
            Name of column containing the within-subject identifier.
        :param subject_col: String
            Name of column containing the subject identifier. This is mandatory if subgroup_col is used.
        :param parametric: Boolean
            If True (default), use the parametric ttest() function. If False, use [pingouin.wilcoxon()](https://pingouin-stats.org/build/html/generated/pingouin.wilcoxon.html#pingouin.wilcoxon) or [pingouin.mwu()](https://pingouin-stats.org/build/html/generated/pingouin.mwu.html#pingouin.mwu)
            for paired or unpaired samples, respectively.
        :param kwargs: dict
            Additional keywords arguments that are passed to [pingouin.pairwise_tests()](https://pingouin-stats.org/build/html/generated/pingouin.pairwise_tests.html#pingouin.pairwise_tests).
        :return: pandas.DataFrame
        """

        result_df = pg.pairwise_tests(
            data=self.data,
            dv=value_col,
            between=group_col,
            within=within_subject_col,
            subject=subject_col,
            parametric=parametric,
            **kwargs,
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    """
    Output P-Values as a matrix in Pandas DataFrame
    """

    @staticmethod
    def get_p_matrix(data, test=None, group_col1=None, group_col2=None, order=None):
        """
        Convert dataframe of statistic results into a matrix. Group columns must be indicated. Group 2 is optional and
        depends on test used (i.e. pairwise vs Mann-Whitney U). Final DataFrame output can be used with the
        Plots.p_matrix() function to generate a heatmap of p-values.

        :param data: pandas.DataFrame
            Input DataFrame. Must be of already computed test results.
        :param group_col1: String
            Name of column containing the group
        :param group_col2: String
            Name of column containing the second group. This variable is optional.
        :param test: String
            Name of the test used to calculate statistics.
        :param order: List or String == "alpha"
            Reorder the groups for the final table. If input is string "alpha", the order of the groups will be
            alphabetized.
        :return:
        """

        matrix_df = utils.multi_group(data, group_col1, group_col2, test, order)

        return matrix_df

    """
    Function to detail significance column meaning
    """

    @staticmethod
    def explain_significance():
        """
        Print out DataFrame containing explanations for star values. This is used for reference. See [GraphPad](https://www.graphpad.com/support/faq/what-is-the-meaning-of--or--or--in-reports-of-statistical-significance-from-prism-or-instat/)

        :return: pandas.DataFrame
        """

        df = pd.DataFrame(
            {
                "pvalue": [
                    "p > 0.05",
                    "p ≤ 0.05",
                    " p ≤ 0.01",
                    "p ≤ 0.001",
                    "p ≤ 0.0001",
                ],
                "p_value": ["No Significance (n.s.)", "*", "**", "***", "****"],
            }
        )

        return df


class Plots(Stats):

    def __init__(self, data):
        super().__init__(data)

    @staticmethod
    def list_test():
        """
        List all tests available for plotting

        :return:
        """
        print(
            "List of tests available for plotting: 'tukey', 'gameshowell', 'pairwise-parametric', 'pairwise-rm', 'pairwise-mixed', 'pairwise-nonparametric', 'wilcoxon', 'mannu', 'kruskal'"
            "'kruskal'"
        )

    def boxplot(
        self,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subject_col=None,
        within_subject_col=None,
        pairs=None,
        pvalue_label=None,
        hide_ns=False,
        palette=None,
        orient="v",
        loc="inside",
        whis=1.5,  # boxplot whiskers
        return_df=None,
        **kwargs,
    ):
        """
        Draw a boxplot from the input DataFrame.

        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup_col: String
            Name of the column containing the subgroup for the group column. This is associated with the hue parameters
            in Seaborn.
        :param subject_col: String
            Name of the column containing the subject column.
        :param within_subject_col: String
            Name of the column containing the within subject column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_label: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param hide_ns: bool
            Automatically hide groups with no significance from plot.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param whis: Int
            Set length of whiskers on plot.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with annotated pairs.

        :return: Fig
        """

        global stat_df
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.boxplot)
        valid_annot = utils.get_kwargs(Annotator)

        sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
        annot_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_annot
        }

        # Perform Stat calculations and get pairs and pvalue for annotation
        pairs, pvalue, stat_df = Plots._get_test(
            self,
            group_col,
            kwargs,
            pairs,
            subgroup_col,
            subject_col,
            within_subject_col,
            test,
            value_col,
        )

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # If set to True, only show plots with significance
        if hide_ns is True:
            # Filter n.s. from pvalue and pairs
            hidden_sigfig_data = [
                (item1, item2) for item1, item2 in zip(pvalue, pairs) if item1 != "n.s."
            ]

            try:
                # Unzip the hidden_sigfig_data and separate into pvalue and pairs variables
                pvalue, pairs = zip(*hidden_sigfig_data)
                # # to troubleshoot
                # print(pvalue)
                # print(pairs)
            except:
                warnings.warn("No Significant Values. hide_ns will be set to False!")
                if all(sigfig == "n.s." for sigfig in pvalue):
                    hide_ns = False

        # set orientation for plot and Annotator
        orient = orient.lower()
        if orient == "v":
            ax = sns.boxplot(
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                whis=whis,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup_col,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.boxplot(
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                whis=whis,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                hue=subgroup_col,
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # Optional input to make custom labels
        if pvalue_label:
            pvalue = pvalue_label

        # Location of annotations
        if loc not in ["inside", "outside"]:
            raise ValueError("Invalid loc! Only 'inside' or 'outside' are accepted!")

        if loc == "inside":
            annotator.configure(loc=loc, test=None)
        else:
            annotator.configure(loc=loc, test=None)

        # Make sure the pairs and pvalue lists match
        if len(pairs) != len(pvalue):
            raise Exception("pairs and pvalue_order length does not match!")
        else:
            annotator.set_custom_annotations(pvalue)
            annotator.annotate(**annotate_kwargs)

        # Adjust title and title fontsize from kwargs
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "title" and "fontsize" in kwargs:
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])

        # Return DataFrame AND figure
        if return_df:
            return stat_df, annotator

        return annotator

    def barplot(
        self,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subject_col=None,
        within_subject_col=None,
        pairs=None,
        pvalue_label=None,
        hide_ns=False,
        palette=None,
        orient="v",
        loc="inside",
        errorbar="sd",
        capsize=0.1,
        return_df=None,
        **kwargs,
    ):
        """
        Draw a barplot from the input DataFrame.

        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup_col: String
            Name of the column containing the subgroup for the group column. This is associated with the hue parameters
            in Seaborn.
        :param subject_col: String
            Name of the column containing the subject column.
        :param within_subject_col: String
            Name of the column containing the within subject column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_label: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param hide_ns: bool
            Automatically hide groups with no significance from plot.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param errorbar: String
            Set confidence interval on plot.
        :param capsize: Int
            Set cap size on plot.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with annotated pairs.

        :return:
        """
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.barplot)
        valid_annot = utils.get_kwargs(Annotator)

        sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
        annot_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_annot
        }

        # Perform Stat calculations and get pairs and pvalue for annotation
        pairs, pvalue, stat_df = Plots._get_test(
            self,
            group_col,
            kwargs,
            pairs,
            subgroup_col,
            subject_col,
            within_subject_col,
            test,
            value_col,
        )

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # If set to True, only show plots with significance
        if hide_ns is True:
            # Filter n.s. from pvalue and pairs
            hidden_sigfig_data = [
                (item1, item2) for item1, item2 in zip(pvalue, pairs) if item1 != "n.s."
            ]

            try:
                # Unzip the hidden_sigfig_data and separate into pvalue and pairs variables
                pvalue, pairs = zip(*hidden_sigfig_data)
                # # to troubleshoot
                # print(pvalue)
                # print(pairs)
            except:
                warnings.warn("No Significant Values. hide_ns will be set to False!")
                if all(sigfig == "n.s." for sigfig in pvalue):
                    hide_ns = False

        # set orientation for plot and Annotator
        orient = orient.lower()
        if orient == "v":
            ax = sns.barplot(
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                errorbar=errorbar,  # errorbar
                capsize=capsize,  # errorbar
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup_col,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.barplot(
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                errorbar=errorbar,  # errorbar
                capsize=capsize,  # errorbar
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                hue=subgroup_col,
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # optional input for custom annotations
        if pvalue_label:
            pvalue = pvalue_label

        # Location of annotations
        if loc not in ["inside", "outside"]:
            raise ValueError("Invalid loc! Only 'inside' or 'outside' are accepted!")

        if loc == "inside":
            annotator.configure(loc=loc, test=None)
        else:
            annotator.configure(loc=loc, test=None)

        # Make sure the pairs and pvalue lists match
        if len(pairs) != len(pvalue):
            raise Exception("pairs and pvalue_order length does not match!")
        else:
            annotator.set_custom_annotations(pvalue)
            annotator.annotate(**annotate_kwargs)

        # Adjust title and title fontsize from kwargs
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "title" and "fontsize" in kwargs:
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])

        # Return DataFrame AND figure
        if return_df:
            return stat_df, annotator

        return annotator

    def violinplot(
        self,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subject_col=None,
        within_subject_col=None,
        pairs=None,
        pvalue_label=None,
        hide_ns=False,
        palette=None,
        orient="v",
        loc="inside",
        return_df=None,
        **kwargs,
    ):
        """
        Draw a biolinplot from the input DataFrame.

        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup_col: String
            Name of the column containing the subgroup for the group column. This is associated with the hue parameters
            in Seaborn.
        :param subject_col: String
            Name of the column containing the subject column.
        :param within_subject_col: String
            Name of the column containing the within subject column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_label: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param hide_ns: bool
            Automatically hide groups with no significance from plot.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with annotated pairs.

        :return:
        """

        global stat_df
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.violinplot)
        valid_annot = utils.get_kwargs(Annotator)

        sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
        annot_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_annot
        }

        # Perform Stat calculations and get pairs and pvalue for annotation
        pairs, pvalue, stat_df = Plots._get_test(
            self,
            group_col,
            kwargs,
            pairs,
            subgroup_col,
            subject_col,
            within_subject_col,
            test,
            value_col,
        )

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # If set to True, only show plots with significance
        if hide_ns is True:
            # Filter n.s. from pvalue and pairs
            hidden_sigfig_data = [
                (item1, item2) for item1, item2 in zip(pvalue, pairs) if item1 != "n.s."
            ]

            try:
                # Unzip the hidden_sigfig_data and separate into pvalue and pairs variables
                pvalue, pairs = zip(*hidden_sigfig_data)
                # # to troubleshoot
                # print(pvalue)
                # print(pairs)
            except:
                warnings.warn("No Significant Values. hide_ns will be set to False!")
                if all(sigfig == "n.s." for sigfig in pvalue):
                    hide_ns = False

        # set orientation for plot and Annotator
        orient = orient.lower()
        if orient == "v":
            ax = sns.violinplot(
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup_col,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.violinplot(
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                hue=subgroup_col,
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # optional input for custom annotations
        if pvalue_label:
            pvalue = pvalue_label

        # Location of annotations
        if loc not in ["inside", "outside"]:
            raise ValueError("Invalid loc! Only 'inside' or 'outside' are accepted!")

        if loc == "inside":
            annotator.configure(loc=loc, test=None)
        else:
            annotator.configure(loc=loc, test=None)

        # Make sure the pairs and pvalue lists match
        if len(pairs) != len(pvalue):
            raise Exception("pairs and pvalue_order length does not match!")
        else:
            annotator.set_custom_annotations(pvalue)
            annotator.annotate(**annotate_kwargs)

        # Adjust title and title fontsize from kwargs
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "title" and "fontsize" in kwargs:
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])

        # Return DataFrame AND figure
        if return_df:
            return stat_df, annotator
        else:
            return annotator

    def swarmplot(
        self,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subject_col=None,
        within_subject_col=None,
        pairs=None,
        pvalue_label=None,
        hide_ns=False,
        palette=None,
        orient="v",
        loc="inside",
        return_df=None,
        **kwargs,
    ):
        """
        Draw a swarmplot from the input DataFrame.

        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup_col: String
            Name of the column containing the subgroup for the group column. This is associated with the hue parameters
            in Seaborn.
        :param subject_col: String
            Name of the column containing the subject column.
        :param within_subject_col: String
            Name of the column containing the within subject column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_label: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param hide_ns: bool
            Automatically hide groups with no significance from plot.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with annotated pairs.

        :return:
        """

        # remove palette/hue warning
        warnings.filterwarnings("ignore", category=FutureWarning)

        global stat_df
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.boxplot)
        valid_annot = utils.get_kwargs(Annotator)

        sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
        annot_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_annot
        }

        # Perform Stat calculations and get pairs and pvalue for annotation
        pairs, pvalue, stat_df = Plots._get_test(
            self,
            group_col,
            kwargs,
            pairs,
            subgroup_col,
            subject_col,
            within_subject_col,
            test,
            value_col,
        )

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # If set to True, only show plots with significance
        if hide_ns is True:
            # Filter n.s. from pvalue and pairs
            hidden_sigfig_data = [
                (item1, item2) for item1, item2 in zip(pvalue, pairs) if item1 != "n.s."
            ]

            try:
                # Unzip the hidden_sigfig_data and separate into pvalue and pairs variables
                pvalue, pairs = zip(*hidden_sigfig_data)
                # # to troubleshoot
                # print(pvalue)
                # print(pairs)
            except:
                warnings.warn("No Significant Values. hide_ns will be set to False!")
                if all(sigfig == "n.s." for sigfig in pvalue):
                    hide_ns = False

        # To color code plots:
        subgroup_col_plot = None
        if subgroup_col is None:
            subgroup_col_plot = group_col

        # set orientation for plot and Annotator
        orient = orient.lower()
        if orient == "v":
            ax = sns.swarmplot(
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col_plot,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup_col,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.swarmplot(
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col_plot,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                hue=subgroup_col,
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # optional input for custom annotations
        if pvalue_label:
            pvalue = pvalue_label

        # # For debugging pairs and pvalue list orders
        # print(pairs)
        # print(pvalue)

        # Location of annotations
        if loc not in ["inside", "outside"]:
            raise ValueError("Invalid loc! Only 'inside' or 'outside' are accepted!")

        if loc == "inside":
            annotator.configure(loc=loc, test=None)
        else:
            annotator.configure(loc=loc, test=None)

        # Make sure the pairs and pvalue lists match
        if len(pairs) != len(pvalue):
            raise Exception("pairs and pvalue_order length does not match!")
        else:
            annotator.set_custom_annotations(pvalue)
            annotator.annotate(**annotate_kwargs)

        # Turn off legend by default. Can be assessed outside using the plt.legend()
        plt.legend().set_visible(False)

        # Adjust title and title fontsize from kwargs
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "title" and "fontsize" in kwargs:
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])

        # Return DataFrame AND figure
        if return_df:
            return stat_df, annotator

        return annotator

    def stripplot(
        self,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subject_col=None,
        within_subject_col=None,
        pairs=None,
        pvalue_label=None,
        hide_ns=False,
        palette=None,
        orient="v",
        loc="inside",
        return_df=None,
        **kwargs,
    ):
        """
        Draw a stripplot from the input DataFrame.

        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup_col: String
            Name of the column containing the subgroup for the group column. This is associated with the hue parameters
            in Seaborn.
        :param subject_col: String
            Name of the column containing the subject column.
        :param within_subject_col: String
            Name of the column containing the within subject column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_label: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param hide_ns: bool
            Automatically hide groups with no significance from plot.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with annotated pairs.

        :return:
        """

        # remove palette/hue warning
        warnings.filterwarnings("ignore", category=FutureWarning)

        global stat_df
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.stripplot)
        valid_annot = utils.get_kwargs(Annotator)

        sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
        annot_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_annot
        }

        # Perform Stat calculations and get pairs and pvalue for annotation
        pairs, pvalue, stat_df = Plots._get_test(
            self,
            group_col,
            kwargs,
            pairs,
            subgroup_col,
            subject_col,
            within_subject_col,
            test,
            value_col,
        )

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # If set to True, only show plots with significance
        if hide_ns is True:
            # Filter n.s. from pvalue and pairs
            hidden_sigfig_data = [
                (item1, item2) for item1, item2 in zip(pvalue, pairs) if item1 != "n.s."
            ]

            try:
                # Unzip the hidden_sigfig_data and separate into pvalue and pairs variables
                pvalue, pairs = zip(*hidden_sigfig_data)
                # # to troubleshoot
                # print(pvalue)
                # print(pairs)
            except:
                warnings.warn("No Significant Values. hide_ns will be set to False!")
                if all(sigfig == "n.s." for sigfig in pvalue):
                    hide_ns = False

        # To color code plots:
        subgroup_col_plot = None
        if subgroup_col is None:
            subgroup_col_plot = group_col

        # set orientation for plot and Annotator
        orient = orient.lower()
        if orient == "v":
            ax = sns.stripplot(
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col_plot,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup_col,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.stripplot(
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col_plot,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                hue=subgroup_col,
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # optional input for custom annotations
        if pvalue_label:
            pvalue = pvalue_label

        # # For debugging pairs and pvalue list orders
        # print(pairs)
        # print(pvalue)

        # Location of annotations
        if loc not in ["inside", "outside"]:
            raise ValueError("Invalid loc! Only 'inside' or 'outside' are accepted!")

        if loc == "inside":
            annotator.configure(loc=loc, test=None)
        else:
            annotator.configure(loc=loc, test=None)

        # Make sure the pairs and pvalue lists match
        if len(pairs) != len(pvalue):
            raise Exception("pairs and pvalue_order length does not match!")
        else:
            annotator.set_custom_annotations(pvalue)
            annotator.annotate(**annotate_kwargs)

        # Turn off legend by default. Can be assessed outside using hte plt.legend()
        plt.legend().set_visible(False)

        # Adjust title and title fontsize from kwargs
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "title" and "fontsize" in kwargs:
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])

        # Return DataFrame AND figure
        if return_df:
            return stat_df, annotator

        return annotator

    def boxenplot(
        self,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subject_col=None,
        within_subject_col=None,
        pairs=None,
        pvalue_label=None,
        hide_ns=False,
        palette=None,
        orient="v",
        loc="inside",
        return_df=None,
        **kwargs,
    ):
        """
        Draw a boxenplot from the input DataFrame.

        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup_col: String
            Name of the column containing the subgroup for the group column. This is associated with the hue parameters
            in Seaborn.
        :param subject_col: String
            Name of the column containing the subject column.
        :param within_subject_col: String
            Name of the column containing the within subject column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_label: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param hide_ns: bool
            Automatically hide groups with no significance from plot.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with annotated pairs.

        :return:
        """

        global stat_df
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.boxenplot)
        valid_annot = utils.get_kwargs(Annotator)

        sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
        annot_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_annot
        }

        # Perform Stat calculations and get pairs and pvalue for annotation
        pairs, pvalue, stat_df = Plots._get_test(
            self,
            group_col,
            kwargs,
            pairs,
            subgroup_col,
            subject_col,
            within_subject_col,
            test,
            value_col,
        )

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # If set to True, only show plots with significance
        if hide_ns is True:
            # Filter n.s. from pvalue and pairs
            hidden_sigfig_data = [
                (item1, item2) for item1, item2 in zip(pvalue, pairs) if item1 != "n.s."
            ]

            try:
                # Unzip the hidden_sigfig_data and separate into pvalue and pairs variables
                pvalue, pairs = zip(*hidden_sigfig_data)
                # # to troubleshoot
                # print(pvalue)
                # print(pairs)
            except:
                warnings.warn("No Significant Values. hide_ns will be set to False!")
                if all(sigfig == "n.s." for sigfig in pvalue):
                    hide_ns = False

        # set orientation for plot and Annotator
        orient = orient.lower()
        if orient == "v":
            ax = sns.boxenplot(
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup_col,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.boxenplot(
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                hue=subgroup_col,
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # optional input for custom annotations
        if pvalue_label:
            pvalue = pvalue_label

        # # For debugging pairs and pvalue list orders
        # print(pairs)
        # print(pvalue)

        # Location of annotations
        if loc not in ["inside", "outside"]:
            raise ValueError("Invalid loc! Only 'inside' or 'outside' are accepted!")

        if loc == "inside":
            annotator.configure(loc=loc, test=None)
        else:
            annotator.configure(loc=loc, test=None)

        # Make sure the pairs and pvalue lists match
        if len(pairs) != len(pvalue):
            raise Exception("pairs and pvalue_order length does not match!")
        else:
            annotator.set_custom_annotations(pvalue)
            annotator.annotate(**annotate_kwargs)

        # Adjust title and title fontsize from kwargs
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "title" and "fontsize" in kwargs:
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])

        # Return DataFrame AND figure
        if return_df:
            return stat_df, annotator

        return annotator

    # todo add support for lineplot
    def _lineplot(
        self,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subject_col=None,
        within_subject_col=None,
        pairs=None,
        pvalue_label=None,
        palette=None,
        orient="v",
        loc="inside",
        ci="sd",
        capsize=0.1,
        return_df=None,
        **kwargs,
    ):
        """
        Draw a lineplot from the input DataFrame.

        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup_col: String
            Name of the column containing the subgroup for the group column. This is associated with the hue parameters
            in Seaborn.
        :param subject_col: String
            Name of the column containing the subject column.
        :param within_subject_col: String
            Name of the column containing the within subject column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_label: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param ci: String
            Set confidence interval on plot.
        :param capsize: Int
            Set cap size on plot.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with annotated pairs.

        :return:
        """

        global stat_df
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.boxplot)
        valid_annot = utils.get_kwargs(Annotator)

        sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
        annot_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_annot
        }

        # Perform Stat calculations and get pairs and pvalue for annotation
        pairs, pvalue, stat_df = Plots._get_test(
            self,
            group_col,
            kwargs,
            pairs,
            subgroup_col,
            subject_col,
            within_subject_col,
            test,
            value_col,
        )

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.lineplot(
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                ci=ci,  # errorbar
                capsize=capsize,  # errorbar
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup_col,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.lineplot(
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                hue=subgroup_col,
                ci=ci,  # errorbar
                capsize=capsize,  # errorbar
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=self.data,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                hue=subgroup_col,
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # optional input for custom annotations
        if pvalue_label:
            pvalue = pvalue_label

        # Location of annotations
        if loc not in ["inside", "outside"]:
            raise ValueError("Invalid loc! Only 'inside' or 'outside' are accepted!")

        if loc == "inside":
            annotator.configure(loc=loc, test=None)
        else:
            annotator.configure(loc=loc, test=None)

        # Make sure the pairs and pvalue lists match
        if len(pairs) != len(pvalue):
            raise Exception("pairs and pvalue_order length does not match!")
        else:
            annotator.set_custom_annotations(pvalue)
            annotator.annotate(**annotate_kwargs)

        # Adjust title and title fontsize from kwargs
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "title" and "fontsize" in kwargs:
            plt.title(kwargs["title"], fontsize=kwargs["fontsize"])

        # Return DataFrame AND figure
        if return_df:
            return stat_df, annotator

        return annotator

    def p_matrix(
        self,
        # data,
        cmap=None,
        title=None,
        title_fontsize=14,
        linewidths=0.01,
        linecolor="gray",
        **kwargs,
    ):
        """
        Wrapper function for scikit_posthoc heatmap.

        :param data: Pandas.Dataframe
            Input table must be a matrix calculated using the stats.get_p_matrix().
        :param cmap: List
            A list of colors. Can be color names or hex codes.
        :param title: String
            Input title for figure.
        :param title_fontsize: Int
            Set size of figure legend.
        :param linewidths: Int
            Set line width of figure.
        :param linecolor: String
            Set line color. Can be color name or hex code.
        :param kwargs: Optional
            Keyword arguemnts associated with [scikit-posthocs](https://scikit-posthocs.readthedocs.io/en/latest/)

        :return: Pyplot figure
        """

        if title:
            plt.title(title, fontsize=title_fontsize)

        if cmap is None:
            cmap = ["1", "#fbd7d4", "#005a32", "#238b45", "#a1d99b"]
            fig = sp.sign_plot(
                self.data,
                cmap=cmap,
                linewidths=linewidths,
                linecolor=linecolor,
                **kwargs,
            )
        else:
            fig = sp.sign_plot(
                self.data,
                cmap=cmap,
                linewidths=linewidths,
                linecolor=linecolor,
                **kwargs,
            )

        # Display plot
        return fig

    """
    Functions to plot data distribution
    """

    def distribution(self, val_col=None, type="histplot", **kwargs):
        """

        :param self: Pandas.Dataframe
            Input data.
        :param val_col: String
            The name of the column containing the dependent variable.
        :param type: String
            The type of figure drawn. For distribution, only "histplot" or "qqplot" supported
        :param kwargs: Optional
            keyword arguments for seaborn or pg.qqplot.

        :return: figure
        """

        # Incorporate params from sns.histplot and pg.qq
        valid_hist = utils.get_kwargs(sns.histplot)
        valid_qq = utils.get_kwargs(pg.qqplot)
        hist_kwargs = {key: value for key, value in kwargs.items() if key in valid_hist}
        qq_kwargs = {key: value for key, value in kwargs.items() if key in valid_qq}

        if type == "histplot":
            fig = sns.histplot(data=self.data, x=val_col, **hist_kwargs)
        elif type == "qqplot":
            fig = pg.qqplot(self.data[val_col], dist="norm", **qq_kwargs)
        else:
            raise ValueError(
                "For test parameter, only 'histplot' or 'qqplot' available"
            )

        return fig

    def _get_test(
        self,
        group_col,
        kwargs,
        pairs,
        subgroup_col,
        subject_col,
        within_subject_col,
        test,
        value_col,
    ):
        """
        Function to obtain the pvalues and pairs for annotating the plot.
        :param self: self.df
        :param group_col: group_col
        :param kwargs: kwargs
        :param pairs: pairs
        :param subgroup_col: subgroup_col
        :param test: test
        :param value_col: value_col
        :return:
        """
        global stat_df, pvalue
        # Check input test and run calculation
        if test == "tukey":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.pairwise_tukey)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_tukey(
                value_col=value_col, group_col=group_col, **pg_kwargs
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-tukey"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "gameshowell":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.pairwise_gameshowell)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_gameshowell(
                value_col=value_col, group_col=group_col, **pg_kwargs
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["pval"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "pairwise-rm":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.pairwise_tests)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}
            # print(pg_kwargs)

            stat_df = Stats(self.data).get_pairwise_rm(
                value_col=value_col,
                group_col=group_col,
                within_subject_col=within_subject_col,
                subject_col=subject_col,
                parametric=True,
                **pg_kwargs,
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-unc"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "pairwise-mixed":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.pairwise_tests)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_pairwise_mixed(
                value_col=value_col,
                group_col=group_col,
                within_subject_col=within_subject_col,
                subgroup_col=subject_col,
                parametric=True,
                **pg_kwargs,
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-unc"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "pairwise-nonparametric":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.pairwise_tests)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_pairwise_tests(
                value_col=value_col,
                group_col=group_col,
                within_subject_col=within_subject_col,
                subject_col=subject_col,
                parametric=False,
                **pg_kwargs,
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-unc"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "pairwise-parametric":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.pairwise_tests)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_pairwise_tests(
                value_col=value_col,
                group_col=group_col,
                within_subject_col=within_subject_col,
                subject_col=subgroup_col,
                parametric=True,
                **pg_kwargs,
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-unc"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "wilcoxon":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.wilcoxon)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_wilcoxon(
                value_col=value_col,
                group_col=group_col,
                subgroup_col=subgroup_col,
                **pg_kwargs,
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-val"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "mannu":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.mwu)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_mannu(
                value_col=value_col,
                group_col=group_col,
                subgroup_col=subgroup_col,
                alternative="two-sided",
                **pg_kwargs,
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-val"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]

        elif test == "kruskal":
            # Get kwargs for pingouin
            valid_pg = utils.get_kwargs(pg.kruskal)
            pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

            stat_df = Stats(self.data).get_kruskal(
                value_col=value_col, group_col=group_col, **pg_kwargs
            )

            """Get pvalue and pairs from table"""
            # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
            stat_df = _get_pair_subgroup(stat_df, hue=pairs)

            pvalue = [utils.star_value(value) for value in stat_df["p-unc"].tolist()]
            pairs = [(a, b) for a, b in zip(stat_df["A"], stat_df["B"])]
        else:
            print(f"Plotting not supported for {test}!")
        return pairs, pvalue, stat_df


def _get_pair_subgroup(df, hue=None):
    """Generate pairs by group_col and hue. Hue will designate which input rows to keep for plotting."""

    if hue is None:
        hue = _get_pairs(df)

    # Convert filter_values to a set of tuples. Both directions are generated for checking df pairs.
    forward_set = {tuple(x) for x in hue}
    reverse_set = {(y, x) for (x, y) in forward_set}

    # Combine columns A and B into a single column of tuples
    df["AB"] = list(zip(df["A"], df["B"]))

    # Filtering DataFrame based on filter values
    filtered_df = (
        df[df["AB"].isin(forward_set) | df["AB"].isin(reverse_set)]
        .copy()
        .reset_index(drop=True)
    )

    # Make pairs between groups and subgroups by df
    filtered_df = _sort_df(filtered_df, hue)

    # Drop the combined column AB if not needed in the final output
    filtered_df.drop("AB", axis=1, inplace=True)

    return filtered_df


def _get_pairs(df):
    # Support function to make pairs form dataframe into a list of tuples
    pairs = [(a, b) for a, b in zip(df["A"], df["B"])]

    return pairs


# Custom sorting function
def _pair_sort(list_order, row):
    # Support function to make pairs between groups and subgroups by df
    try:
        # Check both possible orders of the tuple
        index = list_order.index((row["A"], row["B"]))
    except ValueError:
        try:
            index = list_order.index((row["B"], row["A"]))
        except ValueError:
            # If the row tuple is not found in the desired_order list, assign a high index
            index = len(list_order)

    return index


# Sort the DataFrame based on the custom sorting function
def _sort_df(df, list_order):
    # Support function to make pairs between groups and subgroups by df
    sorted_indices = df.apply(lambda row: _pair_sort(list_order, row), axis=1)
    return df.iloc[sorted_indices.argsort()]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
