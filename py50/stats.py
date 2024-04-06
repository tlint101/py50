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

    def __init__(self):
        pass

    @staticmethod
    def get_normality(data, value_col=None, group_col=None, method="shapiro", **kwargs):
        """
        Test data normality of dataset.

        :param data:pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the grouping variable.
        :param method: String
            Normality test. ‘shapiro’ (default). Additional tests can be found with [pingouin.normality()](https://pingouin-stats.org/build/html/generated/pingouin.normality.html)
        :param kwargs: optional
            Other options available with pingouin.normality()
        :return: Pandas.DataFrame
        """

        result_df = pg.normality(
            data=data, dv=value_col, group=group_col, method=method, **kwargs
        )
        return result_df

    @staticmethod
    def get_homoscedasticity(
        data, value_col=None, group_col=None, method="levene", **kwargs
    ):
        """
        Test for data variance.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the grouping variable.
        :param method: String
            Statistical test. ‘levene’ (default). Additional tests can be found with [pingouin.homoscedasticity()](https://pingouin-stats.org/build/html/generated/pingouin.homoscedasticity.html#pingouin.homoscedasticity)
        :param kwargs: optional
            Other options available with pingouin.homoscedasticity()
        :return: Pandas.DataFrame
        """

        result_df = pg.homoscedasticity(
            data=data, dv=value_col, group=group_col, method=method, **kwargs
        )
        return result_df

    """
    Parametric posts below
    """

    # @staticmethod
    # def get_t_test(df, paired=True, stars=False, decimals=4, **kwargs):
    #     """
    #     Calculate pairwise t-test.
    #
    #     :param df:
    #     :param paired:
    #     :param stars:
    #     :param decimals:
    #     :param kwargs:
    #     :return:
    #     """
    #     result_df = pg.ptests(
    #         data=df, paired=paired, stars=stars, decimals=decimals, **kwargs
    #     )
    #     return result_df

    # @staticmethod
    # def get_t_test(df, dv=None, between=None, within=None, **kwargs):
    #
    #     result_df = pg.pairwise_tests(
    #         data=df, dv=dv, between=between, within=within, **kwargs
    #     )
    #     return result_df

    @staticmethod
    def get_anova(data, value_col=None, group_col=None, **kwargs):
        """
        One-way and N-way ANOVA.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String or list of strings
            Name of columnName of column containing the grouping variable.
        :param kwarts: optional
            Other options available with [pingouin.anova()](https://pingouin-stats.org/build/html/generated/pingouin.anova.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.anova(data=data, dv=value_col, between=group_col, **kwargs)

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    @staticmethod
    def get_welch_anova(data, value_col=None, group_col=None):
        """
        One-way Welch ANOVA

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of column containing the grouping variable.
        :return: Pandas.DataFrame
        """

        result_df = pg.welch_anova(data=data, dv=value_col, between=group_col)

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    # todo add params
    @staticmethod
    def get_rm_anova(
        data,
        value_col=None,
        within_subject_col=None,
        subject_col=None,
        correction="auto",
        detailed=False,
        effsize="ng2",
    ):
        """
        One-way and two-way repeated measures ANOVA.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param within_subject_col: String
            Name of column containing the within factor.
        :param subject_col: String
            Name of column containing the subject identifier.
        :param kwargs: optional
            Other options available with [pingouin.rm_anova()](https://pingouin-stats.org/build/html/generated/pingouin.rm_anova.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.rm_anova(
            data=data,
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

    @staticmethod
    def get_mixed_anova(
        data,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        **kwargs,
    ):
        """
        Mixed-design ANOVA.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of column containing the between factor.
        :param within_subject_col: String
            Name of column containing the within-subject factor (repeated measurements).
        :param subject_col:
            Name of column containing the between-subject identifier.
        :param kwargs: optional
            Other options available with [pingouin.mixed_anova()](https://pingouin-stats.org/build/html/generated/pingouin.mixed_anova.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.mixed_anova(
            data=data,
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

    @staticmethod
    def get_tukey(data, value_col=None, group_col=None, effsize="hedges"):
        """
        Pairwise Tukey post-hoc test.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the between factor.
        :param effsize: String or None
            Effect size. Additional methods can be found with [pingouin.pairwise_tukey()](https://pingouin-stats.org/build/html/generated/pingouin.pairwise_tukey.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.pairwise_tukey(
            data=data, dv=value_col, between=group_col, effsize=effsize
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-tukey"]]
        result_df["significance"] = pvalue

        return result_df

    @staticmethod
    def get_gameshowell(data, value_col=None, group_col=None, effsize="hedges"):
        """
        Pairwise Games-Howell post-hoc test

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of columnName of column containing the between factor.
        :param effsize: String or None
            Effect size. Additional methods can be found with [pingouin.pairwise_gameshowell()](https://pingouin-stats.org/build/html/generated/pingouin.pairwise_gameshowell.html)
        :return: Pandas.DataFrame
        """

        result_df = pg.pairwise_gameshowell(
            data=data, dv=value_col, between=group_col, effsize=effsize
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["pval"]]
        result_df["significance"] = pvalue

        return result_df

    """
    non-parametric tests below
    """

    @staticmethod
    def get_wilcoxon(
        data,
        value_col=None,
        group_col=None,
        subgroup=None,
        alternative="two-sided",
        **kwargs,
    ):
        """
        Calculate wilcoxon tests. This is non-parametric version of paired T-test. Data number must be uniform to work.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Columns containing values for testing.
        :param group_col: String
            Column containing group name.
        :param subgroup: String
            Column containing subgroup name.
        :param alternative: String
            Defines the alternative hypothesis, or tail of the test. Must be one of “two-sided”. Must be one of
            “two-sided” (default), “greater” or “less”.
        :param kwargs: Optional
            Other options available with [pingouin.wilcoxon()](https://pingouin-stats.org/build/html/generated/pingouin.wilcoxon.html)
        :return: Pandas.DataFrame
        """

        # ignore Wilcoxon warnings
        warnings.filterwarnings(
            "ignore",
            message="Exact p-value calculation does not work if there are zeros.*",
        )

        if subgroup:
            # Convert 'Name' and 'Status' columns to string
            data[group_col] = data[group_col].astype(str)
            data[subgroup] = data[subgroup].astype(str)
            data["subgroup"] = data[group_col] + "-" + data[subgroup]

            subgroup_list = data["subgroup"].unique().tolist()
            subgroup_df = data[data["subgroup"].isin(subgroup_list)].copy()

            # Get unique pairs between group and subgroup
            group = subgroup_df["subgroup"].unique()

            # From unique items in group list, generate pairs
            pairs = list(combinations(group, 2))

            results_list = []
            for pair in pairs:
                # Get items from pair list and split by hyphen
                group1, subgroup1 = pair[0].split("-")
                group2, subgroup2 = pair[1].split("-")

                # # For troubleshooting
                # print("first:", data[(data[group_col] == group1)][value_col].shape)
                # print("second:", data[(data[group_col] == group2)][value_col].shape)

                # Check length of groups
                group1_length = data[data[group_col] == group1][value_col]
                group2_length = data[data[group_col] == group2][value_col]

                # print(len(group1_length), len(group2_length)) # For troubleshooting

                if len(group1_length) != len(group2_length):
                    raise ValueError(
                        "The lengths of the groups in group_col are not equal!"
                    )

                # Perform Wilcoxon signed-rank test
                result = pg.wilcoxon(
                    data[(data[group_col] == group1) & (data[subgroup] == subgroup1)][
                        value_col
                    ],
                    data[(data[group_col] == group2) & (data[subgroup] == subgroup2)][
                        value_col
                    ],
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
            result_df["A"] = result_df["A"].apply(lambda x: tuple(x.split("-")))
            result_df["B"] = result_df["B"].apply(lambda x: tuple(x.split("-")))

            return result_df
        else:
            """
            No subgroups found. Tests single group and values.
            """
            # Get unique pairs from group
            group = data[group_col].unique()

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
                group1_length = data[data[group_col] == group1][value_col]
                group2_length = data[data[group_col] == group2][value_col]

                # print(len(group1_length), len(group2_length)) # For troubleshooting

                if len(group1_length) != len(group2_length):
                    raise ValueError(
                        "The lengths of the groups in group_col are not equal!"
                    )

                # Perform wilcoxon
                result = pg.wilcoxon(
                    data[(data[group_col] == group1)][value_col],
                    data[(data[group_col] == group2)][value_col],
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

    @staticmethod
    def get_mannu(
        data,
        value_col=None,
        group_col=None,
        subgroup=None,
        alternative="two-sided",
        **kwargs,
    ):
        """
        Calculate Mann-Whitney U Test. This is a non-parametric version of the independent T-test.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Columns containing values for testing.
        :param group_col: String
            Column containing group name.
        :param subgroup: String
            Column containing subgroup name.
        :param alternative: String
            Defines the alternative hypothesis, or tail of the test. Must be one of “two-sided”. Must be one of
            “two-sided” (default), “greater” or “less”.
        :param kwargs: Optional
            Other options available with [pingouin.mwu()](https://pingouin-stats.org/build/html/generated/pingouin.mwu.html)
        :return: Pandas.DataFrame
        """

        if subgroup:
            # Convert 'Name' and 'Status' columns to string
            data[group_col] = data[group_col].astype(str)
            data[subgroup] = data[subgroup].astype(str)
            data["subgroup"] = data[group_col] + "-" + data[subgroup]

            subgroup_list = data["subgroup"].unique().tolist()
            subgroup_df = data[data["subgroup"].isin(subgroup_list)].copy()

            # Get unique pairs between group and subgroup
            group = subgroup_df["subgroup"].unique()

            # From unique items in group list, generate pairs
            pairs = list(combinations(group, 2))

            results_list = []
            for pair in pairs:
                # Get items from pair list and split by hyphen
                group1, subgroup1 = pair[0].split("-")
                group2, subgroup2 = pair[1].split("-")

                # Perform mwu
                result = pg.mwu(
                    data[(data[group_col] == group1) & (data[subgroup] == subgroup1)][
                        value_col
                    ],
                    data[(data[group_col] == group2) & (data[subgroup] == subgroup2)][
                        value_col
                    ],
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
            result_df = pd.DataFrame(results_list)

            # Split values into and separate by comma
            result_df["A"] = result_df["A"].apply(lambda x: tuple(x.split("-")))
            result_df["B"] = result_df["B"].apply(lambda x: tuple(x.split("-")))

            return result_df
        else:
            """
            No subgroups found. Tests single group and values.
            """
            # Get unique pairs from group
            group = data[group_col].unique()

            # From unique items in group list, generate pairs
            pairs = list(combinations(group, 2))

            results_list = []
            for pair in pairs:
                # Get items from pair list and split by hyphen
                group1 = pair[0]
                group2 = pair[1]
                # Perform mwu
                result = pg.mwu(
                    data[(data[group_col] == group1)][value_col],
                    data[(data[group_col] == group2)][value_col],
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
            result_df = pd.DataFrame(results_list)

            return result_df

    @staticmethod
    def get_kruskal(data, value_col=None, group_col=None, detailed=False):
        """
        Calculate Kruskal-Wallis H-test for independent samples.

        :param data: pandas.DataFrame
            Input DataFrame.
        :param value_col: String
            Name of column containing the dependent variable.
        :param group_col: String
            Name of column containing the between factor.
        :param detailed: Boolean
            Ouput additional details from Kruskal-Wallis H-test.
        :return: Pandas.DataFrame
        """

        result_df = pg.kruskal(
            data=data, dv=value_col, between=group_col, detailed=detailed
        )

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    @staticmethod
    def get_cochran(data, value_col=None, group_col=None, subgroup_col=None):
        """
        Calculate Cochran Q Test. This is used when the dependent variable, or value_col, is binary. For details between
        groups, posthoc test will be needed.

        :param data: pandas.DataFrame
            Input DataFrame.
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
                data=data, dv=value_col, within=subgroup_col, subject=group_col
            )
        else:
            result_df = pg.cochran(data=data, dv=value_col, within=group_col)

        # Add significance asterisk
        pvalue = [utils.star_value(value) for value in result_df["p-unc"]]
        result_df["significance"] = pvalue

        return result_df

    @staticmethod
    def get_friedman(
        data=None, group_col=None, value_col=None, subgroup_col=None, method="chisq"
    ):
        """
        Calculate Friedman Test. Determines if distributions of two or more paired samples are equal. For details between
        groups, posthoc test (get_pairwise_tests(parametric=False)) will be needed.

        :param data: pandas.DataFrame
            Input DataFrame.
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

        result_df = pg.friedman(
            data=data,
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

    @staticmethod
    def get_pairwise_tests(
        data,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        parametric=True,
        **kwargs,
    ):
        """
        Posthoc test for parametric or nonparametric statistics. By default, the parametric parameter is set as True.

        :param data: pandas.DataFrame
            Input DataFrame.
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
            data=data,
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

    @staticmethod
    def get_pairwise_rm(
        data,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        parametric=True,
        **kwargs,
    ):
        """
        Posthoc test for repeated measures.

        :param data: pandas.DataFrame
            Input DataFrame.
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
            data=data,
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

    @staticmethod
    def get_pairwise_mixed(
        data,
        value_col=None,
        group_col=None,
        within_subject_col=None,
        subject_col=None,
        parametric=True,
        **kwargs,
    ):
        """
        Posthoc test for mixed ANOVA.

        :param data: pandas.DataFrame
            Input DataFrame.
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
            data=data,
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

    # todo update documentation
    @staticmethod
    def get_p_matrix(data, test=None, group_col1=None, group_col2=None, **kwargs):
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
        :param kwargs:
        :return:
        """

        # Run tests based on test parameter input
        # todo add options for additional test and ensure format matches
        if test == "tukey":
            matrix_df = utils.multi_group(data, group_col1, group_col2, test)
        elif test == "mannu" or test == "wilcoxon":
            matrix_df = utils.single_group(df=data, group_col=group_col1, test=test)
        else:
            raise NameError(
                "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
            )

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


class Plots:

    # todo update list output
    @staticmethod
    def list_test(list=True):
        """
        List all tests available for plotting
        :param list:
        :return:
        """
        if list:
            print(
                "List of tests available for plotting: 'tukey', 'gameshowell', 'ttest-within', 'ttest-between', 'ttest-mixed', 'wilcoxon', 'mannu', 'para-ttest "
                "'kruskal'"
            )
        else:
            print("Input Test Not Valid!")

    @staticmethod
    def boxplot(
        data,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subgroup_pairs=None,  # The minute this is a parameter, the program goes heywire. Added as variable to _plot_variables()
        pairs=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        loc="inside",
        whis=1.5,  # boxplot whiskers
        return_df=None,
        **kwargs,
    ):
        """
        Draw a boxplot from the input DataFrame.

        :param data: Pandas.DataFrame
            Input data DataFrame.
        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup: String
            Name of the column containing the subgroup for the grou column. This is associated with the hue parameters
            in Seaborn.
        :param subgroup_pairs: String
            Name of the column containing the subgroups to the group column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_order: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param whis: Int
            Set length of whiskers on plot.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with associated pairs.

        :return: Fig
        """
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.boxplot)
        valid_annot = utils.get_kwargs(Annotator)

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        pair_order = pairs

        # Get plot variables
        # If plotting more pairs than needed, issues is with the pairs
        pairs, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            data,
            group_col,
            pair_order,
            test,
            value_col,
            valid_sns,
            valid_annot,
            subgroup_col,
            subgroup_pairs,
            **kwargs,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # Set title and size of plot
        title = kwargs.pop("title", None)
        title_fontsize = kwargs.pop("title_fontsize", None)

        # Set title if provided
        if title:
            plt.title(title, fontsize=title_fontsize)

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.boxplot(
                data=data,
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
                data=data,
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
                data=data,
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
                data=data,
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
        if pvalue_order:
            pvalue = pvalue_order

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

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def barplot(
        data,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subgroup_pairs=None,  # The minute this is a parameter, the program goes heywire. Added as variable to _plot_variables()
        pairs=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        loc="inside",
        ci="sd",
        capsize=0.1,
        return_df=None,
        **kwargs,
    ):
        """
        Draw a boxplot from the input DataFrame.

        :param data: Pandas.DataFrame
            Input data DataFrame.
        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup: String
            Name of the column containing the subgroup for the grou column. This is associated with the hue parameters
            in Seaborn.
        :param subgroup_pairs: String
            Name of the column containing the subgroups to the group column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_order: List.
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
            Returns a DataFrame of calculated results. If pairs used, only return rows with associated pairs.

        :return: Fig
        """
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.barplot)
        valid_annot = utils.get_kwargs(Annotator)

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        pair_order = pairs

        # Get plot variables
        # If plotting more pairs than needed, issues is with the pairs
        pairs, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            data,
            group_col,
            pair_order,
            test,
            value_col,
            valid_sns,
            valid_annot,
            subgroup_col,
            subgroup_pairs,
            **kwargs,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # Set title and size of plot
        title = kwargs.pop("title", None)
        title_fontsize = kwargs.pop("title_fontsize", None)

        # Set title if provided
        if title:
            plt.title(title, fontsize=title_fontsize)

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.barplot(
                data=data,
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
                data=data,
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
                data=data,
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
                data=data,
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
        if pvalue_order:
            pvalue = pvalue_order

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

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def violinplot(
        data,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subgroup_pairs=None,  # The minute this is a parameter, the program goes heywire. Added as variable to _plot_variables()
        pairs=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        loc="inside",
        return_df=None,
        **kwargs,
    ):
        """
        Draw a boxplot from the input DataFrame.

        :param data: Pandas.DataFrame
            Input data DataFrame.
        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup: String
            Name of the column containing the subgroup for the grou column. This is associated with the hue parameters
            in Seaborn.
        :param subgroup_pairs: String
            Name of the column containing the subgroups to the group column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_order: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with associated pairs.

        :return: Fig
        """
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.violinplot)
        valid_annot = utils.get_kwargs(Annotator)

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        pair_order = pairs

        # Get plot variables
        # If plotting more pairs than needed, issues is with the pairs
        pairs, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            data,
            group_col,
            pair_order,
            test,
            value_col,
            valid_sns,
            valid_annot,
            subgroup_col,
            subgroup_pairs,
            **kwargs,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # Set title and size of plot
        title = kwargs.pop("title", None)
        title_fontsize = kwargs.pop("title_fontsize", None)

        # Set title if provided
        if title:
            plt.title(title, fontsize=title_fontsize)

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.violinplot(
                data=data,
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
                data=data,
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
                data=data,
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
                data=data,
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
        if pvalue_order:
            pvalue = pvalue_order

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

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def swarmplot(
        data,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subgroup_pairs=None,  # The minute this is a parameter, the program goes heywire. Added as variable to _plot_variables()
        pairs=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        loc="inside",
        return_df=None,
        **kwargs,
    ):
        """
        Draw a boxplot from the input DataFrame.

        :param data: Pandas.DataFrame
            Input data DataFrame.
        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup: String
            Name of the column containing the subgroup for the grou column. This is associated with the hue parameters
            in Seaborn.
        :param subgroup_pairs: String
            Name of the column containing the subgroups to the group column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_order: List.
            A list containing specific pvalue labels. This order must match the length of pairs list.
        :param palette: String or List.
            Color palette used for the plot. Can be given as common color name or in hex code.
        :param orient: String
            Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param loc: String
            Set location of annotations. Only "inside" or "outside" are supported.
        :param return_df: Boolean
            Returns a DataFrame of calculated results. If pairs used, only return rows with associated pairs.

        :return: Fig
        """
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.swarmplot)
        valid_annot = utils.get_kwargs(Annotator)

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        pair_order = pairs

        # Get plot variables
        # If plotting more pairs than needed, issues is with the pairs
        pairs, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            data,
            group_col,
            pair_order,
            test,
            value_col,
            valid_sns,
            valid_annot,
            subgroup_col,
            subgroup_pairs,
            **kwargs,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # Set title and size of plot
        title = kwargs.pop("title", None)
        title_fontsize = kwargs.pop("title_fontsize", None)

        # Set title if provided
        if title:
            plt.title(title, fontsize=title_fontsize)

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.swarmplot(
                data=data,
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
                data=data,
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
                data=data,
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
                data=data,
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
        if pvalue_order:
            pvalue = pvalue_order

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

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def lineplot(
        data,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup_col=None,
        subgroup_pairs=None,  # The minute this is a parameter, the program goes heywire. Added as variable to _plot_variables()
        pairs=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        loc="inside",
        ci="sd",
        capsize=0.1,
        return_df=None,
        **kwargs,
    ):
        """
        Draw a boxplot from the input DataFrame.

        :param data: Pandas.DataFrame
            Input data DataFrame.
        :param test: String
            Name of test for calculations. Names must match the test names from the py50.Stats()
        :param group_col: String
            Name of column containing groups. This should be the between depending on the selected test.
        :param value_col: String
            Name of the column containing the values. This is the dependent variable.
        :param group_order: List.
            Place the groups in a specific order on the plot.
        :param subgroup: String
            Name of the column containing the subgroup for the grou column. This is associated with the hue parameters
            in Seaborn.
        :param subgroup_pairs: String
            Name of the column containing the subgroups to the group column.
        :param pairs: List
            A list containing specific pairings for annotation on the plot.
        :param pvalue_order: List.
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
            Returns a DataFrame of calculated results. If pairs used, only return rows with associated pairs.

        :return: Fig
        """
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.lineplot)
        valid_annot = utils.get_kwargs(Annotator)

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        pair_order = pairs

        # Get plot variables
        # If plotting more pairs than needed, issues is with the pairs
        pairs, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            data,
            group_col,
            pair_order,
            test,
            value_col,
            valid_sns,
            valid_annot,
            subgroup_col,
            subgroup_pairs,
            **kwargs,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # Set title and size of plot
        title = kwargs.pop("title", None)
        title_fontsize = kwargs.pop("title_fontsize", None)

        # Set title if provided
        if title:
            plt.title(title, fontsize=title_fontsize)

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.lineplot(
                data=data,
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
                data=data,
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
                data=data,
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
                data=data,
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
        if pvalue_order:
            pvalue = pvalue_order

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

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def p_matrix(matrix_df, cmap=None, title=None, title_fontsize=14, **kwargs):
        """
        Wrapper function for scikit_posthoc heatmap.

        :param matrix_df: Pandas.Dataframe
            Input table must be a matrix calculated using the stats.get_p_matrix()
        :param cmap: List
            A list of colors. Can be color names or hex codes
        :param title: String
            Input title for figure
        :param title_fontsize: Int
            Set size of figure legend
        :param kwargs: Optional
            Keyword arguemnts associated with [scikit-posthocs](https://scikit-posthocs.readthedocs.io/en/latest/)

        :return: Pyplot figure
        """
        if title:
            plt.title(title, fontsize=title_fontsize)

        if cmap is None:
            cmap = ["1", "#fbd7d4", "#005a32", "#238b45", "#a1d99b"]
            fig = sp.sign_plot(matrix_df, cmap=cmap, **kwargs)
        else:
            fig = sp.sign_plot(matrix_df, cmap=cmap, **kwargs)

        # Display plot
        return fig

    @staticmethod
    def ttest_bar_plot():
        # Function will mirror above. Need to format shape to fit Statannotation
        pass

    """
    Functions to plot data distribution
    """

    @staticmethod
    def distribution(data, val_col=None, type="histplot", **kwargs):
        """

        :param data: Pandas.Dataframe
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
            fig = sns.histplot(data=data, x=val_col, **kwargs)
        elif type == "qqplot":
            fig = pg.qqplot(data[val_col], dist="norm", **kwargs)
        else:
            raise ValueError(
                "For test parameter, only 'histplot' or 'qqplot' available"
            )

        return fig


def _get_test(
    test,
    data=None,
    group_col=None,
    value_col=None,
    subgroup_col=None,
    subgroup_pairs=None,
    subject_col=None,
    pair_order=None,
    **kwargs,
):
    """
    Function to utilize a specific statistical test. This will output the results in a dataframe and also the pvalues as
    a list. This function is primarily used for the plot functions in the stats.Plots() class.

    :param test:
    :param data:
    :param x_axis:
    :param y_axis:
    :param kwargs:
    :return:
    """

    global pairs
    if test == "tukey":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_tukey)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        result_df = Stats.get_tukey(
            data, value_col=value_col, group_col=group_col, **pg_kwargs
        )

        # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
        result_df = _get_pair_subgroup(result_df, hue=pair_order)

        pvalue = [utils.star_value(value) for value in result_df["p-tukey"].tolist()]
        pairs = [(a, b) for a, b in zip(result_df["A"], result_df["B"])]

    elif test == "gameshowell":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_gameshowell)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        result_df = Stats.get_gameshowell(
            data, value_col=value_col, group_col=group_col, **pg_kwargs
        )

        # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
        result_df = _get_pair_subgroup(result_df, hue=pair_order)

        pvalue = [utils.star_value(value) for value in result_df["pval"].tolist()]
        pairs = [(a, b) for a, b in zip(result_df["A"], result_df["B"])]

    # Parametric T-Test
    elif test == "pairwise-parametric":
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        result_df = Stats.get_pairwise_tests(
            data,
            value_col=value_col,
            group_col=group_col,
            within_subject_col=subgroup_col,
            subject_col=subject_col,
            parametric=True,
            **pg_kwargs,
        )

        # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
        result_df = _get_pair_subgroup(result_df, hue=pair_order)

        # Obtain pvalues and pairs and split them from test_df for passing into Annotator
        pvalue = [utils.star_value(value) for value in result_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(result_df["A"], result_df["B"])]

    elif test == "pairwise-rm":
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        result_df = Stats.get_pairwise_rm(
            data,
            value_col=value_col,
            group_col=None,
            within_subject_col=group_col,
            subject_col=subject_col,
            parametric=True,
            **pg_kwargs,
        )

        # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
        result_df = _get_pair_subgroup(result_df, hue=pair_order)

        # Obtain pvalues and pairs and split them from test_df for passing into Annotator
        pvalue = [utils.star_value(value) for value in result_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(result_df["A"], result_df["B"])]

    # todo pairwise-mixed needs to be modified for plotting
    elif test == "pairwise-mixed":
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        print(subgroup_col)

        # run test
        result_df = Stats.get_pairwise_mixed(
            data,
            value_col=value_col,
            group_col=group_col,
            within_subject_col=subgroup_col,
            subject_col=subject_col,
            parametric=True,
            **pg_kwargs,
        )

        # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
        result_df = _get_pair_subgroup(result_df, hue=pair_order)

        # Obtain pvalues and pairs and split them from test_df for passing into Annotator
        pvalue = [utils.star_value(value) for value in result_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(result_df["A"], result_df["B"])]

    # Non-parametric T-Test
    elif test == "pairwise-nonparametric":
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        result_df = Stats.get_pairwise_tests(
            data,
            value_col=value_col,
            group_col=group_col,
            within_subject_col=subgroup_col,
            subject_col=subject_col,
            parametric=True,
            **pg_kwargs,
        )

        # result_df has removed rows with n.s. This is only needed if plot has specific pairs input
        result_df = _get_pair_subgroup(result_df, hue=pair_order)

        # Obtain pvalues and pairs and split them from test_df for passing into Annotator
        pvalue = [utils.star_value(value) for value in result_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(result_df["A"], result_df["B"])]

    elif test == "wilcoxon":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.wilcoxon)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        result_df = Stats.get_wilcoxon(
            data, group_col=group_col, value_col=value_col, **pg_kwargs
        )
        pvalue = [utils.star_value(value) for value in result_df["p-val"].tolist()]
        # Obtain pairs and split them from Wilcox result DF for passing into Annotator
        pairs = []
        for item in result_df["Comparison"].tolist():
            parts = item.split("-")
            pairs.append((parts[0], parts[1]))

    elif test == "mannu":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.mwu)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # Obtain pairs and split them from Wilcox result DF for passing into Annotator
        if subgroup_col:
            # run test
            result_df = Stats.get_mannu(
                data,
                group_col=group_col,
                value_col=value_col,
                subgroup=subgroup_col,
                **pg_kwargs,
            )

            # Make pairs between groups and subgroups by df
            result_df = _get_pair_subgroup(result_df, hue=subgroup_pairs)
            result_df = result_df.reset_index(drop=True)

            # Obtain pvalues and pairs and split them from test_df for passing into Annotator
            pvalue = [utils.star_value(value) for value in result_df["p-val"].tolist()]
            pairs = _get_pairs(result_df, hue=subgroup_pairs)
        else:
            # run test
            result_df = Stats.get_mannu(
                data, group_col=group_col, value_col=value_col, **pg_kwargs
            )

            result_df = _get_pair_subgroup(result_df, hue=pair_order)

            # Obtain pvalues and pairs and split them from test_df for passing into Annotator
            pvalue = [utils.star_value(value) for value in result_df["p-val"].tolist()]
            pairs = _get_pairs(result_df, hue=pair_order)

    elif test == "kruskal":  # kurskal does not give posthoc. modify
        result_df = Stats.get_kruskal(
            data, value_col=value_col, group_col=group_col, detailed=False
        )
        pvalue = [utils.star_value(value) for value in result_df["p-unc"].tolist()]
    else:
        raise ValueError("Test not recognized!")

    # elif test == "ttest":
    #     test_df = Stats.get_t_test(df, paired=False, x=None, y=None, **kwargs) # todo determine how to select column to return as list

    return pvalue, result_df, pairs, subgroup_col


def _plot_variables(
    data,
    group_col,
    pair_order,
    test,
    value_col,
    valid_sns,
    valid_annot,
    subgroup_col=None,
    subgroup_pairs=None,
    subject_col=None,
    **kwargs,
):
    """
    Output plot variables for use inside plots in Plots() class
    :param data:
    :param group_col:
    :param kwargs:
    :param pair_order: input pairs. this will output which data to keep for plotting.
    :param test:
    :param value_col:
    :return:
    """
    # Get kwarg for sns and annot. If printed, should only appear if kwargs found within module.
    sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
    annot_kwargs = {key: value for key, value in kwargs.items() if key in valid_annot}

    # Run tests based on test parameter input
    if test is not None:
        pvalue, test_df, pairs, subgroup = _get_test(
            test=test,
            data=data,
            group_col=group_col,
            value_col=value_col,
            pair_order=pair_order,
            subgroup_col=subgroup_col,
            subgroup_pairs=subgroup_pairs,
            subject_col=subject_col,
            **kwargs,
        )
    else:
        raise NameError(
            "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
        )

    # set custom pair order
    if pair_order:
        pairs = pair_order

    # return pairs, palette, pvalue, sns_kwargs, annot_kwargs, test_df
    return pairs, pvalue, sns_kwargs, annot_kwargs, test_df


def _get_pair_subgroup(df, hue=None):
    """Generate pairs by group_col and hue. Hue will designate which input rows to keep for plotting."""

    if hue is None:
        hue = _get_pairs(df, hue)
    else:
        hue = hue
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


def _get_pairs(df, hue):
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
