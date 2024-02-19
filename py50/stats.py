"""
Script to calculate statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import pingouin as pg
from statannotations.Annotator import Annotator
from py50 import utils

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
    def get_normality(df, dv=None, group=None, **kwargs):
        """
        Test data normality

        :param df:
        :param dv:
        :param group:
        :param method:
        :return:
        """

        result_df = pg.normality(data=df, dv=dv, group=group, **kwargs)
        return result_df

    @staticmethod
    def get_homoscedasticity(df, dv=None, group=None, **kwargs):
        """
        Test for data variance.

        :param df:
        :param dv:
        :param group:
        :param kwargs:
        :return:
        """

        result_df = pg.homoscedasticity(data=df, dv=dv, group=group, **kwargs)
        return result_df

    """
    Parametric posts below
    """

    @staticmethod
    def get_t_test(df, paired=True, stars=False, decimals=4, **kwargs):
        """
        Calculate pairwise t-test.

        :param df:
        :param paired:
        :param stars:
        :param decimals:
        :param kwargs:
        :return:
        """
        result_df = pg.ptests(
            data=df, paired=paired, stars=stars, decimals=decimals, **kwargs
        )
        return result_df

    @staticmethod
    def get_anova(df, dv=None, between=None, **kwargs):
        """

        :param df:
        :param dv:
        :param between:
        :param type:
        :return:
        """

        result_df = pg.anova(data=df, dv=dv, between=between, **kwargs)
        return result_df

    # todo add welch anova
    @staticmethod
    def get_welch_anova():
        pass

    @staticmethod
    def get_tukey(df, dv=None, between=None, **kwargs):
        """

        :param df:
        :param dv:
        :param between:
        :return:
        """
        recognized_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in pg.pairwise_tukey.__code__.co_varnames
        }

        result_df = pg.pairwise_tukey(
            data=df, dv=dv, between=between, **recognized_kwargs
        )
        return result_df

    @staticmethod
    def get_gameshowell(df, dv=None, between=None, **kwargs):
        result_df = pg.pairwise_gameshowell(data=df, dv=dv, between=between, **kwargs)
        return result_df

    @staticmethod
    def get_rm_anova(df, dv=None, within=None, subject=None, **kwargs):
        result_df = pg.rm_anova(
            data=df, dv=dv, within=within, subject=subject, **kwargs
        )
        return result_df

    @staticmethod
    def get_pairwise_test(df, dv=None, between=None, **kwargs):
        """
        Calculate pairwise_tests
        :param df:
        :param dv:
        :param between:
        :param effsize:
        :param kwargs:
        :return:
        """
        result_df = pg.pairwise_tests(data=df, dv=dv, between=between, **kwargs)
        return result_df

    @staticmethod
    def get_mixed_anova(df, dv=None, within=None, subject=None, **kwargs):
        result_df = pg.mixed_anova(
            data=df, dv=dv, within=within, subject=subject, **kwargs
        )
        return result_df

    """
    non-parametric tests below
    """

    @staticmethod
    def get_wilcoxon(df, group_col=None, value_col=None):
        """
        Calculate wilcoxon
        :param df:
        :param group_col:
        :param value_col:
        :return:
        """
        # Get unique pairs from group
        group = df[group_col].unique()

        # Empty list to store results
        results_list = []

        # Perform Wilcoxon signed-rank test for each pair
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                group1 = group[i]
                group2 = group[j]
                value1 = df[df[group_col] == group1][value_col]
                value2 = df[df[group_col] == group2][value_col]

                # Ensure same length for each condition
                min_length = min(len(value1), len(value2))
                value1 = value1[:min_length]
                value2 = value2[:min_length]

                # Perform Wilcoxon signed-rank test
                result = pg.wilcoxon(value1, value2)

                # Store the results in the list
                key = f"{group1}-{group2}"
                results_list.append(
                    {
                        "Comparison": key,
                        "W-val": result["W-val"].iloc[0],
                        "p-val": result["p-val"].iloc[0],
                        "RBC": result["RBC"].iloc[0],
                        "CLES": result["CLES"].iloc[0],
                    }
                )
        # Convert the list of dictionaries to a DataFrame
        result_df = pd.DataFrame(results_list)

        return result_df

    @staticmethod
    def get_mannu(df, group_col=None, value_col=None):
        """
        Calculate Mann-Whitney U Test
        :param df:
        :param group_col:
        :param value_col:
        :return:
        """
        # Get unique pairs from group
        group = df[group_col].unique()

        # Empty list to store results
        results_list = []

        # Perform Mann-Whitney U Test signed-rank test for each pair
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                group1 = group[i]
                group2 = group[j]
                value1 = df[df[group_col] == group1][value_col]
                value2 = df[df[group_col] == group2][value_col]

                # Ensure same length for each condition
                min_length = min(len(value1), len(value2))
                value1 = value1[:min_length]
                value2 = value2[:min_length]

                # Perform Wilcoxon signed-rank test
                result = pg.mwu(value1, value2)

                # Store the results in the list
                key = f"{group1}-{group2}"
                results_list.append(
                    {
                        "Comparison": key,
                        "U-val": result["U-val"].iloc[0],
                        "p-val": result["p-val"].iloc[0],
                        "RBC": result["RBC"].iloc[0],
                        "CLES": result["CLES"].iloc[0],
                    }
                )
        # Convert the list of dictionaries to a DataFrame
        result_df = pd.DataFrame(results_list)

        return result_df

    @staticmethod
    def get_kruskal(df, dv=None, between=None, detailed=False):
        """
        Calculate Mann-Whitney U Test
        :param df:
        :param group_col:
        :param value_col:
        :return:
        """
        result_df = pg.kruskal(data=df, dv=dv, between=between, detailed=detailed)
        return result_df

    """
    Output P-Values as a matrix in Pandas DataFrame
    """

    @staticmethod
    def get_p_matrix(df, test=None, group_col1=None, group_col2=None):
        """
        Convert dataframe results into a matrix. Group columns must be indicated. Group 2 is optional and depends on test
        used (i.e. pairwise vs Mann-Whitney U). Final DataFrame output can be used with the Plots.p_matrix() function to
        generate a heatmap of p-values.
        :param df:
        :param group1:
        :param group2:
        :param test:
        :param kwargs:
        :return:
        """
        # Run tests based on test parameter input
        if test == "tukey":
            matrix_df = utils.multi_group(df, group_col1, group_col2, test)
        elif test == "mannu" or test == "wilcoxon":
            matrix_df = utils.single_group(df=df, group_col=group_col1, test=test)
        else:
            raise NameError(
                "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
            )

        return matrix_df


class Plots:

    @staticmethod
    def box_plot(
        df,
        test=None,
        group_col=None,
        value_col=None,
        palette=None,
        orient="v",
        savepath=None,
        **kwargs,
    ):
        """

        :param df:
        :param x_axis:
        :param y_axis:
        :param group_col:
        :param test:
        :param return_df: Will return dataframe of calculated results
        :param palette:
        :return:
        """
        # todo add kwarg option for pair order

        # Run tests based on test parameter input
        if test is not None:
            pvalue, test_df, pairs = _get_test(
                test=test,
                df=df,
                group_col=group_col,
                value_col=value_col,
                **kwargs,
            )
        else:
            raise NameError(
                "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
            )

        # set default color palette
        if palette is not None:
            palette = utils.palette(palette)
        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.boxplot(
                data=df,
                x=group_col,
                y=value_col,
                order=df[group_col].unique(),
                palette=palette,
                **kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=group_col,
                y=value_col,
                verbose=False,
                orient="v",
            )
        elif orient == "h":
            ax = sns.boxplot(
                data=df,
                x=value_col,
                y=group_col,
                order=df[group_col].unique(),
                palette=palette,
                **kwargs,
            )
            # flip x and y annotations for horizontal orientation
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=value_col,
                y=group_col,
                verbose=False,
                orient="h",
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # Set custom annotations and annotate
        annotator.set_custom_annotations(pvalue)
        annotator.annotate()

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")

        # if return_df:
        #     return test_df  # return calculated df. Change name for more description

    @staticmethod
    def bar_plot(
        df,
        x_axis=None,
        y_axis=None,
        group_col=None,
        test=None,
        return_df=None,
        palette=None,
        savepath=None,
        **kwargs,
    ):
        """

        :param df:
        :param x_axis:
        :param y_axis:
        :param group_col:
        :param test:
        :param return_df: Will return dataframe of calculated results
        :param palette:
        :return:
        """

        groups = df[group_col].unique()

        # set default color palette
        if palette is not None:
            palette = utils.palette(palette)
        ax = sns.barplot(
            data=df, x=x_axis, y=y_axis, order=groups, palette=palette, **kwargs
        )

        try:
            # Run tests based on test parameter input
            if test is not None:
                pvalue, test_df = _get_test(
                    test=test, df=df, x_axis=x_axis, y_axis=y_axis, **kwargs
                )
            else:
                raise NameError(
                    "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
                )  # todo modify by adding other tests

            # todo add kwarg option for pair order
            # get pairs of groups (x-axis)
            pair_plot = kwargs.get("pair_plot")
            # print(pair_plot)
            if pair_plot is None:
                pair_plot = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]
            # print(pair_plot)

            # add annotations
            annotator = Annotator(
                ax, pair_plot, data=df, x=x_axis, y=y_axis, verbose=False
            )
            annotator.set_custom_annotations(pvalue)
            annotator.annotate()

            if savepath:
                plt.savefig(savepath, dpi=300, bbox_inches="tight")

        except ValueError:
            print("Input test type! i.e. 'tukey', 'gameshowell', or 'ttest'")

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def violin_plot(
        df,
        x_axis=None,
        y_axis=None,
        group_col=None,
        test=None,
        return_df=None,
        palette=None,
        savepath=None,
        **kwargs,
    ):
        """

        :param df:
        :param x_axis:
        :param y_axis:
        :param group_col:
        :param test:
        :param return_df: Will return dataframe of calculated results
        :param palette:
        :return:
        """

        groups = df[group_col].unique()

        # set default color palette
        if palette is not None:
            palette = utils.palette(palette)
        ax = sns.violinplot(
            data=df, x=x_axis, y=y_axis, order=groups, palette=palette, **kwargs
        )

        try:
            # Run tests based on test parameter input
            if test is not None:
                pvalue, test_df = _get_test(
                    test=test, df=df, x_axis=x_axis, y_axis=y_axis, **kwargs
                )
            else:
                raise NameError(
                    "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
                )

            # todo add kwarg option for pair order
            # get pairs of groups (x-axis)
            pair_plot = kwargs.get("pair_plot")
            # print(pair_plot)
            if pair_plot is None:
                pair_plot = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]
            # print(pair_plot)

            # add annotations
            annotator = Annotator(
                ax, pair_plot, data=df, x=x_axis, y=y_axis, verbose=False
            )
            annotator.set_custom_annotations(pvalue)
            annotator.annotate()

            if savepath:
                plt.savefig(savepath, dpi=300, bbox_inches="tight")

        except ValueError:
            print("Input test type! i.e. 'tukey', 'gameshowell', or 'ttest'")

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def swarmplot(
        df,
        x_axis=None,
        y_axis=None,
        group_col=None,
        test=None,
        return_df=None,
        palette=None,
        savepath=None,
        **kwargs,
    ):
        """

        :param df:
        :param x_axis:
        :param y_axis:
        :param group_col:
        :param test:
        :param return_df: Will return dataframe of calculated results
        :param palette:
        :return:
        """

        groups = df[group_col].unique()

        # todo TypeError: swarmplot() got multiple values for argument 'x_axis'
        # set default color palette
        if palette is not None:
            palette = utils.palette(palette)
        ax = sns.swarmplot(
            data=df, x=x_axis, y=y_axis, order=groups, palette=palette, **kwargs
        )

        try:
            # Run tests based on test parameter input
            if test is not None:
                pvalue, test_df = _get_test(
                    test=test, df=df, x_axis=x_axis, y_axis=y_axis, **kwargs
                )
            else:
                raise NameError(
                    "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
                )

            # todo add kwarg option for pair order
            # get pairs of groups (x-axis)
            pair_plot = kwargs.get("pair_plot")
            # print(pair_plot)
            if pair_plot is None:
                pair_plot = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]
            # print(pair_plot)

            # add annotations
            annotator = Annotator(
                ax, pair_plot, data=df, x=x_axis, y=y_axis, verbose=False
            )
            annotator.set_custom_annotations(pvalue)
            annotator.annotate()

            if savepath:
                plt.savefig(savepath, dpi=300, bbox_inches="tight")

        except ValueError:
            print("Input test type! i.e. 'tukey', 'gameshowell', or 'ttest'")

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def p_matrix(matrix_df, cmap=None, **kwargs):
        """
        Wrapper function for scikit_posthoc heatmap.
        :return:
        """
        if cmap is None:
            cmap = ["1", "#fb6a4a", "#08306b", "#4292c6", "#c6dbef"]
            matrix_fig = sp.sign_plot(matrix_df, cmap=cmap, **kwargs)
        else:
            matrix_fig = sp.sign_plot(matrix_df, cmap=cmap, **kwargs)

        return matrix_fig

    # todo bar plot
    @staticmethod
    def ttest_bar_plot():
        # Function will mirror above. Need to format shape to fit Statannotation
        pass

    """
    Functions to plot data distribution
    """

    @staticmethod
    def distribution(df, val_col=None, type="histplot", **kwargs):
        """

        :param df:
        :param val_col:
        :param type:
        :param kwargs: key-word arguments for seaborn or matplotlib plotting. Arguments depend on test type.
        :return:
        """
        if type == "histplot":
            fig = sns.histplot(data=df, x=val_col, kde=True, **kwargs)
        elif type == "qqplot":
            fig = pg.qqplot(df[val_col], dist="norm", **kwargs)
        else:
            raise ValueError(
                "For test parameter, only 'histplot' or 'qqplot' available"
            )

        return fig


def _get_test(test, df=None, group_col=None, value_col=None, **kwargs):
    """
    Function to utilize a specific statistical test. This will output the results in a dataframe and also the pvalues as
    a list. This function is primarily used for the plot functions in the stats.Plots() class.

    :param test:
    :param df:
    :param x_axis:
    :param y_axis:
    :param kwargs:
    :return:
    """

    # todo add function to sort pairs by user input from the plot kwarg
    if test == "tukey":
        test_df = Stats.get_tukey(df, dv=value_col, between=group_col, **kwargs)
        pvalue = [utils.star_value(value) for value in test_df["p-tukey"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    elif test == "gameshowell":
        test_df = Stats.get_gameshowell(df, dv=value_col, between=group_col, **kwargs)
        pvalue = [utils.star_value(value) for value in test_df["pval"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    elif test == "ptest": # todo update ptest
        ptest_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in pg.pairwise_tests.__code__.co_varnames
        }
        test_df = Stats.get_pairwise_test(
            df, dv=value_col, between=group_col, **ptest_kwargs
        )
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
    elif test == "wilcoxon":
        test_df = Stats.get_wilcoxon(df, group_col=group_col, value_col=value_col)
        pvalue = [utils.star_value(value) for value in test_df["p-val"].tolist()]
        # Obtain pairs and split them from Wilcox result DF for passing into Annotator
        pairs = []
        for item in test_df["Comparison"].tolist():
            parts = item.split("-")
            pairs.append((parts[0], parts[1]))

    elif test == "mannu":
        test_df = Stats.get_mannu(df, group_col=group_col, value_col=value_col)
        pvalue = [utils.star_value(value) for value in test_df["p-val"].tolist()]
        # Obtain pairs and split them from Wilcox result DF for passing into Annotator
        pairs = []
        for item in test_df["Comparison"].tolist():
            parts = item.split("-")
            pairs.append((parts[0], parts[1]))

    elif test == "kruskal": # kurskal does not give posthoc. modify
        test_df = Stats.get_kruskal(df, dv=value_col, between=group_col, detailed=False)
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
    else:
        raise ValueError("Test not recognized!")

    # elif test == "ttest":
    #     test_df = Stats.get_t_test(df, paired=False, x=None, y=None, **kwargs) # todo determine how to select column to return as list
    return (pvalue, test_df, pairs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
