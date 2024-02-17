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


class Stats:
    """
    Class contains wrappers for pingouin module. These are statistical outputs for a given input. The functions are in
    a format needed for plotting in class Plots(), however they can also be used individually to output single DataFrame
    with statistical calculations for a given dataset.
    """

    def __init__(self):
        pass

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
    def get_normality(df, dv=None, group=None, **kwargs):
        """

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

        :param df:
        :param dv:
        :param group:
        :param kwargs:
        :return:
        """

        result_df = pg.homoscedasticity(data=df, dv=dv, group=group, **kwargs)
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
    def get_p_test(df, dv=None, between=None, **kwargs):
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
    def get_rm_anova(df, dv=None, within=None, subject=None, **kwargs):
        result_df = pg.rm_anova(
            data=df, dv=dv, within=within, subject=subject, **kwargs
        )
        return result_df

    @staticmethod
    def get_mixed_anova(df, dv=None, within=None, subject=None, **kwargs):
        result_df = pg.mixed_anova(
            data=df, dv=dv, within=within, subject=subject, **kwargs
        )
        return result_df

    # todo pg.ttest may not be needed for now?
    @staticmethod
    def get_t_test(df, paired=False, **kwargs):
        result_df = pg.ttest(paired=paired, **kwargs)
        return result_df


class Plots:

    @staticmethod
    def box_plot(
        df,
        x_axis=None,
        y_axis=None,
        group_col=None,
        test=None,
        return_df=None,
        palette=None,
        savepath=None,
        **kwargs
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
        ax = sns.boxplot(data=df, x=x_axis, y=y_axis, order=groups, palette=palette)

        try:
            # Run tests based on test parameter input
            if test is not None:
                pvalue, test_df = _get_test(
                    test=test, df=df, x_axis=x_axis, y_axis=y_axis, **kwargs
                )
            else:
                raise NameError("Must include test as: 'tukey', 'gameshowell', 'ptest'")

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
    def bar_plot(
        df,
        x_axis=None,
        y_axis=None,
        group_col=None,
        test=None,
        return_df=None,
        palette=None,
        savepath=None,
        **kwargs
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
                    "Must include test as: 'tukey', 'gameshowell', 'ptest'"
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
        **kwargs
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
                raise NameError("Must include test as: 'tukey', 'gameshowell', 'ptest'")

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
        **kwargs
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
                raise NameError("Must include test as: 'tukey', 'gameshowell', 'ptest'")

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

    # todo add documentation for p_matrix
    # todo move logic into the utils?
    @staticmethod
    def p_matrix(
        df,
        x_axis=None,
        y_axis=None,
        test=None,
        **kwargs
    ):
        # Run tests based on test parameter input
        if test is not None:
            pvalue, test_df = _get_test(
                test=test, df=df, x_axis=x_axis, y_axis=y_axis, **kwargs
            )
        else:
            raise NameError("Must include test as: 'tukey', 'gameshowell', 'ptest'")

        groups = sorted(set(test_df["A"]) | set(test_df["B"]))
        matrix_df = pd.DataFrame(index=groups, columns=groups)

        # Fill the matrix with p-values
        for i, row in test_df.iterrows():
            matrix_df.loc[row["A"], row["B"]] = row["p-tukey"]
            matrix_df.loc[row["B"], row["A"]] = row["p-tukey"]

        # Fill NaN cells with NS (Not Significant)
        matrix_df.fillna(1, inplace=True)

        return matrix_df

    @staticmethod
    def plot_sig_matrix():
        return None

    @staticmethod
    def posthoc_plot(df, test=None, val_col=None, group_col=None, **kwargs):

        tests = {'tukey':  sp.posthoc_tukey(df, val_col=val_col, group_col=group_col)

        }
        cmap = kwargs.get("cmap")
        if cmap:
            cmap = cmap
        test_df = tests.get(test)
        test_df = sp.posthoc_tukey(df, val_col=val_col, group_col=group_col)
        return test_df

    @staticmethod
    def ttest_bar_plot():
        # Fucntion will mirror above. Need to format shape to fit Statannotation
        pass


def _get_test(test, df=None, x_axis=None, y_axis=None, **kwargs):
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
        test_df = Stats.get_tukey(df, dv=y_axis, between=x_axis, **kwargs)
        pvalue = [utils.star_value(value) for value in test_df["p-tukey"].tolist()]
    elif test == "gameshowell":
        test_df = Stats.get_gameshowell(df, dv=y_axis, between=x_axis, **kwargs)
        pvalue = [utils.star_value(value) for value in test_df["pval"].tolist()]
    elif test == "ptest":
        ptest_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in pg.pairwise_tests.__code__.co_varnames
        }
        test_df = Stats.get_p_test(df, dv=y_axis, between=x_axis, **ptest_kwargs)
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
    # elif test == "ttest":
    #     test_df = Stats.get_t_test(df, paired=False, x=None, y=None, **kwargs) # todo determine how to select column to return as list
    return (pvalue, test_df)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
