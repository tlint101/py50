"""
Script to calculate statistics.
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import combinations
import pingouin as pg
from statannotations.Annotator import Annotator
from py50 import stat_util as util


def normality(df):
    pass


class Stats:
    def __init__(self):
        pass

    @staticmethod
    def get_normality(df, dv=None, group=None, method="shapiro"):

        result_df = pg.normality(data=df, dv=dv, group=group, method=method)
        return result_df

    @staticmethod
    def get_homoscedasticity(df, dv=None, group=None, **kwargs):
        method = kwargs.get("method")
        if method is None:
            method = "levene"
        result_df = pg.homoscedasticity(
            data=df, dv=dv, group=group, method=method, **kwargs
        )
        return result_df

    @staticmethod
    def get_anova(df, dv=None, between=None, type=2):
        """

        :param df:
        :param dv:
        :param between:
        :param type:
        :return:
        """
        result_df = pg.anova(data=df, dv=dv, between=between, ss_type=2)
        return result_df

    @staticmethod
    def get_tukey(df, dv=None, between=None):
        result_df = pg.pairwise_tukey(data=df, dv=dv, between=between)
        return result_df

    # todo double check star_value
    @staticmethod
    def star_value(p_value):
        if p_value < 0.0001:
            return "****"
        elif p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "n.s."


class Plots:
    def __init__(self):
        pass

    @staticmethod
    def box_plot(df, x_axis=None, y_axis=None, group_col=None, test=None, return_df=None):
        """

        :param df:
        :param x_axis:
        :param y_axis:
        :param group_col:
        :param test:
        :param return_df: Will return dataframe of calculated results
        :return:
        """

        global pvalue, test_value
        pairs = list(combinations(df[group_col].unique(), 2))
        groups = df[group_col].unique()

        ax = sns.boxplot(data=df, x=x_axis, y=y_axis, order=groups)

        # todo create function to format pvalue into star values
        stat = Stats()

        try:
            # Run tests based on test parameter input
            if test is not None:
                if test == "tukey":
                    test_value = stat.get_tukey(df, dv=y_axis, between=x_axis)
                    pvalue = util.tukey_plot_logic(stat, test_value)
                elif test == "anova":
                    print("anova!")
            else:
                print("Test input not available. Try: ")

            # add annotations
            annotator = Annotator(
                ax, pairs, data=df, x=x_axis, y=y_axis, verbose=False
            )
            annotator.set_custom_annotations(pvalue)
            annotator.annotate()
        except ValueError:
            print("Test type needed!")

        if return_df:
            return test_value

    # todo generate state.util script
    # @staticmethod
    # def tukey_plot_logic(stat, test_value):
    #     global pvalue
    #     pvalue = [
    #         f"p={pvalue:.2e}" for pvalue in test_value["p-tukey"].tolist()
    #     ]
    #     pvalue = [
    #         stat.star_value(value)
    #         for value in test_value["p-tukey"].tolist()
    #     ]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
