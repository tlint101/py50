"""
The following script holds plot logic for the indicated tests. This was created to reduce the "look" of the stats file
for maintainability.
"""
import pandas as pd


def tukey_plot_logic(stat, test_value):
    """

    :param stat:
    :param test_value:
    :return:
    """
    pvalue = [stat.star_value(value) for value in test_value["p-tukey"].tolist()]
    return pvalue
