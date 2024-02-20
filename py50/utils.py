"""
The following script holds plot logic for the indicated tests. This was created to reduce the "look" of the stats file
for maintainability.
"""

import inspect
from itertools import combinations
import pandas as pd
import seaborn as sns


def star_value(p_value):
    """
    Scrip to convert p values from tables into traditional star (*) format for plotting.
    :param p_value: A list of p-values obtained from a DataFrame.
    :return: A list containing the corresponding star (*) or "n.s." string.
    """

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


def palette(list=None):
    if list:
        colors = sns.color_palette(list)
    else:
        colors = sns.color_palette()
    return colors


# todo may remove
def get_pairs(df, group_col=None, value_col=None, pairs=2):
    """Input DataFrame and generate pairs for each category column"""

    # Generate pairs of categories
    pairs = list(combinations(df[group_col], pairs))

    # Remove duplicates
    sorted_pairs = [tuple(sorted(pair)) for pair in pairs]

    # Remove duplicates
    unique_pairs = list(set(sorted_pairs))

    return unique_pairs


def tukey_plot_logic(test_value):
    """

    :param stat:
    :param test_value:
    :return:
    """
    pvalue = [star_value(value) for value in test_value["p-tukey"].tolist()]
    return pvalue


def gameshowell_plot_logic(test_value):
    """

    :param stat:
    :param test_value:
    :return:
    """
    pvalue = [star_value(value) for value in test_value["pval"].tolist()]
    return pvalue


def multi_group(df, group_col1=None, group_col2=None, test=None):
    """
    Logic for obtaining p matrix. This is for tests outputs a multiple column with categorical data.
    :param df:
    :param kwargs:
    :param test:
    :param x_axis:
    :param y_axis:
    :return:
    """
    global p_col
    if test == "tukey":
        p_col = "p-tukey"
    groups = sorted(set(df[group_col1]) | set(df[group_col2]))
    matrix_df = pd.DataFrame(index=groups, columns=groups)

    # Fill the matrix with p-values
    for i, row in df.iterrows():
        matrix_df.loc[row[group_col1], row[group_col2]] = row[p_col]
        matrix_df.loc[row[group_col2], row[group_col1]] = row[p_col]

    # Fill NaN cells with NS (Not Significant)
    matrix_df.fillna(1, inplace=True)

    return matrix_df


def single_group(df, group_col=None, test=None):
    """
    Logic for obtaining p matrix. This is for tests that only outputs a single column with categorical data.
    :param df:
    :param group_col:
    :param p_col:
    :return:
    """
    if test is None:
        p_col = "p-val"
    else:
        p_col = "p-val"
    # Get unique groups
    groups = sorted(set(df[group_col]))

    # Create an empty DataFrame with groups as both index and columns
    matrix_df = pd.DataFrame(index=groups, columns=groups)

    # Fill the matrix with values
    for i, row in df.iterrows():
        matrix_df.loc[row[group_col], row[group_col]] = row[p_col]

    # Fill NaN cells if any
    matrix_df.fillna(1, inplace=True)

    return matrix_df


def get_kwargs(func):
    """To obtain kwargs for a given submodule. Usage is the following:
    sns_kwargs = get_kwargs(sns.boxplot)
    """
    return inspect.signature(func).parameters.keys()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
