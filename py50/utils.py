"""
The following script holds plot logic for the indicated tests. This was created to reduce the "look" of the stats file
for maintainability.
"""

import inspect
from itertools import combinations
import numpy as np
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

    :param test_value:
    :return:
    """
    pvalue = [star_value(value) for value in test_value["p-tukey"].tolist()]
    return pvalue


def gameshowell_plot_logic(test_value):
    """

    :param test_value:
    :return:
    """
    pvalue = [star_value(value) for value in test_value["pval"].tolist()]
    return pvalue


def multi_group(df, group_col1=None, group_col2=None, test=None, order=None):
    """
    Logic for obtaining p matrix. This is for tests outputs a multiple column with categorical data.
    :param df: pandas.DataFrame
        Input DataFrame.
    :param group_col1: String
        Column with first group of data.
    :param group_col2: String
        Column with second group of data.
    :param test: String
        Test type. This will extract pvalue column from statistics table.
    :param order: List
        Reorder the groups for the final table. If input is string "alpha", the order of the groups will be
            alphabetized.

    :return:
    """
    global p_col, row_order_list, column_order_list
    if test == "tukey":
        p_col = "p-tukey"
        group_col1 = "A"
        group_col2 = "B"
    elif test == "gameshowell":
        p_col = "pval"
        group_col1 = "A"
        group_col2 = "B"
    elif test == "mannu" or test == "wilcoxon":
        p_col = "p-val"
        group_col1 = "A"
        group_col2 = "B"
    elif test == "pairwise-parametric" or test == "pairwise-nonparametric":
        p_col = "p-unc"
        group_col1 = "A"
        group_col2 = "B"

    # Create empty matrix table
    groups = sorted(set(df[group_col1]) | set(df[group_col2]))
    matrix_df = pd.DataFrame(index=groups, columns=groups)

    # Fill the matrix with p-values

    # If table utilizes subgroup
    if isinstance(matrix_df.index[0], tuple):
        matrix_df.index = matrix_df.index.map(lambda x: str(x))
        dummy_df = pd.DataFrame()
        for i, row in df.iterrows():
            dummy_df.loc[str(row[group_col1]), str(row[group_col2])] = row[p_col]
            dummy_df.loc[str(row[group_col2]), str(row[group_col1])] = row[p_col]

        # Convert index and header into (group1, group2) format
        dummy_df.index = dummy_df.index.map(lambda x: eval(x))
        dummy_df.columns = dummy_df.columns.map(lambda x: eval(x))

        # Rename and flatten index and column headers
        dummy_df.index = [f"{idx[0]}-{idx[1]}" for idx in dummy_df.index]
        dummy_df.columns = [f"{col[0]}-{col[1]}" for col in dummy_df.columns]

        # Ensure that the matrix contains 1 diagonally. Row Index and Column Headers should be in same order
        # Create a permutation
        permutation = np.random.permutation(dummy_df.index)
        # Reindex the DataFrame with the permutation for both index and columns
        dummy_df = dummy_df.reindex(index=permutation, columns=permutation)

        matrix_df = dummy_df

    # If table does not use subgroup
    else:
        for i, row in df.iterrows():
            matrix_df.loc[row[group_col1], row[group_col2]] = row[p_col]
            matrix_df.loc[row[group_col2], row[group_col1]] = row[p_col]

    # Fill NaN cells with NS (Not Significant)
    matrix_df.fillna(1, inplace=True)

    # sort table order
    if isinstance(order, list):
        row_order_list = order
        column_order_list = order
    # else sort alphabetically
    elif order == "alpha":
        row_order_list = sorted(matrix_df.index)
        column_order_list = sorted(matrix_df.columns)
    elif order is None:
        pass
    else:
        print("order param can only be alpha or a list")

    if isinstance(order, list) or order == "alpha":
        matrix_df = matrix_df.loc[row_order_list, column_order_list]

    # # For trouble shooting
    # print(matrix_df)

    return matrix_df


"""
Can further test the subgroup matrix by manually modifying the order as follows:
row_index_list = matrix.index.tolist()
order_list = ['Lunch-Fri', 'Dinner-Sat', 'Dinner-Sun', 'Dinner-Thur', 'Lunch-Thur', 'Dinner-Fri']

matrix_sorted = matrix.loc[order_list]
"""


def single_group(df, group_col=None, test=None):
    """
    Logic for obtaining p matrix. This is for tests that only outputs a single column with categorical data.
    :param test:
    :param df:
    :param group_col:
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
