"""
The following script holds plot logic for the indicated tests. This was created to reduce the "look" of the stats file
for maintainability.
"""

import inspect
from itertools import combinations
from typing import Optional
import numpy as np
import pandas as pd


def _normalize_units(unit: Optional[str] = None) -> str:
    """Normalize concentration unit aliases to 'nM', 'µM', or 'pM'."""
    if unit is None or unit in ("nM", "nm"):
        return "nM"
    if unit in ("µM", "uM", "um", "μM", "μm", "µm"):
        return "µM"
    if unit in ("pM", "pm"):
        return "pM"
    raise ValueError(
        f"Unsupported concentration unit {unit!r}. "
        "Supported units are 'nM', 'µM' ('uM'), or 'pM'."
    )


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
    import seaborn as sns

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
    Convert Tukey HSD p-values into significance star labels.

    :param test_value: pandas.DataFrame
        Output DataFrame from ``Stats.get_tukey()`` / ``pingouin.pairwise_tukey()``.
    :return: list[str]
    """
    pvalue = [star_value(value) for value in test_value["p_tukey"].tolist()]
    return pvalue


def gameshowell_plot_logic(test_value):
    """

    :param test_value:
    :return:
    """
    pvalue = [star_value(value) for value in test_value.pval.tolist()]
    return pvalue


def multi_group(df, group_col1=None, group_col2=None, test=None, order=None):
    """
    Logic for obtaining p matrix. This is for tests outputs a multiple column with categorical data.
    :param df: pandas.DataFrame
        Input DataFrame.
    :param group_col1: String
        Column with first group of data. Defaults to ``"A"``.
    :param group_col2: String
        Column with second group of data. Defaults to ``"B"``.
    :param test: String
        Test type. This will extract pvalue column from statistics table.
    :param order: List or String
        Reorder the groups for the final table. If input is string ``"alpha"``, the order of the groups will be
        alphabetized.

    :return: pandas.DataFrame
    """
    test_p_col = {
        "tukey": "p_tukey",
        "gameshowell": "pval",
        "mannu": "p-val",
        "wilcoxon": "p-val",
        "pairwise-parametric": "p_unc",
        "pairwise-nonparametric": "p_unc",
        "pairwise-rm": "p_unc",
        "pairwise-mixed": "p_unc",
    }
    if test not in test_p_col:
        raise ValueError(
            f"Unsupported test {test!r}. Supported tests are: {sorted(test_p_col)}."
        )

    p_col = test_p_col[test]
    if group_col1 is None:
        group_col1 = "A"
    if group_col2 is None:
        group_col2 = "B"

    if df.empty:
        return pd.DataFrame(dtype=float)

    def _format_label(value):
        if isinstance(value, tuple):
            return "-".join(str(part) for part in value)
        return value

    # Deterministic first-seen order (no random permutation, no eval())
    labels = list(
        dict.fromkeys(
            [_format_label(v) for v in df[group_col1]]
            + [_format_label(v) for v in df[group_col2]]
        )
    )
    matrix_df = pd.DataFrame(1.0, index=labels, columns=labels, dtype=float)

    for _, row in df.iterrows():
        lbl1 = _format_label(row[group_col1])
        lbl2 = _format_label(row[group_col2])
        pval = float(row[p_col])
        matrix_df.loc[lbl1, lbl2] = pval
        matrix_df.loc[lbl2, lbl1] = pval

    # Sort table order
    if order is None:
        pass
    elif order == "alpha":
        sorted_labels = sorted(matrix_df.index, key=str)
        matrix_df = matrix_df.loc[sorted_labels, sorted_labels]
    elif isinstance(order, list):
        matrix_df = matrix_df.loc[order, order]
    else:
        raise ValueError(
            f"Invalid order {order!r}. order param must be None, 'alpha', or a list."
        )

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

    # Create a float64 matrix filled with 1.0 (not significant)
    matrix_df = pd.DataFrame(1.0, index=groups, columns=groups, dtype=float)

    # Fill the matrix with values
    for _, row in df.iterrows():
        matrix_df.loc[row[group_col], row[group_col]] = float(row[p_col])

    return matrix_df


def get_kwargs(func):
    """To obtain kwargs for a given submodule. Usage is the following:
    sns_kwargs = get_kwargs(sns.boxplot)
    """
    return inspect.signature(func).parameters.keys()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
