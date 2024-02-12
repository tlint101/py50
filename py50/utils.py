"""
The following script holds plot logic for the indicated tests. This was created to reduce the "look" of the stats file
for maintainability.
"""

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
