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
    def get_normality(df, group_col=None, value_col=None, **kwargs):
        """
        Test data normality

        :param df:
        :param value_col:
        :param group_col:
        :param method:
        :return:
        """

        result_df = pg.normality(data=df, dv=value_col, group=group_col, **kwargs)
        return result_df

    @staticmethod
    def get_homoscedasticity(df, group_col=None, value_col=None,  **kwargs):
        """
        Test for data variance.

        :param df:
        :param value_col:
        :param group_col:
        :param kwargs:
        :return:
        """

        result_df = pg.homoscedasticity(data=df, dv=value_col, group=group_col, **kwargs)
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
    def get_anova(df, group_col=None, value_col=None, **kwargs):
        """
        Classic ANOVA

        :param df:
        :param value_col:
        :param group_col:
        :param type:
        :return:
        """

        result_df = pg.anova(data=df, dv=value_col, between=group_col, **kwargs)
        return result_df

    @staticmethod
    def get_welch_anova(df, group_col=None, value_col=None):
        result_df = pg.welch_anova(data=df, dv=value_col, between=group_col)
        return result_df

    @staticmethod
    def get_tukey(df, group_col=None, value_col=None, **kwargs):
        """

        :param df:
        :param value_col:
        :param group_col:
        :return:
        """
        result_df = pg.pairwise_tukey(data=df, dv=value_col, between=group_col, **kwargs)
        return result_df

    @staticmethod
    def get_gameshowell(df, group_col=None, value_col=None, **kwargs):
        result_df = pg.pairwise_gameshowell(data=df, dv=value_col, between=group_col, **kwargs)
        return result_df

    @staticmethod
    def get_rm_anova(df, value_col=None, within=None, subject=None, **kwargs):
        "Repeated measures anova"
        result_df = pg.rm_anova(
            data=df, dv=value_col, within=within, subject=subject, **kwargs
        )
        return result_df

    # todo add if/else for between and within. Rename as "factor"
    @staticmethod
    def get_pairwise_test(df, value_col=None, between=None, within=None, **kwargs):
        """
        Calculate pairwise_tests
        :param within:
        :param df:
        :param value_col:
        :param between:
        :param effsize:
        :param kwargs:
        :return:
        """
        result_df = pg.pairwise_tests(
            data=df, dv=value_col, between=between, within=within, **kwargs
        )
        return result_df

    @staticmethod
    def get_mixed_anova(df, value_col=None, within=None, subject=None, **kwargs):
        result_df = pg.mixed_anova(
            data=df, dv=value_col, within=within, subject=subject, **kwargs
        )
        return result_df

    """
    non-parametric tests below
    """

    @staticmethod
    def get_wilcoxon(df, group_col=None, value_col=None, **kwargs):
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
                value1 = value1.iloc[:min_length]
                value2 = value2.iloc[:min_length]

                # Perform Wilcoxon signed-rank test
                result = pg.wilcoxon(value1, value2, **kwargs)

                # Store the results in the list
                results_list.append(
                    {
                        "A": group1,
                        "B": group2,
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
    def get_mannu(df, group_col=None, value_col=None, subgroup=None, **kwargs):
        """
        Calculate Mann-Whitney U Test
        :param df:
        :param group_col:
        :param value_col:
        :return:
        """
        if subgroup:
            # Convert 'Name' and 'Status' columns to string
            df[group_col] = df[group_col].astype(str)
            df[subgroup] = df[subgroup].astype(str)
            df["subgroup"] = df[group_col] + "-" + df[subgroup]

            subgroup_list = df["subgroup"].unique().tolist()
            subgroup_df = df[df["subgroup"].isin(subgroup_list)].copy()

            # Get unique pairs between group and subgroup
            group = subgroup_df["subgroup"].unique()

            # Empty list to store results
            results_list = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    group1 = group[i]
                    group2 = group[j]
                    value1 = df[df["subgroup"] == group1][value_col]
                    value2 = df[df["subgroup"] == group2][value_col]

                    # Ensure same length for each condition
                    min_length = min(len(value1), len(value2))
                    value1 = value1.iloc[:min_length]
                    value2 = value2.iloc[:min_length]

                    # Perform Wilcoxon signed-rank test
                    result = pg.mwu(x=value1, y=value2, **kwargs)

                    # Store the results in the list
                    results_list.append(
                        {
                            "A": group1,
                            "B": group2,
                            "U-val": result["U-val"].iloc[0],
                            "p-val": result["p-val"].iloc[0],
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
                    value1 = value1.iloc[:min_length]
                    value2 = value2.iloc[:min_length]

                    # Perform Wilcoxon signed-rank test
                    result = pg.mwu(x=value1, y=value2)

                    # Store the results in the list
                    results_list.append(
                        {
                            "A": group1,
                            "B": group2,
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
    def get_kruskal(df, group_col=None, value_col=None, detailed=False):
        """
        Calculate Mann-Whitney U Test
        :param df:
        :param group_col:
        :param value_col:
        :return:
        """
        result_df = pg.kruskal(data=df, dv=value_col, between=group_col, detailed=detailed)
        return result_df

    # todo add if/else for between and within. Rename as "factor"
    @staticmethod
    def get_nonpara_test(
        df, dv=None, between=None, within=None, parametric=True, **kwargs
    ):
        """
        Posthoc test for nonparametric statistics. Used after Kruskal test.
        :param df:
        :param dv:
        :param between:
        :param within:
        :param parametric:
        :param kwargs:
        :return:
        """

        result_df = pg.pairwise_tests(
            data=df,
            dv=dv,
            between=between,
            within=within,
            parametric=parametric,
            **kwargs,
        )
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
        # todo add options for additional test and ensure format matches
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

    # todo, if box_plot_test is successful, use initializer
    # # Initializer for future use
    # def __init__(self, df):
    #     if not isinstance(df, pd.DataFrame):
    #         raise ValueError("Input must be a DataFrame")
    #     self.df = df

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
    def box_plot(
        df,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        subgroup=None,
        pair_order=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        return_df=None,
        **kwargs,
    ):
        """
        :param df: Input DataFrame.
        :param test: Name of test to use for calculations.
        :param group_col: Column containing groups.
        :param value_col: Column containing values. This is the dependent variable.
        :param group_order: List. Order the groups for in the plot.
        :param pair_order: List. Order of group pairs. This will modify the way the plot will be annotated.
        :param pvalue_order: List. Order the pvalue labels. This order must match the pairorder.
        :param palette: List. Palette used for the plot. Can be given as common color name or in hex code.
        :param orient: Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param return_df: Boolean to return dataframe of calculated results.
        :return:
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

        # Get plot variables
        pairs, palette, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            df,
            group_col,
            kwargs,
            pair_order,
            palette,
            test,
            value_col,
            valid_sns,
            valid_annot,
            subgroup,
            pair_hue=None,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # Set pairs for each hue/subgroup
        if "pair_hue" in kwargs:
            pairs = kwargs.get("pair_hue")

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.boxplot(
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                hue=subgroup,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                hue=subgroup,
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.boxplot(
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            # flip x and y annotations for horizontal orientation
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # Set custom annotations and annotate
        if pvalue_order:
            pvalue = pvalue_order

        annotator.set_custom_annotations(pvalue)
        annotator.annotate(**annotate_kwargs)

        if return_df:
            return test_df  # return calculated df. Change name for more description

    # todo update below plot with annot_kwargs
    @staticmethod
    def bar_plot(
        df,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        pair_order=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        return_df=None,
        **kwargs,
    ):
        """
        :param df: Input DataFrame.
        :param test: Name of test to use for calculations.
        :param group_col: Column containing groups.
        :param value_col: Column containing values. This is the dependent variable.
        :param group_order: List. Order the groups for in the plot.
        :param pair_order: List. Order of group pairs. This will modify the way the plot will be annotated.
        :param pvalue_order: List. Order the pvalue labels. This order must match the pairorder.
        :param palette: List. Palette used for the plot. Can be given as common color name or in hex code.
        :param orient: Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param return_df: Boolean to return dataframe of calculated results.
        :return:
        """
        # separate kwargs for sns and sns
        valid_sns = utils.get_kwargs(sns.boxplot)
        valid_annot = utils.get_kwargs(Annotator)
        print(valid_sns)

        # Set kwargs dictionary for line annotations
        annotate_kwargs = {}
        if "line_offset_to_group" in kwargs and "line_offset" in kwargs:
            # Get kwargs from input
            line_offset_to_group = kwargs["line_offset_to_group"]
            line_offset = kwargs["line_offset"]
            # Add to dictionary
            annotate_kwargs["line_offset_to_group"] = line_offset_to_group
            annotate_kwargs["line_offset"] = line_offset

        # Get plot variables
        pairs, palette, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            df,
            group_col,
            kwargs,
            pair_order,
            palette,
            test,
            value_col,
            valid_sns,
            valid_annot,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # Get plt kwargs to pass into the sns_kwargs
        capsize = kwargs.pop("capsize", None)  # Extract capsize if present
        ci = kwargs.pop("ci", None)
        if capsize is not None:
            sns_kwargs["capsize"] = capsize  # Update sns_kwargs with capsize
        if ci is not None:  # deprecated in newer sns version
            sns_kwargs["ci"] = ci

        print(sns_kwargs)

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.barplot(
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.barplot(
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            # flip x and y annotations for horizontal orientation
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # Set custom annotations and annotate
        if pvalue_order:
            pvalue = pvalue_order
        annotator.set_custom_annotations(pvalue)
        annotator.annotate(**annotate_kwargs)

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def violin_plot(
        df,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        pair_order=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        return_df=None,
        **kwargs,
    ):
        """
        :param df: Input DataFrame.
        :param test: Name of test to use for calculations.
        :param group_col: Column containing groups.
        :param value_col: Column containing values. This is the dependent variable.
        :param group_order: List. Order the groups for in the plot.
        :param pair_order: List. Order of group pairs. This will modify the way the plot will be annotated.
        :param pvalue_order: List. Order the pvalue labels. This order must match the pairorder.
        :param palette: List. Palette used for the plot. Can be given as common color name or in hex code.
        :param orient: Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param return_df: Boolean to return dataframe of calculated results.
        :return:
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

        # Get plot variables
        pairs, palette, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            df,
            group_col,
            kwargs,
            pair_order,
            palette,
            test,
            value_col,
            valid_sns,
            valid_annot,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.violinplot(
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.violinplot(
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            # flip x and y annotations for horizontal orientation
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # Set custom annotations and annotate
        if pvalue_order:
            pvalue = pvalue_order
        annotator.set_custom_annotations(pvalue)
        annotator.annotate(**annotate_kwargs)

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def swarmplot(
        df,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        pair_order=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        return_df=None,
        **kwargs,
    ):
        """
        :param df: Input DataFrame.
        :param test: Name of test to use for calculations.
        :param group_col: Column containing groups.
        :param value_col: Column containing values. This is the dependent variable.
        :param group_order: List. Order the groups for in the plot.
        :param pair_order: List. Order of group pairs. This will modify the way the plot will be annotated.
        :param pvalue_order: List. Order the pvalue labels. This order must match the pairorder.
        :param palette: List. Palette used for the plot. Can be given as common color name or in hex code.
        :param orient: Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param return_df: Boolean to return dataframe of calculated results.
        :return:
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

        # Get plot variables
        pairs, palette, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            df,
            group_col,
            kwargs,
            pair_order,
            palette,
            test,
            value_col,
            valid_sns,
            valid_annot,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.swarmplot(
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.swarmplot(
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            # flip x and y annotations for horizontal orientation
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # Set custom annotations and annotate
        if pvalue_order:
            pvalue = pvalue_order
        annotator.set_custom_annotations(pvalue)
        annotator.annotate(**annotate_kwargs)

        if return_df:
            return test_df  # return calculated df. Change name for more description

    # todo doublecheck if lineplot works
    @staticmethod
    def lineplot(
        df,
        test=None,
        group_col=None,
        value_col=None,
        group_order=None,
        pair_order=None,
        pvalue_order=None,
        palette=None,
        orient="v",
        return_df=None,
        **kwargs,
    ):
        """
        :param df: Input DataFrame.
        :param test: Name of test to use for calculations.
        :param group_col: Column containing groups.
        :param value_col: Column containing values. This is the dependent variable.
        :param group_order: List. Order the groups for in the plot.
        :param pair_order: List. Order of group pairs. This will modify the way the plot will be annotated.
        :param pvalue_order: List. Order the pvalue labels. This order must match the pairorder.
        :param palette: List. Palette used for the plot. Can be given as common color name or in hex code.
        :param orient: Orientation of the plot. Only "v" and "h" are for vertical and horizontal, respectively, is supported
        :param return_df: Boolean to return dataframe of calculated results.
        :return:
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

        # Get plot variables
        pairs, palette, pvalue, sns_kwargs, annot_kwargs, test_df = _plot_variables(
            df,
            group_col,
            kwargs,
            pair_order,
            palette,
            test,
            value_col,
            valid_sns,
            valid_annot,
        )

        # Set order for groups on plot
        if group_order:
            group_order = group_order

        # set orientation for plot and Annotator
        if orient == "v":
            ax = sns.lineplot(
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=group_col,
                y=value_col,
                order=group_order,
                verbose=False,
                orient="v",
                **annot_kwargs,
            )
        elif orient == "h":
            ax = sns.lineplot(
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                palette=palette,
                **sns_kwargs,
            )
            # flip x and y annotations for horizontal orientation
            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df,
                x=value_col,
                y=group_col,
                order=group_order,
                verbose=False,
                orient="h",
                **annot_kwargs,
            )
        else:
            raise ValueError("Orientation must be 'v' or 'h'!")

        # Set custom annotations and annotate
        if pvalue_order:
            pvalue = pvalue_order
        annotator.set_custom_annotations(pvalue)
        annotator.annotate(**annotate_kwargs)

        if return_df:
            return test_df  # return calculated df. Change name for more description

    @staticmethod
    def p_matrix(matrix_df, cmap=None, title=None, title_size=14, **kwargs):
        """
        Wrapper function for scikit_posthoc heatmap.
        :return:
        """
        if title:
            plt.title(title, fontsize=title_size)

        if cmap is None:
            cmap = ["1", "#fb6a4a", "#08306b", "#4292c6", "#c6dbef"]
            sp.sign_plot(matrix_df, cmap=cmap, **kwargs)
        else:
            sp.sign_plot(matrix_df, cmap=cmap, **kwargs)

        return plt

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


def _get_test(
    test,
    df=None,
    group_col=None,
    value_col=None,
    subgroup=None,
    pair_hue=None,
    **kwargs,
):
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

    global pairs
    if test == "tukey":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_tukey)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        test_df = Stats.get_tukey(df, value_col=value_col, group_col=group_col, **pg_kwargs)
        pvalue = [utils.star_value(value) for value in test_df["p-tukey"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    elif test == "gameshowell":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_gameshowell())
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        test_df = Stats.get_gameshowell(
            df, value_col=value_col, group_col=group_col, **pg_kwargs
        )
        pvalue = [utils.star_value(value) for value in test_df["pval"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    elif test == "ttest-within":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        test_df = Stats.get_pairwise_test(
            df, value_col=value_col, within=group_col, **pg_kwargs
        )
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    elif test == "ttest-between":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        test_df = Stats.get_pairwise_test(
            df, value_col=value_col, between=group_col, **pg_kwargs
        )
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    # todo find example to test ttest-mixed
    # requires BOTH between and within groups
    elif test == "ttest-mixed":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        test_df = Stats.get_pairwise_test(
            df, value_col=value_col, between=group_col, within=group_col, **pg_kwargs
        )
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    elif test == "wilcoxon":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.wilcoxon)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        test_df = Stats.get_wilcoxon(
            df, group_col=group_col, value_col=value_col, **pg_kwargs
        )
        pvalue = [utils.star_value(value) for value in test_df["p-val"].tolist()]
        # Obtain pairs and split them from Wilcox result DF for passing into Annotator
        pairs = []
        for item in test_df["Comparison"].tolist():
            parts = item.split("-")
            pairs.append((parts[0], parts[1]))

    # todo tested for mannu-test. Document and add for others
    # todo condense subgroup if portion
    elif test == "mannu":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.mwu)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # Obtain pairs and split them from Wilcox result DF for passing into Annotator
        if subgroup:
            # run test
            test_df = Stats.get_mannu(
                df,
                group_col=group_col,
                value_col=value_col,
                subgroup=subgroup,
                **pg_kwargs,
            )

            test_df = _get_pair_hue(test_df, hue=pair_hue)

            pvalue = [utils.star_value(value) for value in test_df["p-val"].tolist()]
            pairs = _get_pairs(test_df, hue=pair_hue)
        else:
            # run test
            test_df = Stats.get_mannu(
                df, group_col=group_col, value_col=value_col, **pg_kwargs
            )
            pvalue = [utils.star_value(value) for value in test_df["p-val"].tolist()]
            # Obtain pairs and split them from Wilcox result DF for passing into Annotator
            pairs = _get_pairs(test_df, hue=pair_hue)

    elif test == "para-ttest":
        # get kwargs
        valid_pg = utils.get_kwargs(pg.pairwise_tests)
        pg_kwargs = {key: value for key, value in kwargs.items() if key in valid_pg}

        # run test
        test_df = Stats.get_nonpara_test(
            df, dv=value_col, between=group_col, **pg_kwargs
        )
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
        pairs = [(a, b) for a, b in zip(test_df["A"], test_df["B"])]

    elif test == "kruskal":  # kurskal does not give posthoc. modify
        test_df = Stats.get_kruskal(df, value_col=value_col, group_col=group_col, detailed=False)
        pvalue = [utils.star_value(value) for value in test_df["p-unc"].tolist()]
    else:
        raise ValueError("Test not recognized!")

    # elif test == "ttest":
    #     test_df = Stats.get_t_test(df, paired=False, x=None, y=None, **kwargs) # todo determine how to select column to return as list

    return pvalue, test_df, pairs, subgroup


def _plot_variables(
    df,
    group_col,
    kwargs,
    pair_order,
    palette,
    test,
    value_col,
    valid_sns,
    valid_annot,
    subgroup=None,
    pair_hue=None,
):
    """
    Output plot variables for use inside plots in Plots() class
    :param df:
    :param group_col:
    :param kwargs:
    :param pair_order:
    :param palette:
    :param test:
    :param value_col:
    :return:
    """
    # get kwarg for sns plot
    sns_kwargs = {key: value for key, value in kwargs.items() if key in valid_sns}
    annot_kwargs = {key: value for key, value in kwargs.items() if key in valid_annot}

    # Run tests based on test parameter input
    if test is not None:
        pvalue, test_df, pairs, subgroup = _get_test(
            test=test,
            df=df,
            group_col=group_col,
            value_col=value_col,
            subgroup=subgroup,
            **kwargs,
        )
    else:
        raise NameError(
            "Must include a post-hoc test like: 'tukey', 'gameshowell', 'ptest', 'mannu', etc"
        )

    # set default color palette
    if palette:
        palette = palette(palette)

    # set custom pair order
    if pair_order:
        pairs = pair_order

    return pairs, palette, pvalue, sns_kwargs, annot_kwargs, test_df


# todo add documentation
def _get_pair_hue(df, hue=None):
    """Generate pairs by group_col and hue"""

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
    # Drop the combined column AB if not needed in the final output
    filtered_df.drop("AB", axis=1, inplace=True)
    return filtered_df

    # Define a function to split the string and create a tuple


# Add documentation
def _get_pairs(df, hue):
    # Convert DataFrame to a list of tuples
    pairs = [(a, b) for a, b in zip(df["A"], df["B"])]
    return pairs


if __name__ == "__main__":
    import doctest

    doctest.testmod()
