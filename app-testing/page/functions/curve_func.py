import streamlit as st
from matplotlib import pyplot as plt


class Plot_Logic:

    def label_plot_options(
        label_options,
        plot_title_size,
        plot_title,
        font,
        axis_fontsize,
        xlabel,
        ylabel,
        ymax,
        ymin,
    ):
        """
        Function to organize plat label  options
        """
        if label_options:
            # set the font name for a font family
            if font is None:
                plt.rcParams.update({"font.sans-serif": "DejaVu Sans"})
            else:
                plt.rcParams.update({"font.sans-serif": font})

            # Default label sizes
            plot_title_size = plot_title_size if plot_title_size else 16
            axis_fontsize = axis_fontsize if axis_fontsize else 14

            # Y axis limit
            ymax = int(ymax) if ymax is not None else None
            ymin = int(ymin) if ymin is not None else None

            # Return options
            options = {
                "plot_title": plot_title,
                "plot_title_size": plot_title_size,
                "axis_fontsize": axis_fontsize,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "ymax": ymax,
                "ymin": ymin,
            }
            return options
        else:
            options = {
                "plot_title": None,
                "plot_title_size": 16,
                "axis_fontsize": 14,
                "xlabel": "",
                "ylabel": "",
                "ymax": None,
                "ymin": None,
            }
            return options

    def label_options_true(label_options):
        """
        Function to for label_options input
        """
        plot_title = st.sidebar.text_input(
            label="Plot Title", placeholder="Input Plot Title"
        )
        font = st.sidebar.text_input(
            label="Figure Font", value=None, placeholder="DejaVu Sans"
        )
        plot_title_size = st.sidebar.number_input(
            label="Title Size", value=None, placeholder="16"
        )
        xlabel = st.sidebar.text_input(
            label="X-Axis Label (Default in µM)", placeholder="Input X Label"
        )
        ylabel = st.sidebar.text_input(
            label="Y-Axis Label", placeholder="Input Y Label"
        )
        axis_label_fontsize = st.sidebar.number_input(
            label="Axis Fontsize", value=None, placeholder="14"
        )
        ymax = st.sidebar.number_input(
            label="Y-Axis Max", value=None, placeholder="Y Max"
        )
        ymin = st.sidebar.number_input(
            label="Y-Axis Min", value=None, placeholder="Y Min"
        )

        return (
            label_options,
            plot_title_size,
            plot_title,
            font,
            axis_label_fontsize,
            xlabel,
            ylabel,
            ymax,
            ymin,
        )
