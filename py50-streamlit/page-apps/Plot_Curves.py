import streamlit as st
import io
import pandas as pd
import matplotlib.pyplot as plt
from py50.plotcurve import PlotCurve
from py50.plot_settings import CBMARKERS, CBPALETTE

# # Set page config
# st.set_page_config(page_title='py50: Plot Curves', page_icon='📈', layout='centered')

# Adjust hyperlink colorscheme
links = """<style>
a:link , a:visited{
color: 3081D0;
background-color: transparent;
}

a:hover,  a:active {
color: forestgreen;
background-color: transparent;
}
"""
st.markdown(links, unsafe_allow_html=True)


# housekeeping functions
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
    ylabel = st.sidebar.text_input(label="Y-Axis Label", placeholder="Input Y Label")
    axis_label_fontsize = st.sidebar.number_input(
        label="Axis Fontsize", value=None, placeholder="14"
    )
    ymax = st.sidebar.number_input(label="Y-Axis Max", value=None, placeholder="Y Max")
    ymin = st.sidebar.number_input(label="Y-Axis Min", value=None, placeholder="Y Min")

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


def line_options(line_color_option, line_color, line_width, marker):
    """
    Function for line options
    """
    if line_color_option:
        if line_color:
            line_color = line_color
        else:
            line_color = "Black"
        if line_width:
            line_width = line_width
        else:
            line_width = 1.5
        if marker:
            marker = marker
        else:
            marker = "o"

        return line_color, line_width, marker


def xscale_options(xoptions):
    """
    Function for xscale options
    """
    if xoptions is True:
        conc_unit = st.sidebar.radio("Set units of X Axis", ["µM", "nM"])
        xscale = st.sidebar.checkbox(label="X Axis Linear Scale")
        xscale_ticks_input = st.sidebar.text_input(
            label="Set X-Axis boundaries (i.e. the exponent)",
            placeholder="separate number with comma",
        )
        if xscale is True:
            xscale = "linear"
        else:
            xscale = "log"
        if xscale_ticks_input:
            # Split the user input into two values
            xscale_min, xscale_max = xscale_ticks_input.split(",")
            # Convert the strings to floats
            xscale_min = float(xscale_min)
            xscale_max = float(xscale_max)
            xscale_ticks = (xscale_min, xscale_max)
        else:
            xscale_ticks = (-2.5, 2.5)

        return conc_unit, xscale, xscale_ticks


def hline_vline_logic(hline, hline_color, vline, vline_color):
    """
    Logic for the hline and vline highlights
    """
    hline_highlight = st.sidebar.checkbox(label="Horizontal Line")
    if hline_highlight is True:
        hline = st.sidebar.number_input(
            label="Response Percentage", value=None, placeholder="0"
        )
        hline = hline  # Overwrite hline
        hline_color = st.sidebar.text_input(
            label="Hline Color (Color Name or Hex Code)",
            value="gray",
            placeholder="Color Name or Hex Code",
        )
        hline_color = hline_color  # Overwrite color
    if hline_color == "":
        hline_color = "gray"
        # st.write('## :red[I need a hline color bro!]') # for optional laughs

    vline_highlight = st.sidebar.checkbox(label="Vertical Line")
    if vline_highlight is True:
        vline = st.sidebar.number_input(
            label="Concentration (Must Match X-Axis Units)",
            value=None,
            placeholder="X-Axis",
        )
        vline = vline  # Overwrite hline
        vline_color = st.sidebar.text_input(
            label="Vline Color (Color Name or Hex Code)",
            value="gray",
            placeholder="Color Name or Hex Code",
        )
        vline_color = vline_color  # Overwrite color
    if vline_color == "":
        vline_color = "gray"
        # st.write('## :red[I need a vline color bro!]') # for optional laughs

    return hline, hline_color, vline, vline_color


def legend_logic(legend):
    """
    Logic function for determining the legend location.
    Legend input will be checked with acceptable list from matplotlib.
    """
    if legend is True:
        # Set list of legend options to be checked
        legend_list = [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
        ]
        st.sidebar.subheader("Legend Options")
        legend_loc = st.sidebar.text_input(
            label="Legend Location", placeholder="best", value="best"
        )

        # Logic to ensure it fits with matplotlib requirements, mapped to legend_list
        if any(string in legend_loc for string in legend_list):
            legend_loc = legend_loc
        else:
            error = st.write(
                "Legend Location can only use the following (input without quotes): ",
                legend_list,
            )

    return legend_loc


def download_button(fig, file_name):
    # Figure must be converted into a temporary file in memory
    buf = io.BytesIO()
    # plt.savefig(buf, format='png', dpi=300)
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # Create a download button
    st.download_button(
        "Download Figure", data=buf.read(), file_name=file_name, mime="image/png"
    )


def plot_program(df=None, paste=True):
    """
    Code for plotting. there are if/else statements nested within depending on the 3 types of plots
    """
    # Logic based on paste or CSV input
    global fig_type
    if paste is True:
        # Select columns for calculation
        drug_query = df
        col_header = drug_query.columns.to_list()
        drug_name = st.selectbox("Drug Name:", (col_header))
        compound_conc = st.selectbox(
            "Drug Concentration:", (col_header), index=1
        )  # Index to auto select column
        ave_response = st.selectbox(
            "Average Response column:", (col_header), index=2
        )  # Index to auto select column
    else:
        drug_query = df
        col_header = drug_query.columns.to_list()
        drug_name = st.selectbox("Drug Name:", (col_header))
        compound_conc = st.selectbox("Drug Concentration:", (col_header))
        ave_response = st.selectbox("Average Response column:", (col_header))

    # Set conditions before calculations
    conditions = {drug_name, compound_conc, ave_response}

    if len(conditions) != 3:
        st.write("### :red[Select Drug, Concentration, and Response Columns!]")
    else:
        st.write("## Selected Columns")

        # Ensure columns are float
        drug_query[compound_conc] = drug_query[compound_conc].astype(float)
        drug_query[ave_response] = drug_query[ave_response].astype(float)

        df_calc = drug_query.filter(
            items=(drug_name, compound_conc, ave_response), axis=1
        )

    st.write("**Selected Columns Check**")
    df_calc = drug_query.filter(items=(drug_name, compound_conc, ave_response), axis=1)
    st.data_editor(df_calc, num_rows="dynamic")

    plot_data = PlotCurve(drug_query)

    if len(conditions) != 3:
        pass
    else:
        # figure type
        st.write("### :rainbow[Select Figure Type]")
        fig_type = st.radio(
            "label will be collapsed",
            ["Single Plot", "Multi-Curve Plot", "Grid Plot"],
            captions=[
                "Your Classic Vanilla Dose-Response Plot",
                "One Plot, All Curves",
                "Multiple Plots, But All Tidy",
            ],
            label_visibility="collapsed",
        )

    # Drug name conditions if figure type is Single PLot
    if fig_type == "Single Plot":
        # Confirm DataFrame only contains 1 drug
        if len(df_calc[drug_name].unique()) == 1:
            name = df_calc[drug_name].unique()
            drug_name = ", ".join(map(str, name))
            st.markdown("**Only 1 Drug in table:**")
            st.write("Looking at: ", drug_name)
        else:
            st.markdown("## 🚨Multiple Drugs Detected in Table!🚨")
            st.markdown(
                "Input target drug name or try the **Multi-Curve Plot** or the **Grid Plot**"
            )

            # constrain text_input into column to match figure
            (
                col1,
                col2,
            ) = st.columns(2)
            with col1:
                drug_name = st.text_input(
                    label="Input Target Drug Name", placeholder="Input Drug Name"
                )
            with col2:
                st.header("")

        # Add plotting options to the sidebar
        st.sidebar.header(":green[Single Plot Options:]")

        # Label Options
        label_options = st.sidebar.checkbox(label="Labels")
        if label_options is True:
            # call in label options
            (
                label_options,
                plot_title_size,
                plot_title,
                font,
                axis_label_fontsize,
                xlabel,
                ylabel,
                ymax,
                ymin,
            ) = label_options_true(label_options=label_options)

            # logic based on plot options above
            label_plot_options(
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
        else:
            plot_title = None
            plot_title_size = 16
            axis_label_fontsize = 14
            xlabel = ""
            ylabel = ""
            ymax = None
            ymin = None

        # Set line color options if checked
        line_color_option = st.sidebar.checkbox(label="Line Options")
        if line_color_option is True:
            line_color = st.sidebar.text_input(
                label="Line Color", placeholder="Color Name or Hex Code"
            )
            line_width = st.sidebar.text_input(label="Line Width", placeholder="1.5")
            marker = st.sidebar.text_input(
                label="Marker Style", placeholder="matplotlib styles"
            )

            # logic based on line_color_options above
            line_color, line_width, marker = line_options(
                line_color_option, line_color, line_width, marker
            )
        else:
            line_color = "Black"
            line_width = 1.5
            marker = "o"

        # Legend settings
        legend = st.sidebar.checkbox(label="Legend")
        if legend is True:
            legend_loc = legend_logic(legend)
        else:
            legend_loc = "best"

        # xscale settings
        xoptions = st.sidebar.checkbox(label="X Axis Options")
        if xoptions is True:
            conc_unit, xscale, xscale_ticks = xscale_options(xoptions)
        else:
            conc_unit = "µM"
            xscale = "log"
            xscale_ticks = (-2.5, 2.5)

        # Box settings
        highlight_options = st.sidebar.checkbox(label="Highlight IC Value")
        if highlight_options is True:
            box_highlight = st.sidebar.checkbox(label="Box Highlight")

            # Set variables that will be changed by box setting logic
            box = False
            box_color = "gray"
            box_intercept = 50
            x_concentration = None
            hline = None
            hline_color = "gray"
            vline = None
            vline_color = "gray"

            if box_highlight is True:
                # Add logic for checked box_highlight
                box = True
                # Logic for box color
                box_color = st.sidebar.text_input(
                    label="Box Color", placeholder="Color Name or Hex Code"
                )
                if box_color == "":
                    box_color = "gray"
                else:
                    box_color = box_color
                # Logic for box intercept
                box_intercept = st.sidebar.number_input(
                    label="Response Percentage", value=50, placeholder="50"
                )
                if box_intercept:
                    box_intercept = box_intercept
                # Logic for specific concentration
                x_concentration = st.sidebar.number_input(
                    label="Optional: Highlight By Specific Concentration (will override "
                    "Y-Axis, input must match x-axis units)",
                    value=None,
                    placeholder=None,
                )
                if x_concentration:
                    x_concentration = x_concentration
            else:
                box = False

            # Logic for hline and vline
            hline, hline_color, vline, vline_color = hline_vline_logic(
                hline, hline_color, vline, vline_color
            )
        else:
            # Set variables that will be changed by logic
            box = False
            box_color = "gray"
            box_intercept = 50
            x_concentration = None
            hline = None
            hline_color = "gray"
            vline = None
            vline_color = "gray"

        fig_width = st.sidebar.slider(
            label="Figure Width:", min_value=1, max_value=50, value=6
        )
        fig_height = st.sidebar.slider(
            label="Figure Height:", min_value=1, max_value=50, value=4
        )

        # py50 plot function
        figure = plot_data.single_curve_plot(
            concentration_col=compound_conc,
            response_col=ave_response,
            plot_title=plot_title,
            plot_title_size=plot_title_size,
            drug_name=drug_name,
            xlabel=xlabel,
            ylabel=ylabel,
            ymax=ymax,
            ymin=ymin,
            axis_fontsize=axis_label_fontsize,
            legend=legend,
            legend_loc=legend_loc,
            xscale=xscale,
            conc_unit=conc_unit,
            xscale_ticks=xscale_ticks,
            line_color=line_color,
            line_width=line_width,
            marker=marker,
            box=box,
            box_color=box_color,
            box_intercept=box_intercept,
            conc_target=x_concentration,
            hline=hline,
            hline_color=hline_color,
            vline=vline,
            vline_color=vline_color,
            figsize=(fig_width, fig_height),
            verbose=True,
        )
        # Display figure
        # To reduce the size of the generated figure from stretching the width of the screen due to page layout set to
        # wide, a column is inserted to 'constrain' the image.
        (
            col1,
            col2,
        ) = st.columns(2)
        with col1:
            if conc_unit == "nM":
                st.markdown("Plot scale will be in nM")
            else:
                st.markdown("Plot scale will be in µM")
            st.pyplot(figure)
        with col2:
            st.header("")

        # Add download button
        download_button(fig=figure, file_name="single_curve.png")

    elif fig_type == "Multi-Curve Plot":
        # Confirm DataFrame only contains multiple drugs
        if len(df_calc[drug_name].unique()) > 1:
            name = df_calc[drug_name].unique()
            name = ", ".join(map(str, name))
            st.markdown("### Multiple Drugs Detected in File:")
            st.write("Looking at: ", name)
            # st.write('**Note** Input drugs are converted into µM by default')  # todo warning message double check
        elif len(df_calc[drug_name].unique()) <= 1:
            st.markdown("### 🚨Only 1 Drug Detected in File!!!!🚨")
        else:
            st.write("Is the input file correct?")

        # Add plotting options to the sidebar
        st.sidebar.header(":green[Multi-Curve Plot Options:]")

        # Label Options
        label_options = st.sidebar.checkbox(label="Labels")
        if label_options is True:
            # call in label options
            (
                label_options,
                plot_title_size,
                plot_title,
                font,
                axis_label_fontsize,
                xlabel,
                ylabel,
                ymax,
                ymin,
            ) = label_options_true(label_options=label_options)

            # logic based on plot options above
            label_plot_options(
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
        else:
            plot_title = None
            plot_title_size = 16
            axis_label_fontsize = 14
            xlabel = ""
            ylabel = ""
            ymax = None
            ymin = None

        # Set line color options if checked
        line_color_option = st.sidebar.checkbox(label="Line Options")
        if line_color_option is True:
            line_color = st.sidebar.text_input(
                label="Line Color (separate by comma)",
                placeholder="Color Name or Hex Code",
            )
            line_width = st.sidebar.text_input(label="Line Width", placeholder="1.5")
            marker = st.sidebar.text_input(
                label="Marker Style (separate by comma)",
                placeholder="matplotlib styles",
            )
            if line_color:
                line_color = line_color
                line_color = [
                    word.strip() for word in line_color.split(",")
                ]  # Convert string of words into list
            else:
                line_color = CBPALETTE
            if line_width:
                line_width = line_width
            else:
                line_width = 1.5
            if marker:
                marker = marker
                marker = [word.strip() for word in marker.split(",")]
            else:
                marker = CBMARKERS
        else:
            line_color = CBPALETTE
            line_width = 1.5
            marker = CBMARKERS

        # Legend Settings
        legend = st.sidebar.checkbox(label="Legend")
        if legend is True:
            legend_loc = legend_logic(legend)
        else:
            legend_loc = "best"

        # xscale settings
        xoptions = st.sidebar.checkbox(label="X Axis Options")
        if xoptions is True:
            conc_unit = st.sidebar.radio("Set units of X Axis", ["µM", "nM"])
            xscale = st.sidebar.checkbox(label="X Axis Linear Scale")
            xscale_ticks_input = st.sidebar.text_input(
                label="Set X-Axis boundaries", placeholder="separate number with comma"
            )
            if xscale is True:
                xscale = "linear"
            else:
                xscale = "log"
            if xscale_ticks_input:
                # Split the user input into two values
                xscale_min, xscale_max = xscale_ticks_input.split(",")
                # Convert the strings to floats
                xscale_min = float(xscale_min)
                xscale_max = float(xscale_max)
                xscale_ticks = (xscale_min, xscale_max)
            else:
                xscale_ticks = (-2.5, 2.5)
        else:
            conc_unit = "µM"
            xscale = "log"
            xscale_ticks = (-2.5, 2.5)

        # Set variables that will be changed by box setting logic below
        box = False
        box_color = "gray"
        box_intercept = 50
        x_concentration = None
        hline = None
        hline_color = "gray"
        vline = None
        vline_color = "gray"

        # Box settings
        highlight_options = st.sidebar.checkbox(label="Highlight IC Value")
        if highlight_options is True:
            box_target = st.sidebar.text_input(
                label="Drug Target to Highlight", placeholder="Drug Name"
            )
            box_color = st.sidebar.text_input(
                label="Box Color", placeholder="Color Name or Hex Code"
            )
            box_intercept = st.sidebar.number_input(
                label="Response Percentage", value=None, placeholder="50"
            )

            if box_target:
                box_target = box_target
            else:
                box_target = None
            if box_color:
                box_color = box_color
            else:
                box_color = "gray"
            if box_intercept:
                box_intercept = box_intercept
            else:
                box_intercept = 50

            # Logic for hline and vline
            hline, hline_color, vline, vline_color = hline_vline_logic(
                hline, hline_color, vline, vline_color
            )
        else:
            box_target = False
            box_color = "gray"
            box_intercept = 50

        fig_width = st.sidebar.slider(
            label="Figure Width:", min_value=1, max_value=50, value=6
        )
        fig_height = st.sidebar.slider(
            label="Figure Height:", min_value=1, max_value=50, value=4
        )

        figure = plot_data.multi_curve_plot(
            concentration_col=compound_conc,
            response_col=ave_response,
            name_col=drug_name,
            plot_title=plot_title,
            plot_title_size=plot_title_size,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            conc_unit=conc_unit,
            xscale_ticks=xscale_ticks,
            ymax=ymax,
            ymin=ymin,
            axis_fontsize=axis_label_fontsize,
            line_color=line_color,
            marker=marker,
            line_width=line_width,
            legend=legend,
            legend_loc=legend_loc,
            box_target=box_target,
            box_color=box_color,
            box_intercept=box_intercept,
            hline=hline,
            hline_color=hline_color,
            vline=vline,
            vline_color=vline_color,
            figsize=(fig_width, fig_height),
        )

        # Display figure
        (
            col1,
            col2,
        ) = st.columns(2)
        with col1:
            if conc_unit == "nM":
                st.markdown("Plot scale will be in nM")
            else:
                st.markdown("Plot scale will be in µM")
            st.pyplot(figure)
        with col2:
            st.header("")

        # Add download button
        download_button(fig=figure, file_name="multi_curve.png")

    # Grid options
    elif fig_type == "Grid Plot":
        # Confirm DataFrame only contains multiple drugs
        if len(df_calc[drug_name].unique()) > 1:
            name = df_calc[drug_name].unique()
            name = ", ".join(map(str, name))
            st.markdown("### Multiple Drugs Detected in File:")
            st.write("Looking at: ", name)
        elif len(df_calc[drug_name].unique()) % 2 != 0:
            st.markdown("### 🚨Odd Number of Drugs!!!!🚨")
        else:
            st.write("Is the input file correct?")

        # Add plotting options to the sidebar
        st.sidebar.header(":green[Grid Plot Options:]")

        # Label Options
        label_options = st.sidebar.checkbox(label="Labels")
        if label_options is True:
            # call in label options
            (
                label_options,
                plot_title_size,
                plot_title,
                font,
                axis_label_fontsize,
                xlabel,
                ylabel,
                ymax,
                ymin,
            ) = label_options_true(label_options=label_options)

            # logic based on plot options above
            label_plot_options(
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
        else:
            plot_title = None
            plot_title_size = 16
            axis_label_fontsize = 14
            xlabel = ""
            ylabel = ""
            ymax = None
            ymin = None

        # Set line color options if checked
        line_color_option = st.sidebar.checkbox(label="Line Options")
        if line_color_option is True:
            line_color = st.sidebar.text_input(
                label="Line Color (separate by comma)",
                placeholder="Color Name or Hex Code",
            )
            line_width = st.sidebar.text_input(label="Line Width", placeholder="1.5")
            marker = st.sidebar.text_input(
                label="Marker Style (separate by comma)",
                placeholder="matplotlib styles",
            )
            if line_color:
                line_color = line_color
                line_color = [
                    word.strip() for word in line_color.split(",")
                ]  # Convert string of words into list
            else:
                line_color = CBPALETTE
            if line_width:
                line_width = line_width
            else:
                line_width = 1.5
            if marker:
                marker = marker
                marker = [word.strip() for word in marker.split(",")]
            else:
                marker = CBMARKERS
        else:
            line_color = CBPALETTE
            line_width = 1.5
            marker = CBMARKERS

        # xscale settings
        xoptions = st.sidebar.checkbox(label="X Axis Options")
        if xoptions is True:
            conc_unit = st.sidebar.radio("Set units of X Axis", ["µM", "nM"])
            xscale = st.sidebar.checkbox(label="X Axis Linear Scale")
            xscale_ticks_input = st.sidebar.text_input(
                label="Set X-Axis boundaries", placeholder="separate number with comma"
            )
            if xscale is True:
                xscale = "linear"
            else:
                xscale = "log"
            if xscale_ticks_input:
                # Split the user input into two values
                xscale_min, xscale_max = xscale_ticks_input.split(",")
                # Convert the strings to floats
                xscale_min = float(xscale_min)
                xscale_max = float(xscale_max)
                xscale_ticks = (xscale_min, xscale_max)
            else:
                xscale_ticks = (-2.5, 2.5)
        else:
            conc_unit = "µM"
            xscale = "log"
            xscale_ticks = (-2.5, 2.5)

        # Set variables that will be changed by logic
        box_highlight = False
        box = False
        box_color = "gray"
        box_intercept = 50
        x_concentration = None
        hline = None
        hline_color = "gray"
        vline = None
        vline_color = "gray"

        # Box settings
        highlight_options = st.sidebar.checkbox(label="Highlight IC Value")
        if highlight_options is True:
            box_highlight = st.sidebar.checkbox(label="Box Highlight")

            if box_highlight is True:
                # Add logic for checked box_highlight
                box = True
                # Logic for box color
                box_color = st.sidebar.text_input(
                    label="Box Color", placeholder="Color Name or Hex Code"
                )
                if box_color == "":
                    box_color = "gray"
                else:
                    box_color = box_color
                # Logic for box intercept
                box_intercept = st.sidebar.number_input(
                    label="Response Percentage", value=50, placeholder="50"
                )
                if box_intercept:
                    box_intercept = box_intercept
                # Logic for specific concentration
                x_concentration = st.sidebar.number_input(
                    label="Optional: Highlight By Specific Concentration (will override "
                    "Y-Axis, input must match x-axis units)",
                    value=None,
                    placeholder=None,
                )
                if x_concentration:
                    x_concentration = x_concentration
            else:
                box = False

                # Logic for hline and vline
            hline, hline_color, vline, vline_color = hline_vline_logic(
                hline, hline_color, vline, vline_color
            )
        else:
            box = False
            box_color = "gray"
            box_intercept = 50
            x_concentration = None
            hline = None
            hline_color = "gray"
            vline = None
            vline_color = "gray"

        fig_width = st.sidebar.slider(
            label="Figure Width:", min_value=1, max_value=50, value=8
        )
        fig_height = st.sidebar.slider(
            label="Figure Height:", min_value=1, max_value=50, value=4
        )

        figure = plot_data.grid_curve_plot(
            concentration_col=compound_conc,
            response_col=ave_response,
            name_col=drug_name,
            plot_title=plot_title,
            plot_title_size=plot_title_size,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            conc_unit=conc_unit,
            xscale_ticks=xscale_ticks,
            ymax=ymax,
            ymin=ymin,
            line_color=line_color,
            line_width=line_width,
            box=box_highlight,
            box_color=box_color,
            box_intercept=box_intercept,
            hline=hline,
            hline_color=hline_color,
            vline=vline,
            vline_color=vline_color,
            figsize=(fig_width, fig_height),
            verbose=True,
        )
        # Display figure
        (
            col1,
            col2,
        ) = st.columns(2)
        with col1:
            if conc_unit == "nM":
                st.markdown("Plot scale will be in nM")
            else:
                st.markdown("Plot scale will be in µM")
            st.pyplot(figure)
        with col2:
            st.header("")

        # Add download button
        download_button(fig=figure, file_name="grid_curve.png")
    else:
        st.write(
            "## :red[**Something is wrong with the app! It is the end of days!!!!**]"
        )


"""
Page layout begins below
"""
# # add logo
# st.sidebar.image('img/py50_logo_only.png', width=150)

# Page text
datasets = "https://github.com/tlint101/py50/tree/main/dataset"
st.markdown("# Generate Dose-Response Curves")
st.write(
    "This page will plot a dose-response curve. The plot points will be calculated for each query."
)
st.write("The program requires at least three columns:")
st.write("- Drug Name")
st.write("- Drug Concentration")
st.write("- Average Response")
st.write("Sample datasets can be found [here](%s)" % datasets)
st.write("")

st.markdown("## Select an option to get started:")

# User selects type of interface
option = st.radio(
    "Options are paste or .csv file upload",
    (
        "Paste Data",
        "Upload CSV File",
    ),
)

if option == "Upload CSV File":
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload .csv file")

    # Check if a CSV file has been uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        drug_query = pd.read_csv(uploaded_file)
        st.write("## Input Table")
        st.data_editor(
            drug_query, num_rows="dynamic"
        )  # visualize dataframe in streamlit app
    else:
        # Display a message if no CSV file has been uploaded
        st.warning("Please upload a .csv file.")

    # Select columns for calculation
    if uploaded_file is not None:  # nested in if/else to remove initial traceback error
        st.write("## Select Columns for Calculation")
        plot_program(df=drug_query, paste=False)

# Editable DataFrame
elif option == "Paste Data":
    st.markdown("### Paste Data in Table:")
    # Make dummy dataframe
    df = pd.DataFrame(
        [
            {"Drug Name": "", "Concentration": "", "Response": ""},
        ]
    )

    edited_df = st.data_editor(df, num_rows="dynamic")

    if (edited_df == "").all().all():
        st.markdown(":red[**Table is currently empty**]")
    else:
        plot_program(df=edited_df, paste=True)
