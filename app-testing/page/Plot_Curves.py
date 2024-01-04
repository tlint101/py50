import streamlit as st
import io
import pandas as pd
import matplotlib.pyplot as plt
from py50.plotcurve import PlotCurve
from py50.plot_settings import CBMARKERS, CBPALETTE

# Set page config
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

# # add logo
# st.sidebar.image('img/py50_logo_only.png', width=150)

# Page text
datasets = 'https://github.com/tlint101/py50/tree/main/dataset'
st.markdown('# Generate Dose-Response Curves')
st.write('This page will plot a dose-response curve. The plot points will be calculated for each query.')
st.write('The program requires at least three columns:')
st.write('- Drug Name')
st.write('- Drug Concentration')
st.write('- Average Response')
st.write('Sample datasets can be found [here](%s)' % datasets)
st.write('')

# Upload the CSV file
uploaded_file = st.file_uploader('Upload .csv file')

# Check if a CSV file has been uploaded
if uploaded_file is not None:
    # Section header
    st.write('## Input Table')

    # Read the CSV file into a DataFrame
    drug_query = pd.read_csv(uploaded_file)
    st.dataframe(drug_query, hide_index=True)  # visualize dataframe in streamlit app
else:
    # Display a message if no CSV file has been uploaded
    st.warning('Please upload a .csv file.')

if uploaded_file is not None:  # nested in if/else to remove initial traceback error
    # Section header
    st.write('## Select Columns for Calculation')

    # Select columns for calculation
    col_header = drug_query.columns.to_list()
    drug_name = st.selectbox('Drug Name:', (col_header))
    drug_conc = st.selectbox('Drug Concentration:', (col_header))
    response = st.selectbox('Average Response column:', (col_header))

    st.write('**Selected Columns**')
    df_calc = drug_query.filter(items=(drug_name, drug_conc, response), axis=1)
    st.dataframe(df_calc)

    plot_data = PlotCurve(drug_query)

    # figure type
    fig_type = st.radio(
        '**Select figure type**',
        ['Single Plot', 'Multi-Curve Plot', 'Grid Plot'])

    # todo place each plot type in own python file for maintainability
    # Drug name conditions if figure type is Single PLot
    if fig_type == 'Single Plot':
        # Confirm DataFrame only contains 1 drug
        if len(df_calc[drug_name].unique()) == 1:
            name = df_calc[drug_name].unique()
            drug_name = ', '.join(map(str, name))
            st.markdown('**Only 1 Drug in table:**')
            st.write('Looking at: ', drug_name)
        else:
            st.markdown('## 🚨Multiple Drugs Detected in File!🚨')
            st.markdown('Input target drug name or try the **Multi-Curve Plot** or the **Grid Plot**')
            drug_name = st.text_input(label='Input Target Drug Name', placeholder='Input Drug Name')

        # Add plotting options to the sidebar
        st.sidebar.header("Single Plot Options:")

        # Label Options
        label_options = st.sidebar.checkbox(label='Label Options')
        if label_options is True:
            # Plot title
            plot_title = st.sidebar.text_input(label='Plot Title', placeholder='Input Plot Title')
            font = st.sidebar.text_input(label='Figure Font', value=None, placeholder='DejaVu Sans')
            plot_title_size = st.sidebar.number_input(label='Title Size', value=None, placeholder='16')
            axis_fontsize = st.sidebar.number_input(label='Axis Fontsize', value=None, placeholder='14')
            xlabel = st.sidebar.text_input(label='X-Axis Label (Default in µM)', placeholder='Input X Label')
            ylabel = st.sidebar.text_input(label='Y-Axis Label', placeholder='Input Y Label')
            ylimit = st.sidebar.number_input(label="Y-Axis Limit", value=None, placeholder='Input Limit')

            # set the font name for a font family
            if font is None:
                plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
            else:
                plt.rcParams.update({'font.sans-serif': font})

            # Default label sizes
            if plot_title_size:
                plot_title_size = plot_title_size
            else:
                plot_title_size = 16
            if axis_fontsize:
                axis_fontsize = axis_fontsize
            else:
                axis_fontsize = 14

            # ylabel limit
            if ylimit is None:
                ylimit = None
            else:
                ylimit = int(ylimit)
        else:
            plot_title = None
            plot_title_size = 16
            axis_fontsize = 14
            xlabel = ''
            ylabel = ''
            ylimit = None

        # Set line color options if checked
        line_color_option = st.sidebar.checkbox(label='Line Options')
        if line_color_option is True:
            line_color = st.sidebar.text_input(label='Line Color', placeholder='Color Name or Hex Code')
            line_width = st.sidebar.text_input(label='Line Width', placeholder='1.5')
            marker = st.sidebar.text_input(label='Marker Style', placeholder='matplotlib styles')
            if line_color:
                line_color = line_color
            else:
                line_color = 'Black'
            if line_width:
                line_width = line_width
            else:
                line_width = 1.5
            if marker:
                marker = marker
            else:
                marker = 's'
        else:
            line_color = 'Black'
            line_width = 1.5
            marker = 'o'

        # Set Legend options if checked
        legend = st.sidebar.checkbox(label='Legend')
        if legend is True:
            st.sidebar.subheader('Legend Options')
            legend_loc = st.sidebar.text_input(label='Legend Location', placeholder='lower right')
            if legend_loc:
                legend_loc = legend_loc
            else:
                legend_loc = 'best'
        else:
            legend_loc = 'best'

        # xscale settings
        xoptions = st.sidebar.checkbox(label='X Axis Options')
        if xoptions is True:
            conc_unit = st.sidebar.radio(
                'Set units of X Axis',
                ['µM', 'nM'])
            xscale = st.sidebar.checkbox(label='X Axis Linear Scale')
            xscale_ticks_input = st.sidebar.text_input(label='Set X-Axis boundaries (i.e. the exponent)',
                                                       placeholder='separate number with comma')
            if conc_unit is 'µM':
                st.write('Plot scale is in µM!')
            else:
                st.write('Plot scale is in nM!')

            if xscale is True:
                xscale = 'linear'
            else:
                xscale = 'log'
            if xscale_ticks_input:
                # Split the user input into two values
                xscale_min, xscale_max = xscale_ticks_input.split(',')
                # Convert the strings to floats
                xscale_min = float(xscale_min)
                xscale_max = float(xscale_max)
                xscale_ticks = (xscale_min, xscale_max)
            else:
                xscale_ticks = (-2.5, 2.5)
        else:
            conc_unit = 'µM'
            xscale = 'log'
            xscale_ticks = (-2.5, 2.5)

        # Box settings
        highlight_options = st.sidebar.checkbox(label='Highlight IC Value')
        if highlight_options is True:
            box_highlight = st.sidebar.checkbox(label='Box Highlight')

            # Set variables that will be changed by logic
            box = False
            box_color = 'gray'
            box_intercept = 50
            x_concentration = None
            hline = None
            hline_color = 'gray'
            vline = None
            vline_color = 'gray'

            if box_highlight is True:
                # Add logic for checked box_highlight
                box = True
                # Logic for box color
                box_color = st.sidebar.text_input(label='Box Color', placeholder='Color Name or Hex Code')
                if box_color == '':
                    box_color = 'gray'
                else:
                    box_color = box_color
                # Logic for box intercept
                box_intercept = st.sidebar.number_input(label='Response Percentage', value=50, placeholder='50')
                if box_intercept:
                    box_intercept = box_intercept
                # Logic for specific concentration
                x_concentration = st.sidebar.number_input(
                    label='Optional: Highlight By Specific Concentration (will override '
                          'Y-Axis, input must match x-axis units)', value=None,
                    placeholder=None)
                if x_concentration:
                    x_concentration = x_concentration
            else:
                box = False

            # Logic for hline and vline # todo reorder logic
            hline_highlight = st.sidebar.checkbox(label='Horizontal Line')
            if hline_highlight is True:
                hline = st.sidebar.number_input(label='Response Percentage', value=50, placeholder='0')
                hline = hline  # Overwrite hline
                hline_color = st.sidebar.text_input(label='Hline Color', value='gray', placeholder='Color Name or Hex Code')
                hline_color = hline_color  # Overwrite color

            vline_highlight = st.sidebar.checkbox(label='Vertical Line')
            if vline_highlight is True:
                vline = st.sidebar.number_input(label='Concentration', value=0.1, placeholder='X-Axis')
                vline = vline  # Overwrite hline
                vline_color = st.sidebar.text_input(label='Vline Color', value='gray', placeholder='Color Name or Hex Code')
                vline_color = vline_color  # Overwrite color

        else:
            # Set variables that will be changed by logic
            box = False
            box_color = 'gray'
            box_intercept = 50
            x_concentration = None
            # todo try to remove these variables
            hline = None
            hline_color = 'gray'
            vline = None
            vline_color = 'gray'

        fig_width = st.sidebar.slider(label='Figure Width:', min_value=1, max_value=50, value=8)
        fig_height = st.sidebar.slider(label='Figure Height:', min_value=1, max_value=50, value=6)

        # py50 plot function
        figure = plot_data.single_curve_plot(concentration_col=drug_conc,
                                             response_col=response,
                                             plot_title=plot_title,
                                             plot_title_size=plot_title_size,
                                             drug_name=drug_name,
                                             xlabel=xlabel,
                                             ylabel=ylabel,
                                             ylimit=ylimit,
                                             axis_fontsize=axis_fontsize,
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
                                             verbose=True)
        # Display figure
        # To reduce the size of the generated figure from stretching the width of the screen due to page layout set to
        # wide, a column is inserted to 'constrain' the image.
        # todo repeat this for the other images
        col1, col2, = st.columns(2)
        with col1:
            st.pyplot(figure)
        with col2:
            st.header("")


        # Figure must be converted into a temporary file in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Create a download button
        st.download_button("Download Figure", data=buf.read(), file_name="single_curve.png", mime="image/png")

    # Multi Curve options # todo place into own py file for maintainability
    elif fig_type == 'Multi-Curve Plot':
        # Confirm DataFrame only contains multiple drugs
        if len(df_calc[drug_name].unique()) > 1:
            name = df_calc[drug_name].unique()
            name = ', '.join(map(str, name))
            st.markdown('### Multiple Drugs Detected in File:')
            st.write('Looking at: ', name)
            st.write('**Note** Input drugs are converted into µM by default') # todo warning message
        elif len(df_calc[drug_name].unique()) <= 1:
            st.markdown('### 🚨Only 1 Drug Detected in File!!!!🚨')
        else:
            st.write('Is the input file correct?')

        # Add plotting options to the sidebar
        st.sidebar.header("Multi-Curve Plot Options:")

        # Label Options
        label_options = st.sidebar.checkbox(label='Modify Labels')
        if label_options is True:
            # Plot title
            plot_title = st.sidebar.text_input(label='Plot Title', placeholder='Input Plot Title')
            font = st.sidebar.text_input(label='Figure Font', value=None, placeholder='DejaVu Sans')
            plot_title_size = st.sidebar.number_input(label='Title Size', value=None, placeholder='16')
            axis_fontsize = st.sidebar.number_input(label='Axis Fontsize', value=None, placeholder='14')
            xlabel = st.sidebar.text_input(label='X-Axis Label (Default in µM)', placeholder='Input X Label')
            ylabel = st.sidebar.text_input(label='Y-Axis Label', placeholder='Input Y Label')
            ylimit = st.sidebar.number_input(label="Y-Axis Limit", value=None, placeholder='Input Limit')

            # set the font name for a font family
            if font is None:
                plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
            else:
                plt.rcParams.update({'font.sans-serif': font})

            # Default label sizes
            if plot_title_size:
                plot_title_size = plot_title_size
            else:
                plot_title_size = 16
            if axis_fontsize:
                axis_fontsize = axis_fontsize
            else:
                axis_fontsize = 14

            # ylabel limit
            if ylimit is None:
                ylimit = None
            else:
                ylimit = int(ylimit)
        else:
            plot_title = None
            plot_title_size = 16
            axis_fontsize = 14
            xlabel = ''
            ylabel = ''
            ylimit = None

        # Set line color options if checked
        line_color_option = st.sidebar.checkbox(label='Line Options')
        if line_color_option is True:
            line_color = st.sidebar.text_input(label='Line Color (separate by comma)', placeholder='Color Name or Hex Code')
            line_width = st.sidebar.text_input(label='Line Width', placeholder='1.5')
            marker = st.sidebar.text_input(label='Marker Style (separate by comma)', placeholder='matplotlib styles')
            if line_color:
                line_color = line_color
                line_color = [word.strip() for word in line_color.split(',')]  # Convert string of words into list
            else:
                line_color = CBPALETTE
            if line_width:
                line_width = line_width
            else:
                line_width = 1.5
            if marker:
                marker = marker
                marker = [word.strip() for word in marker.split(',')]
            else:
                marker = CBMARKERS
        else:
            line_color = CBPALETTE
            line_width = 1.5
            marker = CBMARKERS

        # Set Legend options if checked
        legend = st.sidebar.checkbox(label='Legend')
        if legend is True:
            st.sidebar.subheader('Legend Options')
            legend_loc = st.sidebar.text_input(label='Legend Location', placeholder='lower right')
            if legend_loc:
                legend_loc = legend_loc
            else:
                legend_loc = 'best'
        else:
            legend_loc = 'best'

        # xscale settings
        xoptions = st.sidebar.checkbox(label='X Axis Options')
        if xoptions is True:
            conc_unit = st.sidebar.radio(
                'Set units of X Axis',
                ['µM', 'nM'])
            xscale = st.sidebar.checkbox(label='X Axis Linear Scale')
            xscale_ticks_input = st.sidebar.text_input(label='Set X-Axis boundaries',
                                                       placeholder='separate number with comma')
            if xscale is True:
                xscale = 'linear'
            else:
                xscale = 'log'
            if xscale_ticks_input:
                # Split the user input into two values
                xscale_min, xscale_max = xscale_ticks_input.split(',')
                # Convert the strings to floats
                xscale_min = float(xscale_min)
                xscale_max = float(xscale_max)
                xscale_ticks = (xscale_min, xscale_max)
            else:
                xscale_ticks = (-2.5, 2.5)
        else:
            conc_unit = 'µM'
            xscale = 'log'
            xscale_ticks = (-2.5, 2.5)

        # todo reorder logic
        # Set variables that will be changed by logic
        hline = 0
        hline_color = 'gray'
        vline = 0
        vline_color = 'gray'

        # Box settings
        highlight_options = st.sidebar.checkbox(label='Highlight IC Value (Must include Legend)')
        if highlight_options is True:
            box_target = st.sidebar.text_input(label='Drug Target to Highlight', placeholder='Drug Name')
            box_color = st.sidebar.text_input(label='Box Color', placeholder='Color Name or Hex Code')
            box_intercept = st.sidebar.number_input(label='Response Percentage', value=None, placeholder='50')
            if box_target:
                box_target = box_target
                st.sidebar.write('This is the box_target:', box_target)
            else:
                box_target = None
            if box_color:
                box_color = box_color
            else:
                box_color = 'gray'
            if box_intercept:
                box_intercept = box_intercept
            else:
                box_intercept = 50

        # todo reorder logic
        # Add hline and vline options
        if highlight_options is True:
            # Logic for hline and vline
            hline_highlight = st.sidebar.checkbox(label='Horizontal Line')
            if hline_highlight is True:
                hline = st.sidebar.number_input(label='Response Percentage', value=0, placeholder='0')
                hline = hline  # Overwrite hline
                hline_color = st.sidebar.text_input(label='Hline Color', value='gray',
                                                    placeholder='Color Name or Hex Code')
                hline_color = hline_color  # Overwrite color

            vline_highlight = st.sidebar.checkbox(label='Vertical Line')
            if vline_highlight is True:
                vline = st.sidebar.number_input(label='Concentration', value=0.1, placeholder='X-Axis')
                vline = vline  # Overwrite hline
                vline_color = st.sidebar.text_input(label='Vline Color', value='gray',
                                                    placeholder='Color Name or Hex Code')
                vline_color = vline_color  # Overwrite color

        else:
            box_target = False
            box_color = 'gray'
            box_intercept = 50

        fig_width = st.sidebar.slider(label='Figure Width:', min_value=1, max_value=50, value=6)
        fig_height = st.sidebar.slider(label='Figure Height:', min_value=1, max_value=50, value=4)

        figure = plot_data.multi_curve_plot(concentration_col=drug_conc,
                                            response_col=response,
                                            name_col=drug_name,
                                            plot_title=plot_title,
                                            plot_title_size=plot_title_size,
                                            xlabel=xlabel,
                                            ylabel=ylabel,
                                            xscale=xscale,
                                            conc_unit=conc_unit,
                                            xscale_ticks=xscale_ticks,
                                            ylimit=ylimit,
                                            axis_fontsize=axis_fontsize,
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
                                            figsize=(fig_width, fig_height))

        # Display figure
        col1, col2, = st.columns(2)
        with col1:
            st.pyplot(figure)
        with col2:
            st.header("")

        # Figure must be converted into a temporary file in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Create a download button
        st.download_button("Download Figure", data=buf.read(), file_name="multi_curve.png", mime="image/png")

    # Grid options
    elif fig_type == 'Grid Plot':
        # Confirm DataFrame only contains multiple drugs
        if len(df_calc[drug_name].unique()) > 1:
            name = df_calc[drug_name].unique()
            name = ', '.join(map(str, name))
            st.markdown('### Multiple Drugs Detected in File:')
            st.write('Looking at: ', name)
        elif len(df_calc[drug_name].unique()) % 2 != 0:
            st.markdown('### 🚨Odd Number of Drugs!!!!🚨')
        else:
            st.write('Is the input file correct?')

        # Add plotting options to the sidebar
        st.sidebar.header("Grid Plot Options:")

        # Label Options
        label_options = st.sidebar.checkbox(label='Modify Labels')
        if label_options is True:
            # Plot title
            plot_title = st.sidebar.text_input(label='Plot Title', placeholder='Input Plot Title')
            font = st.sidebar.text_input(label='Figure Font', value=None, placeholder='DejaVu Sans')
            plot_title_size = st.sidebar.number_input(label='Title Size', value=None, placeholder='16')
            axis_fontsize = st.sidebar.number_input(label='Axis Fontsize', value=None, placeholder='14')
            xlabel = st.sidebar.text_input(label='X-Axis Label (Default in µM)', placeholder='Input X Label')
            ylabel = st.sidebar.text_input(label='Y-Axis Label', placeholder='Input Y Label')
            ylimit = st.sidebar.number_input(label="Y-Axis Limit", value=None, placeholder='Input Limit')

            # set the font name for a font family
            if font is None:
                plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
            else:
                plt.rcParams.update({'font.sans-serif': font})

            # Default label sizes
            if plot_title_size:
                plot_title_size = plot_title_size
            else:
                plot_title_size = 16
            if axis_fontsize:
                axis_fontsize = axis_fontsize
            else:
                axis_fontsize = 14

            # ylabel limit
            if ylimit is None:
                ylimit = None
            else:
                ylimit = int(ylimit)
        else:
            plot_title = None
            plot_title_size = 16
            axis_fontsize = 14
            xlabel = ''
            ylabel = ''
            ylimit = None

        # Set line color options if checked
        line_color_option = st.sidebar.checkbox(label='Line Options')
        if line_color_option is True:
            line_color = st.sidebar.text_input(label='Line Color (separate by comma)', placeholder='Color Name or Hex Code')
            line_width = st.sidebar.text_input(label='Line Width', placeholder='1.5')
            marker = st.sidebar.text_input(label='Marker Style (separate by comma)', placeholder='matplotlib styles')
            if line_color:
                line_color = line_color
                line_color = [word.strip() for word in line_color.split(',')]  # Convert string of words into list
            else:
                line_color = CBPALETTE
            if line_width:
                line_width = line_width
            else:
                line_width = 1.5
            if marker:
                marker = marker
                marker = [word.strip() for word in marker.split(',')]
            else:
                marker = CBMARKERS
        else:
            line_color = CBPALETTE
            line_width = 1.5
            marker = CBMARKERS

        # Set Legend options if checked
        legend = st.sidebar.checkbox(label='Legend')
        if legend is True:
            st.sidebar.subheader('Legend Options')
            legend_loc = st.sidebar.text_input(label='Legend Location', placeholder='lower right')
            if legend_loc:
                legend_loc = legend_loc
            else:
                legend_loc = 'best'
        else:
            legend_loc = 'best'

        # xscale settings
        xoptions = st.sidebar.checkbox(label='X Axis Options')
        if xoptions is True:
            conc_unit = st.sidebar.radio(
                'Set units of X Axis',
                ['µM', 'nM'])
            xscale = st.sidebar.checkbox(label='X Axis Linear Scale')
            xscale_ticks_input = st.sidebar.text_input(label='Set X-Axis boundaries',
                                                       placeholder='separate number with comma')
            if xscale is True:
                xscale = 'linear'
            else:
                xscale = 'log'
            if xscale_ticks_input:
                # Split the user input into two values
                xscale_min, xscale_max = xscale_ticks_input.split(',')
                # Convert the strings to floats
                xscale_min = float(xscale_min)
                xscale_max = float(xscale_max)
                xscale_ticks = (xscale_min, xscale_max)
            else:
                xscale_ticks = (-2.5, 2.5)
        else:
            conc_unit = 'µM'
            xscale = 'log'
            xscale_ticks = (-2.5, 2.5)

        hline = 0
        hline_color = 'gray'
        vline = 0
        vline_color = 'gray'

        # Box settings
        highlight_options = st.sidebar.checkbox(label='Highlight IC Value (Must include Legend)')
        if highlight_options is True:
            box_color = st.sidebar.text_input(label='Box Color', placeholder='Color Name or Hex Code')
            box_intercept = st.sidebar.number_input(label='Response Percentage', value=None, placeholder='50')
            if box_color:
                box_color = box_color
            else:
                box_color = 'gray'
            if box_intercept:
                box_intercept = box_intercept
            else:
                box_intercept = 50

        # todo reorder logic
        # Add hline and vline options
        if highlight_options is True:
            # Logic for hline and vline
            hline_highlight = st.sidebar.checkbox(label='Horizontal Line')
            if hline_highlight is True:
                hline = st.sidebar.number_input(label='Response Percentage', value=0, placeholder='0')
                hline = hline  # Overwrite hline
                hline_color = st.sidebar.text_input(label='Hline Color', value='gray',
                                                    placeholder='Color Name or Hex Code')
                hline_color = hline_color  # Overwrite color

            vline_highlight = st.sidebar.checkbox(label='Vertical Line')
            if vline_highlight is True:
                vline = st.sidebar.number_input(label='Concentration', value=0.1, placeholder='X-Axis')
                vline = vline  # Overwrite hline
                vline_color = st.sidebar.text_input(label='Vline Color', value='gray',
                                                    placeholder='Color Name or Hex Code')
                vline_color = vline_color  # Overwrite color

        else:
            highlight_options = False
            box_color = 'gray'
            box_intercept = None

        fig_width = st.sidebar.slider(label='Figure Width:', min_value=1, max_value=50, value=8)
        fig_height = st.sidebar.slider(label='Figure Height:', min_value=1, max_value=50, value=4)

        figure = plot_data.grid_curve_plot(concentration_col=drug_conc,
                                           response_col=response,
                                           name_col=drug_name,
                                           plot_title=plot_title,
                                           plot_title_size=plot_title_size,
                                           xlabel=xlabel,
                                           ylabel=ylabel,
                                           xscale=xscale,
                                           conc_unit=conc_unit,
                                           xscale_ticks=xscale_ticks,
                                           ylimit=ylimit,
                                           line_color=line_color,
                                           line_width=line_width,
                                           box=highlight_options,
                                           box_color=box_color,
                                           box_intercept=box_intercept,
                                           hline=hline,
                                           hline_color=hline_color,
                                           vline=vline,
                                           vline_color=vline_color,
                                           figsize=(fig_width, fig_height))
        # Display figure
        col1, col2, = st.columns(2)
        with col1:
            st.pyplot(figure)
        with col2:
            st.header("")

        # Figure must be converted into a temporary file in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Create a download button
        st.download_button("Download Figure", data=buf.read(), file_name="multi_curve.png", mime="image/png")

    else:
        st.write('Something is wrong with the app! Help!')
