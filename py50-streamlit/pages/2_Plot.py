import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from py50.plotcurve import PlotCurve
from py50.plot_settings import CBMARKERS, CBPALETTE

st.set_page_config(page_title='Generate Dose-Response Curves', page_icon='📈')

# Upload the CSV file
uploaded_file = st.file_uploader('Upload .csv file')

# Check if a CSV file has been uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.dataframe(df, hide_index=True)  # visualize dataframe in streamlit app
else:
    # Display a message if no CSV file has been uploaded
    st.warning('Please upload a .csv file.')

# Select columns for calculation
col_header = df.columns.to_list()
drug_name = st.selectbox('Drug Name:', (col_header))
compound_conc = st.selectbox('Compound Concentration:', (col_header))
ave_response = st.selectbox('Average Response column:', (col_header))

df_calc = df.filter(items=(drug_name, compound_conc, ave_response), axis=1)
st.dataframe(df_calc)

# figure type
fig_type = st.radio(
    'Select figure type',
    ['Single Plot', 'Multi-Curve Plot', 'Grid PLot'])

# Drug name conditions if figure type is Single PLot
if fig_type == 'Single Plot':
    if len(df_calc[drug_name].unique()) == 1:
        name = df_calc[drug_name].unique()
        drug_name = name[0]
        st.write('Only 1 Drug in table:')
        st.write('Looking at: ', drug_name)
    else:
        drug_name = st.text_input(label='Drug Name', placeholder='Input Drug Name')

# Single plot options
if fig_type is 'Single Plot':
    plot_data = PlotCurve(df)

    # Add plotting options to the sidebar
    st.sidebar.header("Plotting Options")

    # Plot title
    plot_title = st.sidebar.text_input(label='Plot Title', placeholder='Input Plot Title')

    # xlabel title
    xlabel = st.sidebar.text_input(label='X Label', placeholder='Input X Label')
    if xlabel is None:
        xlabel = 'Logarithmic Concentration (µM)'

    # ylabel title
    ylabel = st.sidebar.text_input(label='Y Label', placeholder='Input Y Label')
    if ylabel is None:
        ylabel = 'Inhibition %'

    # ylabel limit
    ylimit = st.sidebar.text_input(label="Y Limit", placeholder='Set Y Axis Limit')
    if ylimit:
        ylimit = int(ylimit)
    else:
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
        marker = 's'

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

    # xscale
    xoptions = st.sidebar.checkbox(label='X Axis Options')
    if xoptions is True:
        xscale_unit = st.sidebar.radio(
            'Set units of X Axis',
            ['µM', 'nM'])
        xscale = st.sidebar.checkbox(label='X Axis Linear Scale')
        if xscale is True:
            xscale = 'linear'
        else:
            xscale = 'log'
        xscale_ticks = st.sidebar.text_input(label='Set X-Axis boundaries as (X,X)')
        if xscale_ticks:
            xscale_ticks = xscale_ticks
            st.sidebar.write(xscale_ticks)
        else:
            xscale_ticks = (-2.5, 2.5)
            st.sidebar.write(xscale_ticks)
    else:
        xscale_unit = 'µM'
        xscale = 'log'
        xscale_ticks = (-2.5, 2.5)

    # # Set xscale_units
    # xscale_unit = st.sidebar.radio(
    #     'Set units of X Axis',
    #     ['µM', 'nM'])

    figure = plot_data.single_curve_plot(concentration_col=compound_conc,
                                         response_col=ave_response,
                                         plot_title=plot_title,
                                         drug_name=drug_name,
                                         xlabel=xlabel,
                                         ylabel=ylabel,
                                         ylimit=ylimit,
                                         legend=legend,
                                         legend_loc=legend_loc,
                                         xscale=xscale,
                                         xscale_unit=xscale_unit,
                                         xscale_ticks=xscale_ticks,
                                         line_color=line_color,
                                         line_width=line_width,
                                         marker=marker,
                                         box=True,
                                         box_color='gray',
                                         box_intercept=50,
                                         # x_concentration=.694869,
                                         figsize=(8, 6),
                                         output_filename=None)
    st.pyplot(figure)

else:
    st.write('OH POOPY!')

# # Sidebar info
# with st.sidebar:
#     st.write('Plot Settings')
