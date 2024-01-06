import streamlit as st
import pandas as pd
import numpy as np
from py50.calculator import Calculator

# Set page config
# st.set_page_config(page_title='py50: Calculation', page_icon='🧮', layout='wide')

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

# Set up download button for csv files
def download_button(df, file_name=None):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download table as CSV', data=csv, file_name=file_name, mime='text/csv')

# Page text
tutorial = 'https://github.com/tlint101/py50/blob/main/tutorials/002_absolute_ic50.ipynb'
datasets = 'https://github.com/tlint101/py50/tree/main/dataset'
st.markdown('# Calculate Relative and Absolute IC50')
st.write('This page will calculate a final table containing Relative and Absolute IC50 values for a queried drug.')
st.write('The program requires at least three columns:')
st.write('- Drug Name')
st.write('- Drug Concentration')
st.write('- Average Response')
st.write('For more information about Relative vs. Absolute IC50, please see the tutorial [here](%s)' % tutorial)
st.write('Sample datasets can be found [here](%s)' % datasets)
st.write('')

st.markdown('## Select an option to get started:')
option = st.radio(
    '# Select an option to get started',
    ('Upload CSV File', 'Convert IC50 into pIC50'))

if option == 'Upload CSV File':
    # Upload the CSV file
    uploaded_file = st.file_uploader('Upload .csv file')

    # Check if a CSV file has been uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        drug_query = pd.read_csv(uploaded_file)
        st.write('## Input Table')
        st.dataframe(drug_query, hide_index=True)  # visualize dataframe in streamlit app
    else:
        # Display a message if no CSV file has been uploaded
        st.warning('Please upload a .csv file.')

    # Select columns for calculation
    if uploaded_file is not None:  # nested in if/else to remove initial traceback error
        # add space between option
        st.sidebar.markdown('')
        st.sidebar.markdown('**Calculator Options:**')
        col_header = drug_query.columns.to_list()
        drug_name = st.sidebar.selectbox('Drug Name:', (col_header))
        compound_conc = st.sidebar.selectbox('Drug Concentration:', (col_header))
        ave_response = st.sidebar.selectbox('Average Response column:', (col_header))

        units = st.sidebar.radio('Input Concentration Units',
                                 options=['nM', 'µM'],
                                 captions=['Nanomolor', 'Micromolar'])

        # Set conditions before calculations
        conditions = {drug_name, compound_conc, ave_response}
        if len(conditions) != 3:
            st.write('### :red[Select Drug, Concentration, and Response Columns!]')
        else:
            st.write('## Filtered Table')
            df_calc = drug_query.filter(items=(drug_name, compound_conc, ave_response), axis=1)

        # Output filtered table for calculation
        st.dataframe(df_calc)

        # Calculate IC50
        data = Calculator(drug_query)

        absolute = data.calculate_absolute_ic50(name_col=drug_name,
                                                concentration_col=compound_conc,
                                                response_col=ave_response,
                                                input_units=units)
        st.markdown('## Calculated Results')

        # Absolute IC50 Table
        st.dataframe(absolute, hide_index=True)
        download_button(absolute, file_name='py50_ic50.csv')

        # output unit message
        st.markdown(f'Calculated Results are in {units}')

        convert = st.checkbox('Calculate values to pIC50?')

        if convert is True:
            conversion = data.calculate_pic50(name_col=drug_name,
                                        concentration_col=compound_conc,
                                        response_col=ave_response,
                                        input_units=units)
            # Output pIC50 table
            st.dataframe(conversion, hide_index=True)
            download_button(conversion, file_name='py50_pic50.csv')
    else:
        pass
else:
    st.markdown('### Insert your IC50 Value (in nM):')
    input_ic50 = st.number_input('Insert IC50 Value (in nM)', step=1e-6)

    # Conditional if input is ≤ 0
    if input_ic50 > 0:
        pic50 = -np.log10(input_ic50 * 0.000000001)
    else:
        st.write(':red[**Input cannot be 0 or a negative number!**]')

    # Generate a DataFrame from input
    data = pd.DataFrame({'IC50 (nM)': input_ic50, 'pIC50': pic50}, index=[0])

    # Output table
    st.dataframe(data)
    download_button(data, file_name='py50_pic50.csv')
