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

# # add logo
# st.sidebar.header("Calculate IC50")
# st.sidebar.image('img/py50_logo_only.png', width=150)

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


# todo This is the update
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

        st.write('## Filtered Table')
        df_calc = drug_query.filter(items=(drug_name, compound_conc, ave_response), axis=1)
        # todo add option for decreasing order
        st.dataframe(df_calc)

        # Calculate IC50
        data = Calculator(drug_query)

        absolute = data.calculate_absolute_ic50(name_col=drug_name,
                                                concentration_col=compound_conc,
                                                response_col=ave_response,
                                                input_units=units)
        st.markdown('## Calculated Results')

        st.dataframe(absolute, hide_index=True)

        # output unit message
        st.markdown(f'Calculated Results are in {units}')

        convert = st.checkbox('Convert to pIC50?')

        if convert:
            calculation = data.calculate_pic50(name_col=drug_name,
                                               concentration_col=compound_conc,
                                               response_col=ave_response,
                                               input_units=units)
            st.dataframe(calculation, hide_index=True)

    else:
        pass
else:
    st.markdown('### Insert your IC50 Value (in nM):')
    input_ic50 = st.number_input('', step=1e-6)

    if input_ic50 is not None or input_ic50 == 0:
        pic50 = -np.log10(input_ic50 * 0.000000001)
    else:
        st.write('# Cannot convert 0 or negative value')
        # todo error message not writing

    # Generate a DataFrame from input
    data = pd.DataFrame({'IC50 (nM)': input_ic50, 'pIC50': pic50}, index=[0])
    # todo add a download table button

    st.dataframe(data)
