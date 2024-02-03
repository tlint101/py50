import streamlit as st
import pandas as pd
import numpy as np
from page.functions.calculator_func import Calc_Logic

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


'''
Page layout begins below
'''
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
    'Options are paste or .csv file upload',
    ('Paste Data', 'Upload CSV File', 'IC50 to pIC50 Calculator'))

calc = Calc_Logic()

if option == 'Upload CSV File':
    # Upload the CSV file
    uploaded_file = st.file_uploader('Upload .csv file')

    # Check if a CSV file has been uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        drug_query = pd.read_csv(uploaded_file)
        st.write('## Input Table')
        st.data_editor(drug_query, num_rows='dynamic')  # visualize dataframe in streamlit app
    else:
        # Display a message if no CSV file has been uploaded
        st.warning('Please upload a .csv file.')

    # Select columns for calculation
    if uploaded_file is not None:  # nested in if/else to remove initial traceback error
        calc.calculator_program(df=drug_query, paste=False)

# Editable DataFrame
elif option == 'Paste Data':
    st.markdown('### Paste Data in Table:')
    # Make dummy dataframe
    df = pd.DataFrame([{"Drug Name": '', 'Concentration': '', 'Response': ''}, ])

    edited_df = st.data_editor(df, num_rows='dynamic')

    if (edited_df == '').all().all():
        st.write('Table is currently empty')
    else:
        calc.calculator_program(df=edited_df, paste=True)

# pIC50 calculator
else:
    st.markdown('### Insert your IC50 Value (in nM):')
    input_ic50 = st.number_input('Insert IC50 Value (in nM)', step=1e-6)

    # Conditional if input is ≤ 0
    if input_ic50 > 0:
        pic50 = -np.log10(input_ic50 * 0.000000001)
    else:
        st.write(':red[**Input cannot be 0 or a negative number!**]')

    # Generate a DataFrame from input
    data = pd.DataFrame([{'Notes': '', 'IC50 (nM)': input_ic50, 'pIC50': pic50}])

    # Output table
    edited_df = st.data_editor(data, num_rows='dynamic')
    calc.download_button(edited_df, file_name='py50_pic50.csv')
