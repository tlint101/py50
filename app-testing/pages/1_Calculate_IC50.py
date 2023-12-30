import streamlit as st
import pandas as pd
from py50.calculator import Calculator

# Set page config
st.set_page_config(page_title='py50: Calculation', page_icon='🧮', layout='wide')

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

# add logo
st.sidebar.header("Calculate IC50")
st.sidebar.image('img/py50_logo_only.png', width=150)

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

# Upload the CSV file
uploaded_file = st.file_uploader('Upload .csv file')

# Check if a CSV file has been uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    drug_query = pd.read_csv(uploaded_file)
    st.dataframe(drug_query, hide_index=True)  # visualize dataframe in streamlit app
else:
    # Display a message if no CSV file has been uploaded
    st.warning('Please upload a .csv file.')

# Select columns for calculation
if uploaded_file is not None:  # nested in if/else to remove initial traceback error
    col_header = drug_query.columns.to_list()
    drug_name = st.sidebar.selectbox('Drug Name:', (col_header))
    compound_conc = st.sidebar.selectbox('Drug Concentration:', (col_header))
    ave_response = st.sidebar.selectbox('Average Response column:', (col_header))

    units = st.sidebar.radio('Input Concentration Units',
                                options=['nM', 'µM'],
                                captions=['Nanomolor', 'Micromolar'])

    st.write('## Filter Table')
    df_calc = drug_query.filter(items=(drug_name, compound_conc, ave_response), axis=1)
    # todo add option for decreasing order
    st.dataframe(df_calc)

    # Calculate IC50
    data = Calculator(drug_query)

    absolute = data.calculate_absolute_ic50(name_col=drug_name,
                                            concentration_col=compound_conc,
                                            response_col=ave_response,
                                            input_units=units)

    st.markdown('## Calculation Results')
    # todo add st.markdown reminder that units are converted into µM

    st.dataframe(absolute, hide_index=True)
else:
    pass
