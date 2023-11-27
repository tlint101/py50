import streamlit as st
import pandas as pd
from py50.calculate import Calculate

st.set_page_config(page_title='Calculate Relative and Absolute IC50', page_icon='🧮')

# add logo
st.sidebar.image('img/py50_logo_only.png', width=150)

st.markdown('# Calculate Relative and Absolute IC50')
st.sidebar.header("Calculate IC50")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

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
drug_name = st.sidebar.selectbox('Drug Name:', (col_header))
compound_conc = st.sidebar.selectbox('Compound Concentration:', (col_header))
ave_response = st.sidebar.selectbox('Average Response column:', (col_header))

st.write('## Filter Table')
df_calc = df.filter(items=(drug_name, compound_conc, ave_response), axis=1)
st.dataframe(df_calc)

# Calculate IC50
data = Calculate(df)
absolute = data.calculate_absolute_ic50(name_col=drug_name,
                                        concentration_col=compound_conc,
                                        response_col=ave_response)

st.markdown('## Calculation Results')

st.dataframe(absolute, hide_index=True)
