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
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV", data=csv, file_name=file_name, mime="text/csv"
    )


def calculator_program(df=None, paste=True):
    """
    Program to run calculations. It must include nested if/else depending
    on whether the input dataframe is pasted or a csv file.

    :param df: Input dataframe
    :return: editable dataframe
    """
    st.sidebar.markdown(":green[**Calculator Options:**]")

    # Logic based on paste or CSV input
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

    # sidebar options
    units = st.sidebar.radio(
        "Input Concentration Units",
        options=["nM", "µM"],
        captions=["Nanomolor", "Micromolar"],
    )

    # Set conditions for calculations
    conditions = {drug_name, compound_conc, ave_response}
    if len(conditions) != 3:
        st.write("### :red[Select Drug, Concentration, and Response Columns!]")
    else:
        st.write("## Filtered Table")
        df_calc = drug_query.filter(
            items=(drug_name, compound_conc, ave_response), axis=1
        )
        # Ensure columns are float
        df_calc[compound_conc] = df_calc[compound_conc].astype(float)
        df_calc[ave_response] = df_calc[ave_response].astype(float)

        # Output filtered table for calculation
        st.data_editor(df_calc, num_rows="dynamic")

        # Calculate IC50
        data = Calculator(df_calc)

        absolute = data.calculate_absolute_ic50(
            name_col=drug_name,
            concentration_col=compound_conc,
            response_col=ave_response,
            input_units=units,
        )
        st.markdown("## Calculated Results")

        # Absolute IC50 Table
        st.data_editor(absolute, num_rows="dynamic")
        download_button(absolute, file_name="py50_ic50.csv")

        # output unit message
        st.markdown(f"Calculated Results are in {units}")

        convert = st.checkbox("Calculate values to pIC50?")

        if convert is True:
            st.markdown("## Calculated Results with pIC50")
            conversion = data.calculate_pic50(
                name_col=drug_name,
                concentration_col=compound_conc,
                response_col=ave_response,
                input_units=units,
            )
            # Output pIC50 table
            st.data_editor(conversion, num_rows="dynamic")
            download_button(conversion, file_name="py50_pic50.csv")


"""
Page layout begins below
"""
tutorial = (
    "https://github.com/tlint101/py50/blob/main/tutorials/002_absolute_ic50.ipynb"
)
datasets = "https://github.com/tlint101/py50/tree/main/dataset"
st.markdown("# Calculate Relative and Absolute IC50")
st.write(
    "This page will calculate a final table containing Relative and Absolute IC50 values for a queried drug."
)
st.write("The program requires at least three columns:")
st.write("- Drug Name")
st.write("- Drug Concentration")
st.write("- Average Response")
st.write(
    "For more information about Relative vs. Absolute IC50, please see the tutorial [here](%s)"
    % tutorial
)
st.write("Sample datasets can be found [here](%s)" % datasets)
st.write("")

st.markdown("## Select an option to get started:")
option = st.radio(
    "Options are paste or .csv file upload",
    ("Paste Data", "Upload CSV File", "IC50 to pIC50 Calculator"),
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
        calculator_program(df=drug_query, paste=False)

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
        calculator_program(df=edited_df, paste=True)

# pIC50 calculator
else:
    st.markdown("### Insert your IC50 Value (in nM):")
    input_ic50 = st.number_input("Insert IC50 Value (in nM)", step=1e-6)

    # Conditional if input is ≤ 0
    if input_ic50 > 0:
        pic50 = -np.log10(input_ic50 * 0.000000001)
    else:
        st.write(":red[**Input cannot be 0 or a negative number!**]")

    # Generate a DataFrame from input
    data = pd.DataFrame([{"Notes": "", "IC50 (nM)": input_ic50, "pIC50": pic50}])

    # Output table
    edited_df = st.data_editor(data, num_rows="dynamic")
    download_button(edited_df, file_name="py50_pic50.csv")
