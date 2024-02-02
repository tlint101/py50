import streamlit as st
import pandas as pd
from page.functions.curve_func import Plot_Logic

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

# Todo double check functions on streamlit app

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

program = Plot_Logic()

if option == "Upload CSV File":
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload .csv file")

    # Check if a CSV file has been uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("## Input Table")
        st.data_editor(
            df, num_rows="dynamic"
        )  # visualize dataframe in streamlit app
    else:
        # Display a message if no CSV file has been uploaded
        st.warning("Please upload a .csv file.")

    # Select columns for calculation
    if uploaded_file is not None:  # nested in if/else to remove initial traceback error
        st.write("## Select Columns for Calculation")
        program.plot_program(df=df, paste=False)

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
        st.write("Table is currently empty")
    else:
        program.plot_program(df=edited_df, paste=True)
