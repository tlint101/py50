import streamlit as st

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


"""
CHANGELOG BELOW
"""

st.markdown('# :red[**Changelog**]')

st.markdown('## :rainbow[**2024.01.18**]')
st.markdown('''
##### :green[**Major Changes**] 🎉
- Updated py50 Web Application to [py50 v0.3.4](https://github.com/tlint101/py50/releases)
- Added copy/paste feature for DataTables (Tables constrained by Drug Name, Concentration, Response)
- Added pIC50 calculator
    - Option to scale IC50 columns into pIC50
    - Standalone pIC50 converter
        - optional output as .csv file
##### :orange[**Minor Changes**] 📋
-Changed page selection into a selectbox
- Added CSV button to DataTables for easier download
- Increased image resolution for images downloaded using the "Download" button. Now 300 dpi by default.
- Removed legend options for grid plot
- Added messages in the calculator to signal units (nM or µM)
- Added messages to remind user what units (nM or µM) the plot is in
- Adjusted size of the molecule drawer canvas 
- Autoselect columns for Calculator and Plot Curve (for pasting info only)
- Backend -> reformatted code for maintainability
    - Added logic condigions for Calculator and Plot Curve
##### :red[**Bug fixes 🪲**]
:orange[**General**]
- Converted all DataTables into editable DataTables
- Fixed pIC50 calculator to handle both nM or µM values 
            ''')