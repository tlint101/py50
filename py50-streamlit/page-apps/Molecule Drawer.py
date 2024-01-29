import streamlit as st
from streamlit_ketcher import st_ketcher
from streamlit_extras.add_vertical_space import add_vertical_space

# # Set page config
# st.set_page_config(layout="wide")
#
# # add logo
# st.sidebar.image('img/py50_logo_only.png', width=150)

# Page text
datasets = "https://github.com/tlint101/py50/tree/main/dataset"
st.markdown("# Draw Molecular Structures")
st.write(
    "This page will allow users to draw chemical structures of their molecules. Users can translate a smiles string"
    " into a 2D image or vice versa."
)
add_vertical_space(1)

(
    col1,
    col2,
) = st.columns(2)
with col1:
    molecule = st.text_input(label="", placeholder="Input Smiles String")
    smile_code = st_ketcher(molecule, height=600)
    st.markdown(f"Smile string: {smile_code}")
with col2:
    st.header("")
