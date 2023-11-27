import streamlit as st
from streamlit_extras.app_logo import add_logo
from PIL import Image

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

# Place logo image at top of page
image = Image.open('img/py50_full.png')
st.image(image)
st.write('# Welcome to Py50!')  # add a title

st.markdown(
    """
    Fill this out with something fun you doofus!!!!!!!
"""
)


# add logo
st.sidebar.image('img/py50_logo_only.png', width=150)
st.sidebar.success("Select page above to get started!")
