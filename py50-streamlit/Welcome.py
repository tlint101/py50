import streamlit as st

st.set_page_config(
    page_title="py50",
    page_icon="👋",
)

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

# Place logo image at top of page
st.image('img/py50_full.png')
st.write('# Welcome to py50!')  # add a title

github = 'https://github.com/tlint101/py50'
documentation = 'https://py50.readthedocs.io/en/latest/?badge=latest'
zenodo = 'https://zenodo.org/records/10183941'

st.markdown(
    """
    py50 is a program to calculate IC50 values and to generate dose-response curves. The program utilizes the Four 
    parameter logistic (4PL) regression model. 
    
    """)
st.markdown('Further information for py50 can be found on the GitHub repository [here](%s).' % github)
st.markdown('Documentation can be found [here](%s).' % documentation)
st.markdown('If you are interested in citing py50, you are welcome to use the zenodo link [here](%s).' % zenodo)

# add logo
st.sidebar.image('img/py50_logo_only.png', width=150)
st.sidebar.success("Select page above to get started!")
