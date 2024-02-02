"""
Utility functions
"""

import streamlit as st
import io


class Fig_Buttons:
    def __init__(self):
        pass

    def download_button(self, fig, file_name):
        # Figure must be converted into a temporary file in memory
        buf = io.BytesIO()
        # plt.savefig(buf, format='png', dpi=300)
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        # Create a download button
        st.download_button(
            "Download Figure", data=buf.read(), file_name=file_name, mime="image/png"
        )
