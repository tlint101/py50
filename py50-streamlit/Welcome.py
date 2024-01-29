import os
import streamlit as st
from streamlit_activities_menu import get_available_activities, build_activities_menu

st.set_page_config(page_title="py50", page_icon="👋", layout="wide")


def run():
    working_directory = os.path.dirname(os.path.abspath(__file__))

    # add logo
    image = os.path.join(working_directory, "img/py50_logo_only.png")
    st.sidebar.image(image, width=200)

    # Load the available services
    page_settings_path = os.path.join(working_directory, "page_settings.yaml")
    page_path = os.path.join(working_directory, "page-apps/")

    # Load the yaml with core services as activities
    core_activities = get_available_activities(
        activities_filepath=os.path.abspath(page_settings_path)
    )

    page_option, _ = build_activities_menu(
        activities_dict=core_activities,
        label="**Pages:**",
        key="activitiesMenu",
        activities_dirpath=os.path.abspath(page_path),
        disabled=False,
    )

    if page_option == "Home":
        st.sidebar.markdown(":green[**Select page above to get started!**]")
    else:
        pass


if __name__ == "__main__":
    run()
else:
    st.error("The app failed initialization. Report issue to maintainers in github")
