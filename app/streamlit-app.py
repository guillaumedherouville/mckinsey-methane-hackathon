import streamlit as st
from methane_detect import methane_detection
from map_app import historical_data
from clean_r import cleanr_display


def main():
    st.sidebar.title("Choose Display")
    display_option = st.sidebar.radio(
        "Select an option", ("Methane Detection", "Data Analysis", "CleanR")
    )

    if display_option == "Data Analysis":
        historical_data()
    elif display_option == "Methane Detection":
        methane_detection()
    elif display_option == "CleanR":
        cleanr_display()


if __name__ == "__main__":
    main()
