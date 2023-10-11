import streamlit as st
from methane_detect import methane_detection
from map_app import historical_data

# Set the page background color
st.markdown(
    """
    <style>
    body {
        background-color: #FFFAF1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the sidebar background color
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #B3F502;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.sidebar.title("Choose Display")
    display_option = st.sidebar.radio("Select an option", ("Methane Detection", "Historical Data"))

    if display_option == "Historical Data":
        historical_data()
    elif display_option == "Methane Detection":
        methane_detection()

if __name__ == "__main__":
    main()
