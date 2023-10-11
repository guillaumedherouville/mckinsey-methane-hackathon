import streamlit as st

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

def historical_data():
    st.title("Historical Data")

    # Default latitude and longitude for Paris
    default_latitude = 48.8566
    default_longitude = 2.3522
    
    # User input for latitude and longitude
    latitude = st.number_input("Enter Latitude:", value=default_latitude)
    longitude = st.number_input("Enter Longitude:", value=default_longitude)

    # Display the map with the user's point
    st.map(data={"LAT": [latitude], "LON": [longitude]})  # Use 'LAT' and 'LON' for column names

    # Add a note
    st.markdown("Please choose the latitude and longitude of the place where the satellite image was taken.")


def methane_detection():
    st.title("Methane Detection")

    # Upload an image for methane detection
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "tif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

def main():
    st.sidebar.title("Choose Display")
    display_option = st.sidebar.radio("Select an option", ("Historical Data", "Methane Detection"))

    if display_option == "Historical Data":
        historical_data()
    elif display_option == "Methane Detection":
        methane_detection()

if __name__ == "__main__":
    main()
