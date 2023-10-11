import streamlit as st
import pandas as pd
import folium

# from geopy.geocoders import Nominatim

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


def display_city_name(city):
    st.markdown(f"Selected City: {city}")

def display_map_with_location(latitude, longitude):
    st.map(data={"LAT": [latitude], "LON": [longitude]})
    st.markdown(f"Latitude: {latitude}, Longitude: {longitude}")

def historical_data():
    st.title("Historical Data")

    df = pd.read_csv("../data/dataset/train_data/metadata.csv")
    plume_yes_df = df[df['plume'] == 'yes'].drop_duplicates()
    plume_no_df = df[df['plume'] == 'no'].drop_duplicates()

    # Create a Streamlit web app
    st.title("Map with Plume Data")

    # Add a selectbox to allow the user to choose between "plume = yes" and "plume = no"
    option = st.selectbox("Select data to display:", ("Show locations with plume", "Show locations without plume"))

    if option == "Show locations with plume":
        st.map(plume_yes_df, use_container_width=True)  # Display the map with plume = yes
    else:
        st.map(plume_no_df, use_container_width=True)  # Display the map with plume = no

        # Read the CSV file
    cities = pd.read_csv('locations_with_cities.csv')

    # Extract unique cities from the 'city' column
    unique_cities = cities['city'].unique()
    unique_cities = ['Select Location', "Add New Location",] + [city for city in unique_cities if str(city) not in ['N/A', 'nan']


    # User input: choose city from a dropdown or set location with latitude and longitude
    location_option = st.selectbox("Choose an option:", unique_cities)

    if location_option == "Select Location":
        pass
    elif location_option == "Add New Location":
        # User input for latitude and longitude
        default_latitude = 48.8566  # Default latitude for Paris
        default_longitude = 2.3522  # Default longitude for Paris
        latitude = st.number_input("Enter Latitude:", value=default_latitude)
        longitude = st.number_input("Enter Longitude:", value=default_longitude)
        display_map_with_location(latitude, longitude)
        
    else:
        display_city_name(location_option)


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
