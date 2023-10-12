import streamlit as st
import pandas as pd

# Function to display a map with plume data


def display_map_with_plume(df):
    st.map(df[df['plume'] == 'yes'])

# Function to display a map without plume data


def display_map_without_plume(df):
    st.map(df[df['plume'] == 'no'])

# Streamlit app


def historical_data():
    st.title("Historical Data")

    df = pd.read_csv("data/dataset/train_data/metadata.csv").drop_duplicates()

    st.title("Map with Plume Data")

    if st.button("Show locations with plume"):
        display_map_with_plume(df)
    if st.button("Show locations without plume"):
        display_map_without_plume(df)

    cities = pd.read_csv('app/locations_with_cities.csv')

    unique_cities = cities['city'].unique()
    unique_cities = ['Select Location', "Add New Location",] + \
        [city for city in unique_cities if str(city) not in ['N/A', 'nan']]

    location_option = st.selectbox("Choose an option:", unique_cities)

    if location_option == "Select Location":
        pass
    elif location_option == "Add New Location":
        default_latitude = 48.8566
        default_longitude = 2.3522
        latitude = st.number_input("Enter Latitude:", value=default_latitude)
        longitude = st.number_input(
            "Enter Longitude:", value=default_longitude)
        st.map({"lat": [latitude], "lon": [longitude]})
    else:
        # Zoom into the selected city
        selected_city_data = cities[cities['city'] == location_option]
        if not selected_city_data.empty:
            city_latitude = selected_city_data.iloc[0]['lat']
            city_longitude = selected_city_data.iloc[0]['lon']
            st.map({"lat": [city_latitude], "lon": [city_longitude]})


if __name__ == '__main__':
    historical_data()
