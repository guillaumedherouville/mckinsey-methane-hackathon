import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def get_lat_lon(city_name, data):
    location = data[data["city"] == city_name]
    if not location.empty:
        return location.iloc[0]["lat"], location.iloc[0]["lon"]
    else:
        return None

def display_historical_data_for_city(city, city_data, data):
    """
    need to prettify the plot
    """
    st.markdown(f"Selected City: {city}")

    latitude, longitude = get_lat_lon(city, city_data)
    st.map(data={"LAT": [latitude], "LON": [longitude]})
    city_data = data[(data["lat"] == latitude) & (data["lon"] == longitude)]
    city_data["date"] = pd.to_datetime(city_data["date"], format="%Y%m%d")
    st.dataframe(city_data)
    if len(city_data[city_data["plume"] == "yes"]) == 0:
        st.markdown(f"The presence of methane was never detected at this location! 🥳")
    if len(city_data[city_data["plume"] == "no"]) == 0:
        st.markdown(f"The presence of methane is stable in this region 😔")
    if len(city_data[city_data["plume"] == "no"]) > 0 and city_data[city_data['date'] == city_data['date'].max()]['plume'].values[0] == "yes":
        st.markdown(f"🚨 Methane was detected in this area recently 🚨")


    city_data["plume"] = city_data["plume"].apply(
        lambda x: 1 if x == "yes" else 0)
    city_data = city_data.sort_values(by="date")
    plt.plot(figsize=(10, 4))
    plt.plot(city_data["date"], city_data["plume"], marker="o")
    plt.title(f"Methane Detection over Time in {city}")
    plt.xlabel("Date")
    plt.ylabel("Plume (1 for yes, 0 for no)")
    plt.grid()
    plt.show()
    st.pyplot(plt)

def display_map_with_location(latitude, longitude):
    st.map(data={"LAT": [latitude], "LON": [longitude]})
    st.markdown(f"Latitude: {latitude}, Longitude: {longitude}")

def dummydata():
    data = pd.DataFrame({'date': ['2023-02-28 00:00:00', '2023-02-13 00:00:00', '2023-03-05 00:00:00'],
    'id_coord': ['id_0001', 'id_0001', 'id_0001'],
    'plume': ['no', 'no', 'yes'],
    'set': ['train', 'train', 'train'],
    'lat': [48.86, 48.86, 48.86],
    'lon': [2.35, 2.35, 2.35],
    'coord_x': [48.86, 48.86, 48.86],
    'coord_y': [2.35, 2.35, 2.35],
    'path': ['images/mock-data', 'images/mock-data', 'images/mock-data']})
    return data

def display_city_name_1(city, city_data):
    """
    need to prettify the plot
    """
    st.markdown(f"Selected City: {city}")
    st.map(data=city_data) 
    st.dataframe(city_data)
    city_data["date"] = pd.to_datetime(city_data["date"])
    city_data["plume"] = city_data["plume"].apply(lambda x: 1 if x == "yes" else 0)
    city_data = city_data.sort_values(by="date")
    plt.figure(figsize=(10, 4))
    plt.plot(city_data["date"], city_data["plume"], marker="o")
    plt.title(f"Methane Detection over Time in {city}")
    plt.xlabel("Date")
    plt.ylabel("Plume (1 for yes, 0 for no)")
    plt.grid()
    plt.show()
    st.pyplot(plt)

def historical_data():
    st.title("Methane Data Analysis")

    df = pd.read_csv("data/dataset/train_data/metadata.csv")
    df_plum = df[["lat", "lon", "plume"]]

    """
    maybe remove these two maps with one map image that is made not on streamplt

    """
    plume_all_df = df_plum.copy()
    plume_all_df["color"] = plume_all_df['plume'].map({'no': '#008000', 'yes': '#FF0000'})
    plume_yes_df = df_plum[df_plum["plume"] == "yes"].drop_duplicates()
    plume_no_df = df_plum[df_plum["plume"] == "no"].drop_duplicates()

    # Add a selectbox to allow the user to choose between "plume = yes" and "plume = no"
    option = st.selectbox(
        "Select data to display:",
        ("Show all locations", "Show locations with plume", "Show locations without plume"), label_visibility='collapsed'
    )

    if option == "Show locations with plume":
        st.map(plume_yes_df, use_container_width=True)  # Display the map with plume = yes
    elif option == "Show all locations":
        st.map(plume_all_df, use_container_width=True, color='color')  # Display the map with plume = yes
    else:
        st.map(plume_no_df, use_container_width=True, color='#008000')  # Display the map with plume = no

    # Extract unique cities from the train data
    cities_data = pd.read_csv("app/locations_with_cities.csv")
    unique_cities = cities_data["city"].unique()

    # User input: choose city from a dropdown or set location with latitude and longitude
    unique_cities = [
        "Select Location",
        "Find Location",
        "Example (mock data): Paris",
    ] + sorted([city for city in unique_cities if str(city) not in ["N/A", "nan"]])
    location_option = st.selectbox("Choose an option:", unique_cities, label_visibility='collapsed')

    if location_option == "Select Location":
        st.info('Please choose a location')
    elif location_option == "Example (mock data): Paris":
        mock_data = dummydata()
        display_city_name_1("Mock data - Paris", mock_data)
        """add dummy data with both plum and no plum that plot looks good"""
    elif location_option == "Find Location":
        default_latitude = 48.8566  # Default latitude for Paris
        default_longitude = 2.3522  # Default longitude for Paris
        latitude = st.number_input("Enter Latitude:", value=default_latitude)
        longitude = st.number_input(
            "Enter Longitude:", value=default_longitude)
        display_map_with_location(latitude, longitude)
    else:
        display_historical_data_for_city(location_option, cities_data, df[["date", "id_coord", "plume", "lat", "lon"]])



