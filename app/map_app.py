import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def get_lat_lon(city_name, data):
    location = data[data["city"] == city_name]
    if not location.empty:
        return location.iloc[0]["lat"], location.iloc[0]["lon"]
    else:
        return None


def display_city_name(city, city_data, data):
    """
    need to prettify the plot

    """
    st.markdown(f"Selected City: {city}")
    latitude, longitude = get_lat_lon(city, city_data)
    st.map(data={"LAT": [latitude], "LON": [longitude]})
    city_data = data[(data["lat"] == latitude) & (data["lon"] == longitude)]
    city_data["date"] = pd.to_datetime(city_data["date"], format="%Y%m%d")
    st.dataframe(city_data)
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


def display_map_with_location(latitude, longitude):
    st.map(data={"LAT": [latitude], "LON": [longitude]})
    st.markdown(f"Latitude: {latitude}, Longitude: {longitude}")


def historical_data():
    st.title("Methane Data Analysis")

    df = pd.read_csv("../data/dataset/train_data/metadata.csv")
    df_plum = df[["lat", "lon", "plume"]]

    """
    maybe remove these two maps with one map image that is made not on streamplt

    """
    plume_yes_df = df_plum[df_plum["plume"] == "yes"].drop_duplicates()
    plume_no_df = df_plum[df_plum["plume"] == "no"].drop_duplicates()

    # Add a selectbox to allow the user to choose between "plume = yes" and "plume = no"
    option = st.selectbox(
        "Select data to display:",
        ("Show locations with plume", "Show locations without plume"),
    )

    if option == "Show locations with plume":
        st.map(
            plume_yes_df, use_container_width=True
        )  # Display the map with plume = yes
    else:
        st.map(plume_no_df, use_container_width=True)  # Display the map with plume = no

    # Extract unique cities from the train data
    cities_data = pd.read_csv("locations_with_cities.csv")
    unique_cities = cities_data["city"].unique()

    # User input: choose city from a dropdown or set location with latitude and longitude
    unique_cities = [
        "Select Location",
        "Add New Location",
    ] + sorted([city for city in unique_cities if str(city) not in ["N/A", "nan"]])
    location_option = st.selectbox("Choose an option:", unique_cities)

    if location_option == "Select Location":
        pass
    if location_option == "Example Data":
        """add dummy data with both plum and no plum that plot looks good"""
    elif location_option == "Add New Location":
        default_latitude = 48.8566  # Default latitude for Paris
        default_longitude = 2.3522  # Default longitude for Paris
        latitude = st.number_input("Enter Latitude:", value=default_latitude)
        longitude = st.number_input("Enter Longitude:", value=default_longitude)
        display_map_with_location(latitude, longitude)

    else:
        display_city_name(location_option, cities_data, df)
