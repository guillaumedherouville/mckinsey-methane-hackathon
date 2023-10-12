import re
import openai
import streamlit as st
import pandas as pd
from map_app import display_historical_data_for_city

OPEN_API_KEY = ""


def generate_manufacturers_list_from_location(latitude, longitude, city, api_key):
    openai.api_key = api_key
    if city is None:
        prompt = (
            f"I will give you latitude and longitude, i want you to say the name of the city or region and give me lists of 3 main companies-manufacturers in that region.\n\n"
            f"Latitude: {latitude}\n"
            f"Longitude: {longitude}\n\n"
            f"Desired format:\nRegion:\n"
            f"List of manufacturers in this area:\n"
        )
    else:
        prompt = (
            f"I will give you city, i want you to say the name of the city or region and give me lists of 3 main companies-manufacturers in that region.\n\n"
            f"City: {city}\n\n"
            f"Desired format:\nRegion:\n"
            f"List of manufacturers in this area:\n"
        )
    prediction = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )["choices"][0]
    region_match = re.search(r"Region:\s*(\w+)", prediction["message"]["content"])
    region = region_match.group(1) if region_match else None
    if region is None:
        return None, None

    start_phrase = "List of manufacturers in this area:"
    manufacturers_matches = re.search(
        f"{re.escape(start_phrase)}(.*)", prediction["message"]["content"], re.DOTALL
    )
    manufacturers = (
        manufacturers_matches.group(1)
        if manufacturers_matches
        else "No manufacturers found in this region."
    )

    return region, manufacturers


def generate_regulators_list_from_location(latitude, longitude, city, api_key):
    openai.api_key = api_key
    if city is None:
        prompt = (
            f"I will give you latitude and longitude, i want you to give me lists of 3 ESG regulators and lists of 3 research institutions that may be responsible for this region and may be interested in methane detection. \n\n"
            f"Latitude: {latitude}\n"
            f"Longitude: {longitude}\n\n"
            f"Desired format:\nRegion:\n"
            f"List of regulators in this area:\n"
        )
    else:
        prompt = (
            f"I will give you latitude and longitude, i want you to give me lists of 3 ESG regulators and lists of 3 research institutions that may be responsible for this region and may be interested in methane detection. \n\n"
            f"City: {city}\n\n"
            f"Desired format:\nRegion:\n"
            f"List of regulators in this area:\n"
        )

    prediction = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )["choices"][0]

    regulator_phrase = "List of regulators in this area:"
    research_phrase = "List of research institutions in this area:"

    regulators_matches = re.compile(
        f"{re.escape(regulator_phrase)}(.*?){re.escape(research_phrase)}", re.DOTALL
    ).search(prediction["message"]["content"])
    regulators = (
        regulators_matches.group(1)
        if regulators_matches
        else "No ESG regulators found in this region."
    )

    researchers_matches = re.search(
        f"{re.escape(research_phrase)}(.*)", prediction["message"]["content"], re.DOTALL
    )
    researchers = (
        researchers_matches.group(1)
        if researchers_matches
        else "No ESG regulators found in this region."
    )

    return regulators, researchers


def discover_location(latitude, longitude, city, api_key):
    loading_message = (
        f"Discovering opportunities in {city}..."
        if city is not None
        else f"Discovering new location..."
    )
    with st.spinner(loading_message):
        region_name, manufacturers_list = generate_manufacturers_list_from_location(
            latitude, longitude, city, api_key
        )
    if city is None:
        st.markdown(f"Region: {region_name}")
    st.markdown(f"List of main manufacturers in this region:\n {manufacturers_list}")
    with st.spinner(loading_message):
        regulators_list, researchers_list = generate_regulators_list_from_location(
            latitude, longitude, city, api_key
        )
    st.markdown(f"List of main ESG regulators in this region:\n {regulators_list}")
    st.markdown(
        f"List of main research institutions in this region:\n {researchers_list}"
    )


def cleanr_display():
    api_key = ""
    st.title("CleanR Workspace")

    st.info(
        f"Welcome to CleanR Workspace!\n"
        f"Here, we provide a comprehensive repository of analyzed data, both from past assessments and new locations. "
        f"Whether you're looking to explore previously examined areas or discover fresh sites, "
        f"we offer you valuable insights into potential methane clients.\n\n"
        f"The potential beneficiaries of methane detection model can be categorized into three main groups:\n"
        f"1. Manufacturers: For businesses in the energy sector, staying compliant with Environmental, Social, and Governance standards is vital."
        f"The data can help them monitor methane emissions, attract investment, and enhance operational efficiency.\n\n"
        f"2. Regulators: Regulators play a crucial role in ensuring that producers adhere to environmental regulations. "
        f"Our platform assists regulators in cross-checking reported methane emissions, supporting their efforts in upholding compliance standards.\n\n"
        f"3. Research Institutions: For academic and research institutions, tracking and studying methane leakages "
        f"are essential in understanding environmental patterns and resource utilization. "
        f"Our data enables research institutions to explore, analyze, and gain insights into methane emissions and leakages."
    )

    """
    add some summary info on the data we have - how many locations, how many of them have plum, what countries, cities, etc.

    """

    df = pd.read_csv("data/dataset/train_data/metadata.csv")

    # Extract unique cities from the train data
    cities_data = pd.read_csv("app/locations_with_cities.csv")
    unique_cities = cities_data["city"].unique()

    # User input: choose city from a dropdown or set location with latitude and longitude
    unique_cities = [
        "Select Location",
        "Add New Location",
    ] + sorted([city for city in unique_cities if str(city) not in ["N/A", "nan"]])
    location_option = st.selectbox("Select location or add new one", unique_cities)

    if location_option == "Select Location":
        pass
    elif location_option == "Add New Location":
        api_key = st.text_input("Enter you API_KEY to access new location discovery:")
        latitude = st.number_input("Enter Latitude:")
        longitude = st.number_input("Enter Longitude:")

        if st.button("Discover New Location"):
            if latitude is not None and longitude is not None and api_key != "":
                discover_location(latitude, longitude, None, api_key)

            elif api_key == "":
                st.warning("Please set you API_KEY to access new location discovery.")
            elif latitude is None or longitude is None:
                st.warning(
                    "Please fill in both latitude and longitude before discovering a new location."
                )

    else:
        api_key = st.text_input("Enter you API_KEY to access new location discovery:", type='password')
        if api_key == "":
            st.warning("Please set you API_KEY to access new location discovery.")
        else:
            display_historical_data_for_city(location_option, cities_data, df)
            discover_location(None, None, location_option, api_key)
