import streamlit as st
from methane_detect import methane_detection
from map_app import historical_data
from clean_r import cleanr_display


def home_page():
    st.title("Home Page")

    st.info(
        f"Welcome to the future of environmental stewardship!\n\n"
        f"Methane reduction is paramount to a sustainable future, and we believe in working together to make a difference. "
        f"Our app empowers you to harness the power of satellite imagery for a greener world. "
        f"Whether you're an environmental enthusiast, a researcher, or a business leader, you'll find this app"
        f" to be your trusted ally in the battle against methane emissions.\n\n"
        f"Explore Our Three Portals:\n\n"
        f"💪 Methane Detection Portal: Have your own satellite image to analyze? "
        f"Simply upload it, and our state-of-the-art model will swiftly detect the presence of methane. "
        f"Curious to see how it works? No problem! We provide preloaded images for you to grasp the model's capabilities.\n\n"
        f"🌱 Data Analysis Portal: Dive into historical data for various locations where we've gathered information. "
        f"Our intuitive graphs and visuals make it easy to interpret the methane detection results, "
        f"helping you gain valuable insights.\n\n"
        f"🖇️ CleanR Workspace Portal: This exclusive space is designed specifically for our CleanR team. "
        f"Beyond the app's standard features, we assist the CleanR team in connecting with potential clients "
        f"who share our vision for a methane-reduced world. Together, we build a brighter and cleaner future.\n\n"
        f"Join us in the fight against methane emissions, and let's work together to protect our planet.\n\n"
        f"Your journey begins here 🌿"
    )


def main():
    st.sidebar.title("Choose Display")
    display_option = st.sidebar.radio(
        "Select an option",
        ("Home Page", "Methane Detection", "Data Analysis", "CleanR Workspace"),
    )

    if display_option == "Home Page":
        home_page()
    if display_option == "Data Analysis":
        historical_data()
    elif display_option == "Methane Detection":
        methane_detection()
    elif display_option == "CleanR":
        cleanr_display()


if __name__ == "__main__":
    main()
