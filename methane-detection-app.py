import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import urllib
import time
import cv2 as cv
from app.setup import GUI, DataManager

# ------------------------------------------------------#
# ------------------------------------------------------#

def imageWebApp(guiParam):
    """ """
    # Load the image according to the selected option
    conf = DataManager(guiParam)
    image = conf.load_image_source()
    left_co, cent_1_co, cent_2_co, last_co = st.columns(4)
    with cent_1_co:
        st.image(image, width=300)

def main():
    """ """
    # Get the parameter entered by the user from the GUI
    guiParam = GUI().getGuiParameters()
    imageWebApp(guiParam)

# ------------------------------------------------------#
# ------------------------------------------------------#

main()
