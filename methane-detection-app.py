import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import urllib
import time
import cv2 as cv
from app.set-up import GUI #, AppManager, DataManager

# ------------------------------------------------------#
# ------------------------------------------------------#

def imageWebApp(guiParam):
    """ """
    # Load the image according to the selected option
    conf = DataManager(guiParam)
    image = conf.load_image_source()

def main():
    """ """
    # Get the parameter entered by the user from the GUI
    guiParam = GUI().getGuiParameters()
    imageWebApp(guiParam)

# ------------------------------------------------------#
# ------------------------------------------------------#

if __name__ == "methane-detection-app":
    main()
