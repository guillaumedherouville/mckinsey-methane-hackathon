import datetime
import hashlib
import os
import time
import urllib

import cv2 as cv
import numpy as np
import pafy
import pandas as pd
import streamlit as st
import wget
import youtube_dl
from imutils.video import FPS, FileVideoStream, WebcamVideoStream
from PIL import Image


class GUI:
    """
    This class is dedicated to manage to user interface of the website. 
    It contains methods to edit the sidebar for the selected application as well as the front page.
    """

    def __init__(self):

        self.list_of_apps = [
            "Empty",
            "Methane Detection",
        ]
        self.guiParam = {}

    # ----------------------------------------------------------------

    def getGuiParameters(self):
        self.common_config()
        self.appDescription()
        return self.guiParam

    # ------------------------------------a----------------------------

    def common_config(self, title="Methane detection app"):  # (Beta version :golf:)
        """
        User Interface Management: Sidebar
        """
        st.image("./data/demo_images/logo_qb.png", "QuantumBlack", width=450)

        st.title(title)

        st.sidebar.markdown("### :arrow_right: Settings")

        self.dataSource = st.sidebar.radio(
            "Please select the source of your image :",
            ["Demo", "Upload", "TBD"],
        )

        # Get the application from the GUI
        self.selectedApp = st.sidebar.selectbox(
            "Chose an AI Application", self.list_of_apps
        )

        if self.selectedApp is "Empty":
            st.sidebar.warning("Select an application from the list")

        # Update the dictionnary
        self.guiParam.update(
            dict(
                selectedApp=self.selectedApp,
                dataSource=self.dataSource,
            )
        )

    # -------------------------------------------------------------------------

    def appDescription(self):

        st.header(" :arrow_right: Application: {}".format(self.selectedApp))

        if self.selectedApp == "Methane Detection":
            st.info(
                "This application performs methane detection using advanced deep learning models."
            )
            self.sidebarFaceDetection()

        else:
            st.info(
                "To start using QB Team 6 dashboard you must first select an application from the sidebar menu"
            )

    # --------------------------------------------------------------------------
    def sidebarEmpty(self):
        pass

    # --------------------------------------------------------------------------

    def sidebarFaceDetection(self):
        """ """

        st.sidebar.markdown("### :arrow_right: Model")
        # --------------------------------------------------------------------------
        model = st.sidebar.selectbox(
            label="Select the model",
            options=("DL_model 1", "DL_model 2"),
        )

        st.sidebar.markdown("### :arrow_right: Parameters")
        # --------------------------------------------------------------------------
        confThresh = st.sidebar.slider(
            "Confidence", value=0.80, min_value=0.0, max_value=1.00, step=0.05
        )

        self.guiParam.update(dict(confThresh=confThresh, model=model))

# ------------------------------------------------------------------
# ------------------------------------------------------------------

class DataManager:
    """ """

    def __init__(self, guiParam):
        self.guiParam = guiParam

        self.demo_image_examples = {
            "No plume 1": "./data/demo_images/no_plume/20230101_methane_mixing_ratio_id_2384.tif",
            "No plume 2": "./data/demo_images/no_plume/20230101_methane_mixing_ratio_id_4690.tif",
            "No plume 3": "./data/demo_images/no_plume/20230101_methane_mixing_ratio_id_5510.tif",

            "Plume 1": "./data/demo_images/plume/20230101_methane_mixing_ratio_id_4928.tif",
            "Plume 2": "./data/demo_images/plume/20230102_methane_mixing_ratio_id_1465.tif",
            "Plume 3": "./data/demo_images/plume/20230102_methane_mixing_ratio_id_4928.tif",
        }

        self.image = None
        self.data = None

    #################################################################
    #################################################################

    def load_image_source(self):
        """ """
        if self.guiParam["dataSource"] == "Demo":

            @st.cache_data()
            def load_image_from_path(image_path):
                image = cv.imread(image_path, cv.IMREAD_COLOR)
                # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                return image

            file_path_idx = st.selectbox(
                "Select a demo image from the list",
                list(self.demo_image_examples.keys()),
            )
            file_path = self.demo_image_examples[file_path_idx]
            self.image = load_image_from_path(image_path=file_path)
            # --------------------------------------------#
            # --------------------------------------------#

        elif self.guiParam["dataSource"] == "Upload":

            @st.cache_data()
            def load_image_from_upload(file):
                tmp = np.fromstring(file.read(), np.uint8)
                return cv.imdecode(tmp, 1)

            file_path = st.file_uploader("Upload an image", type=["png", "jpg"])

            if file_path is not None:
                self.image = load_image_from_upload(file_path)
            elif file_path is None:
                raise ValueError("[Error] Please upload a valid image ('png', 'jpg')")
            # --------------------------------------------#
            # --------------------------------------------#

        else:
            raise ValueError("Please select one source from the list")

        return self.image
