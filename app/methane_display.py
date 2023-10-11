import streamlit as st
import numpy as np
import cv2 as cv

def image_upload():
    """ """
    st.title("Methane Detection")

    @st.cache_data()    # Basic caching + rewriting to allow tif images
    def load_image_from_upload(file):
        tmp = np.fromstring(file.read(), np.uint8)
        return cv.imdecode(tmp, 1)

    file_path = st.file_uploader("john", label_visibility='collapsed', type=["png", "jpg", "tif"])

    if file_path is not None:
        left_co, cent_1_co, cent_2_co, last_co = st.columns(4)
        with cent_1_co:
            st.image(load_image_from_upload(file_path), width=300)
    else:
        st.info("Please upload an image to get started")
    
def model_choice():
    """ """
    with st.sidebar:
        st.text('')
        # Model choice
        st.markdown("### :arrow_right: Model")
        model = st.selectbox(
            label="Select the model",
            options=("DL_model 1", "DL_model 2"),)
        # Confidence choice
        st.text('')
        st.markdown("### :arrow_right: Parameters")
        confThresh = st.sidebar.slider("Confidence", value=0.80, min_value=0.0, max_value=1.00, step=0.05)
    return model, confThresh


def methane_detection():
    image_upload()
    model_choice()
