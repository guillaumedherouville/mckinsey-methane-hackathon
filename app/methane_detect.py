import streamlit as st
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import io
import rasterio
import cv2

def image_upload():
    """Upload images to streamlit app with this function

    Returns:
        np.array: The normalized image in greyscale format (64x64)
    """
    # Create a file uploader
    uploaded_file = st.file_uploader("Upload a TIFF Image", type=["tif", "tiff"])

    if uploaded_file is not None:
        # Read the uploaded TIFF image
        tiff_image = Image.open(uploaded_file)
        array_plot = np.array(tiff_image)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(array_plot, cmap="gray")
        ax.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches = 0.05)
        #st.image(buf, width=500)
        min_val = array_plot.min()
        max_val = array_plot.max()
        # Normalize the image data
        image = (array_plot - min_val) / (max_val - min_val)
        return image
    else:
        return None
    
    

def model_choice():
    """ """
    with st.sidebar:
        st.text('')
        # Model choice
        st.markdown("### :arrow_right: Model")
        model = st.selectbox(
            label = "Select the model",
            options = (os.listdir("model/")))
        # Confidence choice
        st.text('')
        #st.markdown("### :arrow_right: Parameters")
        #confThresh = st.sidebar.slider("Confidence", value=0.80, min_value=0.0, max_value=1.00, step=0.05)
    return model

def prediction(model, image):
    """load the model and upload the prediction

    Args:
        model (tf.keras.model): The fitted model instance
        image (np.array): Greyscale image of infrared sattelite image (64x64)

    Returns:
        tf.keras.model: The loaded model
    """
    with st.status("Predicting label", expanded=True) as status:
        st.header(model)
        running_model = keras.models.load_model("model/" + model)
        predicted_label = round((running_model.predict(np.array([image]))[0][0]), 5)
        label = (1 if predicted_label>=0.5 else 0)
        st.subheader(f"- Predicted Label: {label} ")
        st.subheader(f"- Predicted Probability: {predicted_label}")
        return running_model
        



def heatmap_box(running_model, image):
    """Produce the heatmap

    Args:
        running_model (tf.keras.model): The model to infer with
        image (np.array): The image to plot as heatmap
    """
    from heatmap import heatmap
    
    # get last layer
    last_conv_layer = None
    for layer in running_model.layers[::-1]:
        if isinstance(layer, keras.layers.Conv2D):
            last_conv_layer = layer
            
    # get heatmap
    heatmap_generator = heatmap()
    heatmap_image = heatmap_generator.make_heatmap(np.array([image]), running_model, last_conv_layer.name)
    heatmap_image = cv2.resize(heatmap_image,(64, 64))

    # Display image and heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image/100, cmap="gray")
    ax.axis('off')
    agree = st.checkbox('HeatMap') 
    if agree:
        ax.imshow(heatmap_image, alpha=0.4)
        buffed = io.BytesIO()
        fig.savefig(buffed, format="png", bbox_inches='tight', pad_inches = 0.05)
        st.image(buffed)
        st.markdown("The HeatMap provides insights into which parts of \
                an image were most influential in determining\
                 the model's prediction")
    else:
        buffed = io.BytesIO()
        fig.savefig(buffed, format="png", bbox_inches='tight', pad_inches = 0.05)
        st.image(buffed)
    
def methane_detection():
    model = model_choice()
    image = image_upload()
    if image is None:
        return

    running_model = prediction(model, image)
    heatmap_box(running_model, image)
