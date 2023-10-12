import streamlit as st
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import io
import rasterio
import cv2


def image_choice():
    """
    Gives option of image source to the user : use a demo file or upload a new one

    Returns:
        str: choice of image source
    """
    st.sidebar.markdown("### :arrow_right: Image source")
    dataSource = st.sidebar.radio(
        "Please select the source of your image :",
        ["Demo", "Upload"],
    )
    return dataSource


def image_demo():
    """
    Generates several paths for demo images and shows a select box to choose

    Returns:
        str: path to the chosen demo image
    """
    demo_image_examples = {
        "No plume 1": "./data/dataset/train_data/images/no_plume/20230101_methane_mixing_ratio_id_2384.tif",
        "No plume 2": "./data/dataset/train_data/images/no_plume/20230101_methane_mixing_ratio_id_4690.tif",
        "No plume 3": "./data/dataset/train_data/images/no_plume/20230101_methane_mixing_ratio_id_5510.tif",
        "Plume 1": "./data/dataset/train_data/images/plume/20230101_methane_mixing_ratio_id_4928.tif",
        "Plume 2": "./data/dataset/train_data/images/plume/20230102_methane_mixing_ratio_id_1465.tif",
        "Plume 3": "./data/dataset/train_data/images/plume/20230102_methane_mixing_ratio_id_4928.tif",
    }
    file_path_idx = st.selectbox(
        "Select a demo image from the list",
        list(demo_image_examples.keys()),
    )
    file_path = demo_image_examples[file_path_idx]
    return file_path


def image_upload(input_type):
    """
    Display a TIFF image uploaded by the user.
    If an image is uploaded, it is displayed in the Streamlit app.

    Returns:
        np.array: The normalized image in greyscale format (64x64)
    """
    if input_type == "Upload":
        source_file = st.file_uploader("Upload a TIFF Image", type=["tif", "tiff"])
    else:
        source_file = image_demo()

    if source_file is not None:
        # Read the uploaded TIFF image
        tiff_image = Image.open(source_file)
        array_plot = np.array(tiff_image)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(array_plot, cmap="gray")
        ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)

        # Normalize the image
        min_val = array_plot.min()
        max_val = array_plot.max()
        image = (array_plot - min_val) / (max_val - min_val)
        return image
    else:
        return None


def model_choice():
    """
    Display a selection box in the Streamlit sidebar to choose a model.

    Returns:
        str: The selected model's filename.
    """
    with st.sidebar:
        st.text("")

        # Model choice
        st.markdown("### :arrow_right: Model")
        models = [model for model in os.listdir("model/")]
        models.remove('__init__.py')
        model = st.selectbox(label="Select the model", options=(models))
    return model


def prediction(model, image):
    """
    Predict a label and probability for an image using a given model.

    Args:
        model (str): The model filename to load and use for prediction.
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        tensorflow.keras.Model: The loaded model.
    """
    # This adds for a status bar
    with st.status("Predicting label", expanded=True) as status:
        st.header(model)

        # Load the selected model and predict label
        running_model = keras.models.load_model("model/" + model)
        predicted_label = round((running_model.predict(np.array([image]))[0][0]), 5)
        label = 1 if predicted_label >= 0.5 else 0

        # Display prediction
        st.subheader(f"- Predicted Label: {label} ")
        st.subheader(f"- Predicted Probability: {predicted_label}")

        return running_model


def heatmap_box(running_model, image):
    """
    Generate and display a heatmap overlay on an input image.

    Args:
        running_model (tensorflow.keras.Model): The loaded model.
        image (numpy.ndarray): The input image as a NumPy array.
    """
    from heatmap import heatmap

    # Find the last convolutional layer in the model

    last_conv_layer = None
    for layer in running_model.layers[::-1]:
        if isinstance(layer, keras.layers.Conv2D):
            last_conv_layer = layer

    # Generate the heatmap
    heatmap_generator = heatmap()
    heatmap_image = heatmap_generator.make_heatmap(
        np.array([image]), running_model, last_conv_layer.name
    )
    heatmap_image = cv2.resize(heatmap_image, (64, 64))

    # Display the original image and heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image / 100, cmap="gray")
    ax.axis("off")
    agree = st.checkbox("HeatMap")

    # Add a checkbox to allow users to toggle the heatmap overlay
    if agree:
        ax.imshow(heatmap_image, alpha=0.4)
        buffed = io.BytesIO()
        fig.savefig(buffed, format="png", bbox_inches="tight", pad_inches=0.05)
        st.image(buffed)
        st.markdown(
            "The HeatMap provides insights into which parts of \
                an image were most influential in determining\
                 the model's prediction"
        )
    else:
        buffed = io.BytesIO()
        fig.savefig(buffed, format="png", bbox_inches="tight", pad_inches=0.05)
        st.image(buffed)


def methane_detection():
    """
    Perform methane detection by selecting a model,
    choosing an image (demo or upload), and displaying results.
    """
    st.title("Methane Detection")
    st.info(
        f"Welcome Methane Detectio portal."
        f"The Methane Monitoring Application harnesses a robust Convolutional Neural Network (CNN) model, "
        f"crafted using advanced software from reputable organizations like Meta. "
        f"Designed for real-time methane plume detection, users can upload satellite images to the app, "
        f"which promptly analyzes them to identify methane presence. Additionally, "
        f"it generates intuitive heatmaps of the images, aiding users in visually identifying "
        f"the concentration and dispersion of methane plumes. This fusion of cutting-edge technology "
        f"and user-centric features, alongside real-time analytical capabilities, positions "
        f"the Methane Monitoring Application as a reliable and effective tool for environmental "
        f"safety and methane emissions monitoring."
    )
    source_image = image_choice()
    model = model_choice()
    image = image_upload(source_image)
    if image is None:
        return

    running_model = prediction(model, image)
    heatmap_box(running_model, image)

