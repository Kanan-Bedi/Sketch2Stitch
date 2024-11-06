import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# Load the generator model
generator = tf.keras.models.load_model('/Users/kananbedi/Desktop/gans/model/generator_2 (2).h5')

# Function to preprocess the image
def preprocess_image(image_path, target_size=(256, 256)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array /= 255.0  # Normalize the pixel values
    image_array = tf.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to generate image using the generator model
def generate_image(image_path, generator_model):
    input_shape = generator_model.input_shape[1:3]
    image_array = preprocess_image(image_path, target_size=input_shape)
    predicted_output = generator_model.predict(image_array)
    predicted_image = tf.keras.preprocessing.image.array_to_img(predicted_output[0])
    return predicted_image

# Streamlit app
def main():
    st.title("GAN Image Generation")
    st.sidebar.title("Settings")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Generate image using the uploaded file
        generated_image = generate_image(uploaded_file, generator)
        st.image(generated_image, caption="Generated Image", use_column_width=True)
    else:
        st.sidebar.info("Please upload an image")

if __name__ == "__main__":
    main()
