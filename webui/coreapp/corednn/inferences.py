""" This module contains functions that run the inference on the input image. """
import os
import numpy as np
import tensorflow as tf
from azure.storage.blob import BlobClient
from PIL import Image, ImageFont, ImageDraw

MODEL_FILEPATH = "coreapp/static/model.h5"


def get_model_from_azure():
    """Download model from azure blob storage.

    Returns:
        bytes: Model as bytes.
    """
    sas_url = r"https://thesis0772655842.blob.core.windows.net/azureml/model?sp=r&st=2023-11-28T12:38:17Z&se=2024-10-31T20:38:17Z&spr=https&sv=2022-11-02&sr=b&sig=zrVEt8ALfQ2mAzB%2F1UYX%2BjpDIjtJp9HpVqkRibUvXMg%3D"
    client = BlobClient.from_blob_url(sas_url)
    model_data = client.download_blob()
    return model_data.readall()


def get_model():
    """If model isn't already downloaded, download it. Otherwise just use it.

    Returns: Tensorflow Model
    """
    if not os.path.exists(MODEL_FILEPATH):
        print("Model is downloading...")
        with open(MODEL_FILEPATH, "wb") as file:
            file.write(get_model_from_azure())
        print("Successfully downloaded.")
    else:
        print("Model already exists.")

    return tf.keras.models.load_model(MODEL_FILEPATH)


def draw_image(image: Image, text: str):
    image = np.array(image.convert("L"))
    image = Image.fromarray(np.repeat(image[..., np.newaxis], 3, axis=2))
    font = ImageFont.truetype("coreapp/static/Roboto-Regular.ttf", 90)
    draw = ImageDraw.Draw(image)
    h, w = image.size
    position = (h // 2 - 70, w // 2 - 90)

    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(37, 150, 190))
    draw.text(position, text, font=font, fill="black")

    return image


def inference(input_image_hash: str, output_image_hash: str):
    """
    Summary: This function reads an image from the static folder, applies a grayscale transformation on it
    and saves the output image back in the static folder.

    It takes as parameters both the hash of the input image, as well as the hash of the output image.
    Even though we receive only the hashes, the full path of the images can be found by adding the prefix
    "coreapp/static/" to the hashes and the suffix ".png".

    Note: No file exists at the output image path when this function is called, but it will be created by
    this function.

    Args:
        input_image_hash (str): Hash of the image on which the inference is run
        output_image_hash (str): Image will be saved with this hash
    """

    prefix = "coreapp/static/"
    suffix = ".png"
    original_image = Image.open(f"{prefix}{input_image_hash}{suffix}")

    # get model from azure blob storage
    model = get_model()

    target_size = (224, 224)
    resized_image = original_image.resize(target_size)

    image_array = np.array(resized_image)
    input_data = np.expand_dims(image_array, axis=0)

    input_data = input_data.astype("float32") / 255.0

    prediction = np.argmax(model.predict(input_data))
    text = ["Benign", "Early Pre-B", "Pre-B", "Pro-B"]

    image = draw_image(original_image, text[prediction])

    image.save(f"{prefix}{output_image_hash}{suffix}")
