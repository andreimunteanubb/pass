""" This module contains functions that run the inference on the input image. """
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def draw_image(image: Image, text: str):
    image = np.array(image.convert('L'))
    image = Image.fromarray(np.repeat(image[..., np.newaxis], 3, axis=2))
    font = ImageFont.truetype("segoeui.ttf", 90)
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
    input_image = Image.open(f"{prefix}{input_image_hash}{suffix}")

    #########################################
    # TODO: Add your inference code here
    #########################################
    text = "Benign"

    image = draw_image(input_image, text)

    image.save(f"{prefix}{output_image_hash}{suffix}")
