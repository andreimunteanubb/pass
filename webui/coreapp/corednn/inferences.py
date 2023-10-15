""" This module contains functions that run the inference on the input image. """

from PIL import Image


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
    output_image = input_image.convert("L")

    output_image.save(f"{prefix}{output_image_hash}{suffix}")
