import random
import os
import string
from PIL import Image

from django.shortcuts import render

# Importing the inference function from the corednn package
from .corednn.inferences import inference


def generate_random_hash():
    """Generate a random hash of length 40"""
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(40)
    )


# List of all session variables
session_variables = ["before_hash", "after_hash"]


def studio_view(request):
    before_hash = request.session.get("before_hash")
    after_hash = request.session.get("after_hash")

    return render(
        request,
        "studio.html",
        {
            "before_hash": before_hash,
            "after_hash": after_hash,
        },
    )


def delete_session_view(request):
    if request.method == "POST":
        for session_variable in session_variables:
            if session_variable in request.session:
                if request.session[session_variable]:
                    try:
                        os.remove(
                            f"coreapp/static/{request.session[session_variable]}.png"
                        )
                    except Exception:
                        pass
                del request.session[session_variable]

    # rerender studio once session is deleted
    return studio_view(request)


def upload_view(request):
    if request.method == "POST" and request.FILES.get("image_file"):
        image_file = request.FILES["image_file"]
        image = Image.open(image_file)
        image_hash = generate_random_hash()
        image.save(f"coreapp/static/{image_hash}.png")
        # only save the hash in the session once the image was saved to disk
        request.session["before_hash"] = image_hash

        # This is used to add a delay between the upload and inference
        return render(
            request,
            "goto_studio.html",
            {
                "title": "Image uploaded successfully!",
                "subtitle": "Nothing to do here. Please go back to the inference studio.",
            },
        )
    # if view is not POST, render the upload page
    return render(request, "upload.html")


def inference_view(request):
    before_hash = request.session["before_hash"]
    after_hash = generate_random_hash()
    request.session["after_hash"] = after_hash

    #########################################
    # TODO: Add your inference code here
    #########################################
    inference(before_hash, after_hash)

    # rerender studio once inference is done
    return studio_view(request)
