import io

import numpy
from PIL import Image # Note: pip install pillow
from io import BytesIO
import base64
import json
import re
import numpy as np
import cv2 as cv2 # Note: pip3 install opencv-python

import flask
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024

model = None

def load_model():
    global model
    # TODO: Modell laden -> architektur muss entsprechend auch in diesem Projekt definiert werden!
    # bzw. am besten ein gesamtes python projekt erstellen idk
    # sodass mit pickle Einlesen möglich ist..
    model = ...


@app.route('/')
def home_page_image_editor():  # Home (index.html)
    return render_template('index.html')


def decode_base64_image_string(base_image_string):
    # https://stackoverflow.com/questions/41957490/send-canvas-image-data-uint8clampedarray-to-flask-server-via-ajax
    image_data = re.sub('^data:image/.+;base64,', '', base_image_string)

    return Image.open(BytesIO(base64.b64decode(image_data))).convert("RGBA")


def encode_base64_image(image):
    # https://stackoverflow.com/questions/11017466/flask-to-return-image-stored-in-database/11017839#11017839
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')

    return base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')


def convert_to_binary_mask(mask_image):
    # https://stackoverflow.com/questions/61918194/how-to-make-a-binary-mask-out-of-an-image-with-a-transparent-background
    cv2_mask_image = cv2.cvtColor(numpy.array(mask_image), cv2.COLOR_RGBA2BGRA)

    # We extract the alpha channel
    alpha_channel = cv2_mask_image[:, :, 3]

    # Binary mask:
    #   - if Pixel is transparent -> 0
    #   - if Pixel is NOT transparent -> 1
    _, binary_mask = cv2.threshold(alpha_channel, 0, 1, cv2.THRESH_BINARY)

    return binary_mask

def perform_opencv_image_inpainting(input_image, mask_image):
    # https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
    cv2_input_image = cv2.cvtColor(numpy.array(input_image), cv2.COLOR_RGBA2BGR)
    binary_mask = convert_to_binary_mask(mask_image)

    inpainted_image = cv2.inpaint(cv2_input_image, binary_mask, 3, cv2.INPAINT_TELEA)
    inpainted_pil_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

    return inpainted_pil_image


@app.route('/processImage', methods=['POST'])
def processImage():
    data = request.get_json()
    input_image = decode_base64_image_string(data['inputImageData'])
    mask_image = decode_base64_image_string(data['maskData'])

    input_image.save("./images/input_image.png", "PNG")
    mask_image.save("./images/mask_image.png", "PNG")

    inpainted_image = perform_opencv_image_inpainting(input_image, mask_image)

    base64_image_string = encode_base64_image(inpainted_image)

    return jsonify({"base64_image_string": f"data:image/png;base64,{base64_image_string}"})


if __name__ == '__main__':
    load_model()
    app.run()
