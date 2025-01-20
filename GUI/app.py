import io
from PIL import Image # Note: pip install pillow
from io import BytesIO
import base64
import json
import re

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


@app.route('/processImage', methods=['POST'])
def processImage():
    data = request.get_json()
    image = decode_base64_image_string(data['inputImageData'])
    mask = decode_base64_image_string(data['maskData'])

    mask.show()
    # Mit image.convert(...) kann das Bild in JPEG oder PNG umgewandelt werden
    #image.convert('RGB').save("./images/canvas.jpg", "JPEG")
    image.save("./images/image.png", "PNG")
    mask.save("./images/mask.png", "PNG")

    # TODO: "image" in entsprechenden datentypen umwandeln
    # Output von Modell in einen Base64 String umwandeln und zurückschicken

    # DUMMMMMYYYYYY
    image = Image.open("./images/image.png", mode='r')
    base64_image_string = encode_base64_image(image)

    return jsonify({"base64_image_string": f"data:image/png;base64,{base64_image_string}"})


if __name__ == '__main__':
    load_model()
    app.run()
