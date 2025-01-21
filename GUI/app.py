import io
import os

import numpy
import torch
from PIL import Image # Note: pip install pillow
from io import BytesIO
import base64
import json
import re
import numpy as np
import cv2 as cv2 # Note: pip3 install opencv-python
import image_inpainting_model
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024

checkpoint_path = "./ddpm_checkpoint.save"
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device =", device)

USE_OPENCV_INPAINTING = False

def load_model():
    global model
    schedule = image_inpainting_model.Schedule(T=100, beta_1=0.0001, beta_T=0.02)
    model = image_inpainting_model.DDPM(schedule)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.compile(model)
    model.to(device)
    model.eval()


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


def convert_input_image_to_tensor(pil_image, target_size=(216, 176)):
    # RGBA -> RGB
    pil_image_rgb = pil_image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    tensor_image = transform(pil_image_rgb)

    # Adding Batch-Dimension
    return tensor_image.unsqueeze(0).to(device)


def convert_to_mask_tensor(pil_mask, target_size=(216, 176)):
    mask_rgba = pil_mask.convert("RGBA")
    cv2_mask_rgba = cv2.cvtColor(np.array(mask_rgba), cv2.COLOR_RGBA2BGRA)
    alpha_channel = cv2_mask_rgba[:, :, 3]

    # Binary mask:
    #   - if Pixel is transparent -> 0
    #   - if Pixel is NOT transparent -> 1
    _, binary_mask = cv2.threshold(alpha_channel, 0, 1, cv2.THRESH_BINARY)

    # Scaling Image
    binary_mask_resized = cv2.resize(binary_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    mask_tensor = torch.from_numpy(binary_mask_resized).float()
    final_mask_tensor = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
    final_mask_tensor = final_mask_tensor.unsqueeze(0).to(device)

    return final_mask_tensor


def convert_model_output_tensor_to_pil_image(tensor_image):
    # Denormalize [-1,1] -> [0,1]
    # x = (x+1)/2
    image = tensor_image.squeeze(0).detach().cpu()
    image = (image + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)

    # Converting to PIL Image
    pil_image = transforms.ToPILImage()(image)

    return pil_image


@torch.no_grad()
def perform_ddpm_inpainting(input_image_pil, mask_image_pil):
    # Input
    input_tensor = convert_input_image_to_tensor(input_image_pil, (216, 176))
    mask_tensor = convert_to_mask_tensor(mask_image_pil, (216, 176))
    mask_tensor = 1 - mask_tensor

    # Inpainting Results
    result_tensor = model.inpaint(input_tensor, mask_tensor, resample_steps=10)
    print("Fertig mit inpainting")
    result_pil_image = convert_model_output_tensor_to_pil_image(result_tensor)

    return result_pil_image


@app.route('/processImage', methods=['POST'])
def processImage():
    data = request.get_json()
    input_image = decode_base64_image_string(data['inputImageData'])
    mask_image = decode_base64_image_string(data['maskData'])

    input_image.save("./images/input_image.png", "PNG")
    mask_image.save("./images/mask_image.png", "PNG")

    if USE_OPENCV_INPAINTING:
        inpainted_image = perform_opencv_image_inpainting(input_image, mask_image)
    else:
        inpainted_image = perform_ddpm_inpainting(input_image, mask_image)
        print("Fertig!")

    base64_image_string = encode_base64_image(inpainted_image)

    return jsonify({"base64_image_string": f"data:image/png;base64,{base64_image_string}"})


if __name__ == '__main__':
    load_model()
    app.run()
