from flask import Blueprint, request, jsonify
from .model import processor, model, generateOutputXML
import base64
from PIL import Image
import io
import numpy as np

bp = Blueprint('routes', __name__)

@bp.route('/process', methods=['POST'])
def process_image():
    data = request.json
    input_image = data['image']
    image_data_bytes = base64.b64decode(input_image)
    image = Image.open(io.BytesIO(image_data_bytes))
    receipt_image_array = np.array(image.convert('RGB'))
    result = generateOutputXML(processor, model, receipt_image_array)
    return jsonify(result)