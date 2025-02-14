from flask import Flask
from flask import request, jsonify
from apple_ocr.ocr import OCR
import base64
from PIL import Image
import io

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_image():
    data = request.json
    input_image = data['data']
    image_data = base64.b64decode(input_image)
    image = Image.open(io.BytesIO(image_data))
    ocr_instance = OCR(image=image)
    dataframe = ocr_instance.recognize()
    prompt = f"""
    {dataframe}
    """
    print(prompt)
    return jsonify(prompt)

if __name__ == '__main__':
    app.run()