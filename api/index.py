from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from flask import request, jsonify
import base64
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
# limiter = Limiter(
#     get_remote_address,
#     app=app,
#     default_limits=["200 per day", "50 per hour"]
# )
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# model_name = "mychen76/invoice-and-receipts_donut_v1"
# processor = DonutProcessor.from_pretrained(model_name)
# model = VisionEncoderDecoderModel.from_pretrained(model_name, torch_dtype=torch.float32, max_memory='1GB')

# # Quantize the model
# model.eval()
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# model = torch.quantization.prepare(model, inplace=True)

# model.to("cuda" if torch.cuda.is_available() else "cpu")

# def generateTextInImage(processor, model, input_image, task_prompt="<s_receipt>"):
#     pixel_values = processor(input_image, return_tensors="pt").pixel_values
#     print("input pixel_values: ", pixel_values.shape)
#     task_prompt = "<s_receipt>"
#     decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     outputs = model.generate(pixel_values.to(device),
#                              decoder_input_ids=decoder_input_ids.to(device),
#                              max_length=model.decoder.config.max_position_embeddings,
#                              early_stopping=True,
#                              pad_token_id=processor.tokenizer.pad_token_id,
#                              eos_token_id=processor.tokenizer.eos_token_id,
#                              use_cache=True,
#                              num_beams=12,
#                              bad_words_ids=[[processor.tokenizer.unk_token_id]],
#                              return_dict_in_generate=True,
#                              output_scores=True,)
#     print()
#     return outputs

# def generateOutputXML(processor, model, input_image, task_start="<s_receipt>"):
#     import re
#     outputs = generateTextInImage(processor, model, input_image, task_prompt=task_start)
#     sequence = processor.batch_decode(outputs.sequences)[0]
#     sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
#     sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
#     return sequence

# def generateOutputJson(processor,model, input_image, task_start="<s_receipt>",task_end="</s_receipt>"):
#     image_data_bytes = base64.b64decode(input_image)
#     image = Image.open(io.BytesIO(image_data_bytes))
#     receipt_image_array = np.array(image.convert('RGB'))
#     xml = generateOutputXML(processor,model, receipt_image_array,task_start=task_start)
#     result=processor.token2json(xml)
#     print(":vampire:",result)
#     return result

@app.route('/', methods=['GET'])
def home():
    """Simple health check route"""
    return "Welcome to BillBro Receipt Processing"

# @app.route('/process', methods=['POST'])
# def process_image():
#     data = request.json
#     input_image = data['image']
#     result = generateOutputJson(processor, model, input_image)
#     return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)