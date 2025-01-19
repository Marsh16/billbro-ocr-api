from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import numpy as np
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

# Initialize the model and processor globally
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "mychen76/invoice-and-receipts_donut_v1"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
model.eval()  # Set to evaluation mode

def process_image(image_base64):
    """Process base64 image and return receipt data"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image if too large
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.size[0] * ratio), int(image.size[1] * ratio)),
                Image.Resampling.LANCZOS
            )
        
        # Convert image to array
        image_array = np.array(image)
        
        # Process image with the processor
        pixel_values = processor(image_array, return_tensors="pt").pixel_values
        
        # Prepare decoder input
        task_prompt = "<s_receipt>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        )["input_ids"]
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode the output
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        
        # Convert to JSON
        result = processor.token2json(sequence)
        return result
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome To BillBro"

@app.route('/api/generate', methods=['POST'])
def generate_text():
    try:
        # Get JSON data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Process image and generate receipt data
        result = process_image(data['image'])
        
        return jsonify({"receipt": result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()