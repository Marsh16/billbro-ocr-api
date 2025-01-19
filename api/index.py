from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import numpy as np
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

def load_model():
    """
    Optimized model loading with explicit configuration management
    """
    device = "cpu"  # Vercel primarily uses CPU
    model_name = "mychen76/invoice-and-receipts_donut_v1"
    
    try:
        # Explicitly set use_fast=True to address processor warning
        processor = DonutProcessor.from_pretrained(
            model_name, 
            use_fast=True  # Explicitly set fast tokenizer
        )
        
        # Load model with careful configuration
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float32  # Explicit dtype specification
        ).to(device)
        
        model.eval()
        return model, processor
    
    except Exception as e:
        print(f"Model loading error: {e}")
        raise

def process_image(image_base64):
    """
    Process base64 encoded image with optimized model configuration
    """
    try:
        # Decode and prepare image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Ensure RGB and resize
        if image.mode != 'RGB':
            image = image.convert('RGB')

        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.size[0] * ratio), int(image.size[1] * ratio)),
                Image.Resampling.LANCZOS
            )

        # Load model with careful configuration
        model, processor = load_model()

        # Prepare image for processing
        image_array = np.array(image)
        pixel_values = processor(
            image_array, 
            return_tensors="pt", 
            # Add any specific preprocessing parameters
            do_resize=True,
            size={"height": 960, "width": 1280}  # Match model's expected input size
        ).pixel_values

        # Prepare decoder input
        task_prompt = "<s_receipt>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, 
            add_special_tokens=False,
            return_tensors="pt",
            # Explicit tokenizer configuration
            truncation=True,
            max_length=768
        )["input_ids"]

        # Generate model output with careful configuration
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=2,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Process and decode model output
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")

        # Convert to JSON
        result = processor.token2json(sequence)
        return result

    except Exception as e:
        print(f"Image processing error: {e}")
        raise Exception(f"Error processing image: {str(e)}")

def handler(request):
    """
    Vercel serverless function handler for processing receipt images
    
    Args:
        request: Flask request object
    
    Returns:
        Flask response with processed receipt data or error message
    """
    # Ensure request is a POST method
    if request.method == 'POST':
        try:
            # Get JSON data
            data = request.get_json()
            
            # Validate input
            if not data or 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400

            # Process image and generate receipt data
            result = process_image(data['image'])
            return jsonify({"receipt": result})

        except Exception as e:
            # Log the error and return a generic error response
            print(f"Request processing error: {e}")
            return jsonify({'error': 'Failed to process receipt image'}), 500
    
    # Handle non-POST requests
    return jsonify({"message": "Send a POST request with an image"}), 405

# Flask app for local development and testing
app = Flask(__name__)

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """Local development route mirroring Vercel serverless function"""
    return handler(request)

@app.route('/', methods=['GET'])
def home():
    """Simple health check route"""
    return "Welcome to BillBro Receipt Processing"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)