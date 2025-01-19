from flask import Flask, request, jsonify
import generate
import model

app = Flask(__name__)

@app.route('/', methods=['GET']) 
def home(): 
    return "Welcome To BillBro"

@app.route('/api/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    image_base64 = data.get('image')

    try:
        result = generate.generateOutputJson(model.tokenizer, model.model, image_base64)
        return jsonify({"receipt": result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__": 
    app.run()
