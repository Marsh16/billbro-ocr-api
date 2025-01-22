from flask import Flask
from flask import request, jsonify
import modelbit

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "Welcome to BillBro Receipt Processing"

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    input_image = data['data']
    result = modelbit.get_inference(
        region="us-east-1.aws",
        workspace="marshaalexis",
        deployment="predict_receipt",
        api_key="milHpxF4VK:msa/paZoJNdpTROEkd5OM/Zp2IlG/i7MRBCv4jjcHjjXM=",
        data=input_image
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run()