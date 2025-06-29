from flask import Flask, request, jsonify
from model_loader import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    if not data or 'inputs' not in data:
        return jsonify({'error': 'Missing input text'}), 400

    try:
        text = data['inputs']
        prediction = predict(text)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2500)
