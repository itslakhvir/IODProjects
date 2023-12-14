
from flask import Flask, request, jsonify
import pickle
import os
import logging
from utils import preprocess_text  # assuming there's a preprocessing function in utils.py

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the pre-trained model
model_path = os.path.join(os.getcwd(), 'stress_detection_model.pkl')
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    logging.error("Model file not found.")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text_input = data['text']
        # Preprocess the text input if necessary
        preprocessed_text = preprocess_text(text_input)

        # Prediction
        prediction = model.predict([preprocessed_text])[0]

        return jsonify({"prediction": prediction}), 200
    except KeyError:
        return jsonify({"error": "Invalid input format. 'text' key is required."}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
