from flask import Flask, request, jsonify
import pickle

# Load the pre-trained model
model_path = 'stress_detection_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text_input = data['text']

    # Assuming your model takes a string and returns 'stress' or 'no stress'
    prediction = model.predict([text_input])[0]

    # Convert prediction to a human-readable format
    result = 'stress' if prediction == 1 else 'no stress'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
