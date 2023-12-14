
from flask import Flask, request, render_template
import model  # Assuming model.py contains necessary model functions
import utils  # Assuming utils.py contains necessary utility functions
import os
import joblib
import pandas as pd

app = Flask(__name__)

with open("stress_detection_model.pkl", "rb") as file:
    model = joblib.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Assuming a function in model.py to make predictions
        prediction = model.predict(pd.Series(text))[0]  
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(debug=True, host='0.0.0.0', port=port)
