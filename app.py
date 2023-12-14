
from flask import Flask, request, render_template
import model  # Assuming model.py contains necessary model functions
import utils  # Assuming utils.py contains necessary utility functions
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Assuming a function in model.py to make predictions
        prediction = model.predict_stress(text)  
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
    
