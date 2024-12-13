from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input features from the form
        features = [float(x) for x in request.form.values()]
        # Predict using the model
        prediction = model.predict([features])
        species = prediction[0]
        return render_template('index.html', prediction_text=f'Predicted Species: {species}')

if __name__ == '__main__':
    app.run(debug=True)