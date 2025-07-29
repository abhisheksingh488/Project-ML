from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('gld_price_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        spx = float(request.form['spx'])
        gld = float(request.form['gld'])
        uso = float(request.form['uso'])
        slv = float(request.form['slv'])

        # Prepare input data for prediction
        input_data = np.array([[spx, gld, uso, slv]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Predicted EUR/USD: {prediction:.4f}')
    except:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)
