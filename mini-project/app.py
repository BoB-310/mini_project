# We need redirect and url_for to send the user back to the main page
from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('grade_predictor_model.pkl')
except FileNotFoundError:
    print("FATAL ERROR: grade_predictor_model.pkl not found. Please run train_model.py first.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

# --- THIS IS THE UPDATED PART ---
# The /predict route now accepts both POST and GET methods
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # If the request is a POST, it means the form was submitted.
    if request.method == 'POST':
        try:
            form_data = [float(x) for x in request.form.values()]
            prediction = model.predict([form_data])
            score = round(prediction[0], 2)
            return render_template('result.html', score=score)
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An error occurred. Please go back and try again."
    
    # If the request is a GET, it means someone refreshed the page.
    # We will redirect them back to the main form page ('home').
    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    print("--- Server is starting... ---")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)