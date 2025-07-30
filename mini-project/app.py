from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# --- 1. Create the Flask App ---
app = Flask(__name__)
CORS(app)


# --- 2. Load the Trained Model ---
try:
    # We are loading the new, smarter model we just created.
    model = joblib.load('grade_predictor_model.pkl')
    print("Updated model loaded successfully.")
except FileNotFoundError:
    print("Error: grade_predictor_model.pkl not found. Run train_model.py first.")
    exit()


# --- 3. Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"Received data: {data}")

    try:
        # Extract all five features, including the new one.
        target_sem = float(data['target_semester'])
        prev_sem_1 = float(data['prev_sem_1_score'])
        prev_sem_2 = float(data['prev_sem_2_score'])
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])

        # Prepare the data in the correct order for the model.
        # The order must match the order in train_model.py
        features_to_predict = np.array([[target_sem, prev_sem_1, prev_sem_2, study_hours, attendance]])

        # Use the model to make a prediction.
        prediction = model.predict(features_to_predict)
        predicted_score = round(prediction[0], 2)
        print(f"Predicted score: {predicted_score}")

        return jsonify({'predicted_score': predicted_score})

    except (KeyError, TypeError, ValueError) as e:
        print(f"Error processing data: {e}")
        return jsonify({'error': 'Invalid input data. Please check all fields.'}), 400


# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)