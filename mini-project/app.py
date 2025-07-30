from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json # Import the json library

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('grade_predictor_model.pkl')
    print("--- Server Started: Model loaded successfully. ---")
except Exception as e:
    print(f"FATAL ERROR: Could not load model. Error: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    print("\n--- Received a new request ---")
    try:
        data = request.get_json()
        print(f"1. Received raw data: {data}")

        # --- Data Extraction ---
        target_sem = float(data['target_semester'])
        prev_sem_1 = float(data['prev_sem_1_score'])
        prev_sem_2 = float(data['prev_sem_2_score'])
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        print("2. Data successfully converted to numbers.")

        # --- Prediction ---
        features = np.array([[target_sem, prev_sem_1, prev_sem_2, study_hours, attendance]])
        prediction_result = model.predict(features)
        print(f"3. Model prediction raw output: {prediction_result}, Type: {type(prediction_result)}")

        # --- Data Type Conversion (Robust Fix) ---
        # Convert the entire numpy array to a standard Python list
        prediction_list = prediction_result.tolist()
        # Get the first (and only) number from the list
        score = prediction_list[0]
        print(f"4. Converted to Python list: {prediction_list}, Extracted score: {score}")
        
        # Round the final score
        final_score = round(score, 2)
        print(f"5. Final rounded score: {final_score}")

        # --- Create JSON Response ---
        response_data = {'predicted_score': final_score}
        print(f"6. Preparing to send this JSON back to browser: {response_data}")
        
        return jsonify(response_data)

    except Exception as e:
        print(f"!!! AN ERROR OCCURRED: {e} !!!")
        return jsonify({'error': 'An error occurred on the server.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)