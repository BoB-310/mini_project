from flask import Flask, request, render_template
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Read the data from the form and convert to numbers
        target_sem = float(request.form['target_semester'])
        prev_sem_1 = float(request.form['prev_sem_1_score'])
        prev_sem_2 = float(request.form['prev_sem_2_score'])
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])

        # 2. --- THIS IS THE NEW VALIDATION BLOCK ---
        # Check if the numbers are within a logical range
        if not (0 <= prev_sem_1 <= 100 and
                0 <= prev_sem_2 <= 100 and
                0 <= study_hours <= 24 and
                0 <= attendance <= 100):
            # If any value is out of range, return an error message
            return "Invalid input. Please go back and enter values within the correct range (e.g., scores 0-100, hours 0-24)."

        # 3. If data is valid, proceed with prediction
        features = [target_sem, prev_sem_1, prev_sem_2, study_hours, attendance]
        prediction = model.predict([features])
        score = round(prediction[0], 2)

        # 4. Render the result page
        return render_template('result.html', score=score)

    except ValueError:
        # This will catch errors if the user enters text instead of numbers
        return "Invalid input. Please enter numbers only."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An unexpected error occurred. Please try again."

if __name__ == '__main__':
    print("--- Server is starting... ---")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)