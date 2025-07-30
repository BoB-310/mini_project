import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# --- 1. Load the Dataset ---
try:
    data = pd.read_csv('grades.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: grades.csv not found. Make sure the file is in the same folder as this script.")
    exit()

# --- 2. Prepare the Data ---
# The features now include the new 'target_semester' column.
# This is the key change!
features = data[['target_semester', 'prev_sem_1_score', 'prev_sem_2_score', 'study_hours', 'attendance']]

# The target remains the 'actual_score' column.
target = data['actual_score']

print("Data prepared for training with new 'target_semester' feature.")

# --- 3. Create and Train the Model ---
model = LinearRegression()
model.fit(features, target)
print("Model re-training complete.")

# --- 4. Save the Updated Model ---
# This will overwrite the old model file with our new, smarter model.
joblib.dump(model, 'grade_predictor_model.pkl')

print("New, updated model saved successfully as grade_predictor_model.pkl")