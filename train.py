import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Step 1: Generate Synthetic Data
np.random.seed(42)

num_drivers = 1000
event_counts = np.random.randint(0, 16, num_drivers)
total_durations = np.random.randint(0, 121, num_drivers)

# Define classification criteria
def classify_driver(events, duration):
    return 'Bad' if events >= 5 and duration >= 30 else 'Good'

driver_labels = [classify_driver(events, duration) for events, duration in zip(event_counts, total_durations)]

# Create DataFrame
df = pd.DataFrame({
    'DriverID': range(1, num_drivers + 1),
    'EventCount': event_counts,
    'TotalDuration': total_durations,
    'DriverClassification': driver_labels
})

# Step 2: Train the Machine Learning Model
# Prepare features and labels
X = df[['EventCount', 'TotalDuration']]
y = df['DriverClassification']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 3: Save the Trained Model
model_filename = 'driver_classification_model.pkl'
joblib.dump(model, model_filename)

# Step 4: Load the Saved Model
loaded_model = joblib.load(model_filename)

# Step 5: Make Predictions with the Loaded Model
y_pred = loaded_model.predict(X_test)

# Evaluate the Model
print(classification_report(y_test, y_pred))
