import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler  # Or any specific function/class you're using

# Load dataset
df = pd.read_csv("TAIPEI_diabetes.csv")  # Make sure this file exists!

# Drop unnecessary columns if they exist
columns_to_drop = ["BloodSugarLevel"]  # Add any other columns you want to drop here
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Ensure 'Outcome' is present before dropping
if "Outcome" in df.columns:
    df = df.drop(columns=["Outcome"])

# Train StandardScaler on the remaining features
scaler = StandardScaler()
scaler.fit(df)

# Save the scaler as a .pkl file
joblib.dump(scaler, "scaler.pkl")

print("✅ Scaler saved as scaler.pkl")