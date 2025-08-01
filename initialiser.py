import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor


# --- 1. Load Your Dataset ---
try:
    df = pd.read_csv("pricing.csv")
    print("✅ Successfully loaded pricing.csv.")
except FileNotFoundError:
    print(
        "❌ Error: 'pricing.csv' not found. Please make sure it's in the same directory."
    )
    exit()

# --- 2. Define Feature Columns ---
# Based on the actual CSV data structure
numerical_features = [
    "cash_transactions",
    "digital_transactions",
    "num_customers",
    "hours_open",
    "expense",
    "income",
]
categorical_features = ["weather"]
TARGET_COLUMN = "credit_score"  # This is the column you want to predict

# All other columns like 'missed_day' and 'local_event' will be passed through without changes.
print("✅ Feature and target columns defined.")

# --- 3. Create the Preprocessing Pipeline ---
# This pipeline scales numerical data and one-hot encodes categorical data.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ],
    remainder="passthrough",  # Keeps other columns
)

# --- 4. Prepare Data and Model ---
# Separate features (X) from the target variable (y)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Initialize the model. `warm_start=True` is essential for iterative training.
model = SGDRegressor(warm_start=True, random_state=42)
print("✅ Model and preprocessor are ready.")

# --- 5. Fit the Preprocessor and Train the Initial Model ---
print("\n🔥 Starting initial training...")

# Step 1: Fit the preprocessor on your data to learn the scaling and encoding
print("Fitting the preprocessor...")
preprocessor.fit(X)

# Step 2: Transform the data using the fitted preprocessor
X_processed = preprocessor.transform(X)

# Step 3: Train the SGDRegressor model on the processed data
print("Training the base model with partial_fit...")
model.partial_fit(X_processed, y)

print("✅ Initial training complete.")

# --- 6. Save Your Trained Artifacts ---
# These files are what you'll use for iterative training later.
PREPROCESSOR_PATH = "preprocessor.pkl"
MODEL_PATH = "model.pkl"

print(f"\n💾 Saving fitted preprocessor to '{PREPROCESSOR_PATH}'...")
joblib.dump(preprocessor, PREPROCESSOR_PATH)

print(f"💾 Saving trained model to '{MODEL_PATH}'...")
joblib.dump(model, MODEL_PATH)

print(
    "\n✨ All done! Your base model and preprocessor are saved and ready for iterative use."
)
