import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor


try:
    df = pd.read_csv("realistic_pricing.csv")
    print("Successfully loaded realistic_pricing.csv.")
except FileNotFoundError:
    print(
        " Error: 'realistic_pricing.csv' not found. Please make sure it's in the same directory."
    )
    exit()


numerical_features = [
    "cash_transactions",
    "digital_transactions",
    "num_customers",
    "hours_open",
    "expense",
    "income",
]
categorical_features = ["weather"]
TARGET_COLUMN = "credit_score"


print(" Feature and target columns defined.")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ],
    remainder="passthrough",
)


X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]


model = SGDRegressor(warm_start=True, random_state=42)
print(" Model and preprocessor are ready.")


print("\n Starting initial training...")


print("Fitting the preprocessor...")
preprocessor.fit(X)


X_processed = preprocessor.transform(X)


print("Training the base model with partial_fit...")
model.partial_fit(X_processed, y)

print(" Initial training complete.")


PREPROCESSOR_PATH = "preprocessor.pkl"
MODEL_PATH = "model.pkl"

print(f"\n Saving fitted preprocessor to '{PREPROCESSOR_PATH}'...")
joblib.dump(preprocessor, PREPROCESSOR_PATH)

print(f" Saving trained model to '{MODEL_PATH}'...")
joblib.dump(model, MODEL_PATH)

print(
    "\n All done! Your base model and preprocessor are saved and ready for iterative use."
)
