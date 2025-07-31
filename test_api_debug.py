import pandas as pd
import joblib
from preprocessor import preprocess_data
from utils import load_model

# Simulate the exact API request data
api_data = {"features": [100, 200, 50, 8, 500, 1000, 1, 0, 1]}

print("API request data:", api_data)

try:
    # Step 1: Extract features (exactly like API)
    features = api_data["features"]
    print("Extracted features:", features)
    print("Weather value (index 6):", features[6], "Type:", type(features[6]))

    # Step 2: Convert weather from numeric to string (exactly like API)
    weather_mapping = {0: "Sunny", 1: "Rainy", 2: "Cloudy", 3: "Foggy", 4: "Humid"}
    weather_value = features[6]  # weather is at index 6
    if isinstance(weather_value, (int, float)) and weather_value in weather_mapping:
        features[6] = weather_mapping[weather_value]

    print("After weather conversion:", features[6], "Type:", type(features[6]))

    # Step 3: Create DataFrame (exactly like API)
    columns = [
        "cash_transactions",
        "digital_transactions",
        "num_customers",
        "hours_open",
        "expense",
        "income",
        "weather",
        "missed_day",
        "local_event",
    ]
    X = pd.DataFrame([features], columns=columns)
    print("\nDataFrame before type conversion:")
    print(X)
    print("Data types:", X.dtypes)

    # Step 4: Ensure numeric columns are float (exactly like API)
    numeric_columns = [
        "cash_transactions",
        "digital_transactions",
        "num_customers",
        "hours_open",
        "expense",
        "income",
        "missed_day",
        "local_event",
    ]
    for col in numeric_columns:
        X[col] = X[col].astype(float)

    # Step 5: Ensure weather is string (exactly like API)
    X["weather"] = X["weather"].astype(str)

    print("\nDataFrame after type conversion:")
    print(X)
    print("Data types:", X.dtypes)

    # Step 6: Preprocess (exactly like API)
    print("\nCalling preprocess_data...")
    X_processed = preprocess_data(X)
    print("✅ Preprocessing successful!")
    print("Processed shape:", X_processed.shape)

    # Step 7: Load model and predict (exactly like API)
    print("\nLoading model...")
    model = load_model()
    print("✅ Model loaded successfully!")

    print("\nMaking prediction...")
    pred = model.predict(X_processed)
    print("✅ Prediction successful!")
    print("Prediction:", int(pred[0]))

    print("\n🎉 All steps completed successfully!")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback

    traceback.print_exc()
