import pandas as pd
import joblib

# Test the exact same data as the API
features = [100, 200, 50, 8, 500, 1000, 1, 0, 1]

print("Original features:", features)
print("Weather value (index 6):", features[6], "Type:", type(features[6]))

# Convert weather from numeric to string
weather_mapping = {0: "Sunny", 1: "Rainy", 2: "Cloudy", 3: "Foggy", 4: "Humid"}
weather_value = features[6]
if isinstance(weather_value, (int, float)) and weather_value in weather_mapping:
    features[6] = weather_mapping[weather_value]

print("After conversion - Weather value:", features[6], "Type:", type(features[6]))

# Create DataFrame
columns = [
    "cash_transactions", "digital_transactions", "num_customers", 
    "hours_open", "expense", "income", "weather", "missed_day", "local_event"
]

X = pd.DataFrame([features], columns=columns)
print("\nDataFrame:")
print(X)
print("\nData types:")
print(X.dtypes)

# Test preprocessor
try:
    preprocessor = joblib.load("preprocessor.pkl")
    print("\nPreprocessor loaded successfully")
    
    # Check what categories the encoder expects
    cat_encoder = preprocessor.named_transformers_['cat']
    print("Expected weather categories:", cat_encoder.categories_[0])
    
    # Try to transform
    X_processed = preprocessor.transform(X)
    print("✅ Transformation successful!")
    print("Output shape:", X_processed.shape)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 