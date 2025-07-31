from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import insert_data, load_model, retrain_model
from preprocessor import preprocess_data
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins


@app.route("/submit", methods=["POST"])
def submit():
    """
    Store new data for future training.
    Expects: { "features": [...], "label": ... }
    """
    data = request.get_json()
    insert_data(data["features"], data["label"])
    return jsonify({"message": "Data received!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]

        # Convert weather from numeric to string if needed
        weather_mapping = {0: "Sunny", 1: "Rainy", 2: "Cloudy", 3: "Foggy", 4: "Humid"}
        weather_value = features[6]  # weather is at index 6
        if isinstance(weather_value, (int, float)) and weather_value in weather_mapping:
            features[6] = weather_mapping[weather_value]

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

        # Create DataFrame with proper data types
        X = pd.DataFrame([features], columns=columns)

        # Ensure numeric columns are float
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

        # Ensure weather is string
        X["weather"] = X["weather"].astype(str)

        X_processed = preprocess_data(X)
        model = load_model()
        pred = model.predict(X_processed)
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Retrain the model with all stored data.
    """
    retrain_model()
    return jsonify({"message": "Model retrained!"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
