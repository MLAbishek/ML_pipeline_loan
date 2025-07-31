from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import insert_data, load_model, retrain_model
from preprocessor import preprocess_data
import pandas as pd

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
        features = [float(x) for x in data["features"]]
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
