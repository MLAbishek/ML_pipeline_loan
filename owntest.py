import pandas as pd
from utils import load_model, preprocess_data


def predict():
    try:

        features = [2850, 1250, 85, 10, 1100, 4100, "Sunny", 0, 0]
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
        print(pred)

    except Exception as e:
        print("Prediction error:", e)


predict()
