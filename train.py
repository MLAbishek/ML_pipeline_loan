import pandas as pd
from utils import (
    fetch_new_data,
    load_model,
    save_model,
    get_last_trained_id,
    update_last_trained_id,
)
from preprocessor import preprocess_data
from sklearn.linear_model import SGDRegressor
import os


COLUMNS = [
    "id",
    "cash_transactions",
    "digital_transactions",
    "num_customers",
    "hours_open",
    "expense",
    "income",
    "weather",
    "missed_day",
    "local_event",
    "credit_score",  # Use credit_score to match CSV and database
]


def train_model():
    last_id = get_last_trained_id()
    new_rows = fetch_new_data(last_id)

    new_data_df = pd.DataFrame(new_rows, columns=COLUMNS)
    last_training_id = new_data_df["id"].iloc[-1]

    X = new_data_df.drop(["id", "credit_score"], axis=1)
    y = new_data_df["credit_score"]

    X_processed = preprocess_data(X)

    print(f"Training on {len(X_processed)} new entries...")

    if not os.path.exists("model.pkl"):
        print("Initializing new model.")
        model = SGDRegressor(warm_start=True)
    else:
        model = load_model()

    model.partial_fit(X_processed, y)

    save_model(model)
    update_last_trained_id(last_training_id)
    print("Model updated and saved.")


if __name__ == "__main__":
    train_model()
