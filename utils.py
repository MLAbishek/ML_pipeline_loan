# utils.py
import sqlite3, os, joblib, requests
import pandas as pd
from sklearn.linear_model import SGDRegressor
from preprocessor import preprocess_data

MODEL_PATH = "model.pkl"
TRACK_FILE = "last_id.txt"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")

COLUMNS = [
    "cash_transactions",
    "digital_transactions",
    "num_customers",
    "hours_open",
    "expense",
    "income",
    "weather",
    "missed_day",
    "local_event",
    "credit_score",
]


def insert_data(x, y):
    """
    Insert new data into the database.
    x: list of feature values
    y: target value (credit_score)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Insert the new data
    cursor.execute(
        """
        INSERT INTO entries (
            cash_transactions, digital_transactions, num_customers, 
            hours_open, expense, income, weather, missed_day, local_event, credit_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        x + [y],
    )

    conn.commit()
    conn.close()


def fetch_new_data(last_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM entries WHERE id > ?", (last_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_last_trained_id():
    if not os.path.exists(TRACK_FILE):
        return 0
    with open(TRACK_FILE, "r") as f:
        return int(f.read().strip())


def update_last_trained_id(new_id):
    with open(TRACK_FILE, "w") as f:
        f.write(str(new_id))


def load_model():
    return joblib.load(MODEL_PATH)


def save_model(model):
    joblib.dump(model, MODEL_PATH)
