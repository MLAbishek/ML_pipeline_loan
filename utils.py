import sqlite3, os, joblib
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")
MODEL_PATH = "model.pkl"
TRACK_FILE = "last_id.txt"


def insert_data(x, y):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Ensure table exists
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cash_transactions REAL,
            digital_transactions REAL,
            num_customers REAL,
            hours_open REAL,
            expense REAL,
            income REAL,
            weather REAL,
            missed_day REAL,
            local_event REAL,
            credit_score REAL
        )
        """
    )
    # Insert the data (order must match COLUMNS in train.py)
    cursor.execute(
        """
        INSERT INTO entries (
            cash_transactions, digital_transactions, num_customers, hours_open,
            expense, income, weather, missed_day, local_event, credit_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (*x, y),
    )
    conn.commit()
    conn.close()


def fetch_new_data(last_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure table exists first
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cash_transactions REAL,
            digital_transactions REAL,
            num_customers REAL,
            hours_open REAL,
            expense REAL,
            income REAL,
            weather REAL,
            missed_day REAL,
            local_event REAL,
            credit_score REAL
        )
        """
    )

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


def retrain_model():
    from train import train_model

    train_model()
