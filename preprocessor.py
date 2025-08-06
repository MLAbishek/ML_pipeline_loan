import joblib


def preprocess_data(X):
    preprocessor = joblib.load("preprocessor.pkl")

    # Ensure all types are exactly correct
    expected_dtypes = {
        "cash_transactions": float,
        "digital_transactions": float,
        "num_customers": float,
        "hours_open": float,
        "expense": float,
        "income": float,
        "missed_day": float,
        "local_event": float,
        "weather": str,
    }

    for col, dtype in expected_dtypes.items():
        if col in X.columns:
            X[col] = X[col].astype(dtype)

    print("✅ Final X before transform:")
    print(X.dtypes)
    print(X.head())

    return preprocessor.transform(X)
