import joblib


def preprocess_data(X):
    # Load your pre-fitted preprocessor (e.g., scaler, encoder)
    preprocessor = joblib.load("preprocessor.pkl")
    return preprocessor.transform(X)
