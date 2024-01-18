import joblib


def predict(data):
    clf = joblib.load("rf_model.pkl")
    return clf.predict(data)