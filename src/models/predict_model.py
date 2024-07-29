import joblib

def load_model(model_path):
    return joblib.load(model_path)

def predict(df, model, features):
    predictions = model.predict_proba(df[features])
    return [p[1] for p in predictions.round()]
