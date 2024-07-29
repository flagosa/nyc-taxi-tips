from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(df, features, target_col):
    X = df[features]
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X, y)
    joblib.dump(model, 'models/random_forest.joblib')
    return model
