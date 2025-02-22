import joblib
from sklearn.linear_model import LinearRegression
from src.data_preprocessing import load_and_preprocess_data, split_data

def train_model():

    data = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(data)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def save_model(model, file_path='models/crop_yield_predictor.pkl'):

    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

if __name__ == "__main__":
    model = train_model()
    save_model(model)

