import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_preprocessing import load_and_preprocess_data, split_data

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    return y_pred

if __name__ == "__main__":
    data = load_and_preprocess_data()
    _, X_test, _, y_test = split_data(data)
    model = joblib.load('models/crop_yield_predictor.pkl')
    evaluate_model(model, X_test, y_test)

