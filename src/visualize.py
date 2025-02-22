import matplotlib.pyplot as plt
import joblib
from src.data_preprocessing import load_and_preprocess_data, split_data

def plot_predictions(y_test, y_pred):

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Crop Yield')
    plt.ylabel('Predicted Crop Yield')
    plt.title('Actual vs Predicted Crop Yield')
    plt.show()

if __name__ == "__main__":
    data = load_and_preprocess_data()
    _, X_test, _, y_test = split_data(data)
    model = joblib.load('models/crop_yield_predictor.pkl')
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)

