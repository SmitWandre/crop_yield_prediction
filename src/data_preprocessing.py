import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path='data/crop_yield_data.csv'):

    data = pd.read_csv(file_path)
    data = data.dropna()
    return data

def split_data(data, features=['temperature', 'rainfall', 'soil_quality'], target='crop_yield', test_size=0.2, random_state=42):

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(data)
    print("Data loaded and split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

