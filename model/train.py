import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model(): 
    # Load the dataset
    data = pd.read_csv('data/house.csv')

    # Preprocess the dataset
    X = data.drop('price', axis = 1)
    y = data['price']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Train the model
    model = LinearRegression()

    # fit the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'model/house.pkl')
    return model
if __name__ == '__main__':
    train_model()