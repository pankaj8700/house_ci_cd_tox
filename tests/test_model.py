import pytest
from model.train import train_model
from model.predict import predict
import numpy as np

def test_train_model():
    # Train the model
    model = train_model()
    assert model is not None  # Ensure the model is trained

def test_prediction():
    # Test if prediction works with the model
    # Example features for the California Housing dataset:
    # [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    sample_input = [8.3252, 41.0, 6.98412698, 1.02380952, 322.0, 2.55555556, 37.88, -122.23]
    prediction = predict(sample_input)
    assert isinstance(prediction, np.ndarray)  # Ensure prediction is a numpy array
    assert prediction.shape == (1,)  # Ensure prediction has the correct shape