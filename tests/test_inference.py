import numpy as np
import pytest

from unittest.mock import Mock
from app.inference import predict_genre


def test_predict_genre_with_mocked_model():
    # Mock the model and encoder
    model = Mock()
    encoder = Mock()

    features = np.random.rand(1,54)    

    model.predict.return_value = np.array([2])  # Mocked prediction
    encoder.inverse_transform.return_value = np.array(['rock'])  # Mocked decoding

    predicted_genre = predict_genre(features, model, encoder)

    assert predicted_genre == 'rock'

    #check that model recieved the correct features
    model.predict.assert_called_once_with(features)

    #check that encoder recieved the correct prediction
    encoder.inverse_transform.assert_called_once_with(np.array([2]))

def test_predict_with_value_error():
    model = Mock()
    encoder = Mock()

    features = np.zeros((1, 54))

    model.predict.side_effect = ValueError("Prediction failed")

    with pytest.raises(ValueError, match="Prediction failed"):
        predict_genre(features, model, encoder)

    model.predict.assert_called_once_with(features)
    encoder.inverse_transform.assert_not_called()  # Ensure encoder is not called if prediction fails


def test_encoder_with_value_error():
    model = Mock()
    encoder = Mock()

    features = np.random.rand(1, 54)

    # Mock the model to return a valid prediction
    model.predict.return_value = np.array([1])  # Assuming 1 is a valid class label

    # Mock the encoder to raise a ValueError when decoding
    encoder.inverse_transform.side_effect = ValueError("Invalid class label")

    with pytest.raises(ValueError):
        predict_genre(features, model, encoder)