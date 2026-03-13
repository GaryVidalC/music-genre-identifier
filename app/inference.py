import numpy as np
import pickle

def predict_genre(features, model, encoder):

    # Predict genre using the model
    predicted_genre_encoded = model.predict(features)

    # Decode the predicted genre
    predicted_genre = encoder.inverse_transform(predicted_genre_encoded)

    return predicted_genre[0]