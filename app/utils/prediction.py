import joblib
import numpy as np
import tensorflow as tf


class HandDataTransformer:

    def transform(self, hand_landmarks):
        if hand_landmarks is None:
            return None

        base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y

        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append(landmark.x - base_x)
            landmarks.append(landmark.y - base_y)

        return np.array(landmarks).reshape(1, -1)


class GesturePredictor:

    def __init__(self, model_path, scaler_path, label_encoder_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(label_encoder_path)
        except Exception as e:
            raise IOError(f"Error loading model or related files: {e}")

    def predict(self, transformed_data):
        if transformed_data is None:
            return None, None

        scaled_data = self.scaler.transform(transformed_data)

        prediction_probabilities = self.model.predict(scaled_data)

        predicted_index = np.argmax(prediction_probabilities)

        confidence = np.max(prediction_probabilities)

        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]

        return predicted_label, confidence
