import cv2
import mediapipe as mp
from utils.prediction import HandDataTransformer, GesturePredictor


class HandGestureReader:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.transformer = HandDataTransformer()
        self.predictor = GesturePredictor(
            model_path="app/trained_model/signclusive-mediapipe-model.keras",
            scaler_path="app/trained_model/scaler.pkl",
            label_encoder_path="app/trained_model/label_encoder.pkl"
        )

    def run(self):
        with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        transformed_data = self.transformer.transform(hand_landmarks)

                        label, confidence = self.predictor.predict(transformed_data)

                        if label and confidence and confidence > 0.5:
                            if confidence >= 0.85:
                                color = (255, 0, 0)
                            else:
                                color = (0, 0, 255)

                            cv2.putText(
                                frame,
                                f"Prediction: {label} ({confidence:.2f})",
                                (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.5,
                                color,
                                3,
                                cv2.LINE_AA
                            )

                cv2.imshow('Hand Gesture Reader', frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    reader = HandGestureReader()
    reader.run()
