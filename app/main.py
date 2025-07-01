import cv2
import mediapipe as mp
import numpy as np

class HandGestureReader:
    """
    Reads hand gestures using MediaPipe and displays their numerical representation.
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def run(self):
        """
        Starts the main loop for hand gesture reading.
        """
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

                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                # Convert the BGR image to RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                # Draw the hand annotations on the image.
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Display numerical representation of landmarks
                        print("Hand Landmarks (normalized):")
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            # Print x, y, z coordinates for each landmark
                            print(f"  Landmark {i}: x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}")
                        print("\n" + "-"*50 + "\n") # Separator for readability

                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cv2.imshow('Hand Gesture Reader', frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    reader = HandGestureReader()
    reader.run()
