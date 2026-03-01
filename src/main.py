import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("model/gesture_model.pkl")
feature_names = model.feature_names_in_

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Fullscreen window
cv2.namedWindow("Sign Language Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Sign Language Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Sign Language Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
    "Sign Language Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # -------- ROI BOX --------
        box_size = 300
        x1 = w//2 - box_size//2
        y1 = h//2 - box_size//2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Blur background
        blurred = cv2.GaussianBlur(frame, (55, 55), 0)
        frame_blur = blurred.copy()
        frame_blur[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_roi)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw landmarks (shifted to ROI position)
                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * box_size) + x1
                    cy = int(lm.y * box_size) + y1
                    cv2.circle(frame_blur, (cx, cy), 5, (0, 0, 255), -1)

                mp_draw.draw_landmarks(
                    frame_blur,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Extract x,y like dataset
                data = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * box_size)
                    y = int(lm.y * box_size)
                    data.extend([x, y])

                if len(data) == len(feature_names):
                    X = pd.DataFrame([data], columns=feature_names)
                    prediction = model.predict(X)[0]

                    cv2.putText(frame_blur,
                                f"Gesture: {prediction}",
                                (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 255, 0),
                                3)
                else:
                    print("❌ Feature mismatch:", len(data))

        # Draw ROI box
        cv2.rectangle(frame_blur, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame_blur, "Place Hand Here",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.imshow("Sign Language Detection", frame_blur)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()