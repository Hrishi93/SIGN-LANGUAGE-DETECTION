import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time

print("🔄 Loading XYZ model...")
model = joblib.load("../model/gesture_model_xyz.pkl")
feature_names = model.feature_names_in_

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_time = 0

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])  # XYZ

                if len(data) == len(feature_names):

                    X = pd.DataFrame([data], columns=feature_names)
                    prediction = model.predict(X)[0]

                    probs = model.predict_proba(X)[0]
                    confidence = np.max(probs) * 100

                    cv2.putText(frame, f"Gesture: {prediction}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    cv2.putText(frame, f"Confidence: {confidence:.2f}%",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Realtime Sign Detection (XYZ)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()