import cv2
import mediapipe as mp
import joblib

model = joblib.load("hand_gesture_model.pkl")

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

while True:
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    word_detected = "none"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    try:
                        word_detected = model.predict([landmarks])[0]
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        word_detected = "error"

    cv2.putText(
        frame,
        f"Detected word: {word_detected}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
hands.close()
cv2.destroyAllWindows()