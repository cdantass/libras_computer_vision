import cv2
import mediapipe as mp
import joblib

LETTERS = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

current_index = 0
current_letter = LETTERS[current_index]

waiting_open_hand = False

model = joblib.load("hand_gesture_model.pkl")

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def hand_is_open(hand_landmarks):
    fingertips = [8, 12, 16, 20]

    fingers_up = 0

    for tip in fingertips:
        if (
            hand_landmarks.landmark[tip].y
            < hand_landmarks.landmark[tip - 2].y
        ):
            fingers_up += 1

    return fingers_up >= 4


while True:

    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(image, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    word_detected = "-"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            if len(landmarks) == 63:

                try:
                    word_detected = model.predict(
                        [landmarks]
                    )[0]

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    word_detected = "ERROR"

            # Accepted the letter
            if (
                word_detected == current_letter
                and not waiting_open_hand
            ):
                waiting_open_hand = True

            # Advance only with an open hand completely
            if waiting_open_hand:

                if hand_is_open(hand_landmarks):

                    current_index += 1

                    if current_index >= len(LETTERS):
                        current_index = len(LETTERS) - 1

                    current_letter = LETTERS[current_index]

                    waiting_open_hand = False

    # INTERFACE
    cv2.putText(
        frame,
        f"Make the letter: {current_letter}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Detected: {word_detected}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    progress_text = f"Progress: {current_index + 1}/26"

    cv2.putText(
        frame,
        progress_text,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    if waiting_open_hand:

        cv2.putText(
            frame,
            "CORRECT! OPEN YOUR HAND",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    # Barra de progresso

    bar_width = 400

    progress = (current_index + 1) / len(LETTERS)

    filled = int(bar_width * progress)

    cv2.rectangle(
        frame,
        (20, 230),
        (420, 260),
        (255, 255, 255),
        2
    )

    cv2.rectangle(
        frame,
        (20, 230),
        (20 + filled, 260),
        (0, 255, 0),
        -1
    )

    cv2.imshow(
        "Learn Libras",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()