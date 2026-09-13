import cv2
import mediapipe as mp
import joblib
from flask import Flask, Response, render_template

app = Flask(__name__)

LETTERS = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

model = joblib.load("hand_gesture_model.pkl")

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


def gen_frames():
    """Generator que processa cada frame da webcam e produz os bytes
    JPEG prontos para streaming - é o equivalente ao loop 'while True'
    do hand.py original, trocando cv2.imshow por yield."""

    # estado da "soletração", equivalente às globais do hand.py original
    current_index = 0
    current_letter = LETTERS[current_index]
    waiting_open_hand = False

    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(image, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        word_detected = "-"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    try:
                        word_detected = model.predict([landmarks])[0]
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        word_detected = "ERROR"

                # aceita a letra
                if word_detected == current_letter and not waiting_open_hand:
                    waiting_open_hand = True

                # só avança com a mão totalmente aberta
                if waiting_open_hand:
                    if hand_is_open(hand_landmarks):
                        current_index += 1
                        if current_index >= len(LETTERS):
                            current_index = len(LETTERS) - 1
                        current_letter = LETTERS[current_index]
                        waiting_open_hand = False

        # ---------------- INTERFACE DESENHADA NO FRAME ----------------
        cv2.putText(
            frame, f"Make the letter: {current_letter}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"Detected: {word_detected}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        progress_text = f"Progress: {current_index + 1}/26"
        cv2.putText(
            frame, progress_text, (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
        )

        if waiting_open_hand:
            cv2.putText(
                frame, "CORRECT! OPEN YOUR HAND", (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
            )

        bar_width = 400
        progress = (current_index + 1) / len(LETTERS)
        filled = int(bar_width * progress)

        cv2.rectangle(frame, (20, 230), (420, 260), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 230), (20 + filled, 260), (0, 255, 0), -1)

        # ---------------- ENVIO PARA O NAVEGADOR ----------------
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)