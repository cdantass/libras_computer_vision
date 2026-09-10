import os
import cv2
import mediapipe as mp
import pandas as pd

DATASET_PATH = "dataset"

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

dataset = []

for split in ["train", "test"]:

    split_path = os.path.join(DATASET_PATH, split)

    if not os.path.exists(split_path):
        print(f"Pasta não encontrada: {split_path}")
        continue

    for label in os.listdir(split_path):

        label_path = os.path.join(split_path, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Processando letra: {label}")

        for image_file in os.listdir(label_path):

            image_path = os.path.join(label_path, image_file)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                continue

            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            dataset.append(
                landmarks + [label]
            )

hands.close()

columns = []

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

columns.append("label")

df = pd.DataFrame(
    dataset,
    columns=columns
)

df.to_csv(
    "hand_landmarks_dataset.csv",
    index=False
)

print("Dataset successfully saved!")
print(f"Total samples: {len(df)}")
print("Archive saved as: hand_landmarks_dataset.csv")