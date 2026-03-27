# enroll.py
import os
import cv2
from modules.arcface_recognizer import ArcFaceRecognizer
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATASET_DIR = "dataset"
EMBEDDINGS_PATH = "embeddings/known_faces.pkl"


def enroll():
    recognizer = ArcFaceRecognizer(embeddings_path=EMBEDDINGS_PATH)
    known = {}

    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\nenrolling: {person_name}")
        embeddings = []

        for img_file in sorted(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"  skip: cannot read {img_file}")
                continue

            emb = recognizer.get_embedding(img)

            if emb is not None:
                embeddings.append(emb)
                print(f"  ok: {img_file} shape={emb.shape}")
            else:
                print(f"  skip: failed embedding for {img_file}")

        if embeddings:
            known[person_name] = embeddings
            print(f"  {len(embeddings)} embeddings stored for {person_name}")
        else:
            print(f"  warn: no valid embeddings for {person_name}, skipped")

    if not known:
        print("\nerror: no embeddings created, check dataset/ folder")
        return

    recognizer.known_embeddings = known
    recognizer.save_embeddings()
    print(f"\nenrollment complete: {len(known)} people enrolled")
    for name, embs in known.items():
        print(f"  {name}: {len(embs)} embeddings")


if __name__ == "__main__":
    enroll()
