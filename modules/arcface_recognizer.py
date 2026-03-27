import cv2
import numpy as np
import pickle
import os
import onnxruntime as ort


class ArcFaceRecognizer:
    def __init__(self,
                 model_path="models/w600k_r50.onnx",
                 embeddings_path="embeddings/known_faces.pkl",
                 similarity_threshold=0.35):

        self.embeddings_path = embeddings_path
        self.similarity_threshold = similarity_threshold
        self.known_embeddings = {}

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"ArcFace model loaded from {model_path}")

        if os.path.exists(embeddings_path):
            self._load_embeddings()
            print(f"loaded {len(self.known_embeddings)} identities")
        else:
            print("no embeddings found, run enroll.py first")


    def _preprocess(self, face_img):
        img = cv2.resize(face_img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)


    def get_embedding(self, face_img):
        if face_img is None or face_img.size == 0:
            return None
        try:
            inp = self._preprocess(face_img)
            emb = self.session.run(None, {self.input_name: inp})[0][0]
            norm = np.linalg.norm(emb)
            return (emb / norm).astype(np.float32) if norm > 0 else emb
        except Exception as e:
            print(f"embedding error: {e}")
            return None


    def recognize(self, face_img):
        emb = self.get_embedding(face_img)
        if emb is None:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        for name, emb_list in self.known_embeddings.items():
            for known_emb in emb_list:
                score = float(np.dot(emb, known_emb))
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < self.similarity_threshold:
            return "Unknown", best_score
        return best_name, best_score


    def _load_embeddings(self):
        with open(self.embeddings_path, "rb") as f:
            self.known_embeddings = pickle.load(f)


    def save_embeddings(self):
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.known_embeddings, f)
        print(f"saved embeddings: {list(self.known_embeddings.keys())}")
