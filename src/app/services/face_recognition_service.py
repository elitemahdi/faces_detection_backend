from pathlib import Path
import cv2
import numpy as np

YUNET_MODEL_PATH = Path("models/face_detection_yunet.onnx")
SFACE_MODEL_PATH = Path("models/face_recognition_sface.onnx")


class FaceRecognitionService:

    def __init__(self):
        if not YUNET_MODEL_PATH.exists():
            raise RuntimeError(
                f"Face detector model not found at {YUNET_MODEL_PATH}."
            )
        if not SFACE_MODEL_PATH.exists():
            raise RuntimeError(
                f"Face recognizer model not found at {SFACE_MODEL_PATH}."
            )

        # 1. Detector: YuNet
        self.detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

        # 2. Recognizer: SFace (outputs normalized 512-dim embedding)
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(SFACE_MODEL_PATH),
            config="",
        )

    def process_and_embed(
        self, img: np.ndarray
    ) -> list[tuple[np.ndarray, list[float]]]:
        """Detects faces, aligns them, and returns (bbox, 512d_vector) for each face."""
        h, w, _ = img.shape
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        if faces is None:
            return []

        results = []
        for face in faces:
            bbox = face[:4].astype(int)  # [x, y, w, h]

            # Align & crop using detected landmarks
            aligned_face = self.recognizer.alignCrop(img, face)

            # Extract 512-dim feature embedding
            feature = self.recognizer.feature(aligned_face)
            embedding_512d = feature.flatten().tolist()

            results.append((bbox, embedding_512d))

        return results