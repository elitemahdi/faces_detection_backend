from pathlib import Path
import cv2
import numpy as np

YUNET_MODEL_PATH = Path("models/face_detection_yunet.onnx")
SFACE_MODEL_PATH = Path("models/face_recognition_sface.onnx")
ADAFACE_MODEL_PATH = Path("models/adaface_ir50.onnx")

# Standard InsightFace / AdaFace 5-point landmark coordinates for 112x112 aligned crops
REFERENCE_5_POINTS = np.array(
    [
        [38.2946, 51.6963],  # Right eye
        [73.5318, 51.5014],  # Left eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Right mouth corner
        [70.7299, 92.2041],  # Left mouth corner
    ],
    dtype=np.float32,
)


class FaceRecognitionService:
    """Face Recognition Service combining OpenCV YuNet face & 5-point landmark detector,
    OpenCV similarity transform alignment to 112x112, and AdaFace / SFace 512-dim embedding.
    """

    def __init__(self):
        if not YUNET_MODEL_PATH.exists():
            raise RuntimeError(
                f"Face detector model not found at {YUNET_MODEL_PATH}."
            )

        # 1. Detector: OpenCV YuNet
        self.detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

        # 2. Recognizer: Check for AdaFace ONNX, fallback to OpenCV SFace
        self.adaface_session = None
        self.input_name = None
        self._load_adaface_model()

        # OpenCV SFace as recognizer & aligner fallback
        self.recognizer = None
        if SFACE_MODEL_PATH.exists():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=str(SFACE_MODEL_PATH),
                    config="",
                )
            except Exception:
                pass

    def _load_adaface_model(self):
        """Attempts to load AdaFace IR-50 model via onnxruntime or OpenCV DNN."""
        if not ADAFACE_MODEL_PATH.exists():
            return

        # Check if file is valid (> 10MB)
        if ADAFACE_MODEL_PATH.stat().st_size < 10 * 1024 * 1024:
            return

        try:
            import onnxruntime as ort

            self.adaface_session = ort.InferenceSession(
                str(ADAFACE_MODEL_PATH),
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.adaface_session.get_inputs()[0].name
        except Exception:
            try:
                # Fallback to OpenCV DNN
                self.adaface_net = cv2.dnn.readNetFromONNX(str(ADAFACE_MODEL_PATH))
            except Exception:
                pass

    def align_and_crop(self, img: np.ndarray, face_info: np.ndarray) -> np.ndarray:
        """Aligns and crops the face to 112x112 using OpenCV affine similarity transform
        calculated from 5-point facial landmarks.
        """
        # Extract 5 landmarks: right eye, left eye, nose, right mouth, left mouth
        landmarks = np.array(
            [
                [face_info[4], face_info[5]],
                [face_info[6], face_info[7]],
                [face_info[8], face_info[9]],
                [face_info[10], face_info[11]],
                [face_info[12], face_info[13]],
            ],
            dtype=np.float32,
        )

        # Estimate partial affine (similarity) transform matrix: scale, rotation, translation
        tfm, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_5_POINTS)

        if tfm is not None:
            aligned = cv2.warpAffine(
                img,
                tfm,
                (112, 112),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            return aligned

        # Fallback using FaceRecognizerSF alignCrop if available
        if self.recognizer is not None:
            return self.recognizer.alignCrop(img, face_info)

        # Basic bounding box crop & resize fallback
        x, y, w, h = face_info[:4].astype(int)
        ih, iw, _ = img.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, iw - x), min(h, ih - y)
        crop = img[y : y + h, x : x + w]
        if crop.size > 0:
            return cv2.resize(crop, (112, 112))
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def extract_embedding(self, aligned_bgr_112: np.ndarray) -> list[float]:
        """Extracts 512-dim L2-normalized feature embedding from 112x112 aligned face crop."""
        # 1. AdaFace ONNX inference
        if self.adaface_session is not None and self.input_name is not None:
            # Preprocessing: (x - 127.5) / 127.5, shape (1, 3, 112, 112)
            blob = (aligned_bgr_112.astype(np.float32) - 127.5) / 127.5
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]

            output = self.adaface_session.run(None, {self.input_name: blob})[0]
            embedding = output.flatten().astype(np.float32)

            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            return embedding.tolist()

        # 2. SFace fallback inference
        if self.recognizer is not None:
            feature = self.recognizer.feature(aligned_bgr_112).flatten()
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm
            return feature.tolist()

        raise RuntimeError("No face recognizer model (AdaFace or SFace) is active.")

    def process_and_embed(
        self, img: np.ndarray
    ) -> list[tuple[np.ndarray, list[float]]]:
        """Phase A & B Core Pipeline:
        1. Grab/load image with OpenCV
        2. Detect faces & 5-point landmarks with YuNet
        3. Align and crop to 112x112 using OpenCV similarity transform
        4. Pass into AdaFace to generate 512-dim normalized embedding
        5. Returns list of (bbox [x, y, w, h], 512d_embedding)
        """
        h, w, _ = img.shape
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        if faces is None:
            return []

        results = []
        for face in faces:
            bbox = face[:4].astype(int)  # [x, y, w, h]

            # 5-point alignment & 112x112 crop
            aligned_face = self.align_and_crop(img, face)

            # 512-dim AdaFace feature embedding
            embedding_512d = self.extract_embedding(aligned_face)

            results.append((bbox, embedding_512d))

        return results

    @staticmethod
    def cosine_similarity(vec1: list[float] | np.ndarray, vec2: list[float] | np.ndarray) -> float:
        """Calculates cosine similarity between two normalized face embeddings."""
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))
