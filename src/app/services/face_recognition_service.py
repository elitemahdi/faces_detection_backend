from pathlib import Path
import cv2
import numpy as np

YUNET_MODEL_PATH = Path("models/face_detection_yunet.onnx")
SFACE_MODEL_PATH = Path("models/face_recognition_sface.onnx")
ADAFACE_MODEL_PATH = Path("models/adaface_ir50.onnx")

# Standard InsightFace / AdaFace 5-point landmark coordinates for 112x112 aligned crops
# Coordinate-for-coordinate match with YuNet:
# 1. Right Eye: [38.2946, 51.6963]
# 2. Left Eye:  [73.5318, 51.5014]
# 3. Nose Tip:  [56.0252, 71.7366]
# 4. Right Mouth Corner: [41.5493, 92.3655]
# 5. Left Mouth Corner:  [70.7299, 92.2041]
REFERENCE_5_POINTS = np.array(
    [
        [38.2946, 51.6963],  # Right Eye
        [73.5318, 51.5014],  # Left Eye
        [56.0252, 71.7366],  # Nose Tip
        [41.5493, 92.3655],  # Right Mouth
        [70.7299, 92.2041],  # Left Mouth
    ],
    dtype=np.float32,
)


class FaceRecognitionService:
    """Thread-safe Face Recognition Service.
    - Shares read-only thread-safe ONNX Runtime session for AdaFace across all threads.
    - Provides dedicated per-thread OpenCV FaceDetectorYN instances via get_detector().
    - Implements batched tensor inference, quality gating, and vectorized cosine similarity.
    """

    def __init__(self):
        if not YUNET_MODEL_PATH.exists():
            raise RuntimeError(f"Face detector model not found at {YUNET_MODEL_PATH}.")

        self.yunet_path = str(YUNET_MODEL_PATH)
        self.adaface_session = None
        self.input_name = None
        self._load_adaface_model()

        # Fallback SFace recognizer model path
        self.sface_path = str(SFACE_MODEL_PATH) if SFACE_MODEL_PATH.exists() else None
        self.sface_recognizer = None
        if self.sface_path:
            try:
                self.sface_recognizer = cv2.FaceRecognizerSF.create(model=self.sface_path, config="")
            except Exception:
                pass

    def _load_adaface_model(self):
        """Loads AdaFace ONNX runtime session with multi-threaded execution options."""
        if not ADAFACE_MODEL_PATH.exists() or ADAFACE_MODEL_PATH.stat().st_size < 10 * 1024 * 1024:
            return

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 2
            self.adaface_session = ort.InferenceSession(
                str(ADAFACE_MODEL_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.adaface_session.get_inputs()[0].name
        except Exception:
            self.adaface_session = None

    def get_detector(self, initial_size: tuple[int, int] = (320, 320)) -> cv2.FaceDetectorYN:
        """Instantiates a dedicated, thread-safe FaceDetectorYN instance for the calling thread."""
        return cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=initial_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

    def align_and_crop(self, img: np.ndarray, face_info: np.ndarray) -> np.ndarray:
        """Aligns and crops the face to 112x112 using OpenCV similarity affine transform."""
        landmarks = np.array(
            [
                [face_info[4], face_info[5]],    # Right Eye
                [face_info[6], face_info[7]],    # Left Eye
                [face_info[8], face_info[9]],    # Nose Tip
                [face_info[10], face_info[11]],  # Right Mouth
                [face_info[12], face_info[13]],  # Left Mouth
            ],
            dtype=np.float32,
        )

        tfm, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_5_POINTS)
        if tfm is not None:
            return cv2.warpAffine(
                img,
                tfm,
                (112, 112),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        # Fallback using FaceRecognizerSF alignCrop if available
        if self.sface_recognizer is not None:
            return self.sface_recognizer.alignCrop(img, face_info)

        # Fallback bounding box crop & resize with safe boundary clipping
        x, y, w, h = face_info[:4].astype(int)
        ih, iw, _ = img.shape
        x1 = int(np.clip(x, 0, iw - 1))
        y1 = int(np.clip(y, 0, ih - 1))
        x2 = int(np.clip(x + w, x1 + 1, iw))
        y2 = int(np.clip(y + h, y1 + 1, ih))
        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop, (112, 112)) if crop.size > 0 else np.zeros((112, 112, 3), dtype=np.uint8)

    def extract_embedding(self, aligned_bgr_112: np.ndarray) -> list[float]:
        """Extracts a single 512-dim L2-normalized embedding with empty output guard."""
        embeddings = self.extract_batch_embeddings([aligned_bgr_112])
        if not embeddings:
            raise RuntimeError("Failed to extract face embedding.")
        return embeddings[0]

    def extract_batch_embeddings(self, aligned_crops: list[np.ndarray]) -> list[list[float]]:
        """Extracts 512-dim L2-normalized feature embeddings from a batch of 112x112 aligned face crops.
        - Transforms HWC (112, 112, 3) to CHW (3, 112, 112) and normalizes to [-1, 1].
        - Evaluates the batch in a single forward ONNX inference call.
        - Normalizes embeddings across the batch dimension.
        """
        if not aligned_crops:
            return []

        # 1. AdaFace ONNX inference
        if self.adaface_session is not None and self.input_name is not None:
            batch = np.stack(
                [
                    np.transpose((c.astype(np.float32) - 127.5) / 127.5, (2, 0, 1))
                    for c in aligned_crops
                ],
                axis=0,
            )
            outputs = self.adaface_session.run(None, {self.input_name: batch})[0]

            # Vectorized L2 normalization across batch dimension
            norms = np.linalg.norm(outputs, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            normalized = outputs / norms
            return normalized.tolist()

        # 2. SFace fallback
        if self.sface_recognizer is not None:
            embeddings = []
            for crop in aligned_crops:
                feat = self.sface_recognizer.feature(crop).flatten()
                n = np.linalg.norm(feat)
                if n > 1e-6:
                    feat = feat / n
                embeddings.append(feat.tolist())
            return embeddings

        raise RuntimeError("No face recognizer model (AdaFace or SFace) is active.")

    def enroll_face(
        self,
        img: np.ndarray,
        min_confidence: float = 0.90,
        min_face_size: int = 100,
    ) -> tuple[np.ndarray, list[float], float]:
        """Thread-safe Phase A Registration Quality Gate."""
        h, w, _ = img.shape
        detector = self.get_detector((w, h))

        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            raise ValueError("No face detected in the photo. Please provide a clear portrait.")
        if len(faces) > 1:
            raise ValueError(
                f"Multiple faces ({len(faces)}) detected. Please upload a photo with only one person."
            )

        face = faces[0]
        score = float(face[14]) if len(face) > 14 else 1.0
        bx, by, bw, bh = face[:4].astype(int)

        if score < min_confidence:
            raise ValueError(
                f"Face detection confidence is too low ({score:.2f} < {min_confidence:.2f}). Please upload a clearer, well-lit photo."
            )

        if bw < min_face_size or bh < min_face_size:
            raise ValueError(
                f"Detected face resolution is too small ({bw}x{bh} px < {min_face_size}x{min_face_size} px). Please upload a closer portrait."
            )

        aligned_face = self.align_and_crop(img, face)
        embedding = self.extract_embedding(aligned_face)
        return np.array([bx, by, bw, bh]), embedding, score

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

    @staticmethod
    def batch_cosine_similarity(
        query_vec: list[float] | np.ndarray,
        enrolled_matrix: np.ndarray,
    ) -> np.ndarray:
        """Vectorized cosine similarity search across an entire matrix of enrolled embeddings (Q . E^T)."""
        if enrolled_matrix is None or enrolled_matrix.size == 0:
            return np.empty((0,), dtype=np.float32)

        mat = np.atleast_2d(enrolled_matrix).astype(np.float32)
        q = np.array(query_vec, dtype=np.float32).flatten()
        qn = np.linalg.norm(q)
        if qn > 1e-6:
            q = q / qn
        return np.dot(mat, q)


# Global singleton instance (stateless & thread-safe)
face_recognition_service = FaceRecognitionService()
