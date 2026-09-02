from pathlib import Path
from typing import Protocol
import cv2
import numpy as np


class FaceDetectorProtocol(Protocol):
    def detect(self, img: np.ndarray) -> tuple[int, np.ndarray | None]:
        ...

    def setInputSize(self, size: tuple[int, int]) -> None:
        ...


class YoloFaceDetector:
    """YOLOv8-Face / YOLOv11-Face ONNX Detector with 5-Point Facial Landmark Head.
    
    Provides high recall on small, tilted, off-axis, and distant surveillance CCTV faces.
    Outputs standardized (N, 15) numpy array:
    [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, score]
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size  # (width, height)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = None
        self._init_session()

    def _init_session(self):
        if not self.model_path.exists():
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 2
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:
            print(f"[WARN] Failed to load YOLO-Face model at {self.model_path}: {exc}")
            self.session = None

    def setInputSize(self, size: tuple[int, int]) -> None:
        # Retained for API compatibility with FaceDetectorYN
        pass

    @staticmethod
    def _letterbox(
        img: np.ndarray,
        target_shape: tuple[int, int] = (640, 640),
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Resizes and pads image to target_shape while preserving aspect ratio."""
        shape = img.shape[:2]  # (height, width)
        target_w, target_h = target_shape

        r = min(target_w / shape[1], target_h / shape[0])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (target_w - new_unpad[0]) / 2
        dh = (target_h - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img_padded, r, (dw, dh)

    def detect(self, img: np.ndarray) -> tuple[int, np.ndarray | None]:
        """Detects faces and 5-point landmarks in an image.
        Returns (1, faces) where faces is an (N, 15) float32 array:
        [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, score]
        """
        if self.session is None or img is None or img.size == 0:
            return 0, None

        orig_h, orig_w = img.shape[:2]
        img_padded, scale, (dw, dh) = self._letterbox(img, self.input_size)

        # Preprocess: BGR -> RGB -> CHW float32 [0.0, 1.0]
        blob = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        try:
            outputs = self.session.run(None, {self.input_name: blob})[0]
        except Exception:
            return 0, None

        # Output shape is typically (1, channels, num_boxes) or (1, num_boxes, channels)
        if outputs.ndim == 3:
            if outputs.shape[1] < outputs.shape[2]:
                preds = np.transpose(outputs[0], (1, 0))  # (num_boxes, channels)
            else:
                preds = outputs[0]
        else:
            return 0, None

        num_boxes, channels = preds.shape
        if num_boxes == 0:
            return 1, np.empty((0, 15), dtype=np.float32)

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        scores = preds[:, 4]

        # Filter by confidence
        mask = scores >= self.score_threshold
        if not np.any(mask):
            return 1, np.empty((0, 15), dtype=np.float32)

        cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]
        filtered_preds = preds[mask]

        # Convert cx, cy, w, h to x1, y1 in 640 space
        x1_raw = cx - w / 2
        y1_raw = cy - h / 2

        # Convert back to original image space
        x1 = np.clip((x1_raw - dw) / scale, 0, orig_w - 1)
        y1 = np.clip((y1_raw - dh) / scale, 0, orig_h - 1)
        box_w = np.clip(w / scale, 1, orig_w - x1)
        box_h = np.clip(h / scale, 1, orig_h - y1)

        # Extract 5 landmarks
        if channels >= 20:
            lm_indices = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
            raw_lms = filtered_preds[:, lm_indices]
        elif channels >= 15:
            raw_lms = filtered_preds[:, 5:15]
        else:
            # Synthetic landmarks from box geometry if head omitted
            raw_lms = np.zeros((len(scores), 10), dtype=np.float32)
            raw_lms[:, 0] = x1 + box_w * 0.35
            raw_lms[:, 1] = y1 + box_h * 0.40
            raw_lms[:, 2] = x1 + box_w * 0.65
            raw_lms[:, 3] = y1 + box_h * 0.40
            raw_lms[:, 4] = x1 + box_w * 0.50
            raw_lms[:, 5] = y1 + box_h * 0.60
            raw_lms[:, 6] = x1 + box_w * 0.40
            raw_lms[:, 7] = y1 + box_h * 0.78
            raw_lms[:, 8] = x1 + box_w * 0.60
            raw_lms[:, 9] = y1 + box_h * 0.78

        if channels >= 15:
            for i in range(0, 10, 2):
                raw_lms[:, i] = np.clip((raw_lms[:, i] - dw) / scale, 0, orig_w - 1)
                raw_lms[:, i + 1] = np.clip((raw_lms[:, i + 1] - dh) / scale, 0, orig_h - 1)

        # NMS suppression
        boxes_for_nms = [
            [int(x1[i]), int(y1[i]), int(box_w[i]), int(box_h[i])]
            for i in range(len(scores))
        ]
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            scores.tolist(),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

        if len(indices) == 0:
            return 1, np.empty((0, 15), dtype=np.float32)

        indices = indices.flatten()
        faces = np.zeros((len(indices), 15), dtype=np.float32)
        for out_idx, in_idx in enumerate(indices):
            faces[out_idx, 0] = x1[in_idx]
            faces[out_idx, 1] = y1[in_idx]
            faces[out_idx, 2] = box_w[in_idx]
            faces[out_idx, 3] = box_h[in_idx]
            faces[out_idx, 4:14] = raw_lms[in_idx]
            faces[out_idx, 14] = scores[in_idx]

        return 1, faces


class YuNetFaceDetector:
    """Wrapper around OpenCV FaceDetectorYN."""

    def __init__(
        self,
        model_path: str | Path,
        initial_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ):
        self.detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=initial_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.detector.setInputSize(size)

    def detect(self, img: np.ndarray) -> tuple[int, np.ndarray | None]:
        if img is None or img.size == 0:
            return 0, None
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        return self.detector.detect(img)
