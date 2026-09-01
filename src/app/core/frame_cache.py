import numpy as np


class FrameCache:
    """Shared, thread-safe in-memory cache for the latest raw frames captured from active camera streams."""

    _frames: dict[int, np.ndarray] = {}

    @classmethod
    def get(cls, camera_id: int) -> np.ndarray | None:
        return cls._frames.get(camera_id)

    @classmethod
    def set(cls, camera_id: int, frame: np.ndarray) -> None:
        cls._frames[camera_id] = frame

    @classmethod
    def remove(cls, camera_id: int) -> None:
        cls._frames.pop(camera_id, None)


frame_cache = FrameCache
