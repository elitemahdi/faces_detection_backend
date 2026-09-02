from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "facedb"

    # AI Detection & Recognition Configuration
    DETECTOR_BACKBONE: str = "auto"  # "yolo", "yunet", or "auto"
    YOLO_MODEL_PATH: str = "models/yolov8n_face.onnx"
    YUNET_MODEL_PATH: str = "models/face_detection_yunet.onnx"
    RECOGNITION_MODEL: str = "auto"  # "adaface", "sface", or "auto"
    ADAFACE_MODEL_PATH: str = "models/adaface_ir50.onnx"
    SFACE_MODEL_PATH: str = "models/face_recognition_sface.onnx"
    EMBEDDING_DIM: int = 512

    # Surveillance Quality & Distance Gating
    MIN_FACE_SIZE_FOR_RECOGNITION: int = 40  # Minimum face width/height in px before triggering AdaFace
    MATCH_THRESHOLD: float = 0.38
    TRACK_ACTIVATION_THRESHOLD: float = 0.35
    LOST_TRACK_BUFFER: int = 30

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
