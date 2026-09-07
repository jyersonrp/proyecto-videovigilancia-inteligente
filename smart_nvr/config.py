"""Configuration management for Smart NVR using Pydantic Settings.

Loads environment variables from .env with sane defaults for residential and SMB usage.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple, Type, Union
from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class LenientEnvSettingsSource(EnvSettingsSource):
    """Env source that falls back to raw string if json.loads fails on complex types."""

    def decode_complex_value(self, field_name: str, field: FieldInfo, value: Any) -> Any:
        try:
            return json.loads(value)
        except Exception:
            return value


class LenientDotEnvSettingsSource(DotEnvSettingsSource):
    """DotEnv source that falls back to raw string if json.loads fails on complex types."""

    def decode_complex_value(self, field_name: str, field: FieldInfo, value: Any) -> Any:
        try:
            return json.loads(value)
        except Exception:
            return value


class Settings(BaseSettings):
    """Smart NVR system-wide settings."""

    # Storage and Filesystem Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    STORAGE_DIR: Path = Path("storage")
    RECORDINGS_DIR: Path = Path("storage/recordings")
    CLIPS_DIR: Path = Path("storage/recordings/clips")
    SNAPSHOTS_DIR: Path = Path("storage/recordings/snapshots")
    DB_PATH: Path = Path("storage/nvr.db")
    MODELS_DIR: Path = Path("models")
    YOLO_MODEL_PATH: str = "models/yolov8n.onnx"
    MAX_STORAGE_GB: float = 50.0
    RETENTION_DAYS: int = 14

    # Ingestion and Video Streaming
    DEFAULT_FPS: int = 15
    RTSP_TRANSPORT: str = "tcp"  # Options: "tcp", "udp"
    RTSP_STIMEOUT: int = 5000000  # 5 seconds in microseconds
    FRAME_DROP_POLICY: str = "drop_oldest"
    RECONNECT_DELAY_INITIAL: float = 1.0
    RECONNECT_DELAY_MAX: float = 30.0
    RECONNECT_BACKOFF_FACTOR: float = 2.0
    JPEG_QUALITY: int = 75
    STREAM_MAX_LATENCY_MS: int = 150

    # Event Recording & Circular Buffer
    PRE_ROLL_SECONDS: float = 3.0
    POST_ROLL_SECONDS: float = 5.0
    MAX_EVENT_DURATION_SECONDS: float = 120.0

    # Two-Phase Hybrid Detection
    MOG2_HISTORY: int = 500
    MOG2_VAR_THRESHOLD: float = 16.0
    MOG2_DETECT_SHADOWS: bool = True
    MOG2_MIN_CONTOUR_AREA: int = 500
    MOG2_SHADOW_THRESHOLD: int = 200
    MOG2_DOWNSCALE_WIDTH: int = 320
    MOG2_DOWNSCALE_HEIGHT: int = 180
    AI_ENABLED: bool = True
    AI_CONFIDENCE_THRESHOLD: float = 0.50
    AI_TARGET_CLASSES: List[str] = ["person", "car", "motorcycle", "bus", "truck"]
    AI_INFERENCE_FPS: float = 5.0
    AI_ENGINE_TIER: str = "onnx"  # "onnx", "opencv_dnn", "mock"

    # Alerting & Gmail SMTP
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USE_TLS: bool = True
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = ""
    ALERT_RECIPIENTS: List[str] = []
    ALERT_COOLDOWN_SECONDS: int = 60
    ALERT_ENABLED: bool = True

    # Web & REST API
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["*"]
    SECRET_KEY: str = "smart-nvr-secret-key-change-in-production"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            LenientEnvSettingsSource(settings_cls),
            LenientDotEnvSettingsSource(
                settings_cls,
                env_file=settings_cls.model_config.get("env_file"),
                env_file_encoding=settings_cls.model_config.get("env_file_encoding", "utf-8"),
            ),
            file_secret_settings,
        )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Support comma-separated or single-string origins."""
        if isinstance(v, str):
            if not v.strip():
                return ["*"]
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("ALERT_RECIPIENTS", mode="before")
    @classmethod
    def parse_alert_recipients(cls, v: Union[str, List[str]]) -> List[str]:
        """Support comma-separated recipient strings from environment variables."""
        if isinstance(v, str):
            if not v.strip():
                return []
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("AI_TARGET_CLASSES", mode="before")
    @classmethod
    def parse_ai_target_classes(cls, v: Union[str, List[str]]) -> List[str]:
        """Support comma-separated class strings from environment variables."""
        if isinstance(v, str):
            if not v.strip():
                return []
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.STORAGE_DIR,
            self.RECORDINGS_DIR,
            self.CLIPS_DIR,
            self.SNAPSHOTS_DIR,
            self.MODELS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()


settings = get_settings()
