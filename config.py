"""Pipeline settings module backed by pydantic-settings.

Validates environment variables and provides strongly-typed configuration.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Configuration settings for the massive plant detection inference pipeline.

    Example:
        >>> settings = PipelineSettings(input_dir=Path("/data/images"))
        >>> print(settings.tile_size)
        640
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    input_dir: Path = Field(
        default=Path("./sample_images"),
        description="Path to local folder containing input images.",
    )
    recursive_scan: bool = Field(
        default=False,
        description="Scan input directory recursively.",
    )
    allowed_extensions: tuple[str, ...] = Field(
        default=(".jpg", ".jpeg", ".png", ".tif"),
        description="Supported image file extensions.",
    )

    model_name: str = Field(
        default="massive_detection-yolo",
        description="Identifier name for model in run records.",
    )
    model_weights_path: Path = Field(
        default=Path("./best.pt"),
        description="Path to PyTorch YOLO weights file.",
    )
    device: str = Field(
        default="cuda:0",
        description="Target compute device ('cuda:0', 'cpu').",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 half precision for inference.",
    )

    tile_size: int = Field(
        default=1024,
        description="Dimension in pixels for square tiles.",
    )
    tile_overlap_pct: float = Field(
        default=0.25,
        description="Overlap percentage ratio between adjacent tiles.",
    )
    conf_threshold: float = Field(
        default=0.2,
        description="Minimum confidence score threshold.",
    )
    iou_threshold: float = Field(
        default=0.45,
        description="Intersection over Union threshold for NMS.",
    )
    detection_classes: tuple[str, ...] = Field(
        default=("coconut",),
        description="Filter list of target class names to retain.",
    )

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/plant_detection",
        description="PostgreSQL DSN string.",
    )

    GOOGLE_APPLICATION_CREDENTIALS: str | None = Field(
        default=None,
        description="Path to GCP service account credentials JSON file.",
    )
    GCP_BUCKET_NAME: str = Field(
        default="jfs_storage",
        description="Target GCP Cloud Storage bucket name.",
    )
    GCP_PROJECT_ID: str = Field(
        default="jfs-agritech-app",
        description="GCP project identifier.",
    )
    gcp_prefix: str = Field(
        default="annotated/",
        description="Target prefix path within GCP bucket.",
    )

    @property
    def gcp_bucket(self) -> str:
        """Return target GCP bucket name."""
        return self.GCP_BUCKET_NAME

    @property
    def gcp_credentials_path(self) -> Path | None:
        """Return Path object for GCP credentials JSON file if configured."""
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            return None
        return Path(self.GOOGLE_APPLICATION_CREDENTIALS)

    gpu_batch_size: int = Field(
        default=256,
        description="Number of tiles sent per GPU inference batch.",
    )
    gpu_vram_target_pct: float = Field(
        default=0.90,
        description="Target fraction of GPU VRAM to occupy during inference.",
    )
    cpu_preprocess_workers: int = Field(
        default=24,
        description="Number of parallel CPU workers for tiling.",
    )
    cpu_postprocess_workers: int = Field(
        default=12,
        description="Number of parallel CPU workers for drawing.",
    )
    upload_threads: int = Field(
        default=24,
        description="Number of threads for GCP uploads.",
    )
    max_ram_utilization_pct: float = Field(
        default=0.90,
        description="Target maximum RAM utilization ceiling ratio.",
    )
    ram_check_interval_images: int = Field(
        default=25,
        description="Check system RAM every N images.",
    )

    max_attempts_per_image: int = Field(
        default=3,
        description="Maximum retry attempts per image record.",
    )
    max_attempts_per_tile: int = Field(
        default=2,
        description="Maximum retry attempts per tile record.",
    )

    log_dir: Path = Field(
        default=Path("./logs"),
        description="Directory where log files are retained.",
    )
    verbose: bool = Field(
        default=False,
        description="Enable debug logging output.",
    )


def create_default_settings() -> PipelineSettings:
    """Instantiate standard pipeline settings.

    Example:
        >>> settings = create_default_settings()
        >>> isinstance(settings, PipelineSettings)
        True
    """
    return PipelineSettings()
