"""Unit tests for configuration module."""

from pathlib import Path
from config import PipelineSettings, create_default_settings


def test_default_settings_instantiation() -> None:
    """Verify default settings instantiate with expected values."""
    settings = create_default_settings()
    assert settings.tile_size == 640
    assert settings.tile_overlap_pct == 0.15
    assert settings.conf_threshold == 0.3
    assert settings.iou_threshold == 0.45
    assert settings.gpu_batch_size == 64


def test_custom_settings_override() -> None:
    """Verify custom parameter overrides in settings constructor."""
    settings = PipelineSettings(
        input_dir=Path("/custom/path"),
        tile_size=1024,
        verbose=True,
    )
    assert settings.input_dir == Path("/custom/path")
    assert settings.tile_size == 1024
    assert settings.verbose is True


def test_gcp_settings() -> None:
    """Verify GCP credentials and bucket settings."""
    settings = PipelineSettings(
        GCP_BUCKET_NAME="jfs_storage",
        GCP_PROJECT_ID="jfs-agritech-app",
        GOOGLE_APPLICATION_CREDENTIALS="jfs-agritech-app-5aca05c3642c.json",
    )
    assert settings.gcp_bucket == "jfs_storage"
    assert settings.GCP_PROJECT_ID == "jfs-agritech-app"
    assert settings.gcp_credentials_path == Path("jfs-agritech-app-5aca05c3642c.json")

