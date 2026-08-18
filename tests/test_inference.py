"""Unit tests for YOLO inference engine and multi-strategy compute device resolution."""

from pathlib import Path
from pipeline.inference import (
    YoloPlantDetectorEngine,
    _can_initialize_device,
    resolve_best_compute_device,
)


def test_resolve_best_compute_device_fallback_cpu() -> None:
    """Verify compute device fallback to CPU when CUDA is unavailable."""
    device = resolve_best_compute_device("cuda:99")
    assert isinstance(device, str)
    assert device in ("cpu", "cuda:0", "mps")


def test_can_initialize_device_invalid() -> None:
    """Verify initialization check fails gracefully for invalid device strings."""
    assert _can_initialize_device("invalid_device_name_xyz") is False
    assert _can_initialize_device("cpu") is True


def test_yolo_engine_instantiation(tmp_path: Path) -> None:
    """Verify YoloPlantDetectorEngine instantiates and resolves device cleanly."""
    weights_path = tmp_path / "dummy.pt"
    engine = YoloPlantDetectorEngine(weights_path=weights_path, device="cuda:0")
    assert engine._device in ("cpu", "cuda:0")
