"""Unit tests for bounding box rendering module."""

from pathlib import Path
import cv2
import numpy as np
import pytest
from pipeline.drawing import BoundingBoxRenderer
from pipeline.models import DetectionRecord


def test_render_and_save_annotated_image(tmp_path: Path) -> None:
    """Test rendering bounding box annotations and saving file."""
    renderer = BoundingBoxRenderer(box_color=(0, 255, 0), thickness=2)

    src_img = tmp_path / "src.jpg"
    dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(src_img), dummy_array)

    dst_img = tmp_path / "out" / "annotated.jpg"
    detections = [DetectionRecord("plant", 0.90, 10, 10, 50, 50)]

    result_path = renderer.render_and_save(src_img, dst_img, detections)
    assert result_path.exists()
    assert result_path == dst_img


def test_render_missing_source_image_raises(tmp_path: Path) -> None:
    """Test that rendering a non-existent source image raises FileNotFoundError."""
    renderer = BoundingBoxRenderer()
    with pytest.raises(FileNotFoundError, match="Source image not found"):
        renderer.render_and_save(
            tmp_path / "missing.jpg", tmp_path / "out.jpg", []
        )
