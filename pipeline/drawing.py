"""Bounding box visualization and image annotation rendering module."""

from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
from pipeline.models import DetectionRecord


class BoundingBoxRenderer:
    """Renders bounding box rectangles onto image copies and saves annotated files.

    Example:
        >>> renderer = BoundingBoxRenderer(box_color=(0, 255, 0), thickness=3)
        >>> renderer.render_and_save(Path("/in.jpg"), Path("/out.jpg"), detections=[])
    """

    def __init__(
        self,
        box_color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
        font_scale: float = 0.8,
    ) -> None:
        """Initialize renderer with custom drawing options.

        Example:
            >>> renderer = BoundingBoxRenderer(box_color=(0, 255, 0), thickness=2)
        """
        self._box_color = box_color
        self._thickness = thickness
        self._font_scale = font_scale

    def render_and_save(
        self,
        source_image_path: Path,
        destination_path: Path,
        detections: Sequence[DetectionRecord],
    ) -> Path:
        """Draw bounding boxes on image copy and write output file.

        Example:
            >>> out_path = renderer.render_and_save(src_path, dst_path, detections)
        """
        if not source_image_path.exists():
            raise FileNotFoundError(f"Source image not found at '{source_image_path}'")

        img = cv2.imread(str(source_image_path))
        if img is None:
            raise ValueError(f"Unable to read image at '{source_image_path}'. Invalid format.")

        annotated_img = self._draw_detections_on_array(img, detections)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(destination_path), annotated_img)
        if not success:
            raise RuntimeError(f"Failed to save annotated image to '{destination_path}'")

        return destination_path

    def _draw_detections_on_array(
        self, img_bgr: np.ndarray, detections: Sequence[DetectionRecord]
    ) -> np.ndarray:
        """Draw bounding boxes and confidence text directly onto OpenCV BGR image array."""
        annotated = img_bgr.copy()
        for det in detections:
            cv2.rectangle(
                annotated,
                (det.bbox_x1, det.bbox_y1),
                (det.bbox_x2, det.bbox_y2),
                self._box_color,
                self._thickness,
            )
            label = f"{det.class_name} {det.confidence:.2f}"
            self._draw_label_text(annotated, label, det.bbox_x1, det.bbox_y1)
        return annotated

    def _draw_label_text(
        self, img: np.ndarray, label: str, x: int, y: int
    ) -> None:
        """Render filled background box and text label at top-left box corner."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_text = max(y - 8, 15)
        cv2.putText(
            img,
            label,
            (x, y_text),
            font,
            self._font_scale,
            self._box_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
