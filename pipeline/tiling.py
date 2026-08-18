"""Image tiling and cross-tile Non-Maximum Suppression (NMS) module."""

from typing import Sequence
import numpy as np
from pipeline.models import DetectionRecord, TileBox


class ImageTilingEngine:
    """Calculates tile slice grids, coordinate mapping, and cross-tile NMS deduplication.

    Example:
        >>> engine = ImageTilingEngine(tile_size=640, tile_overlap_pct=0.15)
        >>> tiles = engine.calculate_tile_grid(img_w=1920, img_h=1080)
    """

    def __init__(self, tile_size: int = 640, tile_overlap_pct: float = 0.15) -> None:
        """Initialize engine with tile size and overlap percentage.

        Example:
            >>> engine = ImageTilingEngine(tile_size=640, tile_overlap_pct=0.15)
        """
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if not (0.0 <= tile_overlap_pct < 0.9):
            raise ValueError(f"tile_overlap_pct must be in range [0, 0.9), got {tile_overlap_pct}")

        self._tile_size = tile_size
        self._tile_overlap_pct = tile_overlap_pct

    def calculate_tile_grid(self, img_w: int, img_h: int) -> list[TileBox]:
        """Compute grid of tile slice boxes for an image.

        Example:
            >>> grid = engine.calculate_tile_grid(1000, 1000)
        """
        stride = int(self._tile_size * (1.0 - self._tile_overlap_pct))
        stride = max(1, stride)

        x_offsets = self._generate_axis_offsets(img_w, self._tile_size, stride)
        y_offsets = self._generate_axis_offsets(img_h, self._tile_size, stride)

        tiles: list[TileBox] = []
        idx = 0
        for y_off in y_offsets:
            for x_off in x_offsets:
                w = min(self._tile_size, img_w - x_off)
                h = min(self._tile_size, img_h - y_off)
                tiles.append(TileBox(idx, x_off, y_off, w, h))
                idx += 1
        return tiles

    def _generate_axis_offsets(self, dimension: int, tile_sz: int, stride: int) -> list[int]:
        """Generate sequence of origin offsets along one axis."""
        if dimension <= tile_sz:
            return [0]
        offsets = list(range(0, dimension - tile_sz + 1, stride))
        if offsets[-1] + tile_sz < dimension:
            offsets.append(dimension - tile_sz)
        return offsets

    def map_tile_coords_to_full_image(
        self,
        bbox: tuple[int, int, int, int],
        tile_offset: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Map tile local (x1, y1, x2, y2) coords to full-image space.

        Example:
            >>> mapped = engine.map_tile_coords_to_full_image((10, 10, 50, 50), (100, 200))
            >>> mapped
            (110, 210, 150, 250)
        """
        x1, y1, x2, y2 = bbox
        x_off, y_off = tile_offset
        return (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off)

    def apply_cross_tile_nms(
        self,
        detections: Sequence[DetectionRecord],
        iou_threshold: float = 0.45,
    ) -> list[DetectionRecord]:
        """Deduplicate overlapping detections across tile boundaries using NMS.

        Example:
            >>> filtered = engine.apply_cross_tile_nms(detections, iou_threshold=0.45)
        """
        if not detections:
            return []

        boxes = np.array([[d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2] for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)

        keep_indices = self._vectorized_nms(boxes, scores, iou_threshold)
        return [detections[i] for i in keep_indices]

    def _vectorized_nms(
        self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float
    ) -> list[int]:
        """Vectorized Non-Maximum Suppression algorithm."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep
