"""Pipeline domain data models."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class TileBox:
    """Bounding box coordinates and grid location of a tile slice.

    Example:
        >>> tile = TileBox(tile_index=0, x_offset=0, y_offset=0, width=640, height=640)
        >>> tile.tile_index
        0
    """

    tile_index: int
    x_offset: int
    y_offset: int
    width: int
    height: int


@dataclass
class DetectionRecord:
    """Bounding box detection result mapped to full image pixel space.

    Example:
        >>> det = DetectionRecord("plant", 0.92, 100, 150, 200, 250, tile_index=0)
        >>> det.confidence
        0.92
    """

    class_name: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    tile_index: int | None = None
    tile_id: int | None = None


@dataclass
class ImageRecord:
    """Database representation of an image being processed in the pipeline.

    Example:
        >>> rec = ImageRecord(1, UUID("..."), Path("/path/img.jpg"), "pending")
        >>> rec.status
        'pending'
    """

    id: int
    run_id: UUID
    image_path: Path
    status: str
    file_hash: str | None = None
    width: int | None = None
    height: int | None = None
    tiles_total: int | None = None
    tiles_done: int = 0
    detections_count: int = 0
    annotated_local_path: Path | None = None
    gcp_url: str | None = None
    uploaded: bool = False
    attempts: int = 0
    error_message: str | None = None


@dataclass
class PipelineProgressStats:
    """Aggregated counters for tracking live batch execution progress.

    Example:
        >>> stats = PipelineProgressStats(total_images=100)
        >>> stats.succeeded_images
        0
    """

    total_images: int = 0
    succeeded_images: int = 0
    failed_images: int = 0
    skipped_images: int = 0
    failed_tiles: int = 0
    failed_uploads: int = 0
