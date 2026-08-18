"""Named fake classes for external I/O (Postgres, GPU YOLO, GCP Storage) testing."""

from pathlib import Path
from typing import Sequence
from uuid import UUID, uuid4
import numpy as np
from PIL import Image

from config import PipelineSettings
from pipeline.models import DetectionRecord, ImageRecord, TileBox
from pipeline.tiling import ImageTilingEngine


class FakePostgresRepository:
    """Named fake repository operating over in-memory dictionaries.

    Example:
        >>> repo = FakePostgresRepository()
        >>> isinstance(repo, FakePostgresRepository)
        True
    """

    def __init__(self) -> None:
        """Initialize empty in-memory collections."""
        self.runs: dict[UUID, dict] = {}
        self.images: dict[int, ImageRecord] = {}
        self.detections: dict[int, list[DetectionRecord]] = {}
        self._next_image_id: int = 1

    def create_run(self, settings: PipelineSettings) -> UUID:
        """Create fake run record and return UUID.

        Example:
            >>> run_id = repo.create_run(settings)
        """
        run_id = uuid4()
        self.runs[run_id] = {"status": "running", "settings": settings}
        return run_id

    def register_discovered_images(
        self, run_id: UUID, image_paths: Sequence[Path]
    ) -> int:
        """Register image paths into in-memory storage.

        Example:
            >>> count = repo.register_discovered_images(run_id, [Path("/img.jpg")])
        """
        added = 0
        existing_paths = {rec.image_path for rec in self.images.values()}
        for path in image_paths:
            if path in existing_paths:
                continue
            img_id = self._next_image_id
            self._next_image_id += 1
            self.images[img_id] = ImageRecord(
                id=img_id,
                run_id=run_id,
                image_path=path,
                status="pending",
            )
            added += 1
        return added

    def fetch_pending_images(
        self, run_id: UUID, batch_size: int = 100
    ) -> list[ImageRecord]:
        """Return list of pending image records.

        Example:
            >>> pending = repo.fetch_pending_images(run_id)
        """
        results: list[ImageRecord] = []
        for rec in self.images.values():
            if rec.status not in ("done", "failed"):
                results.append(rec)
            if len(results) >= batch_size:
                break
        return results

    def update_image_status(
        self,
        image_id: int,
        status: str,
        width: int | None = None,
        height: int | None = None,
        tiles_total: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update status fields on fake image record.

        Example:
            >>> repo.update_image_status(1, "processing")
        """
        if image_id not in self.images:
            return
        rec = self.images[image_id]
        rec.status = status
        if width is not None:
            rec.width = width
        if height is not None:
            rec.height = height
        if tiles_total is not None:
            rec.tiles_total = tiles_total
        if error_message is not None:
            rec.error_message = error_message

    def record_image_completion(
        self,
        image_id: int,
        annotated_path: Path,
        gcp_url: str | None,
        detections_count: int,
    ) -> None:
        """Mark fake image as completed and uploaded.

        Example:
            >>> repo.record_image_completion(1, Path("/out.jpg"), "https://gcs/...", 5)
        """
        if image_id not in self.images:
            return
        rec = self.images[image_id]
        rec.status = "done"
        rec.uploaded = True
        rec.annotated_local_path = annotated_path
        rec.gcp_url = gcp_url
        rec.detections_count = detections_count

    def save_image_detections(
        self, image_id: int, detections: Sequence[DetectionRecord]
    ) -> None:
        """Save detection records in memory for an image.

        Example:
            >>> repo.save_image_detections(1, [DetectionRecord("plant", 0.9, 1, 1, 5, 5)])
        """
        if image_id not in self.detections:
            self.detections[image_id] = []
        self.detections[image_id].extend(detections)

    def mark_image_failed(self, image_id: int, error_message: str) -> None:
        """Mark fake image as failed with error context.

        Example:
            >>> repo.mark_image_failed(1, "IO Error")
        """
        if image_id not in self.images:
            return
        rec = self.images[image_id]
        rec.status = "failed"
        rec.attempts += 1
        rec.error_message = error_message

    def mark_run_status(self, run_id: UUID, status: str) -> None:
        """Update run status string.

        Example:
            >>> repo.mark_run_status(run_id, "completed")
        """
        if run_id in self.runs:
            self.runs[run_id]["status"] = status


class FakePlantDetectorEngine:
    """Named fake YOLO plant detector returning deterministic synthetic detections.

    Example:
        >>> detector = FakePlantDetectorEngine()
        >>> isinstance(detector, FakePlantDetectorEngine)
        True
    """

    def __init__(self, synthetic_confidence: float = 0.95) -> None:
        """Initialize fake detector with synthetic confidence score.

        Example:
            >>> detector = FakePlantDetectorEngine(synthetic_confidence=0.88)
        """
        self._conf = synthetic_confidence

    def infer_tile_batch(
        self,
        tile_crops: Sequence[np.ndarray | Image.Image],
        tile_boxes: Sequence[TileBox],
        tiling_engine: ImageTilingEngine,
    ) -> list[DetectionRecord]:
        """Return synthetic plant bounding box detection per tile crop.

        Example:
            >>> dets = detector.infer_tile_batch(crops, boxes, tiling)
        """
        results: list[DetectionRecord] = []
        for tile in tile_boxes:
            mapped_bbox = tiling_engine.map_tile_coords_to_full_image(
                (10, 10, 50, 50), (tile.x_offset, tile.y_offset)
            )
            results.append(
                DetectionRecord(
                    class_name="plant",
                    confidence=self._conf,
                    bbox_x1=mapped_bbox[0],
                    bbox_y1=mapped_bbox[1],
                    bbox_x2=mapped_bbox[2],
                    bbox_y2=mapped_bbox[3],
                    tile_index=tile.tile_index,
                )
            )
        return results


class FakeStorageUploader:
    """Named fake GCP Storage uploader returning simulated public URLs.

    Example:
        >>> uploader = FakeStorageUploader(bucket_name="test-bucket")
        >>> isinstance(uploader, FakeStorageUploader)
        True
    """

    def __init__(self, bucket_name: str = "fake-bucket") -> None:
        """Initialize fake uploader with target bucket name.

        Example:
            >>> uploader = FakeStorageUploader("my-bucket")
        """
        self._bucket = bucket_name
        self.uploaded_files: list[Path] = []

    def upload_file(self, local_path: Path, object_key: str | None = None) -> str:
        """Simulate GCP file upload and return synthetic URL.

        Example:
            >>> url = uploader.upload_file(Path("/tmp/img.jpg"))
        """
        self.uploaded_files.append(local_path)
        key = object_key or local_path.name
        return f"https://storage.googleapis.com/{self._bucket}/{key}"

    def delete_all_uploaded_blobs(self) -> int:
        """Delete all simulated uploaded files.

        Example:
            >>> deleted_count = uploader.delete_all_uploaded_blobs()
        """
        count = len(self.uploaded_files)
        self.uploaded_files.clear()
        return count
