"""PostgreSQL repository interface for pipeline persistence and resumability tracking."""

from pathlib import Path
from typing import Sequence
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from config import PipelineSettings
from pipeline.models import DetectionRecord, ImageRecord, TileBox


class PlantDetectionRepository:
    """PostgreSQL data access adapter for pipeline resumability and results tracking.

    Example:
        >>> repo = PlantDetectionRepository("postgresql://user:pass@localhost:5432/db")
        >>> isinstance(repo, PlantDetectionRepository)
        True
    """

    def __init__(self, database_url: str) -> None:
        """Initialize repository with database DSN.

        Example:
            >>> repo = PlantDetectionRepository("postgresql://localhost/db")
        """
        if not database_url.startswith("postgresql://") and not database_url.startswith(
            "postgres://"
        ):
            raise ValueError(
                f"Invalid database_url scheme '{database_url}'. Expected 'postgresql://...'"
            )
        self._database_url = database_url

    def initialize_schema(self, schema_sql_path: Path) -> None:
        """Apply schema DDL to target database.

        Example:
            >>> repo.initialize_schema(Path("db/schema.sql"))
        """
        if not schema_sql_path.exists():
            raise FileNotFoundError(f"Schema file not found at '{schema_sql_path}'")
        sql_content = schema_sql_path.read_text(encoding="utf-8")
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_content)
            conn.commit()

    def create_run(self, settings: PipelineSettings) -> UUID:
        """Register a new pipeline run entry and return its assigned UUID.

        Example:
            >>> run_id = repo.create_run(settings)
            >>> isinstance(run_id, UUID)
            True
        """
        # If a previous run exists for the same input_dir and is still running/aborted,
        # reuse it so the pipeline can resume where it left off.
        lookup_q = """
        SELECT id FROM runs
        WHERE input_dir = %s AND status IN ('running', 'aborted')
        ORDER BY id DESC
        LIMIT 1;
        """
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(lookup_q, (str(settings.input_dir),))
                row = cur.fetchone()
                if row:
                    return row["id"]

        insert_q = """
        INSERT INTO runs (
            input_dir, model_name, tile_size, tile_overlap_pct,
            conf_threshold, iou_threshold, status
        ) VALUES (%s, %s, %s, %s, %s, %s, 'running')
        RETURNING id;
        """
        params = (
            str(settings.input_dir),
            settings.model_name,
            settings.tile_size,
            settings.tile_overlap_pct,
            settings.conf_threshold,
            settings.iou_threshold,
        )
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(insert_q, params)
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(
                        "Failed to obtain returned run UUID from database insert."
                    )
                run_id: UUID = row[0]
            conn.commit()
            return run_id

    def register_discovered_images(
        self, run_id: UUID, image_paths: Sequence[Path]
    ) -> int:
        """Insert discovered image paths into database if not previously registered.

        Example:
            >>> registered = repo.register_discovered_images(run_id, [Path("/img1.jpg")])
        """
        query = """
        INSERT INTO images (run_id, image_path, status)
        VALUES (%s, %s, 'pending')
        ON CONFLICT (image_path) DO NOTHING;
        """
        records = [(run_id, str(path)) for path in image_paths]
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)
                inserted_count = cur.rowcount
            conn.commit()
            return inserted_count

    def count_pending_images(self, run_id: UUID, max_attempts: int = 5) -> int:
        """Count images that are still pending for the active run."""
        query = """
        SELECT COUNT(*)
        FROM images
        WHERE run_id = %s
          AND status NOT IN ('done', 'failed')
          AND attempts < %s;
        """
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (run_id, max_attempts))
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def fetch_pending_images(
        self, run_id: UUID, batch_size: int = 100, max_attempts: int = 5
    ) -> list[ImageRecord]:
        """Fetch a stable batch of runnable images for the active run.

        Example:
            >>> images = repo.fetch_pending_images(run_id, batch_size=50)
        """
        query = """
        SELECT id, run_id, image_path, status, file_hash, width, height,
               tiles_total, tiles_done, detections_count, annotated_local_path,
               gcp_url, uploaded, attempts, error_message
        FROM images
        WHERE run_id = %s
          AND status NOT IN ('done', 'failed')
          AND attempts < %s
        ORDER BY id ASC
        LIMIT %s;
        """
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (run_id, max_attempts, batch_size))
                rows = cur.fetchall()
                return [self._map_image_row(row) for row in rows]

    def _map_image_row(self, row: dict) -> ImageRecord:
        """Map query row dictionary to strongly-typed ImageRecord object."""
        return ImageRecord(
            id=row["id"],
            run_id=row["run_id"],
            image_path=Path(row["image_path"]),
            status=row["status"],
            file_hash=row["file_hash"],
            width=row["width"],
            height=row["height"],
            tiles_total=row["tiles_total"],
            tiles_done=row["tiles_done"],
            detections_count=row["detections_count"],
            annotated_local_path=Path(row["annotated_local_path"])
            if row["annotated_local_path"]
            else None,
            gcp_url=row["gcp_url"],
            uploaded=row["uploaded"],
            attempts=row["attempts"],
            error_message=row["error_message"],
        )

    def update_image_status(
        self,
        image_id: int,
        status: str,
        width: int | None = None,
        height: int | None = None,
        tiles_total: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update processing stage status and metrics for an image record.

        Example:
            >>> repo.update_image_status(1, "processing", width=1920, height=1080)
        """
        query = """
        UPDATE images
        SET status = %s,
            width = COALESCE(%s, width),
            height = COALESCE(%s, height),
            tiles_total = COALESCE(%s, tiles_total),
            error_message = %s,
            updated_at = now()
        WHERE id = %s;
        """
        params = (status, width, height, tiles_total, error_message, image_id)
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()

    def record_image_completion(
        self,
        image_id: int,
        annotated_path: Path,
        gcp_url: str | None,
        detections_count: int,
    ) -> None:
        """Mark an image as fully processed, annotated, and uploaded.

        Example:
            >>> repo.record_image_completion(1, Path("/out/annotated.jpg"), "https://gcs/...", 12)
        """
        query = """
        UPDATE images
        SET status = 'done',
            uploaded = TRUE,
            annotated_local_path = %s,
            gcp_url = %s,
            detections_count = %s,
            finished_at = now(),
            updated_at = now()
        WHERE id = %s;
        """
        params = (str(annotated_path), gcp_url, detections_count, image_id)
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()

    def save_image_detections(
        self, image_id: int, detections: Sequence[DetectionRecord]
    ) -> None:
        """Bulk insert merged bounding box detection records into Postgres.

        Example:
            >>> repo.save_image_detections(1, [DetectionRecord("plant", 0.9, 10, 10, 50, 50)])
        """
        if not detections:
            return

        query = """
        INSERT INTO detections (
            image_id, tile_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        records = [
            (
                image_id,
                d.tile_id,
                d.class_name,
                d.confidence,
                d.bbox_x1,
                d.bbox_y1,
                d.bbox_x2,
                d.bbox_y2,
            )
            for d in detections
        ]
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)
            conn.commit()

    def mark_image_failed(self, image_id: int, error_message: str) -> None:
        """Increment attempt counter and mark image as failed upon fatal error.

        Example:
            >>> repo.mark_image_failed(1, "Corrupt file format")
        """
        query = """
        UPDATE images
        SET status = 'failed',
            attempts = attempts + 1,
            error_message = %s,
            updated_at = now()
        WHERE id = %s;
        """
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (error_message, image_id))
            conn.commit()

    def mark_run_status(self, run_id: UUID, status: str) -> None:
        """Update run execution state ('completed', 'aborted').

        Example:
            >>> repo.mark_run_status(run_id, "completed")
        """
        query = """
        UPDATE runs
        SET status = %s, finished_at = now()
        WHERE id = %s;
        """
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (status, run_id))
            conn.commit()

    def clean_database_tables(self) -> None:
        """Truncate all database tables (detections, tiles, images, runs CASCADE).

        Example:
            >>> repo.clean_database_tables()
        """
        query = "TRUNCATE TABLE detections, tiles, images, runs CASCADE;"
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()
