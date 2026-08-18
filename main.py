"""Main CLI entrypoint for H100 Massive Inference batch plant detection pipeline."""

from pathlib import Path
import typer
from config import PipelineSettings
from db.repository import PlantDetectionRepository
from logging_setup import initialize_pipeline_logger
from pipeline.discovery import ImageFileScanner
from pipeline.drawing import BoundingBoxRenderer
from pipeline.inference import YoloPlantDetectorEngine
from pipeline.resource_monitor import MemoryGuardScheduler
from pipeline.runner import BatchPipelineOrchestrator
from pipeline.tiling import ImageTilingEngine
from pipeline.upload import GcpImageUploader

app: typer.Typer = typer.Typer(
    help="H100/RTX 5090 Massive Plant Detection Batch Pipeline CLI"
)


def start(
    input_dir: Path = typer.Argument(
        Path("./sample_images"),
        help="Path to input directory containing target images.",
    ),
    dry_run: bool = typer.Option(
        False, help="Register images and validate pipeline without running GPU inference."
    ),
    limit: int | None = typer.Option(
        None, help="Optional maximum number of images to process in this run."
    ),
    verbose: bool = typer.Option(
        False, help="Enable verbose debug level console logging."
    ),
) -> None:
    """Run batch plant detection pipeline over target directory.

    Example:
        >>> uv run start /path/to/images --dry-run
    """
    settings = PipelineSettings(input_dir=input_dir, verbose=verbose)
    initialize_pipeline_logger(verbose=settings.verbose, log_dir=settings.log_dir)

    repository = PlantDetectionRepository(database_url=settings.database_url)
    detector = YoloPlantDetectorEngine(
        weights_path=settings.model_weights_path,
        device=settings.device,
        use_fp16=settings.use_fp16,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        target_classes=settings.detection_classes,
    )
    uploader = GcpImageUploader(
        bucket_name=settings.gcp_bucket,
        prefix=settings.gcp_prefix,
        credentials_path=settings.gcp_credentials_path,
    )
    tiling = ImageTilingEngine(
        tile_size=settings.tile_size,
        tile_overlap_pct=settings.tile_overlap_pct,
    )
    renderer = BoundingBoxRenderer()
    scanner = ImageFileScanner(allowed_extensions=settings.allowed_extensions)
    memory_guard = MemoryGuardScheduler(max_ram_pct=settings.max_ram_utilization_pct)

    orchestrator = BatchPipelineOrchestrator(
        settings=settings,
        repository=repository,
        detector_engine=detector,
        uploader=uploader,
        tiling_engine=tiling,
        renderer=renderer,
        scanner=scanner,
        memory_guard=memory_guard,
    )
    orchestrator.run_pipeline(dry_run=dry_run, limit=limit)


def start_cli() -> None:
    """Standalone CLI entrypoint for 'uv run start <dir_path>'.

    Example:
        >>> start_cli()
    """
    typer.run(start)


def migrate(
    schema_path: Path = typer.Option(
        Path("./db/schema.sql"), help="Path to PostgreSQL schema DDL file."
    ),
) -> None:
    """Apply PostgreSQL schema migration DDL script to database.

    Example:
        >>> uv run migrate
    """
    settings = PipelineSettings()
    initialize_pipeline_logger(verbose=settings.verbose, log_dir=settings.log_dir)
    repository = PlantDetectionRepository(database_url=settings.database_url)
    repository.initialize_schema(schema_path)
    typer.echo(f"Successfully applied database schema from '{schema_path}'.")


def migrate_cli() -> None:
    """Standalone CLI entrypoint for 'uv run migrate'.

    Example:
        >>> migrate_cli()
    """
    typer.run(migrate)


def clean_db() -> None:
    """Truncate all PostgreSQL database tables.

    Example:
        >>> uv run clean-db
    """
    settings = PipelineSettings()
    initialize_pipeline_logger(verbose=settings.verbose, log_dir=settings.log_dir)
    repository = PlantDetectionRepository(database_url=settings.database_url)
    repository.clean_database_tables()
    typer.echo("Successfully truncated all database tables (detections, tiles, images, runs).")


def clean_db_cli() -> None:
    """Standalone CLI entrypoint for 'uv run clean-db'.

    Example:
        >>> clean_db_cli()
    """
    typer.run(clean_db)


def clean_gcp() -> None:
    """Delete all uploaded blobs from GCP Cloud Storage bucket prefix.

    Example:
        >>> uv run clean_gcp
    """
    settings = PipelineSettings()
    initialize_pipeline_logger(verbose=settings.verbose, log_dir=settings.log_dir)
    uploader = GcpImageUploader(
        bucket_name=settings.gcp_bucket,
        prefix=settings.gcp_prefix,
        credentials_path=settings.gcp_credentials_path,
    )
    count = uploader.delete_all_uploaded_blobs()
    typer.echo(f"Successfully deleted {count} uploaded objects from GCP bucket '{settings.gcp_bucket}'.")


def clean_gcp_cli() -> None:
    """Standalone CLI entrypoint for 'uv run clean_gcp'.

    Example:
        >>> clean_gcp_cli()
    """
    typer.run(clean_gcp)


if __name__ == "__main__":
    start_cli()
