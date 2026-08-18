"""End-to-end integration tests for batch pipeline orchestrator using named fakes."""

from pathlib import Path
import cv2
import numpy as np
from config import PipelineSettings
from pipeline.discovery import ImageFileScanner
from pipeline.drawing import BoundingBoxRenderer
from pipeline.resource_monitor import MemoryGuardScheduler
from pipeline.runner import BatchPipelineOrchestrator
from pipeline.tiling import ImageTilingEngine
from tests.fakes import (
    FakePlantDetectorEngine,
    FakePostgresRepository,
    FakeStorageUploader,
)


def setup_sample_images(input_dir: Path, count: int = 3) -> None:
    """Create sample dummy image files for end-to-end testing."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img_path = input_dir / f"test_img_{i}.jpg"
        dummy_img = np.zeros((800, 800, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy_img)


def test_pipeline_dry_run_execution(tmp_path: Path) -> None:
    """Test dry-run mode discovers images and registers them without inference."""
    input_dir = tmp_path / "images"
    setup_sample_images(input_dir, count=3)

    settings = PipelineSettings(input_dir=input_dir)
    repo = FakePostgresRepository()
    detector = FakePlantDetectorEngine()
    uploader = FakeStorageUploader()
    tiling = ImageTilingEngine(tile_size=640)
    renderer = BoundingBoxRenderer()
    scanner = ImageFileScanner(allowed_extensions=(".jpg",))
    guard = MemoryGuardScheduler(max_ram_pct=0.90)

    orchestrator = BatchPipelineOrchestrator(
        settings=settings,
        repository=repo,  # pyright: ignore[reportArgumentType]
        detector_engine=detector,  # pyright: ignore[reportArgumentType]
        uploader=uploader,  # pyright: ignore[reportArgumentType]
        tiling_engine=tiling,
        renderer=renderer,
        scanner=scanner,
        memory_guard=guard,
    )

    stats = orchestrator.run_pipeline(dry_run=True)
    assert stats.total_images == 3
    assert len(repo.images) == 3


def test_pipeline_full_e2e_execution(tmp_path: Path) -> None:
    """Test full pipeline end-to-end execution with fake I/O adapters."""
    input_dir = tmp_path / "images"
    setup_sample_images(input_dir, count=2)

    settings = PipelineSettings(input_dir=input_dir)
    repo = FakePostgresRepository()
    detector = FakePlantDetectorEngine(synthetic_confidence=0.92)
    uploader = FakeStorageUploader(bucket_name="test-bucket")
    tiling = ImageTilingEngine(tile_size=640)
    renderer = BoundingBoxRenderer()
    scanner = ImageFileScanner(allowed_extensions=(".jpg",))
    guard = MemoryGuardScheduler(max_ram_pct=0.90)

    orchestrator = BatchPipelineOrchestrator(
        settings=settings,
        repository=repo,  # pyright: ignore[reportArgumentType]
        detector_engine=detector,  # pyright: ignore[reportArgumentType]
        uploader=uploader,  # pyright: ignore[reportArgumentType]
        tiling_engine=tiling,
        renderer=renderer,
        scanner=scanner,
        memory_guard=guard,
    )

    stats = orchestrator.run_pipeline(dry_run=False)
    assert stats.total_images == 2
    assert stats.succeeded_images == 2
    assert len(uploader.uploaded_files) == 2

    for rec in repo.images.values():
        assert rec.status == "done"
        assert rec.uploaded is True
        assert rec.gcp_url is not None
