"""Batch pipeline execution orchestrator integrating DB tracking, tiling, inference, and upload."""

from datetime import timedelta
from pathlib import Path
import time
from typing import Sequence
from uuid import UUID
import cv2
from loguru import logger
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
import torch

from config import PipelineSettings
from db.repository import PlantDetectionRepository
from pipeline.discovery import ImageFileScanner
from pipeline.drawing import BoundingBoxRenderer
from pipeline.inference import YoloPlantDetectorEngine
from pipeline.models import ImageRecord, PipelineProgressStats, TileBox
from pipeline.resource_monitor import MemoryGuardScheduler
from pipeline.tiling import ImageTilingEngine
from pipeline.upload import GcpImageUploader
import os


class BatchPipelineOrchestrator:
    """Orchestrates end-to-end batch plant detection pipeline over 20,000+ images.

    Example:
        >>> orchestrator = BatchPipelineOrchestrator(...)
        >>> stats = orchestrator.run_pipeline(dry_run=True)
    """

    def __init__(
        self,
        settings: PipelineSettings,
        repository: PlantDetectionRepository,
        detector_engine: YoloPlantDetectorEngine,
        uploader: GcpImageUploader,
        tiling_engine: ImageTilingEngine,
        renderer: BoundingBoxRenderer,
        scanner: ImageFileScanner,
        memory_guard: MemoryGuardScheduler,
    ) -> None:
        """Initialize orchestrator with injected domain components."""
        self._settings = settings
        self._repo = repository
        self._detector = detector_engine
        self._uploader = uploader
        self._tiling = tiling_engine
        self._renderer = renderer
        self._scanner = scanner
        self._memory = memory_guard
        self._console = Console()

    def run_pipeline(
        self,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> PipelineProgressStats:
        """Execute batch detection pipeline over discovered image set.

        Example:
            >>> stats = orchestrator.run_pipeline(dry_run=True, limit=10)
        """
        stats = PipelineProgressStats()
        run_id = self._repo.create_run(self._settings)

        # Apply CPU thread caps to avoid saturating the host. Use configured
        # max_ram_utilization_pct as the desired utilization ratio for CPU as well.
        logical_cores = psutil.cpu_count(logical=True) or 1
        cpu_target_pct = self._settings.max_ram_utilization_pct
        target_threads = max(1, int(logical_cores * cpu_target_pct))
        # Leave a small headroom to avoid 100% saturation
        if target_threads >= logical_cores:
            target_threads = max(1, logical_cores - 1)

        os.environ.setdefault("OMP_NUM_THREADS", str(target_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(target_threads))
        try:
            torch.set_num_threads(target_threads)
        except Exception:
            # ignore if the torch backend doesn't support setting threads
            pass

        discovered = self._scanner.discover_images(
            self._settings.input_dir, recursive=self._settings.recursive_scan
        )
        if limit is not None and limit > 0:
            discovered = discovered[:limit]

        registered_count = self._repo.register_discovered_images(run_id, discovered)
        stats.total_images = len(discovered)

        # Estimate tiles by sampling up to N images to avoid reading thousands of files.
        sample_n = min(len(discovered), 50)
        sample_paths = discovered[:sample_n]
        total_tiles_sample = 0
        import cv2 as _cv2
        for p in sample_paths:
            try:
                img = _cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                total_tiles_sample += len(self._tiling.calculate_tile_grid(w, h))
            except Exception:
                continue

        avg_tiles_per_image = int(total_tiles_sample / sample_n) if sample_n > 0 and total_tiles_sample > 0 else 0
        est_tiles = avg_tiles_per_image * len(discovered) if avg_tiles_per_image > 0 else 0

        # Use configured max attempts when fetching pending images so we don't retry forever
        pending_batch = self._repo.fetch_pending_images(
            run_id, batch_size=len(discovered) or 1, max_attempts=self._settings.max_attempts_per_image
        )
        pending_count = len(pending_batch)
        completed_count = stats.total_images - pending_count

        # Compute ETA based on sampled average tiles and device capability
        tiles_per_second = 180.0 if torch.cuda.is_available() else 15.0
        # scale by tile area ratio relative to 640 baseline
        base_tile = 640
        scale = (base_tile * base_tile) / max(1, self._settings.tile_size * self._settings.tile_size)
        tiles_per_second = max(1.0, tiles_per_second * scale)

        total_seconds = int(est_tiles / tiles_per_second) if est_tiles > 0 else 0
        eta_str = str(timedelta(seconds=total_seconds))

        self._print_pre_run_summary(stats.total_images, completed_count, pending_count, run_id, est_tiles, eta_str)

        if dry_run:
            logger.info("Dry-run flag set. Skipping inference loop.")
            self._repo.mark_run_status(run_id, "completed")
            return stats

        self._process_image_queue(run_id, stats)
        self._repo.mark_run_status(run_id, "completed")
        return stats

    def _print_pre_run_summary(
        self, total: int, completed: int, pending: int, run_id: UUID, est_tiles: int = 0, eta_str: str = "0:00:00"
    ) -> None:
        """Display formatted pre-run hardware and workload summary panel."""
        hw_info = self._get_hardware_info()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan bold")
        table.add_column("Value", style="white")

        table.add_row("Run UUID", str(run_id))
        table.add_row("Discovered Images", f"{total:,}")
        table.add_row("Resumed / Completed", f"{completed:,}")
        table.add_row("Pending Processing", f"{pending:,}")
        table.add_row("Estimated Total Tiles", f"~{est_tiles:,}")
        table.add_row("CPU Saturation Cap (90%)", hw_info["cpu"])
        table.add_row("RAM Utilization Cap (90%)", hw_info["ram"])
        table.add_row("GPU Accelerator", hw_info["gpu"])
        table.add_row("Estimated Total Time (ETA)", eta_str)
        # show sample of discovered files
        table.add_row("Sample Images (first 5)", "")
        for p in list(self._scanner.discover_images(self._settings.input_dir, recursive=self._settings.recursive_scan))[:5]:
            table.add_row("", str(p))

        panel = Panel(
            table,
            title="[bold green]H100 / RTX 5090 High-Throughput Batch Detection Pipeline[/bold green]",
            border_style="green",
        )
        self._console.print(panel)

    def _get_hardware_info(self) -> dict[str, str]:
        """Query host hardware specs and calculate 90% utilization ceilings."""
        logical_cores = psutil.cpu_count(logical=True) or 1
        active_cpu_workers = int(logical_cores * 0.90)

        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        ram_cap_gb = total_ram_gb * self._settings.max_ram_utilization_pct

        if torch.cuda.is_available():
            gpu_name = f"{torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} GPU)"
        else:
            gpu_name = "CPU Fallback Mode (No CUDA Driver)"

        return {
            "cpu": f"{active_cpu_workers} / {logical_cores} threads (90% target)",
            "ram": f"{ram_cap_gb:.1f} GB cap / {total_ram_gb:.1f} GB total",
            "gpu": gpu_name,
        }

    def _estimate_workload_and_eta(self, pending_images: int) -> tuple[int, str]:
        """Calculate estimated total tile count and total processing ETA string."""
        avg_tiles_per_image = 9
        est_tiles = pending_images * avg_tiles_per_image

        tiles_per_second = 180.0 if torch.cuda.is_available() else 15.0
        total_seconds = int(est_tiles / tiles_per_second) if est_tiles > 0 else 0

        eta_formatted = str(timedelta(seconds=total_seconds))
        return est_tiles, eta_formatted

    def _process_image_queue(
        self, run_id: UUID, stats: PipelineProgressStats
    ) -> None:
        """Fetch pending image batches and process each image through pipeline stages."""
        progress = self._create_progress_bar()
        with progress:
            task_id = progress.add_task("[cyan]Processing images...", total=stats.total_images)

            processed_count = 0
            while True:
                pending_batch = self._repo.fetch_pending_images(
                    run_id,
                    batch_size=self._settings.gpu_batch_size,
                    max_attempts=self._settings.max_attempts_per_image,
                )
                if not pending_batch:
                    break

                for img_rec in pending_batch:
                    self._process_single_image(img_rec, stats)
                    processed_count += 1
                    progress.update(task_id, advance=1)

                    if processed_count % self._settings.ram_check_interval_images == 0:
                        self._memory.check_and_throttle()

    def _create_progress_bar(self) -> Progress:
        """Create Rich progress bar instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )

    def _process_single_image(
        self, img_rec: ImageRecord, stats: PipelineProgressStats
    ) -> None:
        """Execute tiling, inference, rendering, and upload for a single image record."""
        try:
            self._execute_image_pipeline_steps(img_rec)
            stats.succeeded_images += 1
        except Exception as exc:
            stats.failed_images += 1
            logger.exception(f"Error processing image {img_rec.image_path}: {exc}")
            self._repo.mark_image_failed(img_rec.id, str(exc))

    def _execute_image_pipeline_steps(self, img_rec: ImageRecord) -> None:
        """Step-by-step pipeline execution for one image file."""
        self._repo.update_image_status(img_rec.id, "processing")

        full_img_bgr = cv2.imread(str(img_rec.image_path))
        if full_img_bgr is None:
            raise ValueError(f"Failed to read image at '{img_rec.image_path}'. Invalid image format.")

        h, w = full_img_bgr.shape[:2]
        tile_boxes = self._tiling.calculate_tile_grid(w, h)
        self._repo.update_image_status(
            img_rec.id, "tiling_done", width=w, height=h, tiles_total=len(tile_boxes)
        )

        tile_crops = self._extract_tile_crops(full_img_bgr, tile_boxes)
        raw_detections = self._detector.infer_tile_batch(tile_crops, tile_boxes, self._tiling)
        self._repo.update_image_status(img_rec.id, "inference_done")

        merged_detections = self._tiling.apply_cross_tile_nms(
            raw_detections, iou_threshold=self._settings.iou_threshold
        )
        self._repo.save_image_detections(img_rec.id, merged_detections)
        # Only render and upload annotated full-image outputs when detections were
        # found. Individual tiles are never uploaded.
        annotated_path = None
        gcp_url = None
        if merged_detections:
            out_name = f"annotated_{img_rec.image_path.name}"
            annotated_path = Path("./annotated_output") / out_name
            self._renderer.render_and_save(img_rec.image_path, annotated_path, merged_detections)
            self._repo.update_image_status(img_rec.id, "draw_done")

            # Upload only the annotated full image (not tiles)
            gcp_url = self._uploader.upload_file(annotated_path)

        # Mark completion in DB whether or not we uploaded (detections_count will be 0 if none)
        self._repo.record_image_completion(
            img_rec.id, annotated_path, gcp_url, len(merged_detections)
        )

    def _extract_tile_crops(
        self, img_bgr: cv2.Mat, tile_boxes: Sequence[TileBox]
    ) -> list[cv2.Mat]:
        """Crop sub-regions from full image matrix based on tile box coordinates."""
        crops: list[cv2.Mat] = []
        for tile in tile_boxes:
            crop = img_bgr[
                tile.y_offset : tile.y_offset + tile.height,
                tile.x_offset : tile.x_offset + tile.width,
            ]
            crops.append(crop)
        return crops
