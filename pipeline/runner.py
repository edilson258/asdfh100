"""Batch pipeline execution orchestrator integrating DB tracking, tiling, inference, and upload."""

import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Sequence
from uuid import UUID

import cv2
import psutil
import torch
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from config import PipelineSettings
from db.repository import PlantDetectionRepository
from pipeline.discovery import ImageFileScanner
from pipeline.drawing import BoundingBoxRenderer
from pipeline.inference import YoloPlantDetectorEngine
from pipeline.models import ImageRecord, PipelineProgressStats, TileBox
from pipeline.resource_monitor import MemoryGuardScheduler
from pipeline.tiling import ImageTilingEngine
from pipeline.upload import GcpImageUploader

# Type alias for what the loader stage hands off to the GPU stage.
_LoadedImage = tuple[ImageRecord, list[TileBox], list["cv2.Mat"]]


class BatchPipelineOrchestrator:
    """Orchestrates end-to-end batch plant detection pipeline over 20,000+ images.

    The pipeline is split into three overlapped stages so the GPU is never left
    idle waiting on disk I/O, JPEG decode/encode, DB writes, or GCP uploads:

        loader threads  ->  GPU inference (main thread)  ->  postproc threads
        (read/decode/tile)   (strictly sequential CUDA calls)   (NMS/draw/DB/upload)

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

        # The repository's thread-safety isn't guaranteed by this file's
        # contract, so DB calls made from loader/postproc worker threads are
        # serialized through this lock. If PlantDetectionRepository already
        # uses a connection pool internally, this only adds a (cheap) extra
        # serialization point around individual statements -- it does not
        # change what gets called or how.
        self._repo_lock = threading.Lock()

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

        avg_tiles_per_image = (
            int(total_tiles_sample / sample_n)
            if sample_n > 0 and total_tiles_sample > 0
            else 0
        )
        est_tiles = (
            avg_tiles_per_image * len(discovered) if avg_tiles_per_image > 0 else 0
        )

        # Use configured max attempts when fetching pending images so we don't retry forever
        pending_batch = self._repo.fetch_pending_images(
            run_id,
            batch_size=len(discovered) or 1,
            max_attempts=self._settings.max_attempts_per_image,
        )
        pending_count = len(pending_batch)
        completed_count = stats.total_images - pending_count

        # Compute ETA based on sampled average tiles and device capability
        tiles_per_second = 180.0 if torch.cuda.is_available() else 15.0
        # scale by tile area ratio relative to 640 baseline
        base_tile = 640
        scale = (base_tile * base_tile) / max(
            1, self._settings.tile_size * self._settings.tile_size
        )
        tiles_per_second = max(1.0, tiles_per_second * scale)

        total_seconds = int(est_tiles / tiles_per_second) if est_tiles > 0 else 0
        eta_str = str(timedelta(seconds=total_seconds))

        self._print_pre_run_summary(
            stats.total_images,
            completed_count,
            pending_count,
            run_id,
            est_tiles,
            eta_str,
        )

        if dry_run:
            logger.info("Dry-run flag set. Skipping inference loop.")
            self._repo.mark_run_status(run_id, "completed")
            return stats

        self._process_image_queue(run_id, stats)
        self._repo.mark_run_status(run_id, "completed")
        return stats

    def _print_pre_run_summary(
        self,
        total: int,
        completed: int,
        pending: int,
        run_id: UUID,
        est_tiles: int = 0,
        eta_str: str = "0:00:00",
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
        for p in list(
            self._scanner.discover_images(
                self._settings.input_dir, recursive=self._settings.recursive_scan
            )
        )[:5]:
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
            gpu_name = (
                f"{torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} GPU)"
            )
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

    # ------------------------------------------------------------------
    # Overlapped pipeline: loader threads -> GPU (main thread) -> postproc
    # threads. This is the part that changed to keep the GPU continuously
    # fed instead of idling through decode/tile/NMS/draw/upload each image.
    # ------------------------------------------------------------------

    def _process_image_queue(self, run_id: UUID, stats: PipelineProgressStats) -> None:
        """Fetch pending images and run them through the pipeline with the
        load, inference, and post-process stages overlapped across images.
        """
        progress = self._create_progress_bar()
        stats_lock = threading.Lock()
        progress_lock = threading.Lock()

        # Allow future config overrides without requiring config.py changes
        # right now: falls back to sensible CPU-derived defaults if the
        # settings object doesn't define these fields.
        cpu_count = os.cpu_count() or 4
        num_loader_workers = getattr(self._settings, "loader_workers", None) or max(
            2, min(8, cpu_count)
        )
        num_postproc_workers = getattr(
            self._settings, "postprocess_workers", None
        ) or max(2, min(8, cpu_count))
        window_size = max(2, num_loader_workers * 2)
        postproc_backlog_limit = num_postproc_workers * 4

        task_id = progress.add_task(
            "[cyan]Processing images...", total=stats.total_images
        )

        def fetch_pending_iter():
            """Yield pending ImageRecords in batches from the repository."""
            while True:
                with self._repo_lock:
                    batch = self._repo.fetch_pending_images(
                        run_id,
                        batch_size=self._settings.gpu_batch_size,
                        max_attempts=self._settings.max_attempts_per_image,
                    )
                if not batch:
                    return
                for rec in batch:
                    yield rec

        record_iter = fetch_pending_iter()
        pending_futures: "queue.Queue[Future]" = queue.Queue()
        postproc_backpressure = threading.BoundedSemaphore(postproc_backlog_limit)
        outstanding_postproc: list[Future] = []
        processed_count = 0

        def on_step_done() -> None:
            nonlocal processed_count
            processed_count += 1
            with progress_lock:
                progress.update(task_id, advance=1)
            if processed_count % self._settings.ram_check_interval_images == 0:
                self._memory.check_and_throttle()

        def make_postproc_callback() -> "callable[[Future], None]":
            def _callback(fut: Future) -> None:
                try:
                    success = fut.result()
                except Exception:
                    success = False
                with stats_lock:
                    if success:
                        stats.succeeded_images += 1
                    else:
                        stats.failed_images += 1
                postproc_backpressure.release()

            return _callback

        with (
            progress,
            ThreadPoolExecutor(
                max_workers=num_loader_workers, thread_name_prefix="loader"
            ) as loader_pool,
            ThreadPoolExecutor(
                max_workers=num_postproc_workers, thread_name_prefix="postproc"
            ) as postproc_pool,
        ):

            def submit_next_load() -> bool:
                """Pull the next pending record and submit it for loading.
                Returns False once the record source is exhausted.
                """
                try:
                    img_rec = next(record_iter)
                except StopIteration:
                    return False
                fut = loader_pool.submit(self._load_and_tile_image, img_rec)
                pending_futures.put(fut)
                return True

            # Prime the loader window so the GPU has work waiting as soon as
            # the main loop starts.
            active = 0
            for _ in range(window_size):
                if submit_next_load():
                    active += 1
                else:
                    break

            while active > 0:
                load_future = pending_futures.get()
                active -= 1

                # Immediately backfill the window so loading for future
                # images keeps happening while we run GPU inference below.
                if submit_next_load():
                    active += 1

                loaded = load_future.result()
                if loaded is None:
                    # Failure already logged and recorded in the DB by
                    # _load_and_tile_image.
                    with stats_lock:
                        stats.failed_images += 1
                    on_step_done()
                    continue

                img_rec, tile_boxes, tile_crops = loaded

                # GPU inference is kept strictly on the main thread so CUDA
                # calls stay sequential and back-to-back -- this avoids
                # cross-thread CUDA context contention while the loader and
                # postproc pools keep everything else overlapped around it.
                try:
                    raw_detections = self._run_gpu_inference(
                        img_rec, tile_crops, tile_boxes
                    )
                except Exception as exc:
                    logger.exception(
                        f"GPU inference failed for {img_rec.image_path}: {exc}"
                    )
                    with self._repo_lock:
                        self._repo.mark_image_failed(img_rec.id, str(exc))
                    with stats_lock:
                        stats.failed_images += 1
                    on_step_done()
                    continue

                # Hand NMS + draw + DB write + upload off to the postproc
                # pool so the GPU can move straight to the next image
                # instead of waiting on disk/network I/O.
                postproc_backpressure.acquire()
                pp_future = postproc_pool.submit(
                    self._postprocess_and_upload, img_rec, raw_detections
                )
                pp_future.add_done_callback(make_postproc_callback())
                outstanding_postproc.append(pp_future)

                on_step_done()

            # Drain any postproc work still in flight before returning so
            # stats/DB state are fully settled when run_pipeline() returns.
            for pp_future in outstanding_postproc:
                pp_future.result()

    def _create_progress_bar(self) -> Progress:
        """Create Rich progress bar instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )

    def _load_and_tile_image(self, img_rec: ImageRecord) -> _LoadedImage | None:
        """Read, tile, and crop a single image (loader-thread stage).

        Runs on a loader worker thread so disk I/O and JPEG decode overlap
        with GPU inference happening on other images. Returns None (after
        logging and recording the failure) if the image can't be read.
        """
        try:
            with self._repo_lock:
                self._repo.update_image_status(img_rec.id, "processing")

            full_img_bgr = cv2.imread(str(img_rec.image_path))
            if full_img_bgr is None:
                raise ValueError(
                    f"Failed to read image at '{img_rec.image_path}'. Invalid image format."
                )

            h, w = full_img_bgr.shape[:2]
            tile_boxes = self._tiling.calculate_tile_grid(w, h)
            with self._repo_lock:
                self._repo.update_image_status(
                    img_rec.id,
                    "tiling_done",
                    width=w,
                    height=h,
                    tiles_total=len(tile_boxes),
                )

            tile_crops = self._extract_tile_crops(full_img_bgr, tile_boxes)
            # Release the full-resolution buffer now -- downstream drawing
            # re-reads img_rec.image_path from disk itself, so we don't need
            # to keep this in memory while it waits for the GPU/postproc.
            del full_img_bgr

            return img_rec, tile_boxes, tile_crops
        except Exception as exc:
            logger.exception(f"Error loading/tiling image {img_rec.image_path}: {exc}")
            with self._repo_lock:
                self._repo.mark_image_failed(img_rec.id, str(exc))
            return None

    def _run_gpu_inference(
        self,
        img_rec: ImageRecord,
        tile_crops: list["cv2.Mat"],
        tile_boxes: Sequence[TileBox],
    ) -> list:
        """Run YOLO inference for all tiles of one image (GPU-thread stage).

        Always called from the main thread -- see the note in
        _process_image_queue on why CUDA calls are kept sequential.
        """
        raw_detections = self._detector.infer_tile_batch(
            tile_crops, tile_boxes, self._tiling
        )
        with self._repo_lock:
            self._repo.update_image_status(img_rec.id, "inference_done")
        return raw_detections

    def _postprocess_and_upload(
        self, img_rec: ImageRecord, raw_detections: list
    ) -> bool:
        """NMS, draw, save detections, and upload the annotated image
        (postproc-thread stage). Runs off the GPU thread so drawing and
        network I/O never block inference on the next image.

        Returns True on success; on failure, logs, records the failure in
        the DB, and returns False so the caller can update run stats.
        """
        try:
            merged_detections = self._tiling.apply_cross_tile_nms(
                raw_detections, iou_threshold=self._settings.iou_threshold
            )
            with self._repo_lock:
                self._repo.save_image_detections(img_rec.id, merged_detections)

            # Only render and upload annotated full-image outputs when
            # detections were found. Individual tiles are never uploaded.
            annotated_path = None
            gcp_url = None
            if merged_detections:
                out_name = f"annotated_{img_rec.image_path.name}"
                annotated_path = Path("./annotated_output") / out_name
                self._renderer.render_and_save(
                    img_rec.image_path, annotated_path, merged_detections
                )
                with self._repo_lock:
                    self._repo.update_image_status(img_rec.id, "draw_done")

                # Upload only the annotated full image (not tiles)
                gcp_url = self._uploader.upload_file(annotated_path)

            with self._repo_lock:
                self._repo.record_image_completion(
                    img_rec.id, annotated_path, gcp_url, len(merged_detections)
                )
            return True
        except Exception as exc:
            logger.exception(f"Error post-processing image {img_rec.image_path}: {exc}")
            with self._repo_lock:
                self._repo.mark_image_failed(img_rec.id, str(exc))
            return False

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
