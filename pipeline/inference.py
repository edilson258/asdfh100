"""YOLO model inference wrapper for CUDA batch processing."""

from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from pipeline.models import DetectionRecord, TileBox
from pipeline.tiling import ImageTilingEngine


def resolve_best_compute_device(requested_device: str = "cuda:0") -> str:
    """Resolve operational compute device across CUDA, MPS, and CPU fallback strategies.

    Example:
        >>> device = resolve_best_compute_device("cuda:0")
        >>> isinstance(device, str)
        True
    """
    if "cuda" in requested_device:
        device = _try_cuda_device_strategies(requested_device)
        if device != "cpu":
            _optimize_cuda_runtime()
            return device

    if "mps" in requested_device or _is_mps_available():
        if _can_initialize_device("mps"):
            return "mps"

    return "cpu"


def _try_cuda_device_strategies(requested_device: str) -> str:
    """Attempt multiple CUDA device initialization strategies."""
    if not torch.cuda.is_available():
        return "cpu"

    if _can_initialize_device(requested_device):
        return requested_device

    device_count = torch.cuda.device_count()
    for idx in range(device_count):
        candidate = f"cuda:{idx}"
        if _can_initialize_device(candidate):
            return candidate

    return "cpu"


def _can_initialize_device(device_str: str) -> bool:
    """Test initializing a small dummy PyTorch tensor on target device."""
    try:
        dummy = torch.zeros(1, device=device_str)
        _ = dummy + 1.0
        return True
    except Exception:
        return False


def _is_mps_available() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) is available."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def _optimize_cuda_runtime() -> None:
    """Enable conservative CUDA runtime optimizations for fixed-size tiling."""
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class YoloPlantDetectorEngine:
    """Ultralytics YOLO wrapper optimizing batch inference on RTX 5090 / H100 GPUs.

    Example:
        >>> engine = YoloPlantDetectorEngine(weights_path=Path("best.pt"), device="cuda:0")
        >>> isinstance(engine, YoloPlantDetectorEngine)
        True
    """

    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda:0",
        use_fp16: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Sequence[str] | None = None,
        batch_size: int = 64,
    ) -> None:
        """Initialize engine and load model into GPU resident memory.

        Example:
            >>> engine = YoloPlantDetectorEngine(Path("best.pt"), device="cpu")
        """
        resolved_device = resolve_best_compute_device(device)
        self._device = resolved_device
        self._use_fp16 = use_fp16 and "cuda" in resolved_device
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._batch_size = max(1, batch_size)
        if "cuda" in resolved_device:
            try:
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                if gpu_mem_gb >= 80:
                    self._batch_size = max(self._batch_size, 256)
                elif gpu_mem_gb >= 40:
                    self._batch_size = max(self._batch_size, 128)
            except Exception:
                pass
        # Accept None to mean "no filtering"; otherwise normalize to lowercase set
        self._target_classes = set([t.lower() for t in target_classes]) if target_classes else None

        self._model = self._load_yolo_model(weights_path, resolved_device)

    def _load_yolo_model(self, weights_path: Path, device: str) -> YOLO:
        """Load YOLO PyTorch model weights onto target compute device."""
        weights_str = str(weights_path) if weights_path.exists() else "yolov8n.pt"
        model = YOLO(weights_str)
        try:
            model.to(device)
        except Exception:
            model.to("cpu")
            self._device = "cpu"
            self._use_fp16 = False
        return model

    def infer_tile_batch(
        self,
        tile_crops: Sequence[np.ndarray | Image.Image],
        tile_boxes: Sequence[TileBox],
        tiling_engine: ImageTilingEngine,
    ) -> list[DetectionRecord]:
        """Perform batched GPU inference over tile crops and map coordinates.

        Example:
            >>> detections = engine.infer_tile_batch(crops, boxes, tiling_engine)
        """
        if not tile_crops:
            return []

        results = self._run_model_predict(tile_crops)
        return self._parse_and_map_results(results, tile_boxes, tiling_engine)

    def _run_model_predict(self, tile_crops: Sequence[np.ndarray | Image.Image]) -> list:
        """Execute model predict call under PyTorch FP16 autocast context if enabled."""
        if self._use_fp16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                return self._model.predict(
                    source=list(tile_crops),
                    conf=self._conf_threshold,
                    iou=self._iou_threshold,
                    device=self._device,
                    batch=self._batch_size,
                    verbose=False,
                )
        return self._model.predict(
            source=list(tile_crops),
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=self._device,
            batch=self._batch_size,
            verbose=False,
        )

    def _parse_and_map_results(
        self,
        results: list,
        tile_boxes: Sequence[TileBox],
        tiling_engine: ImageTilingEngine,
    ) -> list[DetectionRecord]:
        """Parse raw model predictions and map bounding box coords to original image space."""
        mapped_detections: list[DetectionRecord] = []
        for res, tile in zip(results, tile_boxes):
            if res.boxes is None or len(res.boxes) == 0:
                continue

            names_dict = res.names or {}
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = str(names_dict.get(cls_id, "")).lower()
                if self._target_classes and cls_name not in self._target_classes:
                    continue

                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                tile_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                mapped_bbox = tiling_engine.map_tile_coords_to_full_image(
                    tile_bbox, (tile.x_offset, tile.y_offset)
                )

                mapped_detections.append(
                    DetectionRecord(
                        class_name=cls_name,
                        confidence=conf,
                        bbox_x1=mapped_bbox[0],
                        bbox_y1=mapped_bbox[1],
                        bbox_x2=mapped_bbox[2],
                        bbox_y2=mapped_bbox[3],
                        tile_index=tile.tile_index,
                    )
                )
        return mapped_detections
