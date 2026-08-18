"""Unit tests for image tiling engine and cross-tile NMS deduplication."""

import pytest
from pipeline.models import DetectionRecord
from pipeline.tiling import ImageTilingEngine


def test_tiling_engine_init_validation() -> None:
    """Test validation of invalid tile size and overlap parameters."""
    with pytest.raises(ValueError, match="tile_size must be positive"):
        ImageTilingEngine(tile_size=0)

    with pytest.raises(ValueError, match="tile_overlap_pct must be in range"):
        ImageTilingEngine(tile_overlap_pct=1.5)


def test_calculate_tile_grid_small_image() -> None:
    """Test grid generation when image fits inside a single tile."""
    engine = ImageTilingEngine(tile_size=640, tile_overlap_pct=0.15)
    grid = engine.calculate_tile_grid(img_w=500, img_h=400)

    assert len(grid) == 1
    assert grid[0].x_offset == 0
    assert grid[0].y_offset == 0
    assert grid[0].width == 500
    assert grid[0].height == 400


def test_calculate_tile_grid_large_image() -> None:
    """Test grid generation for a large high-resolution image."""
    engine = ImageTilingEngine(tile_size=640, tile_overlap_pct=0.15)
    grid = engine.calculate_tile_grid(img_w=1920, img_h=1080)

    assert len(grid) > 1
    for tile in grid:
        assert tile.width <= 640
        assert tile.height <= 640


test_map_tile_coords_to_full_image_data = [
    ((10, 20, 50, 60), (100, 200), (110, 220, 150, 260)),
    ((0, 0, 100, 100), (0, 0), (0, 0, 100, 100)),
]


@pytest.mark.parametrize("bbox,offset,expected", test_map_tile_coords_to_full_image_data)
def test_map_tile_coords_to_full_image(
    bbox: tuple[int, int, int, int],
    offset: tuple[int, int],
    expected: tuple[int, int, int, int],
) -> None:
    """Test mapping tile coordinates to full image coordinate space."""
    engine = ImageTilingEngine(tile_size=640, tile_overlap_pct=0.15)
    result = engine.map_tile_coords_to_full_image(bbox, offset)
    assert result == expected


def test_apply_cross_tile_nms_deduplication() -> None:
    """Test that highly overlapping detections across tiles are deduplicated."""
    engine = ImageTilingEngine(tile_size=640, tile_overlap_pct=0.15)
    det1 = DetectionRecord("plant", 0.95, 100, 100, 200, 200, tile_index=0)
    det2 = DetectionRecord("plant", 0.80, 105, 105, 205, 205, tile_index=1)

    filtered = engine.apply_cross_tile_nms([det1, det2], iou_threshold=0.45)
    assert len(filtered) == 1
    assert filtered[0].confidence == 0.95
