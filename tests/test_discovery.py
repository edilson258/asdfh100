"""Unit tests for image discovery and hashing scanner module."""

from pathlib import Path
import pytest
from pipeline.discovery import ImageFileScanner


def test_discover_images_flat(tmp_path: Path) -> None:
    """Test discovering images in a flat directory structure."""
    scanner = ImageFileScanner(allowed_extensions=(".jpg", ".png"))
    (tmp_path / "img1.jpg").write_bytes(b"data1")
    (tmp_path / "img2.png").write_bytes(b"data2")
    (tmp_path / "ignore.txt").write_bytes(b"data3")

    found = scanner.discover_images(tmp_path, recursive=False)
    assert len(found) == 2
    assert sorted([p.name for p in found]) == ["img1.jpg", "img2.png"]


def test_discover_images_recursive(tmp_path: Path) -> None:
    """Test discovering images recursively across subdirectories."""
    scanner = ImageFileScanner(allowed_extensions=(".jpg",))
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    (tmp_path / "img1.jpg").write_bytes(b"data1")
    (sub_dir / "img2.jpg").write_bytes(b"data2")

    found = scanner.discover_images(tmp_path, recursive=True)
    assert len(found) == 2


def test_compute_sha256(tmp_path: Path) -> None:
    """Test computing SHA256 checksum for a local file."""
    scanner = ImageFileScanner(allowed_extensions=(".jpg",))
    sample_file = tmp_path / "test.jpg"
    sample_file.write_bytes(b"hello world")

    hash_str = scanner.compute_sha256(sample_file)
    assert len(hash_str) == 64
    assert isinstance(hash_str, str)


def test_discover_missing_directory_raises() -> None:
    """Test that scanning a non-existent directory raises ValueError."""
    scanner = ImageFileScanner(allowed_extensions=(".jpg",))
    with pytest.raises(ValueError, match="Input directory"):
        scanner.discover_images(Path("/non/existent/path/xyz"))
