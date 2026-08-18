"""File discovery module for discovering and hashing target images."""

import hashlib
from pathlib import Path
from typing import Sequence


class ImageFileScanner:
    """Discovers images within a local directory and computes file hashes.

    Example:
        >>> scanner = ImageFileScanner(allowed_extensions=(".jpg", ".png"))
        >>> images = scanner.discover_images(Path("/data/images"), recursive=False)
    """

    def __init__(self, allowed_extensions: Sequence[str]) -> None:
        """Initialize scanner with allowed file extensions.

        Example:
            >>> scanner = ImageFileScanner(allowed_extensions=(".jpg", ".png"))
        """
        self._allowed_exts = tuple(ext.lower() for ext in allowed_extensions)

    def discover_images(self, input_dir: Path, recursive: bool = False) -> list[Path]:
        """Scan directory and return sorted list of supported image file paths.

        Example:
            >>> paths = scanner.discover_images(Path("/images"), recursive=True)
        """
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory.")

        pattern = "**/*" if recursive else "*"
        return [
            file_path
            for file_path in sorted(input_dir.glob(pattern))
            if file_path.is_file() and file_path.suffix.lower() in self._allowed_exts
        ]

    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 checksum string for a target file.

        Example:
            >>> hash_str = scanner.compute_sha256(Path("/image.jpg"))
            >>> len(hash_str)
            64
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot hash missing file at '{file_path}'.")

        sha256 = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
