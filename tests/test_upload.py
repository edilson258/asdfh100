"""Unit tests for GCP Storage image uploader module."""

from datetime import datetime, timezone
from pathlib import Path
from tests.fakes import FakeStorageUploader


def test_fake_uploader_delete_all_uploaded_blobs(tmp_path: Path) -> None:
    """Verify deleting uploaded blobs resets storage tracking."""
    uploader = FakeStorageUploader(bucket_name="test-bucket")
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img1.write_bytes(b"data1")
    img2.write_bytes(b"data2")

    uploader.upload_file(img1)
    uploader.upload_file(img2)
    assert len(uploader.uploaded_files) == 2

    deleted_count = uploader.delete_all_uploaded_blobs()
    assert deleted_count == 2
    assert len(uploader.uploaded_files) == 0


def test_upload_date_prefix_format() -> None:
    """Verify UTC date string format YYYY-MM-DD."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert len(date_str) == 10
    assert date_str.count("-") == 2
