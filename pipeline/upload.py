"""GCP Cloud Storage client wrapper for uploading annotated images."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from google.cloud import storage


class GcpImageUploader:
    """Uploads annotated output image files to GCP Cloud Storage.

    Example:
        >>> uploader = GcpImageUploader(bucket_name="my-bucket", prefix="annotated/")
        >>> isinstance(uploader, GcpImageUploader)
        True
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "annotated/",
        credentials_path: Path | None = None,
    ) -> None:
        """Initialize GCS client and target bucket.

        Example:
            >>> uploader = GcpImageUploader("plant-bucket")
        """
        self._bucket_name = bucket_name
        self._prefix = prefix.strip("/") + "/" if prefix else ""
        self._client = self._create_storage_client(credentials_path)

    def _create_storage_client(self, credentials_path: Path | None) -> storage.Client:
        """Instantiate GCS client using credentials file or Application Default Credentials."""
        if credentials_path and credentials_path.exists():
            return storage.Client.from_service_account_json(str(credentials_path))
        return storage.Client()

    def upload_file(self, local_path: Path, object_key: str | None = None) -> str:
        """Upload a local image file to GCP bucket with date prefix and return URL.

        Example:
            >>> url = uploader.upload_file(Path("/tmp/annotated.jpg"))
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file to upload not found at '{local_path}'.")

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = object_key or f"{self._prefix}{date_str}/{local_path.name}"
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(key)

        blob.upload_from_filename(str(local_path))
        return f"https://storage.googleapis.com/{self._bucket_name}/{key}"

    def upload_batch(
        self, local_paths: Sequence[Path], max_threads: int = 16
    ) -> list[str]:
        """Upload a batch of files concurrently using a ThreadPoolExecutor.

        Example:
            >>> urls = uploader.upload_batch([Path("/tmp/img1.jpg")], max_threads=4)
        """
        if not local_paths:
            return []

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(self.upload_file, path) for path in local_paths]
            return [f.result() for f in futures]

    def delete_all_uploaded_blobs(self) -> int:
        """Delete all objects stored in the target GCP bucket under prefix.

        Example:
            >>> deleted_count = uploader.delete_all_uploaded_blobs()
        """
        bucket = self._client.bucket(self._bucket_name)
        blobs = list(bucket.list_blobs(prefix=self._prefix))
        if not blobs:
            return 0
        bucket.delete_blobs(blobs)
        return len(blobs)
