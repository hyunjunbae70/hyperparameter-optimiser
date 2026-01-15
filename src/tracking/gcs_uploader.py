import os
from typing import Optional

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class GCSUploader:
    def __init__(
        self, bucket_name: Optional[str] = None, project_id: Optional[str] = None
    ):
        if not GCS_AVAILABLE:
            print("Warning: google-cloud-storage not available. GCS upload disabled.")
            self.enabled = False
            return

        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

        if not self.bucket_name:
            print("Warning: No GCS bucket specified. GCS upload disabled.")
            self.enabled = False
            return

        try:
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            self.enabled = True
        except Exception as e:
            print(f"Warning: Failed to initialize GCS client: {e}")
            self.enabled = False

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        if not self.enabled:
            return False

        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            print(f"Error uploading {local_path} to GCS: {e}")
            return False

    def upload_directory(self, local_dir: str, remote_prefix: str) -> int:
        if not self.enabled:
            return 0

        uploaded_count = 0

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                remote_path = os.path.join(remote_prefix, relative_path)

                if self.upload_file(local_path, remote_path):
                    uploaded_count += 1

        return uploaded_count

    def download_file(self, remote_path: str, local_path: str) -> bool:
        if not self.enabled:
            return False

        try:
            blob = self.bucket.blob(remote_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"Downloaded gs://{self.bucket_name}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            print(f"Error downloading from GCS: {e}")
            return False
