from minio import Minio
from minio.error import S3Error
import os
from typing import Optional, BinaryIO
from fastapi import HTTPException, status

class MinioClient:
    def __init__(self):
        self.client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        self.bucket_name = "product-images"
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Create the bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize MinIO bucket: {str(e)}"
            )

    def upload_file(self, file_data: BinaryIO, file_name: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload a file to MinIO
        
        Args:
            file_data: File-like object containing the file data
            file_name: Name to give the file in MinIO
            content_type: MIME type of the file
            
        Returns:
            str: URL of the uploaded file
        """
        try:
            # Get file size by seeking to end
            file_data.seek(0, os.SEEK_END)
            file_size = file_data.tell()
            file_data.seek(0)
            
            # Upload the file
            self.client.put_object(
                self.bucket_name,
                file_name,
                file_data,
                file_size,
                content_type=content_type
            )
            
            # Return the URL to access the file
            return f"http://localhost:9000/{self.bucket_name}/{file_name}"
            
        except S3Error as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to MinIO: {str(e)}"
            )

# Create a singleton instance
minio_client = MinioClient()
