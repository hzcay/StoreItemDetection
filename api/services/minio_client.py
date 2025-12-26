from minio import Minio
from minio.error import S3Error
import os
from typing import BinaryIO, Optional
from fastapi import HTTPException, status
import io
from datetime import timedelta
# Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "product-images")


def generate_presigned_url(
    object_name: str,
    expiry_seconds: int = 3600
) -> str:
    return minio_client.presigned_get_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        expires=timedelta(seconds=expiry_seconds),
    )

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,  # Set to True for HTTPS
)

def init_minio():
    """Initialize MinIO client and ensure bucket exists."""
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
        return minio_client
    except S3Error as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize MinIO: {str(e)}"
        )

def put_object(
    bucket_name: str,
    object_name: str,
    data: BinaryIO,
    length: int,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload an object to MinIO.
    
    Args:
        bucket_name: Name of the bucket
        object_name: Name of the object in the bucket
        data: Binary data to upload
        length: Length of the data in bytes
        content_type: MIME type of the object
        
    Returns:
        str: URL of the uploaded object
        
    Raises:
        HTTPException: If upload fails
    """
    try:
        # Ensure we're at the start of the stream
        if hasattr(data, 'seek'):
            data.seek(0)
            
        minio_client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data,
            length=length,
            content_type=content_type
        )
        
        return f"http://{MINIO_ENDPOINT}/{bucket_name}/{object_name}"
        
    except S3Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file to MinIO: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error uploading file: {str(e)}"
        )
