import io
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from core.config import settings
from minio import Minio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


minio_client = Minio(
    settings.MINIO_URL,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=True
)


def upload_image_to_minio(file_bytes: bytes, file_name: str, bucket_name: str) -> str:
    logger.info(f"Preparing to upload image: {file_name} to MinIO bucket: {bucket_name}")

    if not minio_client.bucket_exists(bucket_name):
        logger.info(f"Bucket {bucket_name} does not exist. Creating it now.")
        minio_client.make_bucket(bucket_name)
    else:
        logger.info(f"Bucket {bucket_name} already exists.")

    try:
        logger.info(f"Uploading image: {file_name} to MinIO...")
        minio_client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=file_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type="image/jpeg"
        )
        logger.info(f"Successfully uploaded image: {file_name} to MinIO bucket: {bucket_name}")

        image_url = f"http://{settings.MINIO_URL}/{bucket_name}/{file_name}"
        logger.info(f"Image URL: {image_url}")
        return image_url

    except Exception as e:
        logger.error(f"Error uploading image: {file_name} to MinIO. Error: {str(e)}")
        raise

