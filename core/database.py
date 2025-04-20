import io

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from core.config import settings
from minio import Minio


engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

engine = create_engine(settings.DATABASE_URL)


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
    secure=False
)


def upload_image_to_minio(file_bytes: bytes, file_name: str) -> str:
    if not minio_client.bucket_exists(settings.MINIO_BUCKET):
        minio_client.make_bucket(settings.MINIO_BUCKET)

    minio_client.put_object(
        bucket_name=settings.MINIO_BUCKET,
        object_name=file_name,
        data=io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type="image/jpeg"
    )

    return f"http://{settings.MINIO_URL}/{settings.MINIO_BUCKET}/{file_name}"
