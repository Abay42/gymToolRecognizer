from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os

load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Recognizing GYM Tools")
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 480))
    MINIO_URL: str = os.getenv("MINIO_URL")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY")
    MUSCLE_BUCKET: str = os.getenv("MUSCLE_BUCKET")
    PROFILE_BUCKET: str = os.getenv("PROFILE_IMAGE_BUCKET")
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_PUBLISHABLEKEY: str = os.getenv("STRIPE_PUBLISHABLEKEY")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN")
    GITHUB_MODEL_ENDPOINT: str = os.getenv("GITHUB_MODEL_ENDPOINT")
    MODEL: str = os.getenv("MODEL")
    TOGIS_TOKEN: str = os.getenv("TOGIS_TOKEN")
    TOGIS_URL: str = os.getenv("TOGIS_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()

