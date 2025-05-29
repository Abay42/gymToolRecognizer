import time

from fastapi import APIRouter, Depends, HTTPException, Body, UploadFile, File
from sqlalchemy.orm import Session
import logging

from core.config import Settings
from core.database import get_db, upload_image_to_minio
from core.security import get_current_user
from crud.user import get_profile, update_profile, delete_user, update_profile_image
from schemas.user import UserProfile, UserProfileUpdate
from model.user import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/profile", response_model=UserProfile)
def profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = get_profile(db, current_user.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"Profile accessed: {current_user.email}")
    return user


@router.put("/update", response_model=UserProfile)
def update(
    current_user: User = Depends(get_current_user),
    profile_data: UserProfileUpdate = Body(...),
    db: Session = Depends(get_db)
):
    updated_user = update_profile(db, current_user.email, profile_data)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found or update failed")

    logger.info(f"Profile updated: {current_user.email}")
    return updated_user


@router.post("/upload-profile-image")
def upload_profile_image(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_bytes = file.file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 5MB")

    try:
        filename = f"{current_user}_{int(time.time())}.jpg"

        image_url = upload_image_to_minio(
            file_bytes,
            filename,
            Settings.PROFILE_BUCKET
        )

        updated_user = update_profile_image(db, current_user.email, image_url)

        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(f"Profile image uploaded for user: {current_user.email}")
        return {
            "message": "Profile image uploaded successfully",
            "image_url": image_url
        }

    except Exception as e:
        logger.error(f"Error uploading profile image for {current_user.email}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload image")


@router.delete("/profile-image")
def delete_profile_image(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    updated_user = update_profile_image(db, current_user.email, None)

    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"Profile image deleted for user: {current_user.email}")
    return {"message": "Profile image removed successfully"}


@router.delete("/delete")
def delete(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = delete_user(db, current_user.email)
    if not result:
        raise HTTPException(status_code=404, detail="User not found or deletion failed")

    logger.info(f"User deleted: {current_user.email}")
    return result
