import random
import logging
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from PIL import Image
import io

from core.config import settings
from core.converter import image_to_base64, CLASS_NAMES
from core.database import get_db, get_image_from_minio
from cv.GymToolRecognizer import GymToolRecognizer
from model.gymtool import GymTool
from core.security import get_current_user
from model.user import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["cv"])

recognizer = GymToolRecognizer("cv/model.pth")


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logger.info(f"Received image for prediction from user: {current_user.email}")

    try:
        if not current_user.is_sub:
            if current_user.free_attempts <= 0:
                logger.warning(f"No free attempts left for user: {current_user.email}")
                return {"error": "No free attempts left. Please subscribe to continue."}

            current_user.free_attempts -= 1
            db.commit()
            logger.info(f"Decreased attempt. Remaining: {current_user.free_attempts}")

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        logger.info("Image successfully loaded into memory.")

        gym_tools = db.query(GymTool).all()
        if not gym_tools:
            logger.warning("No gym tools found in database.")
            return {"error": "No gym tools found in database."}

        predicted_id = recognizer.predict_image(image)
        logger.info(f"Model predicted class ID: {predicted_id}")

        predicted_name = CLASS_NAMES.get(predicted_id, None)
        if not predicted_name:
            logger.warning(f"Class name not found for ID: {predicted_id}")
            return {"error": f"Class name not found for ID: {predicted_id}"}

        gym_tool = db.query(GymTool).filter(GymTool.name == predicted_name).first()
        if not gym_tool:
            logger.warning(f"Predicted ID {predicted_id} not found in DB.")
            return {"error": "Predicted gym tool not found."}

        encoded_image = image_to_base64(image)
        muscles_info = []
        for assoc in gym_tool.muscle_associations:
            muscle_data = {
                "name": assoc.muscle.name,
                "primary": assoc.primary_muscles,
                "secondary": assoc.secondary_muscles,
            }

            if assoc.muscle.image_url:
                logger.info(f"Converting muscle image URL to base64: {assoc.muscle.image_url}")
                muscle_image_b64 = get_image_from_minio(assoc.muscle.image_url)
                if muscle_image_b64:
                    muscle_data["image_b64"] = muscle_image_b64
                    logger.info(f"Successfully encoded muscle image for: {assoc.muscle.name}")
                else:
                    muscle_data["image_url"] = assoc.muscle.image_url
                    muscle_data["image_b64"] = None
                    logger.warning(f"Failed to encode muscle image, keeping URL: {assoc.muscle.name}")
            else:
                muscle_data["image_url"] = None
                muscle_data["image_b64"] = None

            muscles_info.append(muscle_data)

        response = {
            "class_id": gym_tool.id,
            "class_name": predicted_name,
            "name": gym_tool.name,
            "description": gym_tool.description,
            "links": gym_tool.links,
            "alternative": gym_tool.alternative,
            "muscles": muscles_info,
            "requested_by": current_user.email,
            "free_attempts_left": current_user.free_attempts if not current_user.is_sub else "∞",
            "image_b64": encoded_image
        }

        logger.info("Prediction response generated successfully.")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": "Internal server error"}

