import random
import logging
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from PIL import Image
import io


from core.database import get_db
from model.gymtool import GymTool
from core.security import get_current_user
from model.user import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["cv"])


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logger.info(f"Received image for prediction from user: {current_user.email}")

    try:
        Image.open(io.BytesIO(await file.read()))
        logger.info("Image successfully loaded into memory.")

        gym_tools = db.query(GymTool).all()
        if not gym_tools:
            logger.warning("No gym tools found in database.")
            return {"error": "No gym tools found in database"}

        gym_tool = random.choice(gym_tools)
        logger.info(f"Predicted gym tool: {gym_tool.name} (ID: {gym_tool.id})")

        muscles_info = [
            {"id": muscle.id, "name": muscle.name, "image_url": muscle.image_url}
            for muscle in gym_tool.muscles
        ]

        response = {
            "class_id": gym_tool.id,
            "name": gym_tool.name,
            "description": gym_tool.description,
            "links": gym_tool.links,
            "alternative": gym_tool.alternative,
            "muscles": muscles_info,
            "requested_by": current_user.email
        }

        logger.info("Prediction response generated successfully.")
        logger.info(f"Response is: {response}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": "Internal server error"}

