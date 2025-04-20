import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from core.database import get_db
from core.security import get_current_user
from crud.user_gym_logs import set_gym_flag, get_user_gym_logs
from datetime import date
from pydantic import BaseModel
from model.user import User

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/gym-log", tags=["gym-log"])


class GymLogRequest(BaseModel):
    date: date
    went: bool


@router.patch("/flag")
def mark_attendance(
    data: GymLogRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"[Attendance] User {current_user.email} is marking {data.date} as {'attended' if data.went else 'missed'}")
    try:
        result = set_gym_flag(db, current_user.id, data.date, data.went)
        logger.info(f"[Attendance] Update successful for {current_user.email} on {data.date}")
        return result
    except Exception as e:
        logger.error(f"[Attendance] Failed to mark attendance for {current_user.email} - Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark attendance")


@router.get("/logs")
def get_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"[Logs] Retrieving gym logs for user: {current_user.email}")
    try:
        logs = get_user_gym_logs(db, current_user.id)
        logger.info(f"[Logs] Retrieved {len(logs)} log entries for {current_user.email}")
        return logs
    except Exception as e:
        logger.error(f"[Logs] Failed to retrieve logs for {current_user.email} - Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")
