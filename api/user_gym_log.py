import logging
import string
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from core.database import get_db
from core.security import get_current_user
from crud.user_gym_logs import set_gym_flag, get_user_gym_logs, get_user_gym_log_by_date
from model.user import User
from schemas.user_gym_log import GymLogRequest

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/gym-log", tags=["gym-log"])


@router.patch("/flag")
def mark_attendance(
    data: GymLogRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"[Attendance] User {current_user.email} is marking {data.date} as {'attended' if data.went else 'missed'}")
    try:
        result = set_gym_flag(db, current_user.id, data)
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


@router.get("/{log_date}")
def get_log_by_date(
    log_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"[Logs] Retrieving gym log for user: {current_user.email} on date: {log_date}")
    try:
        log_entry = get_user_gym_log_by_date(db, current_user.id, log_date)
        if not log_entry:
            raise HTTPException(status_code=404, detail="Log entry not found")
        logger.info(f"[Logs] Retrieved log for {current_user.email} on {log_date}")
        return log_entry
    except Exception as e:
        logger.error(f"[Logs] Failed to retrieve log for {current_user.email} on {log_date} - Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve log entry")
