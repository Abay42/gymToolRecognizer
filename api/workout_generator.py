from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.database import get_db
from core.security import get_current_user
from crud.user_gym_logs import get_user_gym_logs
from crud.workout_generator import generate_workout
from model.user import User
from sqlalchemy.orm import Session


router = APIRouter(prefix="/workout", tags=["workout"])


class WorkoutRequest(BaseModel):
    goal: str


@router.post("/generate")
def generate_workout_plan(
        req: WorkoutRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    prompt = (
        f"Создай программу тренировок для {req.goal}."
    )
    try:
        user_logs = get_user_gym_logs(db, current_user.id)
        workout = generate_workout(prompt, user_logs)
        return {"workout_plan": workout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось создать тренировку: {str(e)}")