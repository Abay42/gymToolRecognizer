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
    goal: str  # e.g., hypertrophy, fat loss
    level: str  # beginner/intermediate/advanced
    focus: str  # chest, full body, etc.
    equipment: str  # dumbbells, machines
    duration_minutes: int


@router.post("/generate")
def generate_workout_plan(
        req: WorkoutRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    prompt = (
        f"Create a {req.goal} workout plan for a {req.level} user "
        f"focusing on {req.focus} using {req.equipment}. "
        f"The session should last {req.duration_minutes} minutes."
    )
    try:
        user_logs = get_user_gym_logs(db, current_user.id)
        workout = generate_workout(prompt, user_logs)
        return {"workout_plan": workout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate workout: {str(e)}")