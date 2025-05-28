from typing import List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.database import get_db
from core.security import get_current_user
from crud.chat import add_workout_to_history, remove_from_favorites, add_to_favorites, \
    get_user_history, get_user_favorites_as_history
from crud.user_gym_logs import get_user_gym_logs
from crud.workout_generator import generate_workout
from model.user import User
from sqlalchemy.orm import Session

from schemas.chat import WorkoutGenerateResponse, ChatHistoryResponse

router = APIRouter(prefix="/workout", tags=["workout"])


class WorkoutRequest(BaseModel):
    goal: str


@router.post("/generate", response_model=WorkoutGenerateResponse)
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

        age = current_user.age
        gender = current_user.gender

        workout = generate_workout(prompt, user_logs, age, gender)

        return add_workout_to_history(db, current_user.id, req.goal, workout)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось создать тренировку: {str(e)}")


@router.get("/history")
def get_workout_history(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    return get_user_history(db, current_user.id)


@router.post("/favorite/{history_id}")
def add_favorite(
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return add_to_favorites(db, current_user.id, history_id)


@router.delete("/favorite/{history_id}")
def remove_favorite(
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = remove_from_favorites(db, current_user.id, history_id)
    if not result:
        raise HTTPException(status_code=404, detail="Не найдено в избранном")
    return {"detail": "Удалено из избранного"}


@router.get("/favorites", response_model=List[ChatHistoryResponse])
def get_favorites(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return get_user_favorites_as_history(db, current_user.id)
