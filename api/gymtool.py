from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.orm import Session
import logging

from core.database import get_db
from crud.gymtool import get_all_gymtools, get_gymtool_by_name, add_muscle_to_gymtool, add_link_to_gymtool
from schemas.gymtool import AddMuscleRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tool", tags=["gymtools"])


@router.get("/all")
def read_all_tools(db: Session = Depends(get_db)):
    return get_all_gymtools(db)


@router.get("/profile")
def read_tool(name: str, db: Session = Depends(get_db)):
    tool = get_gymtool_by_name(db, name)
    if not tool:
        raise HTTPException(status_code=404, detail="Gym tool not found")
    return tool


@router.post("/add-muscle")
def add_muscle(
    data: AddMuscleRequest,
    db: Session = Depends(get_db)
):
    gym_tool = add_muscle_to_gymtool(db, data)
    if not gym_tool:
        raise HTTPException(status_code=400, detail="Failed to add muscle to gym tool")
    return {"message": f"Muscle '{data.muscle_name}' added successfully to '{data.gymtool_name}'"}


@router.post("/add-link")
def add_link(
    gymtool_name: str = Query(..., title="Gym Tool Name"),
    url: str = Query(..., title="Link URL"),
    db: Session = Depends(get_db)
):
    new_link = add_link_to_gymtool(db, gymtool_name, url)
    if not new_link:
        raise HTTPException(status_code=400, detail="Failed to add link to gym tool")
    return {"message": f"Link '{url}' added successfully to '{gymtool_name}'"}