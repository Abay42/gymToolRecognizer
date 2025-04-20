from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
import time

from starlette.responses import StreamingResponse

from core.database import get_db, upload_image_to_minio
from model.muscle import Muscle
import requests
import os
from io import BytesIO

RAPIDAPI_KEY = os.getenv("RAPID_KEY")
RAPIDAPI_HOST = os.getenv("RAPID_HOST")
multiColorUrl = os.getenv("DUAL_COLOR")
singleColorUrl = os.getenv("SINGLE_COLOR")
allMusclesUrl = os.getenv("ALL_MUSCLES")

router = APIRouter(prefix="/muscles", tags=["muscles"])


@router.get("/multImg")
def generate_multiple_muscle_image(primary_muscles: str, secondary_muscles: str = ""):
    querystring = {
        "primaryColor": "240,100,80",
        "secondaryColor": "200,100,80",
        "primaryMuscleGroups": primary_muscles,
        "secondaryMuscleGroups": secondary_muscles,
        "transparentBackground": "0"
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }

    response = requests.get(multiColorUrl, headers=headers, params=querystring)

    print("Status Code:", response.status_code)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch muscle image")

    return StreamingResponse(BytesIO(response.content), media_type="image/png")


@router.get("/img")
def generate_muscle_image(muscle_groups: str):
    querystring = {
        "muscleGroups": muscle_groups,
        "color": "200,100,80",
        "transparentBackground": "0"
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    response = requests.get(singleColorUrl, headers=headers, params=querystring)
    print("Status Code:", response.status_code)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch muscle image")

    return StreamingResponse(BytesIO(response.content), media_type="image/png")


@router.get("/all")
def all_muscles():
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    response = requests.get(allMusclesUrl, headers=headers)

    return response.json()


@router.post("/upload/")
def upload_muscle_image(
        name: str,
        file: UploadFile = File(...),
        db: Session = Depends(get_db)):
    file_bytes = file.file.read()

    filename = f"{name}_{int(time.time())}.jpg"

    image_url = upload_image_to_minio(file_bytes, filename)

    muscle = db.query(Muscle).filter(Muscle.name == name).first()
    if not muscle:
        muscle = Muscle(name=name)
        db.add(muscle)

    muscle.image_url = image_url
    db.commit()
    db.refresh(muscle)

    return {
        "id": muscle.id,
        "name": muscle.name,
        "image_url": muscle.image_url
    }
