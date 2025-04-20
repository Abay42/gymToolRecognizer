from pydantic import BaseModel


class AddMuscleRequest(BaseModel):
    muscle_id: int


class AddLinkRequest(BaseModel):
    url: str
