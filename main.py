from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import router as auth_router
from api.user import router as user_router
from api.gymtool import router as gymtool_router
from api.recognize import router as recognize_router
from api.muscle import router as muscle_router
from api.user_gym_log import router as user_gym_log
from api.payment import router as payment
from api.workout_generator import router as workout

app = FastAPI(strict_slashes=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(gymtool_router)
app.include_router(user_router)
app.include_router(recognize_router)
app.include_router(muscle_router)
app.include_router(payment)
app.include_router(user_gym_log)
app.include_router(workout)
