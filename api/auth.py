import logging
from fastapi import Depends, HTTPException, status, APIRouter
from sqlalchemy.orm import Session
from datetime import timedelta
from core.database import get_db
from core.security import create_access_token
from core.validator import is_valid_email, is_valid_password
from crud.user import create_user, authenticate_user
from schemas.user import UserCreate, UserLogin
from model.user import User
from core import validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"Registration attempt for email: {user.email}")
    #if not is_valid_email(user.email):
        #raise HTTPException(status_code=400, detail="Invalid email format")

    existing_user = db.query(User).filter(User.email == user.email).first()

    if existing_user:
        logger.warning(f"Registration failed: Email {user.email} already registered")
        raise HTTPException(status_code=400, detail="Email already registered")

    #if not is_valid_password(user.password):
        #raise HTTPException(status_code=400, detail="Password does not meet complexity requirements")

    new_user = create_user(db, user)
    logger.info(f"User {user.email} registered successfully with ID {new_user.id}")
    return {"msg": "User created", "user_id": new_user.id}


@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    logger.info(f"Login attempt for email: {user.email}")
    # if not is_valid_email(user.email):
    #     raise HTTPException(status_code=400, detail="Invalid email format")

    user_obj = authenticate_user(db, user.email, user.password)
    if not user_obj:
        logger.warning(f"Login failed for email: {user.email} - Invalid credentials")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user_obj.email}, expires_delta=timedelta(minutes=60))
    logger.info(f"User {user.email} logged in successfully, token issued")
    return {"access_token": access_token, "token_type": "bearer"}
