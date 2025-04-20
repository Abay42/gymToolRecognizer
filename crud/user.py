from sqlalchemy.orm import Session
from model.user import User
from schemas.user import UserCreate, UserProfile
from core.security import hash_password, verify_password
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_user(db: Session, user_data: UserCreate):
    hashed_pw = hash_password(user_data.password)
    new_user = User(email=user_data.email, password=hashed_pw, gender=user_data.gender, age=user_data.age,
                    username=user_data.username, is_sub=False)
    logger.info(f"User is : {new_user}")
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"User created: {new_user.email}")
    return new_user


def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if user and verify_password(password, user.password):
        logger.info(f"User authenticated: {email}")
        return user
    logger.warning(f"Authentication failed for: {email}")
    return None


def get_profile(db: Session, email: str):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.warning(f"Profile not found for email: {email}")
        return None
    logger.info(f"Profile retrieved for email: {email}")
    return user


def update_profile(db: Session, email: str, profile_data: UserProfile):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.warning(f"Update failed - User not found: {email}")
        return None

    user.username = profile_data.username or user.username
    user.age = profile_data.age or user.age
    db.commit()
    db.refresh(user)

    logger.info(f"Profile updated for email: {email}")
    return user


def delete_user(db: Session, email: str):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.warning(f"Delete failed - User not found: {email}")
        return None

    db.delete(user)
    db.commit()
    logger.info(f"User deleted: {email}")
    return {"message": "User deleted successfully"}
