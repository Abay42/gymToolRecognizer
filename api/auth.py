import logging
from fastapi import Depends, HTTPException, status, APIRouter, Request, BackgroundTasks
from jose import JWTError
from sqlalchemy.orm import Session
from datetime import timedelta
from core.database import get_db
from core.email import send_verification_email, send_welcome_email
from core.oauth import oauth_service
from core.security import create_access_token, generate_verification_token
from core.validator import is_valid_email, is_valid_password
from crud.user import create_user, authenticate_user, get_user_by_email, activate_user
from schemas.user import UserCreate, UserLogin
from model.user import User
from core import validator
import asyncio
from core.security import verify_verification_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(user: UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    logger.info(f"Registration attempt for email: {user.email}")
    if not is_valid_email(user.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    existing_user = db.query(User).filter(User.email == user.email).first()

    if existing_user:
        logger.warning(f"Registration failed: Email {user.email} already registered")
        raise HTTPException(status_code=400, detail="Email already registered")

    if not is_valid_password(user.password):
        raise HTTPException(status_code=400, detail="Password does not meet complexity requirements")

    new_user = create_user(db, user, is_active=False)
    verification_token = generate_verification_token(new_user.email)

    def send_verification_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(send_verification_email(
                email=new_user.email,
                username=new_user.username or new_user.email.split('@')[0],
                verification_token=verification_token
            ))
        finally:
            loop.close()

    background_tasks.add_task(send_verification_task)

    logger.info(f"User {user.email} registered successfully with ID {new_user.id}. Verification email sent.")
    return {
        "msg": "User created successfully. Please check your email to verify your account.",
        "user_id": new_user.id
    }


@router.get("/verify-email")
def verify_email(token: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        email = verify_verification_token(token)

        if not email:
            raise HTTPException(status_code=400, detail="Invalid or expired verification token")

        user = get_user_by_email(db, email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.is_active:
            return {"msg": "Email already verified"}

        activated_user = activate_user(db, user.id)

        if activated_user:
            def send_welcome_task():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(send_welcome_email(
                        email=user.email,
                        username=user.username or user.email.split('@')[0]
                    ))
                finally:
                    loop.close()

            background_tasks.add_task(send_welcome_task)

        logger.info(f"Email verified successfully for user: {email}")
        return {"msg": "Email verified successfully. Your account is now active. Welcome email sent!"}

    except HTTPException:
        raise
    except JWTError:
        logger.error("JWT token validation failed")
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
    except Exception as e:
        logger.error(f"Email verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during verification")


@router.post("/resend-verification")
def resend_verification_email(email: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_active:
        raise HTTPException(status_code=400, detail="Email already verified")

    verification_token = generate_verification_token(user.email)

    background_tasks.add_task(
        send_verification_email,
        email=user.email,
        username=user.username if hasattr(user, 'username') else user.email.split('@')[0],
        verification_token=verification_token
    )

    logger.info(f"Verification email resent to: {email}")
    return {"msg": "Verification email sent successfully"}


@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    logger.info(f"Login attempt for email: {user.email}")
    if not is_valid_email(user.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    user_obj = authenticate_user(db, user.email, user.password)
    if not user_obj:
        logger.warning(f"Login failed for email: {user.email} - Invalid credentials")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user_obj.email}, expires_delta=timedelta(minutes=60))
    logger.info(f"User {user.email} logged in successfully, token issued")
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/providers")
def get_oauth_providers():
    providers = oauth_service.get_available_providers()
    return {"providers": providers}


@router.get("/{provider}/login")
def oauth_login(provider: str):
    supported_providers = ["google", "microsoft", "discord"]

    if provider not in supported_providers:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider}' not supported. Available providers: {', '.join(supported_providers)}"
        )

    try:
        authorization_url = oauth_service.initiate_oauth_flow(provider)
        return {"authorization_url": authorization_url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate {provider} OAuth: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate OAuth login")


@router.get("/{provider}/callback")
async def oauth_callback(provider: str, request: Request, db: Session = Depends(get_db)):
    supported_providers = ["google", "microsoft", "discord"]

    if provider not in supported_providers:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider}' not supported. Available providers: {', '.join(supported_providers)}"
        )

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    error_description = request.query_params.get("error_description")

    if error:
        logger.error(f"{provider.title()} OAuth error: {error} - {error_description}")
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing authorization code or state")

    try:
        access_token, user_data = await oauth_service.handle_oauth_callback(
            provider, code, state, db
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{provider.title()} OAuth callback failed: {str(e)}")
        raise HTTPException(status_code=500, detail="OAuth authentication failed")
