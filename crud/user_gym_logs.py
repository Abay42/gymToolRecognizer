from sqlalchemy.orm import Session
from model.user_gym_log import UserGymLog
from datetime import date


def set_gym_flag(db: Session, user_id: int, log_date: date, went: bool):
    log = db.query(UserGymLog).filter_by(user_id=user_id, date=log_date).first()
    if log:
        log.went_to_gym = went
    else:
        log = UserGymLog(user_id=user_id, date=log_date, went_to_gym=went)
        db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_user_gym_logs(db: Session, user_id: int):
    return db.query(UserGymLog).filter_by(user_id=user_id).all()
