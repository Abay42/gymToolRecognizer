from sqlalchemy.orm import Session
from model.user_gym_log import UserGymLog

from schemas.user_gym_log import GymLogRequest


def set_gym_flag(db: Session, current_user_id: int, data: GymLogRequest):
    log = db.query(UserGymLog).filter_by(user_id=current_user_id, date=data.date).first()

    if log:
        log.went_to_gym = data.went
        log.action = data.action
    else:
        log = UserGymLog(
            user_id=current_user_id,
            date=data.date,
            went_to_gym=data.went,
            action=data.action
        )
        db.add(log)

    db.commit()
    db.refresh(log)
    return log

def get_user_gym_logs(db: Session, user_id: int):
    return db.query(UserGymLog).filter_by(user_id=user_id).all()
