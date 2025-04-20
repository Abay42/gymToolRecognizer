from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, index=True)
    age = Column(Integer, nullable=False)
    password = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_sub = Column(Boolean, default=False)
    sub_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    gym_logs = relationship("UserGymLog", back_populates="user", cascade="all, delete-orphan")

