from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from core.database import Base
from model.association import gymtool_muscle_association
from model.muscle import Muscle
from model.gymtoolLink import GymToolLink

class GymTool(Base):
    __tablename__ = "gymtools"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    alternative = Column(String, nullable=True)

    muscles = relationship(
        "Muscle",
        secondary=gymtool_muscle_association,
        back_populates="gym_tools"
    )
    links = relationship("GymToolLink", back_populates="gym_tool", cascade="all, delete-orphan")

