from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from core.database import Base
from model.association import gymtool_muscle_association


class Muscle(Base):
    __tablename__ = "muscles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    image_url = Column(String, nullable=True)

    gym_tools = relationship(
        "GymTool",
        secondary=gymtool_muscle_association,
        back_populates="muscles"
    )
