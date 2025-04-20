from sqlalchemy import Table, Column, Integer, ForeignKey
from core.database import Base

gymtool_muscle_association = Table(
    "gymtool_muscle_association",
    Base.metadata,
    Column("gymtool_id", Integer, ForeignKey("gymtools.id"), primary_key=True),
    Column("muscle_id", Integer, ForeignKey("muscles.id"), primary_key=True)
)
