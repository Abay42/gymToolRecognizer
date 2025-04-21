from typing import Optional
from pydantic import BaseModel, model_validator
from datetime import date


class GymLogRequest(BaseModel):
    date: date
    went: bool
    action: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_if_went(self) -> "GymLogRequest":
        if self.went and (not self.action or self.action.strip() == ""):
            raise ValueError("Action must be provided when 'went' is True")
        if not self.went and self.action:
            raise ValueError("Action must be empty when 'went' is False")
        return self
