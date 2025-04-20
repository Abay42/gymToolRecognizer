from pydantic import BaseModel


class PaymentRequest(BaseModel):
    amount: float
    user_id: int
