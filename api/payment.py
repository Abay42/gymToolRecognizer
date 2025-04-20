import stripe
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.payment import PaymentRequest
from model.user import User

router = APIRouter(prefix="/payment", tags=["payment"])


@router.post("/create-payment-intent/")
def create_payment_intent(
    payment: PaymentRequest,
    db: Session = Depends(get_db)
):
    try:
        intent = stripe.PaymentIntent.create(
            amount=int(payment.amount * 100),
            currency="usd",
            payment_method_types=["card"],
            metadata={"user_id": str(payment.user_id)},  # Add user ID here
        )

        # OPTIONAL: You might want to wait for actual payment confirmation using Stripe webhooks
        # For now, we assume payment is successful here
        user = db.query(User).filter(User.id == payment.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.is_sub = True
        db.commit()
        db.refresh(user)

        return {"client_secret": intent.client_secret, "message": "Payment initiated, user marked as subscribed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
