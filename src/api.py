"""FastAPI entrypoints for AI model inference, training, backtesting, and health checks.

Minimal token-based auth is included. For production, replace token store with a proper auth provider.
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Any
import logging

from .security_utils import verify_token, create_jwt_for_user
from .ml.model import predict, train_model, backtest_model
from .webhook_receiver import app as webhook_app, hmac_signature_required, WEBHOOK_HMAC_SECRET

logger = logging.getLogger('atwz.api')


app = FastAPI(title='AlgoTradingWithZerodha API')
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")



def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail='Invalid or expired token')
    return user

# Example token endpoint (for demo; replace with real user/password check)
from fastapi import status
from fastapi.security import OAuth2PasswordRequestForm
@app.post('/token')
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Replace with real user/password validation
    if form_data.username == "demo" and form_data.password == "demo":
        access_token = create_jwt_for_user(form_data.username)
        return {"access_token": access_token, "token_type": "bearer"}
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


@app.get('/health')
async def health() -> Dict[str, str]:
    return {'status': 'ok'}



@app.post('/predict')
async def api_predict(
    payload: Dict[str, Any],
    user: str = Depends(get_current_user),
    _hmac: None = hmac_signature_required(WEBHOOK_HMAC_SECRET)
) -> Dict[str, Any]:
    try:
        res = predict(payload)
        return {'ok': True, 'result': res}
    except Exception:
        logger.exception('predict failed')
        raise HTTPException(status_code=500, detail='predict failed')



@app.post('/train')
async def api_train(
    payload: Dict[str, Any],
    user: str = Depends(get_current_user),
    _hmac: None = hmac_signature_required(WEBHOOK_HMAC_SECRET)
) -> Dict[str, Any]:
    try:
        res = train_model(payload)
        return {'ok': True, 'model': res}
    except Exception:
        logger.exception('train failed')
        raise HTTPException(status_code=500, detail='train failed')



@app.post('/backtest')
async def api_backtest(
    payload: Dict[str, Any],
    user: str = Depends(get_current_user),
    _hmac: None = hmac_signature_required(WEBHOOK_HMAC_SECRET)
) -> Dict[str, Any]:
    try:
        res = backtest_model(payload)
        return {'ok': True, 'report': res}
    except Exception:
        logger.exception('backtest failed')
        raise HTTPException(status_code=500, detail='backtest failed')


# Mount webhook app under /webhook
app.mount('/webhook', webhook_app)
