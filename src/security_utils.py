"""Security helpers: encryption and token utilities.

This module demonstrates Fernet symmetric encryption for storing secrets and
an extremely lightweight token verification function. Replace with a robust
auth provider and secret management (Vault/KMS) for production.
"""
from cryptography.fernet import Fernet
import os
import logging
from typing import Optional

logger = logging.getLogger('atwz.security_utils')


def generate_key() -> bytes:
    key = Fernet.generate_key()
    return key


def load_or_create_key(path: str = '.fernet.key') -> bytes:
    if os.path.exists(path):
        return open(path, 'rb').read()
    key = generate_key()
    with open(path, 'wb') as f:
        f.write(key)
    return key


def encrypt_secret(value: str, key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(value.encode('utf-8'))


def decrypt_secret(token: bytes, key: bytes) -> str:
    f = Fernet(key)
    return f.decrypt(token).decode('utf-8')



# JWT utilities
import jwt
import datetime
from typing import Dict, Any

JWT_SECRET = os.getenv('JWT_SECRET', 'demo_jwt_secret')
JWT_ALGORITHM = 'HS256'
JWT_EXP_MINUTES = int(os.getenv('JWT_EXP_MINUTES', '60'))

def create_jwt_for_user(user: str, extra: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        'sub': user,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXP_MINUTES),
        'iat': datetime.datetime.utcnow(),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get('sub')
    except jwt.ExpiredSignatureError:
        logger.warning('JWT expired')
        return None
    except jwt.InvalidTokenError:
        logger.warning('Invalid JWT')
        return None
