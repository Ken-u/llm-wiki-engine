"""JWT authentication dependency."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import User, UserApiToken
from app.config import get_config
from app.database import get_db

_bearer = HTTPBearer()

ALGORITHM = "HS256"


def hash_password(raw: str) -> str:
    return bcrypt.hashpw(raw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(raw: str, hashed: str) -> bool:
    return bcrypt.checkpw(raw.encode("utf-8"), hashed.encode("utf-8"))


def hash_api_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_api_token() -> str:
    return f"lwu_{secrets.token_urlsafe(32)}"


def create_access_token(user_id: int) -> str:
    cfg = get_config().auth
    expire = datetime.now(timezone.utc) + timedelta(hours=cfg.jwt_expire_hours)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        cfg.jwt_secret,
        algorithm=ALGORITHM,
    )


def _decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, get_config().auth.jwt_secret, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc
    return user_id


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    raw_token = creds.credentials
    try:
        user_id = _decode_token(raw_token)
    except HTTPException:
        api_token = (
            await db.execute(
                select(UserApiToken).where(UserApiToken.token_hash == hash_api_token(raw_token))
            )
        ).scalar_one_or_none()
        if api_token is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
        user_id = api_token.user_id
        api_token.last_used_at = datetime.now(timezone.utc)
        await db.commit()

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user
