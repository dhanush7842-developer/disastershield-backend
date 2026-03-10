"""
DisasterShield AI — Authentication
JWT-based auth with two built-in users (swap for DB in production).
"""

import os
import hashlib
import hmac
import json
import base64
import time
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

SECRET_KEY   = os.getenv("JWT_SECRET_KEY", "disastershield-dev-secret-change-in-prod")
ALGORITHM    = "HS256"
EXPIRE_SECS  = int(os.getenv("JWT_EXPIRE_SECS", "3600"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

# ── Built-in demo users (replace with DB in production) ───────────────────
# Passwords are sha256 hashed for simplicity in demo; use bcrypt in prod
def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

DEMO_USERS = {
    "analyst": {
        "hashed_pw": _sha256("analyst123"),
        "scopes":    ["user"],
        "full_name": "Disaster Analyst",
    },
    "admin": {
        "hashed_pw": _sha256("admin123"),
        "scopes":    ["user", "admin"],
        "full_name": "System Administrator",
    },
}


# ── Minimal JWT implementation (no external lib needed) ───────────────────
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))


def _create_token(payload: dict) -> str:
    header  = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body    = _b64url_encode(json.dumps(payload).encode())
    sig_input = f"{header}.{body}".encode()
    sig     = hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(sig)}"


def _verify_token(token: str) -> dict:
    try:
        header, body, sig = token.split(".")
        sig_input = f"{header}.{body}".encode()
        expected  = hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(sig), expected):
            raise ValueError("Invalid signature")
        payload = json.loads(_b64url_decode(body))
        if payload.get("exp", 0) < time.time():
            raise ValueError("Token expired")
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_access_token(username: str, scopes: list) -> str:
    return _create_token({
        "sub":    username,
        "scopes": scopes,
        "iat":    int(time.time()),
        "exp":    int(time.time()) + EXPIRE_SECS,
    })


# ── FastAPI dependencies ──────────────────────────────────────────────────
def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    """Allow unauthenticated access in dev mode; enforce in prod."""
    dev_mode = os.getenv("AUTH_REQUIRED", "false").lower() == "false"
    if dev_mode and not token:
        # Return a default dev user when AUTH_REQUIRED=false
        return {"sub": "dev_user", "scopes": ["user", "admin"]}
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _verify_token(token)


def require_admin(user: dict = Depends(get_current_user)):
    if "admin" not in user.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin scope required",
        )
    return user


# ── Auth router ───────────────────────────────────────────────────────────
from fastapi import APIRouter
from .schemas import TokenResponse

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/token", response_model=TokenResponse, summary="Get JWT token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = DEMO_USERS.get(form.username)
    if not user or user["hashed_pw"] != _sha256(form.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(form.username, user["scopes"])
    return TokenResponse(access_token=token, expires_in=EXPIRE_SECS)


@auth_router.get("/me", summary="Get current user info")
async def me(user: dict = Depends(get_current_user)):
    info = DEMO_USERS.get(user.get("sub"), {})
    return {
        "username":  user.get("sub"),
        "scopes":    user.get("scopes", []),
        "full_name": info.get("full_name", "Unknown"),
    }
