"""
auth.py — User authentication for OpenSCAD Copilot.
SQLite-backed with bcrypt password hashing and JWT tokens.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    import jwt
except ImportError:
    import jose.jwt as jwt  # type: ignore

DB_PATH = Path(__file__).resolve().parent / "users.db"
SECRET_KEY = os.getenv("JWT_SECRET", "openscad-copilot-secret-change-in-prod")
ALGORITHM = "HS256"

_bearer = HTTPBearer(auto_error=False)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         TEXT PRIMARY KEY,
                email      TEXT UNIQUE NOT NULL,
                name       TEXT NOT NULL,
                password   TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT 'student',
                company    TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        con.commit()


def create_user(email: str, name: str, password: str, role: str = "student", company: str = "") -> dict:
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO users (id, email, name, password, role, company, created_at) VALUES (?,?,?,?,?,?,?)",
                (user_id, email.lower().strip(), name, hashed, role, company, created_at),
            )
            con.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email already registered")
    return {"id": user_id, "email": email, "name": name, "role": role, "company": company, "created_at": created_at}


def authenticate_user(email: str, password: str) -> dict:
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not bcrypt.checkpw(password.encode(), row["password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return dict(row)


def update_user(user_id: str, name: str, role: str, company: str) -> dict:
    with _conn() as con:
        con.execute(
            "UPDATE users SET name=?, role=?, company=? WHERE id=?",
            (name, role, company, user_id),
        )
        con.commit()
        row = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


def make_token(user_id: str) -> str:
    return jwt.encode({"sub": user_id}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub", "")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return dict(row)
