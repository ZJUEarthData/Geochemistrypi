from datetime import timedelta
from typing import List

from database import get_db
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

# from geochemistrypi.database import get_db
from .constants import ACCESS_TOKEN_EXPIRE_MINUTES
from .dependencies import get_current_active_user
from .schemas import User, UserCreate
from .service import authenticate_user, create_new_user, get_user_by_email, get_user_by_id, get_user_by_username, get_users
from .utils import create_access_token

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=List[User])
async def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = get_users(db, skip=0, limit=100)
    return users


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@router.get("/{user_id}", response_model=User)
async def read_user(user_id: str, db: Session = Depends(get_db)):
    db_user = get_user_by_id(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.post("/register", response_model=User)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_new_user(db=db, user=user)


@router.put("/{username}")
async def update_user(username: str):
    return {"error": "Not implemented"}


@router.post("/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user_email = form_data.username
    user = authenticate_user(db, user_email, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        # data={"sub": user.username}, expires_delta=access_token_expires
        data={"sub": user.email},
        expires_delta=access_token_expires,
    )
    return {"message": "Successfully logged in", "userID": user.id, "access_token": access_token, "token_type": "bearer"}
