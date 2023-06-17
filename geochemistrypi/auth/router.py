from datetime import timedelta

from database import get_db
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .constants import ACCESS_TOKEN_EXPIRE_MINUTES
from .dependencies import get_current_active_user
from .schemas import User, UserCreate
from .service import authenticate_user, create_new_user, get_user_by_email, get_user_by_username
from .utils import create_access_token

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)


# @router.get("/", response_model=List[User])
# async def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     users = get_users(db, skip=0, limit=100)
#     return users


@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    user_info = {"username": current_user.username, "email": current_user.email, "upload_count": current_user.upload_count}
    return user_info


# @router.get("/{user_id}", response_model=User)
# async def read_user(user_id: int, db: Session = Depends(get_db)):
#     db_user = get_user_by_id(db, user_id)
#     if db_user is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return db_user


@router.put("/{username}")
async def update_user(username: str):
    return {"error": "Not implemented"}


@router.post("/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm is a class that has username and password attributes
    # The user is only allowed to login with email, so we need to change the username to email
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
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register")
async def register(email: str, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    db_user = get_user_by_username(db, username=form_data.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    db_user = get_user_by_email(db, email=email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = UserCreate(email=email, username=form_data.username, password=form_data.password)
    create_new_user(db=db, user=user)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        # data={"sub": user.username}, expires_delta=access_token_expires
        data={"sub": user.email},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}
