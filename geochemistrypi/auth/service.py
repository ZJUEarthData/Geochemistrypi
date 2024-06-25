from sqlalchemy.orm import Session

from .schemas import UserCreate
from .sql_models import User
from .utils import get_password_hash, verify_password
import httpx
from .schemas import TokenResponse, ValidationResponse, UserInfoResponse
from .config import loginURL



def get_user_by_id(db: Session, user_id: str):
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()


def create_new_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db, email: str, password: str):
    # user = get_user_by_username(db, username)
    user = get_user_by_email(db, email=email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


async def get_token(code: str) -> str:
    params = {
        "appcode": loginURL["appCode"],
        "code": code,
        "secret": loginURL["secretCode"],
    }
    queryString = "&".join([f"{key}={value}" for key, value in params.items()])
    urlWithParams = f"{loginURL['tokenChange']}?{queryString}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(urlWithParams)
        response.raise_for_status() 
        response_data = response.json()
        token_response = TokenResponse(**response_data)
        return token_response.accessToken

async def validate_token(token: str) -> bool:
    params = {"token": token}
    queryString = "&".join([f"{key}={value}" for key, value in params.items()])
    urlWithParams = f"{loginURL['validate']}?{queryString}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(urlWithParams)
        response.raise_for_status()  # Raise an error for bad responses
        response_data = response.json()
        validation_response = ValidationResponse(**response_data)
        return validation_response.data

async def get_user_info(token: str):
    params = {
        "appcode": loginURL["appCode"],
        "token": token,
        "secret": loginURL["secretCode"],
    }
    queryString = "&".join([f"{key}={value}" for key, value in params.items()])
    urlWithParams = f"{loginURL['infoChange']}?{queryString}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(urlWithParams)
        response.raise_for_status()  # Raise an error for bad responses
        response_data = response.json()
        user_info_response = UserInfoResponse(**response_data)
        return user_info_response