import os

from dotenv import load_dotenv

load_dotenv()

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "5277ee28edc76a2810312a90ab8ab53a6d9fa16be8a4db446ad93e66b2444d02"
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
