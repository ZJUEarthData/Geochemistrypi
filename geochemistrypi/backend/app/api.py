from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import data

app = FastAPI()

app.include_router(data.router)

origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    "localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to Geochemistry Pi!"}
