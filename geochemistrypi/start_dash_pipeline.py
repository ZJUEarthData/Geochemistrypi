import os

import uvicorn
from auth import router as auth_router
from auth import sql_models as auth_models
from data_mining import router as data_mining_router
from database import engine
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

load_dotenv()
auth_models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(data_mining_router.router)
app.include_router(auth_router.router)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = allowed_origins.split(",") if allowed_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def read_root():
    return {"message": "Welcome to Geochemistry Pi!"}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Geochemistry Pi",
        version="0.2.1",
        description="Geochemistry π API",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run("start_dash_pipeline:app", host=host, port=port, reload=True)
