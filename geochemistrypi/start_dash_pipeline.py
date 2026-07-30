import os
import sys
from importlib import import_module

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

if not __package__:
    package_parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    package_name = "geochemistrypi"
else:
    package_name = __package__

auth_router = import_module(f"{package_name}.auth.router")
auth_models = import_module(f"{package_name}.auth.sql_models")
data_mining_router = import_module(f"{package_name}.data_mining.router")
get_engine = import_module(f"{package_name}.database").get_engine

load_dotenv()

app = FastAPI()
app.include_router(data_mining_router.router)
app.include_router(auth_router.router)


@app.on_event("startup")
def create_database_tables() -> None:
    """Initialize tables on API startup rather than module import."""
    auth_models.Base.metadata.create_all(bind=get_engine())


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
    uvicorn.run(f"{package_name}.start_dash_pipeline:app", host=host, port=port, reload=True)
