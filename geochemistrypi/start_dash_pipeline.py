import os

import uvicorn
from auth import router as auth_router
from auth import sql_models as auth_models
from data_mining import router as data_mining_router
from data_mining.dash_pipeline import dash_pipeline
from database import engine
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.openapi.utils import get_openapi

# Load environment variables
load_dotenv()

# Create tables in database
auth_models.Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI()

# Add routers
app.include_router(data_mining_router.router)
app.include_router(auth_router.router)

# Get the allowed origins from the environment
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to Geochemistry Pi!"}


# Add Dash app
dash_prefix = os.getenv("DASH_REQUESTS_PATHNAME_PREFIX", "/dash/")
dash_app = dash_pipeline(requests_pathname_prefix=dash_prefix)
app.mount(dash_prefix, WSGIMiddleware(dash_app.server))


api_docs = """
Geochemistry π is a Python framework for data-driven geochemistry discovery. It provides an extendable tool and one-stop shop for geochemical data analysis on tabular data.

### Authentication

This API uses OAuth2 with password flow, so you will need to create a user first by using the `/auth/register` endpoint. After creating a user,
 you can login using the `/auth/login` endpoint to get an access token. You can then use the access token to access the other endpoints.
   When clicking on the `Authorize` button, enter the following credentials:

   username: registered email address <br>
   password: registered password
"""


def custom_openapi():
    # Override OpenAPI schema to change field name in Swagger UI
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Geochemistry Pi",
        version="0.2.1",
        description=api_docs,
        routes=app.routes,
    )

    # openapi_schema["info"]["x-logo"] = {
    #     "url": "https://raw.githubusercontent.com/GeochemistryPi/geochemistrypi/master/docs/logo.png"
    # }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Override OpenAPI schema
app.openapi = custom_openapi

if __name__ == "__main__":
    backend_host = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port = int(os.getenv("BACKEND_PORT", 8000))
    # Run the app
    uvicorn.run("start_dash_pipeline:app", host=backend_host, port=backend_port, reload=True)
