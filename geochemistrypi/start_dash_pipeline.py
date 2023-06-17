import uvicorn
from auth import router as auth_router
from auth import sql_models as auth_models
from data_mining import router as data_mining_router
from data_mining.dash_pipeline import dash_pipeline
from database import engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.openapi.utils import get_openapi

auth_models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(data_mining_router.router)
app.include_router(auth_router.router)

origins = [
    "http://localhost:3001",
    "https://localhost:3001",
    "localhost:3001",
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


dash_app = dash_pipeline(requests_pathname_prefix="/dash/")
app.mount("/dash", WSGIMiddleware(dash_app.server))


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


app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run("start_dash_pipeline:app", host="0.0.0.0", port=8000, reload=True)
