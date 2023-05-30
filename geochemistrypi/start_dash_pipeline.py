import uvicorn
from data_mining import router as data_mining_router
from data_mining.dash_pipeline import dash_pipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware

app = FastAPI()

app.include_router(data_mining_router.router)

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


dash_app = dash_pipeline(requests_pathname_prefix="/dash/")
app.mount("/dash", WSGIMiddleware(dash_app.server))


if __name__ == "__main__":
    uvicorn.run("start_dash_pipeline:app", host="0.0.0.0", port=8000, reload=True)
