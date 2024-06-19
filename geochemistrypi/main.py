from fastapi import FastAPI
import uvicorn
from auth.router import router
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Register routers
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
