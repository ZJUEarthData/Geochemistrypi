import os

import uvicorn

PORT = int(os.environ.get("BACKEND_PORT", 8000))

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=PORT, reload=True)
