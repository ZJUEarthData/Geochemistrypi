import os

import pandas as pd
from fastapi import APIRouter, UploadFile

# Mock the database
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FAKE_DATABASE_DIR = os.path.join(CURRENT_DIR, "fake_database")

router = APIRouter(
    prefix="/data-mining",
    tags=["data-mining"],
    responses={404: {"description": "Not found"}},
)


@router.post("/upload")
async def upload_data(data: UploadFile):
    contents = await data.read()
    df = pd.read_excel(contents)
    # print(df)
    os.makedirs(FAKE_DATABASE_DIR, exist_ok=True)
    df.to_excel(os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx"))

    # Return the processed data to the client
    # processed_data = df.to_json(orient="records")
    # print(processed_data)
    # print(type(processed_data))
    # return processed_data

    return {"message": "Data uploaded successfully"}


@router.get("/get-raw-data")
async def get_data():
    if not os.path.exists(os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx")):
        return {"message": "No data available"}
    df = pd.read_excel(os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx"))
    raw_data = df.to_json(orient="records")
    print(raw_data)
    print(type(raw_data))
    return raw_data


# @router.post("/upload")
# async def upload_data(data: bytes = File(...)):
#     # print(data)
#     print(pd.read_excel(data))
#     # Process the uploaded file as needed
#     # e.g., save it to a specific location or perform data processing
#     return {'message': 'Data uploaded successfully'}
