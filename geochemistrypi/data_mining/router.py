import os

import pandas as pd
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from .service import read_all_datasets, read_basic_datasets_info, read_dataset, remove_dataset, upload_dataset

# import subprocess


# from fastapi.responses import PlainTextResponse


# Mock the database
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FAKE_DATABASE_DIR = os.path.join(CURRENT_DIR, "fake_database")

router = APIRouter(
    prefix="/data-mining",
    tags=["data-mining"],
    responses={404: {"description": "Not found"}},
)


@router.post("/{user_id}/upload-dataset")
async def post_dataset(user_id: int, dataset: UploadFile, db=Depends(get_db)):
    dataset_name = dataset.filename
    data = await dataset.read()
    df = pd.read_excel(data)
    json_df = df.to_json(orient="records")
    db_dataset = upload_dataset(db=db, user_id=user_id, dataset_name=dataset_name, json_dataset=json_df)

    os.makedirs(FAKE_DATABASE_DIR, exist_ok=True)
    df.to_excel(os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx"))

    # Return the processed data to the client
    # processed_data = df.to_json(orient="records")
    # print(processed_data)
    # print(type(processed_data))
    # return processed_data

    return {"message": "Data uploaded successfully", "uploaded_dataset": db_dataset}


@router.delete("/{user_id}/delete-dataset")
async def delete_dataset(user_id: int, dataset_id: int, db=Depends(get_db)):
    try:
        db_dataset = remove_dataset(db=db, user_id=user_id, dataset_id=dataset_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Dataset deleted successfully", "removed_dataset": db_dataset}


@router.get("/{user_id}/get-all-dataset")
async def get_all_datasets(user_id: int, db=Depends(get_db)):
    return read_all_datasets(db=db, user_id=user_id)


@router.get("/{user_id}/basic-datasets-info")
async def get_basic_datasets_info(user_id: int, db=Depends(get_db)):
    return read_basic_datasets_info(db=db, user_id=user_id)


@router.get("/{user_id}/get-dataset")
async def get_dataset(user_id: int, dataset_id: int, db=Depends(get_db)):
    try:
        db_dataset = read_dataset(db=db, user_id=user_id, dataset_id=dataset_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Dataset retrieved successfully", "dataset": db_dataset}


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


# @router.get("/execute-command")
# async def execute_command(command: str):
#     process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
#     output, _ = process.communicate()
#     return PlainTextResponse(output.decode())
