# geochemistrypi/data_mining/router.py
import os
import io
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sklearn.model_selection import train_test_split

from auth.dependencies import get_current_active_user
from database import get_db

from .service import (
    read_all_datasets,
    read_basic_datasets_info,
    read_dataset,
    remove_dataset,
    upload_dataset,
)
from .schemas import Dataset as DatasetOut, BasicDatasetInfo, ClassificationRunRequest
from .sql_models import Dataset as DatasetModel
from .process.classify import ClassificationModelSelection 

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FAKE_DATABASE_DIR = os.path.join(CURRENT_DIR, "fake_database")

router = APIRouter(
    prefix="/data-mining",
    tags=["data-mining"],
    responses={404: {"description": "Not found"}},
)


@router.post("/upload-dataset", response_model=BasicDatasetInfo, status_code=status.HTTP_201_CREATED)
async def post_dataset(
    dataset: UploadFile = File(...),
    current_user=Depends(get_current_active_user),
    db=Depends(get_db),
):
    dataset_name = dataset.filename or "uploaded"
    raw = await dataset.read()
    ext = os.path.splitext(dataset_name)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(raw))
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv/.xlsx/.xls")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    json_df = df.to_json(orient="records", force_ascii=False)
    db_dataset = upload_dataset(
        db=db,
        user_id=current_user.id,
        dataset_name=dataset_name,
        json_dataset=json_df,
    )

    os.makedirs(FAKE_DATABASE_DIR, exist_ok=True)
    if ext == ".csv":
        df.to_csv(os.path.join(FAKE_DATABASE_DIR, "user_data.csv"), index=False)
    else:
        df.to_excel(os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx"), index=False)

    return db_dataset


@router.post("/run-classification", tags=["data-mining"])
async def run_classification_pipeline(
    request: ClassificationRunRequest,
    current_user=Depends(get_current_active_user),
    db=Depends(get_db)
):
    db_dataset = read_dataset(db=db, user_id=current_user.id, dataset_id=request.dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        df = pd.read_json(io.StringIO(db_dataset.json_data), orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset data: {e}")
        
    if request.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found in dataset")
        
    X = df.drop(columns=[request.target_column])
    y = df[[request.target_column]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    name_all = pd.Series(df.index, name="Sample ID")
    name_train = pd.Series(X_train.index, name="Sample ID")
    name_test = pd.Series(X_test.index, name="Sample ID")
    
    transformer_config = {
        "interactive": False,
        "label_mapping": request.label_mapping.dict(exclude_none=True) if request.label_mapping else None
    }
    
    model_selector = ClassificationModelSelection(
        model_name=request.model_name,
        transformer_config=transformer_config
    )
    
    try:
        model_selector.activate(
            X, y, 
            X_train, X_test, 
            y_train, y_test, 
            name_train, name_test, name_all
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
        
    return {"message": "Classification pipeline executed successfully!", "status": "success"}

@router.delete("/delete-dataset", response_model=DatasetOut)
async def delete_dataset(
    dataset_id: int,
    current_user=Depends(get_current_active_user),
    db=Depends(get_db),
):
    obj = read_dataset(db=db, user_id=current_user.id, dataset_id=dataset_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Dataset not found")
    removed = remove_dataset(db=db, user_id=current_user.id, dataset_id=dataset_id)
    return removed


@router.get("/get-all-dataset", response_model=List[DatasetOut])
async def get_all_datasets(
    current_user=Depends(get_current_active_user),
    db=Depends(get_db),
):
    return read_all_datasets(db=db, user_id=current_user.id)


@router.get("/basic-datasets-info", response_model=List[BasicDatasetInfo])
async def get_basic_datasets_info(
    current_user=Depends(get_current_active_user),
    db=Depends(get_db),
):
    return read_basic_datasets_info(db=db, user_id=current_user.id)


@router.get("/get-dataset", response_model=DatasetOut)
async def get_dataset(
    dataset_id: int,
    current_user=Depends(get_current_active_user),
    db=Depends(get_db),
):
    obj = read_dataset(db=db, user_id=current_user.id, dataset_id=dataset_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return obj
@router.get("/get-all-dataset-open", response_model=List[DatasetOut], tags=["data-mining"])
def get_all_datasets_open(db=Depends(get_db)):
    return db.query(DatasetModel).order_by(DatasetModel.sequence).all()