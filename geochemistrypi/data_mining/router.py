# geochemistrypi/data_mining/router.py
import os
import io
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from auth.dependencies import get_current_active_user
from database import get_db

from .service import (
    read_all_datasets,
    read_basic_datasets_info,
    read_dataset,
    remove_dataset,
    upload_dataset,
)
from .schemas import Dataset as DatasetOut, BasicDatasetInfo
from .sql_models import Dataset as DatasetModel

# 本地留档（可选）
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

    # 同时支持 CSV / Excel；把解析错误转成 400，避免 500
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(raw))
        elif ext in (".xlsx", ".xls"):
            # 读 .xlsx 需要 openpyxl
            df = pd.read_excel(io.BytesIO(raw))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv/.xlsx/.xls")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    # 入库
    json_df = df.to_json(orient="records", force_ascii=False)
    db_dataset = upload_dataset(
        db=db,
        user_id=current_user.id,
        dataset_name=dataset_name,
        json_dataset=json_df,
    )

    # 可选：在本地留一份，便于 Dash 或调试
    os.makedirs(FAKE_DATABASE_DIR, exist_ok=True)
    if ext == ".csv":
        df.to_csv(os.path.join(FAKE_DATABASE_DIR, "user_data.csv"), index=False)
    else:
        df.to_excel(os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx"), index=False)

    # 直接返回 ORM；FastAPI 会按 BasicDatasetInfo 裁剪为 {id,name,sequence}
    return db_dataset


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


# ====== 调试接口：用于定位问题（用完可删除） ======

# 不做权限校验、不按 user 过滤，直接返回全表
@router.get("/get-all-dataset-open", response_model=List[DatasetOut], tags=["data-mining"])
def get_all_datasets_open(db=Depends(get_db)):
    return db.query(DatasetModel).order_by(DatasetModel.sequence).all()

# 需要登录，但不按 user 过滤
@router.get("/get-all-dataset-nofilter", response_model=List[DatasetOut], tags=["data-mining"])
def get_all_datasets_nofilter(current_user=Depends(get_current_active_user), db=Depends(get_db)):
    return db.query(DatasetModel).order_by(DatasetModel.sequence).all()
