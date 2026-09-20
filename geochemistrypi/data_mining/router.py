import io
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sklearn.model_selection import train_test_split

from ..auth.dependencies import get_current_active_user
from ..database import get_db
from .model.classification import ClassificationWorkflowBase
from .process.classify import ClassificationModelSelection
from .schemas import BasicDatasetInfo, ClassificationRunRequest
from .schemas import Dataset as DatasetOut
from .service import read_all_datasets, read_basic_datasets_info, read_dataset, remove_dataset, upload_dataset
from .sql_models import Dataset as DatasetModel

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FAKE_DATABASE_DIR = os.path.join(CURRENT_DIR, "fake_database")
DEFAULT_CLASSIFICATION_TEST_RATIO = 0.2


def _resolve_api_metric_average(metric_average: Optional[str], class_count: int) -> Optional[str]:
    if metric_average is not None:
        return metric_average
    if class_count > 2:
        return "weighted"
    return None


def _encode_target_for_split(y: pd.DataFrame, label_mapping: Optional[Dict[str, Any]]) -> Tuple[pd.Series, int]:
    if label_mapping:
        y_encoded, label_config = ClassificationWorkflowBase.customize_label(y, label_mapping=label_mapping, interactive=False, return_config=True)
        return y_encoded.iloc[:, 0], int(label_config["num_classes"])
    return y.iloc[:, 0], int(y.iloc[:, 0].nunique())


def _build_classification_split_parameters(y: pd.DataFrame, label_mapping: Optional[Dict[str, Any]] = None, default_test_ratio: float = DEFAULT_CLASSIFICATION_TEST_RATIO) -> Dict[str, Any]:
    stratify_source, class_count = _encode_target_for_split(y, label_mapping)
    class_counts = stratify_source.value_counts()
    if class_counts.empty or class_counts.min() < 2:
        return {"stratify_target": None, "test_size": default_test_ratio, "class_count": class_count}

    n_samples = len(stratify_source)
    min_test_samples = max(math.ceil(n_samples * default_test_ratio), class_count)
    max_test_samples = n_samples - class_count
    test_size = min(min_test_samples, max_test_samples)
    if test_size < class_count:
        return {"stratify_target": None, "test_size": default_test_ratio, "class_count": class_count}
    return {"stratify_target": stratify_source, "test_size": test_size, "class_count": class_count}


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
    db=Depends(get_db),
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

    label_mapping = request.label_mapping.dict(exclude_none=True) if request.label_mapping else None
    try:
        split_parameters = _build_classification_split_parameters(y, label_mapping=label_mapping)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_parameters["test_size"],
            random_state=42,
            stratify=split_parameters["stratify_target"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Failed to split classification data: {e}")

    name_all = pd.Series(df.index, name="Sample ID")
    name_train = pd.Series(X_train.index, name="Sample ID")
    name_test = pd.Series(X_test.index, name="Sample ID")

    transformer_config = {
        "interactive": False,
        "label_mapping": label_mapping,
    }
    metric_average = _resolve_api_metric_average(request.metric_average, split_parameters["class_count"])

    model_selector = ClassificationModelSelection(
        model_name=request.model_name,
        transformer_config=transformer_config,
        metric_average=metric_average,
    )

    try:
        model_selector.activate(X, y, X_train, X_test, y_train, y_test, name_train, name_test, name_all)
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
