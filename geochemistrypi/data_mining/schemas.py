from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

try:
    from pydantic import ConfigDict

    _P2 = True
except Exception:
    _P2 = False


class Dataset(BaseModel):
    id: int
    name: Optional[str] = None
    json_data: Optional[str] = None
    sequence: Optional[int] = None
    user_id: Optional[int] = None

    if _P2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True


class Diagram(BaseModel):
    id: int
    name: Optional[str] = None
    image: Optional[bytes] = None
    dataset_id: Optional[int] = None

    if _P2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True


class BasicDatasetInfo(BaseModel):
    id: int
    name: str
    sequence: int


class LabelMappingConfig(BaseModel):
    """用于接收前端传来的多分类映射规则"""

    type: str
    bins: Optional[List[float]] = None
    labels: Optional[List[str]] = None
    num_classes: Optional[int] = None
    mapping: Optional[Dict[str, str]] = None


class ClassificationRunRequest(BaseModel):
    """用于接收前端触发机器学习训练的请求体"""

    dataset_id: int
    target_column: str
    model_name: str
    label_mapping: Optional[LabelMappingConfig] = None
    metric_average: Optional[Literal["micro", "macro", "weighted"]] = None
