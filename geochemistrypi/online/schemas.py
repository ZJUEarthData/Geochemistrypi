"""Response models for the Online API."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class InputColumnItem(BaseModel):
    name: str
    label: str
    description: str
    data_type: str
    unit: str
    example: float | int | str
    required: bool = True
    minimum: float | None = None
    exclusive_minimum: bool = False


class MethodCatalogItem(BaseModel):
    name: str
    description: str
    elements: list[str]
    status: Literal["verified", "testing"]
    status_message: str
    formula: str | None = None
    input_columns: list[InputColumnItem] = Field(default_factory=list)
    input_notes: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)


class TaskCatalogItem(BaseModel):
    name: str
    available: bool
    methods: list[MethodCatalogItem] = Field(default_factory=list)
    error: str | None = None


class CatalogResponse(BaseModel):
    tasks: list[TaskCatalogItem]


class ArtifactResponse(BaseModel):
    name: str
    download_url: str
    size_bytes: int


class RunResponse(BaseModel):
    job_id: str
    status: str
    message: str
    artifacts: list[ArtifactResponse]
