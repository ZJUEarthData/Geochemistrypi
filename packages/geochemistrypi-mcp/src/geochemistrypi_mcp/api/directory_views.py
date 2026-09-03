"""Bounded public views over complete dataset and MLflow directory records."""

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar

from pydantic import Field

from .schemas import DatasetCatalogEntry, ExperimentRunSummary, ExperimentSummary, GetExperimentResponse, ListDatasetsResponse, ListExperimentsResponse, StrictModel

MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES = 64 * 1024
# Full detail is explicitly opt-in and lossless.  The higher bound can carry
# any single record accepted by the existing 2 MB MLflow bridge while still
# preventing an unbounded structured response.
MAX_FULL_DIRECTORY_RESPONSE_JSON_BYTES = 2_100_000
DEFAULT_COMPACT_DIRECTORY_LIMIT = 16
DEFAULT_FULL_DIRECTORY_LIMIT = 50
MAX_DIRECTORY_PAGE_LIMIT = 100
_TEXT_PREFIX_LIMIT = 8
_TEXT_PREFIX_UTF8_BYTES = 512
_VIEW_SCHEMA_VERSION = 1
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DirectoryViewError(ValueError):
    """Raised when one lossless public page cannot satisfy its byte contract."""


class ListDatasetsViewRequest(StrictModel):
    """Select one deterministic bounded view of the complete trusted catalog."""

    source: Literal["all", "builtin", "desktop"] = "all"
    detail: Literal["compact", "full"] = "compact"
    offset: int = Field(0, ge=0)
    limit: int | None = Field(
        None,
        ge=1,
        le=MAX_DIRECTORY_PAGE_LIMIT,
        description="Maximum requested entries; compact defaults to 16 and full defaults to 50.",
    )
    if_view_sha256: str | None = Field(
        None,
        pattern=_SHA256_PATTERN,
        description="Return a short unchanged receipt only for this exact source, detail, offset, and page projection.",
    )


class ListExperimentsViewRequest(StrictModel):
    """Select one deterministic bounded view of active MLflow experiments."""

    maximum_experiments: int = Field(100, ge=1, le=100)
    detail: Literal["compact", "full"] = "compact"
    offset: int = Field(0, ge=0)
    limit: int | None = Field(
        None,
        ge=1,
        le=MAX_DIRECTORY_PAGE_LIMIT,
        description="Maximum requested entries; compact defaults to 16 and full defaults to 50.",
    )
    if_view_sha256: str | None = Field(
        None,
        pattern=_SHA256_PATTERN,
        description="Return a short unchanged receipt only for this exact query, detail, offset, and page projection.",
    )


class GetExperimentViewRequest(StrictModel):
    """Read one experiment and one deterministic bounded page of its runs."""

    experiment_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    maximum_runs: int = Field(50, ge=0, le=100)
    detail: Literal["compact", "full"] = "compact"
    offset: int = Field(0, ge=0)
    limit: int | None = Field(
        None,
        ge=1,
        le=MAX_DIRECTORY_PAGE_LIMIT,
        description="Maximum requested runs; compact defaults to 16 and full defaults to 50.",
    )
    if_view_sha256: str | None = Field(
        None,
        pattern=_SHA256_PATTERN,
        description="Return a short unchanged receipt only for this exact experiment, detail, offset, and run page.",
    )


class DirectoryTextReceipt(StrictModel):
    """One readable UTF-8 prefix bound to the complete text."""

    text: str = Field(max_length=_TEXT_PREFIX_UTF8_BYTES)
    truncated: bool
    sha256: str = Field(pattern=_SHA256_PATTERN)
    total_utf8_bytes: int = Field(ge=0)


class DirectoryTextSequenceReceipt(StrictModel):
    """Bounded ordered text prefix plus complete sequence identity."""

    prefix: tuple[DirectoryTextReceipt, ...] = Field(default=(), max_length=_TEXT_PREFIX_LIMIT)
    total_count: int = Field(ge=0)
    truncated: bool
    sha256: str = Field(pattern=_SHA256_PATTERN)


class CompactDatasetCatalogEntry(StrictModel):
    """Dataset identity and scientific routing fields without an installation path."""

    dataset_id: str
    source: Literal["builtin", "desktop"]
    task: Literal[
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    ] | None = None
    role: Literal["training", "application", "unspecified"]
    file_name: str
    format: Literal["csv", "xlsx"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    row_count: int | None = Field(None, ge=0)
    column_count: int | None = Field(None, ge=0)
    analysis_blockers: DirectoryTextSequenceReceipt
    supported_for_analysis: bool


class CompactExperimentSummary(StrictModel):
    """Stable experiment identity without tags or filesystem locations."""

    experiment_id: str
    name: str
    lifecycle_stage: Literal["active"]


class CompactExperimentRunSummary(StrictModel):
    """Stable run identity and lifecycle times without metrics, parameters, or paths."""

    run_id: str
    run_name: str
    status: str
    start_time: int | None = None
    end_time: int | None = None


class CompactListDatasetsResponse(StrictModel):
    """Default bounded catalog page without absolute installation paths."""

    schema_version: int = 1
    response_detail: Literal["compact"] = "compact"
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_filter: Literal["all", "builtin", "desktop"]
    supported_formats: tuple[Literal["csv", "xlsx"], ...]
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    sort_order: Literal["dataset_id_ascending"] = "dataset_id_ascending"
    datasets: tuple[CompactDatasetCatalogEntry, ...]
    warnings: DirectoryTextSequenceReceipt


class FullListDatasetsResponse(StrictModel):
    """Paged legacy catalog fields retaining every path and entry attribute."""

    schema_version: int = 1
    response_detail: Literal["full"] = "full"
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_filter: Literal["all", "builtin", "desktop"]
    supported_formats: tuple[Literal["csv", "xlsx"], ...]
    desktop_root: str | None = None
    datasets: tuple[DatasetCatalogEntry, ...]
    warnings: tuple[str, ...] = ()
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    sort_order: Literal["dataset_id_ascending"] = "dataset_id_ascending"


class ListDatasetsNotModifiedResponse(StrictModel):
    """Small receipt for one unchanged catalog projection."""

    response_detail: Literal["not_modified"] = "not_modified"
    not_modified: Literal[True] = True
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_filter: Literal["all", "builtin", "desktop"]
    requested_detail: Literal["compact", "full"]
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    requery_required: Literal[False] = False


class CompactListExperimentsResponse(StrictModel):
    """Default bounded experiment directory without tags or tracking paths."""

    schema_version: Literal[1] = 1
    response_detail: Literal["compact"] = "compact"
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_experiments: int = Field(ge=1, le=100)
    experiment_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    sort_order: Literal["experiment_id_ascending"] = "experiment_id_ascending"
    experiments: tuple[CompactExperimentSummary, ...]


class FullListExperimentsResponse(StrictModel):
    """Paged legacy experiment fields retaining tracking paths and all tags."""

    schema_version: Literal[1] = 1
    response_detail: Literal["full"] = "full"
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_experiments: int = Field(ge=1, le=100)
    tracking_root: str
    experiment_count: int = Field(ge=0)
    experiments: tuple[ExperimentSummary, ...]
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    sort_order: Literal["experiment_id_ascending"] = "experiment_id_ascending"


class ListExperimentsNotModifiedResponse(StrictModel):
    """Small receipt for one unchanged experiment-list projection."""

    response_detail: Literal["not_modified"] = "not_modified"
    not_modified: Literal[True] = True
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_experiments: int = Field(ge=1, le=100)
    requested_detail: Literal["compact", "full"]
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    requery_required: Literal[False] = False


class CompactGetExperimentResponse(StrictModel):
    """Default bounded run history without tags, metrics, parameters, or paths."""

    schema_version: Literal[1] = 1
    response_detail: Literal["compact"] = "compact"
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_runs: int = Field(ge=0, le=100)
    experiment: CompactExperimentSummary
    run_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    sort_order: Literal["start_time_descending_then_run_id"] = "start_time_descending_then_run_id"
    runs: tuple[CompactExperimentRunSummary, ...]


class FullGetExperimentResponse(StrictModel):
    """Paged legacy run history retaining tags, metrics, parameters, and paths."""

    schema_version: Literal[1] = 1
    response_detail: Literal["full"] = "full"
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_runs: int = Field(ge=0, le=100)
    tracking_root: str
    experiment: ExperimentSummary
    run_count: int = Field(ge=0)
    runs: tuple[ExperimentRunSummary, ...]
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    sort_order: Literal["start_time_descending_then_run_id"] = "start_time_descending_then_run_id"


class GetExperimentNotModifiedResponse(StrictModel):
    """Small receipt for one unchanged experiment-history projection."""

    response_detail: Literal["not_modified"] = "not_modified"
    not_modified: Literal[True] = True
    view_sha256: str = Field(pattern=_SHA256_PATTERN)
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=128)
    maximum_runs: int = Field(ge=0, le=100)
    requested_detail: Literal["compact", "full"]
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=MAX_DIRECTORY_PAGE_LIMIT)
    next_offset: int | None = Field(None, ge=0)
    requery_required: Literal[False] = False


_Response = TypeVar("_Response", bound=StrictModel)
_Item = TypeVar("_Item")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _response_size(response: StrictModel) -> int:
    return len(_canonical_json_bytes(response.model_dump(mode="json")))


def _response_budget(response: StrictModel) -> int:
    return MAX_FULL_DIRECTORY_RESPONSE_JSON_BYTES if getattr(response, "response_detail", None) == "full" else MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES


def _bounded_text(value: str) -> DirectoryTextReceipt:
    raw = value.encode("utf-8")
    if len(raw) <= _TEXT_PREFIX_UTF8_BYTES:
        text = value
        truncated = False
    else:
        budget = _TEXT_PREFIX_UTF8_BYTES - len("…".encode("utf-8"))
        text = raw[:budget].decode("utf-8", errors="ignore") + "…"
        truncated = True
    return DirectoryTextReceipt(
        text=text,
        truncated=truncated,
        sha256=hashlib.sha256(raw).hexdigest(),
        total_utf8_bytes=len(raw),
    )


def _text_sequence(values: Sequence[str]) -> DirectoryTextSequenceReceipt:
    prefix = tuple(_bounded_text(value) for value in values[:_TEXT_PREFIX_LIMIT])
    return DirectoryTextSequenceReceipt(
        prefix=prefix,
        total_count=len(values),
        truncated=(len(prefix) < len(values) or any(item.truncated for item in prefix)),
        sha256=_canonical_sha256(list(values)),
    )


def _limit(detail: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    return DEFAULT_COMPACT_DIRECTORY_LIMIT if detail == "compact" else DEFAULT_FULL_DIRECTORY_LIMIT


def _with_view_identity(
    response: _Response,
    view_kind: str,
    selector: dict[str, Any],
) -> _Response:
    payload = response.model_dump(mode="json", exclude={"view_sha256"})
    identity = _canonical_sha256(
        {
            "view_schema_version": _VIEW_SCHEMA_VERSION,
            "view_kind": view_kind,
            "selector": selector,
            "payload": payload,
        }
    )
    identified = response.model_copy(update={"view_sha256": identity})
    maximum_bytes = _response_budget(identified)
    if _response_size(identified) > maximum_bytes:
        raise DirectoryViewError(
            f"The requested {identified.response_detail} directory page cannot fit the " f"{maximum_bytes}-byte UTF-8 structured-response budget; request a smaller limit or use compact detail."
        )
    return identified


def _fit_page(
    items: Sequence[_Item],
    *,
    offset: int,
    limit: int,
    builder: Callable[[tuple[_Item, ...], int | None], _Response],
) -> _Response:
    total = len(items)
    available = tuple(items[offset : offset + limit])
    for returned in range(len(available), -1, -1):
        if returned == 0 and offset < total:
            break
        page = available[:returned]
        next_offset = offset + returned if offset + returned < total else None
        try:
            return builder(page, next_offset)
        except DirectoryViewError:
            continue
    raise DirectoryViewError(
        "One complete directory entry exceeds the lossless full-detail structured-response budget. "
        "Use compact detail, or reduce the source record metadata before requesting full detail; no legacy field was dropped."
    )


def _compact_dataset(entry: DatasetCatalogEntry) -> CompactDatasetCatalogEntry:
    return CompactDatasetCatalogEntry(
        dataset_id=entry.dataset_id,
        source=entry.source,
        task=entry.task,
        role=entry.role,
        file_name=entry.file_name,
        format=entry.format,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
        row_count=entry.row_count,
        column_count=entry.column_count,
        analysis_blockers=_text_sequence(entry.analysis_blockers),
        supported_for_analysis=entry.supported_for_analysis,
    )


def list_datasets_response_view(
    source: ListDatasetsResponse,
    request: ListDatasetsViewRequest,
) -> CompactListDatasetsResponse | FullListDatasetsResponse | ListDatasetsNotModifiedResponse:
    """Project the complete canonical catalog only at the public tool boundary."""
    ordered = tuple(sorted(source.datasets, key=lambda item: item.dataset_id))
    limit = _limit(request.detail, request.limit)
    total = len(ordered)

    def build(page: tuple[DatasetCatalogEntry, ...], next_offset: int | None):
        common = dict(
            view_sha256="0" * 64,
            source_filter=source.source_filter,
            supported_formats=source.supported_formats,
            total_count=total,
            returned_count=len(page),
            offset=request.offset,
            limit=limit,
            next_offset=next_offset,
        )
        if request.detail == "full":
            response = FullListDatasetsResponse(
                schema_version=source.schema_version,
                desktop_root=source.desktop_root,
                datasets=page,
                warnings=source.warnings,
                **common,
            )
        else:
            response = CompactListDatasetsResponse(
                schema_version=source.schema_version,
                datasets=tuple(_compact_dataset(item) for item in page),
                warnings=_text_sequence(source.warnings),
                **common,
            )
        return _with_view_identity(
            response,
            "list_datasets",
            {
                "source": request.source,
                "detail": request.detail,
                "offset": request.offset,
                "limit": limit,
            },
        )

    response = _fit_page(
        ordered,
        offset=request.offset,
        limit=limit,
        builder=build,
    )
    if request.if_view_sha256 != response.view_sha256:
        return response
    return ListDatasetsNotModifiedResponse(
        view_sha256=response.view_sha256,
        source_filter=source.source_filter,
        requested_detail=request.detail,
        total_count=response.total_count,
        returned_count=response.returned_count,
        offset=response.offset,
        limit=response.limit,
        next_offset=response.next_offset,
    )


def _compact_experiment(experiment: ExperimentSummary) -> CompactExperimentSummary:
    return CompactExperimentSummary(
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        lifecycle_stage=experiment.lifecycle_stage,
    )


def list_experiments_response_view(
    source: ListExperimentsResponse,
    request: ListExperimentsViewRequest,
) -> CompactListExperimentsResponse | FullListExperimentsResponse | ListExperimentsNotModifiedResponse:
    """Project a complete experiment listing into one identity-bound page."""
    ordered = tuple(sorted(source.experiments, key=lambda item: item.experiment_id))
    limit = _limit(request.detail, request.limit)
    total = len(ordered)

    def build(page: tuple[ExperimentSummary, ...], next_offset: int | None):
        common = dict(
            view_sha256="0" * 64,
            maximum_experiments=request.maximum_experiments,
            experiment_count=source.experiment_count,
            total_count=total,
            returned_count=len(page),
            offset=request.offset,
            limit=limit,
            next_offset=next_offset,
        )
        if request.detail == "full":
            response = FullListExperimentsResponse(
                schema_version=source.schema_version,
                tracking_root=source.tracking_root,
                experiments=page,
                **common,
            )
        else:
            response = CompactListExperimentsResponse(
                schema_version=source.schema_version,
                experiments=tuple(_compact_experiment(item) for item in page),
                **common,
            )
        return _with_view_identity(
            response,
            "list_experiments",
            {
                "maximum_experiments": request.maximum_experiments,
                "detail": request.detail,
                "offset": request.offset,
                "limit": limit,
            },
        )

    response = _fit_page(
        ordered,
        offset=request.offset,
        limit=limit,
        builder=build,
    )
    if request.if_view_sha256 != response.view_sha256:
        return response
    return ListExperimentsNotModifiedResponse(
        view_sha256=response.view_sha256,
        maximum_experiments=request.maximum_experiments,
        requested_detail=request.detail,
        total_count=response.total_count,
        returned_count=response.returned_count,
        offset=response.offset,
        limit=response.limit,
        next_offset=response.next_offset,
    )


def _compact_run(run: ExperimentRunSummary) -> CompactExperimentRunSummary:
    return CompactExperimentRunSummary(
        run_id=run.run_id,
        run_name=run.run_name,
        status=run.status,
        start_time=run.start_time,
        end_time=run.end_time,
    )


def get_experiment_response_view(
    source: GetExperimentResponse,
    request: GetExperimentViewRequest,
) -> CompactGetExperimentResponse | FullGetExperimentResponse | GetExperimentNotModifiedResponse:
    """Project one complete experiment record without performing a list query."""
    ordered = tuple(
        sorted(
            source.runs,
            key=lambda item: (
                -(item.start_time if item.start_time is not None else -1),
                item.run_id,
            ),
        )
    )
    limit = _limit(request.detail, request.limit)
    total = len(ordered)

    def build(page: tuple[ExperimentRunSummary, ...], next_offset: int | None):
        common = dict(
            view_sha256="0" * 64,
            maximum_runs=request.maximum_runs,
            run_count=source.run_count,
            total_count=total,
            returned_count=len(page),
            offset=request.offset,
            limit=limit,
            next_offset=next_offset,
        )
        if request.detail == "full":
            response = FullGetExperimentResponse(
                schema_version=source.schema_version,
                tracking_root=source.tracking_root,
                experiment=source.experiment,
                runs=page,
                **common,
            )
        else:
            response = CompactGetExperimentResponse(
                schema_version=source.schema_version,
                experiment=_compact_experiment(source.experiment),
                runs=tuple(_compact_run(item) for item in page),
                **common,
            )
        return _with_view_identity(
            response,
            "get_experiment",
            {
                "experiment_id": request.experiment_id,
                "maximum_runs": request.maximum_runs,
                "detail": request.detail,
                "offset": request.offset,
                "limit": limit,
            },
        )

    response = _fit_page(
        ordered,
        offset=request.offset,
        limit=limit,
        builder=build,
    )
    if request.if_view_sha256 != response.view_sha256:
        return response
    return GetExperimentNotModifiedResponse(
        view_sha256=response.view_sha256,
        experiment_id=request.experiment_id,
        maximum_runs=request.maximum_runs,
        requested_detail=request.detail,
        total_count=response.total_count,
        returned_count=response.returned_count,
        offset=response.offset,
        limit=response.limit,
        next_offset=response.next_offset,
    )
