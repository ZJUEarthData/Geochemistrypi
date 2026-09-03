import json
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.directory_views import (
    MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES,
    MAX_FULL_DIRECTORY_RESPONSE_JSON_BYTES,
    CompactGetExperimentResponse,
    CompactListDatasetsResponse,
    CompactListExperimentsResponse,
    FullGetExperimentResponse,
    FullListDatasetsResponse,
    FullListExperimentsResponse,
    GetExperimentNotModifiedResponse,
    GetExperimentViewRequest,
    ListDatasetsNotModifiedResponse,
    ListDatasetsViewRequest,
    ListExperimentsNotModifiedResponse,
    ListExperimentsViewRequest,
    get_experiment_response_view,
    list_datasets_response_view,
    list_experiments_response_view,
)
from geochemistrypi_mcp.api.schemas import DatasetCatalogEntry, ExperimentRunSummary, ExperimentSummary, GetExperimentResponse, ListDatasetsResponse, ListExperimentsResponse
from geochemistrypi_mcp.config.settings import McpSettings
from geochemistrypi_mcp.server import create_server
from mcp import Client


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _json_bytes(value) -> bytes:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _dataset(index: int, *, source: str = "builtin", path_repeat: int = 0) -> DatasetCatalogEntry:
    prefix = "builtin" if source == "builtin" else "desktop"
    return DatasetCatalogEntry(
        dataset_id=f"{prefix}:dataset-{index:03d}",
        source=source,
        role="training",
        task="classification",
        file_name=f"岩石-{index:03d}.xlsx",
        path=f"C:/安装目录/{index:03d}/" + ("地球化学/" * path_repeat) + f"dataset-{index:03d}.xlsx",
        format="xlsx",
        size_bytes=10_000 + index,
        sha256=f"{index:064x}",
        row_count=200 + index,
        column_count=12,
        analysis_blockers=("列名需要人工确认。" * 100,),
        supported_for_analysis=True,
    )


def _experiment(index: int, *, name_repeat: int = 1) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=f"exp-{index:03d}",
        name=f"实验-{index:03d}-" + ("玄武岩" * name_repeat),
        lifecycle_stage="active",
        artifact_location=f"file:///tracking/实验-{index:03d}",
        tags={
            "研究区域": "东亚大陆边缘" * 20,
            "完整标签": f"标签-{index:03d}",
        },
    )


def _run(index: int, *, parameter_repeat: int = 1) -> ExperimentRunSummary:
    return ExperimentRunSummary(
        run_id=f"run-{index:03d}",
        run_name=f"运行-{index:03d}",
        status="FINISHED",
        start_time=1_000_000 + index,
        end_time=1_001_000 + index,
        artifact_uri=f"file:///tracking/run-{index:03d}/artifacts",
        metrics={"accuracy": 0.9, "loss": None},
        params={"scientific_contract": "完整参数" * parameter_repeat},
    )


def test_near_one_mib_catalog_defaults_to_bounded_compact_projection() -> None:
    datasets = tuple(_dataset(index, path_repeat=500) for index in reversed(range(100)))
    source = ListDatasetsResponse(
        source_filter="builtin",
        supported_formats=("csv", "xlsx"),
        desktop_root="C:/Users/researcher/Desktop",
        datasets=datasets,
        warnings=("目录警告" * 100,),
    )

    assert 900_000 < len(_json_bytes(source)) < 1_200_000

    compact = list_datasets_response_view(source, ListDatasetsViewRequest())

    assert isinstance(compact, CompactListDatasetsResponse)
    assert compact.returned_count == 16
    assert compact.total_count == 100
    assert compact.next_offset == 16
    assert len(_json_bytes(compact)) <= MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES
    payload = compact.model_dump(mode="json")
    assert "desktop_root" not in payload
    assert all("path" not in item for item in payload["datasets"])
    assert tuple(item.dataset_id for item in compact.datasets) == tuple(f"builtin:dataset-{index:03d}" for index in range(16))
    first = compact.datasets[0]
    assert {
        "dataset_id",
        "source",
        "task",
        "role",
        "file_name",
        "size_bytes",
        "sha256",
        "row_count",
        "column_count",
        "analysis_blockers",
    } <= set(type(first).model_fields)
    assert first.analysis_blockers.truncated is True


def test_dataset_full_pages_are_lossless_sorted_and_without_gaps() -> None:
    datasets = tuple(_dataset(index, path_repeat=400) for index in reversed(range(100)))
    source = ListDatasetsResponse(
        source_filter="builtin",
        supported_formats=("csv", "xlsx"),
        desktop_root="C:/Users/researcher/Desktop",
        datasets=datasets,
        warnings=("legacy warning",),
    )
    expected = sorted(datasets, key=lambda item: item.dataset_id)
    expected_by_id = {item.dataset_id: item for item in expected}
    returned = []
    offset = 0

    while True:
        page = list_datasets_response_view(
            source,
            ListDatasetsViewRequest(detail="full", offset=offset, limit=7),
        )
        assert isinstance(page, FullListDatasetsResponse)
        assert len(_json_bytes(page)) <= MAX_FULL_DIRECTORY_RESPONSE_JSON_BYTES
        assert page.desktop_root == source.desktop_root
        assert page.warnings == source.warnings
        for item in page.datasets:
            assert item == expected_by_id[item.dataset_id]
        returned.extend(page.datasets)
        if page.next_offset is None:
            break
        assert page.next_offset == offset + page.returned_count
        offset = page.next_offset

    assert returned == expected
    assert len({item.dataset_id for item in returned}) == len(returned)


def test_experiment_and_run_directories_paginate_losslessly_with_multibyte_values() -> None:
    experiments = tuple(_experiment(index, name_repeat=25) for index in reversed(range(100)))
    experiment_source = ListExperimentsResponse(
        tracking_root="C:/跟踪目录",
        experiment_count=100,
        experiments=experiments,
    )

    compact_experiments = list_experiments_response_view(
        experiment_source,
        ListExperimentsViewRequest(),
    )
    assert isinstance(compact_experiments, CompactListExperimentsResponse)
    assert compact_experiments.returned_count == 16
    assert len(_json_bytes(compact_experiments)) <= MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES
    compact_payload = compact_experiments.model_dump(mode="json")
    assert "tracking_root" not in compact_payload
    assert set(compact_payload["experiments"][0]) == {
        "experiment_id",
        "name",
        "lifecycle_stage",
    }

    expected_experiments = sorted(experiments, key=lambda item: item.experiment_id)
    returned_experiments = []
    offset = 0
    while True:
        page = list_experiments_response_view(
            experiment_source,
            ListExperimentsViewRequest(detail="full", offset=offset, limit=13),
        )
        assert isinstance(page, FullListExperimentsResponse)
        assert len(_json_bytes(page)) <= MAX_FULL_DIRECTORY_RESPONSE_JSON_BYTES
        returned_experiments.extend(page.experiments)
        if page.next_offset is None:
            break
        offset = page.next_offset
    assert returned_experiments == expected_experiments

    runs = tuple(_run(index, parameter_repeat=100) for index in range(100))
    history_source = GetExperimentResponse(
        tracking_root="C:/跟踪目录",
        experiment=experiments[0],
        run_count=100,
        runs=runs,
    )
    compact_history = get_experiment_response_view(
        history_source,
        GetExperimentViewRequest(experiment_id=experiments[0].experiment_id),
    )
    assert isinstance(compact_history, CompactGetExperimentResponse)
    assert compact_history.returned_count == 16
    assert len(_json_bytes(compact_history)) <= MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES
    compact_run = compact_history.model_dump(mode="json")["runs"][0]
    assert set(compact_run) == {
        "run_id",
        "run_name",
        "status",
        "start_time",
        "end_time",
    }

    expected_runs = sorted(runs, key=lambda item: (-item.start_time, item.run_id))
    returned_runs = []
    offset = 0
    while True:
        page = get_experiment_response_view(
            history_source,
            GetExperimentViewRequest(
                experiment_id=experiments[0].experiment_id,
                detail="full",
                offset=offset,
                limit=11,
            ),
        )
        assert isinstance(page, FullGetExperimentResponse)
        assert len(_json_bytes(page)) <= MAX_FULL_DIRECTORY_RESPONSE_JSON_BYTES
        assert page.experiment == history_source.experiment
        returned_runs.extend(page.runs)
        if page.next_offset is None:
            break
        offset = page.next_offset
    assert returned_runs == expected_runs


def test_conditional_hash_is_isolated_by_query_detail_page_and_experiment() -> None:
    builtin = ListDatasetsResponse(
        source_filter="builtin",
        supported_formats=("csv", "xlsx"),
        datasets=tuple(_dataset(index) for index in range(4)),
    )
    base = list_datasets_response_view(
        builtin,
        ListDatasetsViewRequest(source="builtin", limit=2),
    )
    unchanged = list_datasets_response_view(
        builtin,
        ListDatasetsViewRequest(
            source="builtin",
            limit=2,
            if_view_sha256=base.view_sha256,
        ),
    )
    assert isinstance(unchanged, ListDatasetsNotModifiedResponse)

    desktop = ListDatasetsResponse(
        source_filter="desktop",
        supported_formats=("csv", "xlsx"),
        datasets=tuple(_dataset(index, source="desktop") for index in range(4)),
    )
    changed_dataset_views = (
        list_datasets_response_view(
            desktop,
            ListDatasetsViewRequest(
                source="desktop",
                limit=2,
                if_view_sha256=base.view_sha256,
            ),
        ),
        list_datasets_response_view(
            builtin,
            ListDatasetsViewRequest(
                source="builtin",
                detail="full",
                limit=2,
                if_view_sha256=base.view_sha256,
            ),
        ),
        list_datasets_response_view(
            builtin,
            ListDatasetsViewRequest(
                source="builtin",
                offset=2,
                limit=2,
                if_view_sha256=base.view_sha256,
            ),
        ),
        list_datasets_response_view(
            builtin,
            ListDatasetsViewRequest(
                source="builtin",
                limit=3,
                if_view_sha256=base.view_sha256,
            ),
        ),
    )
    assert all(not isinstance(view, ListDatasetsNotModifiedResponse) for view in changed_dataset_views)

    experiment_source = ListExperimentsResponse(
        tracking_root="C:/tracking",
        experiment_count=4,
        experiments=tuple(_experiment(index) for index in range(4)),
    )
    experiment_base = list_experiments_response_view(
        experiment_source,
        ListExperimentsViewRequest(maximum_experiments=4, limit=2),
    )
    experiment_unchanged = list_experiments_response_view(
        experiment_source,
        ListExperimentsViewRequest(
            maximum_experiments=4,
            limit=2,
            if_view_sha256=experiment_base.view_sha256,
        ),
    )
    assert isinstance(experiment_unchanged, ListExperimentsNotModifiedResponse)
    changed_experiment = list_experiments_response_view(
        experiment_source,
        ListExperimentsViewRequest(
            maximum_experiments=3,
            limit=2,
            if_view_sha256=experiment_base.view_sha256,
        ),
    )
    assert not isinstance(changed_experiment, ListExperimentsNotModifiedResponse)

    first_history = GetExperimentResponse(
        tracking_root="C:/tracking",
        experiment=_experiment(1),
        run_count=4,
        runs=tuple(_run(index) for index in range(4)),
    )
    history_base = get_experiment_response_view(
        first_history,
        GetExperimentViewRequest(experiment_id="exp-001", limit=2),
    )
    history_unchanged = get_experiment_response_view(
        first_history,
        GetExperimentViewRequest(
            experiment_id="exp-001",
            limit=2,
            if_view_sha256=history_base.view_sha256,
        ),
    )
    assert isinstance(history_unchanged, GetExperimentNotModifiedResponse)
    second_history = first_history.model_copy(update={"experiment": _experiment(2)})
    changed_history = get_experiment_response_view(
        second_history,
        GetExperimentViewRequest(
            experiment_id="exp-002",
            limit=2,
            if_view_sha256=history_base.view_sha256,
        ),
    )
    assert not isinstance(changed_history, GetExperimentNotModifiedResponse)


def test_full_detail_preserves_one_record_larger_than_compact_budget() -> None:
    large_run = _run(1, parameter_repeat=30_000)
    source = GetExperimentResponse(
        tracking_root="C:/tracking",
        experiment=_experiment(1),
        run_count=1,
        runs=(large_run,),
    )

    full = get_experiment_response_view(
        source,
        GetExperimentViewRequest(experiment_id="exp-001", detail="full", limit=1),
    )

    assert isinstance(full, FullGetExperimentResponse)
    assert len(_json_bytes(full)) > MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES
    assert len(_json_bytes(full)) <= MAX_FULL_DIRECTORY_RESPONSE_JSON_BYTES
    assert full.runs == (large_run,)
    assert full.runs[0].params == large_run.params


def test_utf8_size_measurement_counts_bytes_not_characters() -> None:
    source = ListExperimentsResponse(
        tracking_root="C:/跟踪目录",
        experiment_count=100,
        experiments=tuple(_experiment(index, name_repeat=100) for index in range(100)),
    )

    response = list_experiments_response_view(
        source,
        ListExperimentsViewRequest(limit=100),
    )
    serialized = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert len(serialized.encode("utf-8")) > len(serialized)
    assert len(serialized.encode("utf-8")) <= MAX_COMPACT_DIRECTORY_RESPONSE_JSON_BYTES


class _ExperimentStore:
    def __init__(self, response: GetExperimentResponse) -> None:
        self.response = response
        self.get_calls = []
        self.list_calls = []

    def get(self, request):
        self.get_calls.append(request)
        return self.response

    def list(self, request):
        self.list_calls.append(request)
        raise AssertionError("get_experiment must not call list_experiments")


class _DirectoryRunManager:
    def __init__(self, experiment_manager: _ExperimentStore) -> None:
        self.experiment_manager = experiment_manager
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DatasetCatalog:
    def __init__(self, response: ListDatasetsResponse) -> None:
        self.response = response
        self.list_calls = []

    def list(self, request):
        self.list_calls.append(request)
        return self.response


class _ListExperimentStore:
    def __init__(self, response: ListExperimentsResponse) -> None:
        self.response = response
        self.list_calls = []

    def list(self, request):
        self.list_calls.append(request)
        return self.response

    def get(self, request):
        raise AssertionError("list_experiments must not call get_experiment")


@pytest.mark.anyio
async def test_public_list_tools_page_only_after_complete_internal_queries(tmp_path: Path) -> None:
    dataset_source = ListDatasetsResponse(
        source_filter="builtin",
        supported_formats=("csv", "xlsx"),
        datasets=tuple(_dataset(index) for index in reversed(range(20))),
    )
    experiment_source = ListExperimentsResponse(
        tracking_root="C:/tracking",
        experiment_count=20,
        experiments=tuple(_experiment(index) for index in reversed(range(20))),
    )
    catalog = _DatasetCatalog(dataset_source)
    store = _ListExperimentStore(experiment_source)
    runs = _DirectoryRunManager(store)
    runs.dataset_catalog = catalog
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        runs,
    )

    async with Client(server) as client:
        datasets = await client.call_tool("list_datasets", {"source": "builtin"})
        experiments = await client.call_tool("list_experiments", {})

    assert datasets.is_error is False
    assert datasets.structured_content["total_count"] == 20
    assert datasets.structured_content["returned_count"] == 16
    assert experiments.is_error is False
    assert experiments.structured_content["total_count"] == 20
    assert experiments.structured_content["returned_count"] == 16
    assert len(catalog.list_calls) == 1
    assert catalog.list_calls[0].source == "builtin"
    assert not hasattr(catalog.list_calls[0], "offset")
    assert len(store.list_calls) == 1
    assert store.list_calls[0].maximum_experiments == 100
    assert not hasattr(store.list_calls[0], "offset")
    assert runs.closed is True


@pytest.mark.anyio
async def test_public_get_experiment_uses_direct_lookup_without_listing(tmp_path: Path) -> None:
    source = GetExperimentResponse(
        tracking_root="C:/tracking",
        experiment=_experiment(7),
        run_count=2,
        runs=(_run(1), _run(2)),
    )
    store = _ExperimentStore(source)
    runs = _DirectoryRunManager(store)
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        runs,
    )

    async with Client(server) as client:
        result = await client.call_tool("get_experiment", {"experiment_id": "exp-007"})

    assert result.is_error is False
    assert result.structured_content["response_detail"] == "compact"
    assert result.structured_content["returned_count"] == 2
    assert store.list_calls == []
    assert len(store.get_calls) == 1
    assert store.get_calls[0].experiment_id == "exp-007"
    assert runs.closed is True


@pytest.mark.anyio
async def test_oversized_lossless_full_record_returns_actionable_public_error(tmp_path: Path) -> None:
    oversized_run = _run(1).model_copy(update={"params": {"scientific_contract": "完整参数" * 300_000}})
    source = GetExperimentResponse(
        tracking_root="C:/tracking",
        experiment=_experiment(7),
        run_count=1,
        runs=(oversized_run,),
    )
    store = _ExperimentStore(source)
    runs = _DirectoryRunManager(store)
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        runs,
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "get_experiment",
            {"experiment_id": "exp-007", "detail": "full", "limit": 1},
        )

    assert result.is_error is True
    assert result.structured_content["result_type"] == "directory_view_rejected"
    assert result.structured_content["retryable"] is True
    assert "detail=compact" in result.structured_content["next_action"]
    assert "silently dropped" in result.structured_content["next_action"]
    assert "internal" not in result.content[0].text.lower()
    assert store.list_calls == []
    assert len(store.get_calls) == 1
    assert runs.closed is True
