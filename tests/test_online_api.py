"""Integration tests for the minimal Online API."""

from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from geochemistrypi.online.app import create_app


def make_kinetic_workbook() -> bytes:
    buffer = BytesIO()
    pd.DataFrame(
        [
            {"c0": 100.0, "k": 0.1, "t": 5.0},
            {"c0": 50.0, "k": 0.2, "t": 3.0},
        ]
    ).to_excel(buffer, index=False)
    return buffer.getvalue()


def make_second_order_workbook() -> bytes:
    buffer = BytesIO()
    pd.DataFrame(
        [
            {"c0": 100.0, "k": 0.01, "t": 5.0},
            {"c0": 50.0, "k": 0.02, "t": 3.0},
            {"c0": 25.0, "k": 0.0, "t": 10.0},
        ]
    ).to_excel(buffer, index=False)
    return buffer.getvalue()


def make_radioactive_decay_workbook() -> bytes:
    buffer = BytesIO()
    pd.DataFrame(
        [
            {"n0": 1000.0, "decay_const": 0.05, "t": 10.0},
            {"n0": 500.0, "decay_const": 0.1, "t": 5.0},
            {"n0": 250.0, "decay_const": 0.0, "t": 12.0},
            {"n0": 0.0, "decay_const": 0.2, "t": 3.0},
        ]
    ).to_excel(buffer, index=False)
    return buffer.getvalue()


def make_fick_diffusion_workbook() -> bytes:
    buffer = BytesIO()
    pd.DataFrame(
        [
            {"D": 1e-9, "dc_dx": -1000.0},
            {"D": 2e-9, "dc_dx": 500.0},
            {"D": 0.0, "dc_dx": 250.0},
        ]
    ).to_excel(buffer, index=False)
    return buffer.getvalue()


def make_chromatography_workbook() -> bytes:
    buffer = BytesIO()
    pd.DataFrame(
        [
            {"tR": 10.0, "sigma": 0.5},
            {"tR": 5.0, "sigma": 1.0},
            {"tR": 0.0, "sigma": 2.0},
        ]
    ).to_excel(buffer, index=False)
    return buffer.getvalue()


def make_vanthoff_workbook() -> bytes:
    buffer = BytesIO()
    pd.DataFrame(
        [
            {"K1": 10.0, "dH": 50000.0, "T1": 298.15, "T2": 350.0},
            {"K1": 10.0, "dH": -50000.0, "T1": 298.15, "T2": 350.0},
            {"K1": 7.5, "dH": 0.0, "T1": 300.0, "T2": 500.0},
            {"K1": 3.0, "dH": 50000.0, "T1": 298.15, "T2": 298.15},
        ]
    ).to_excel(buffer, index=False)
    return buffer.getvalue()


def make_activity_coefficient_workbook() -> bytes:
    buffer = BytesIO()
    pd.DataFrame(
        [
            {"z": 1, "ionic_strength": 0.1},
            {"z": -2, "ionic_strength": 0.5},
            {"z": 0, "ionic_strength": 1.0},
            {"z": 3, "ionic_strength": 0.0},
        ]
    ).to_excel(buffer, index=False)
    return buffer.getvalue()


def make_workbook(rows: list[dict]) -> bytes:
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False)
    return buffer.getvalue()


def post_first_order(client: TestClient, content: bytes, filename: str = "kinetic.xlsx"):
    return client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_kinetic",
            "method": "first_order",
            "element": "Any",
        },
        files={
            "dataset": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def post_second_order(client: TestClient, content: bytes, filename: str = "second-order.xlsx"):
    return client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_kinetic",
            "method": "second_order",
            "element": "Any",
        },
        files={
            "dataset": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def post_radioactive_decay(client: TestClient, content: bytes, filename: str = "radioactive-decay.xlsx"):
    return client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_kinetic",
            "method": "radioactive_decay",
            "element": "Any",
        },
        files={
            "dataset": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def post_fick_diffusion(client: TestClient, content: bytes, filename: str = "fick-diffusion.xlsx"):
    return client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_transport",
            "method": "fick_diffusion",
            "element": "Any",
        },
        files={
            "dataset": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def post_chromatography(client: TestClient, content: bytes, filename: str = "chromatography.xlsx"):
    return client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_transport",
            "method": "chromatography",
            "element": "Any",
        },
        files={
            "dataset": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def post_vanthoff(client: TestClient, content: bytes, filename: str = "vanthoff.xlsx"):
    return client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_thermodynamic",
            "method": "vanthoff",
            "element": "Any",
        },
        files={
            "dataset": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def post_activity_coefficient(
    client: TestClient,
    content: bytes,
    filename: str = "activity-coefficient.xlsx",
):
    return client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_thermodynamic",
            "method": "activity_coefficient",
            "element": "Any",
        },
        files={
            "dataset": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )


def test_health_and_catalog(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    catalog = client.get("/api/chemical-modeling/catalog")
    assert catalog.status_code == 200
    kinetic = next(task for task in catalog.json()["tasks"] if task["name"] == "algo_kinetic")
    assert kinetic["available"] is True
    first_order = next(method for method in kinetic["methods"] if method["name"] == "first_order")
    assert first_order["status"] == "verified"
    assert first_order["formula"] == "C_t = c0 × exp(-k × t)"
    assert first_order["required_columns"] == ["c0", "k", "t"]
    assert first_order["input_columns"][0] == {
        "name": "c0",
        "label": "初始浓度",
        "description": "反应开始时的物质浓度。",
        "data_type": "number",
        "unit": "自定义，须与结果浓度一致",
        "example": 100.0,
        "required": True,
        "minimum": 0.0,
        "exclusive_minimum": False,
    }

    second_order = next(method for method in kinetic["methods"] if method["name"] == "second_order")
    assert second_order["status"] == "verified"
    assert second_order["required_columns"] == ["c0", "k", "t"]
    assert second_order["input_columns"][0]["minimum"] == 0.0
    assert second_order["input_columns"][0]["exclusive_minimum"] is True

    radioactive_decay = next(
        method for method in kinetic["methods"] if method["name"] == "radioactive_decay"
    )
    assert radioactive_decay["status"] == "verified"
    assert radioactive_decay["formula"] == "N_t = n0 × exp(-decay_const × t)"
    assert radioactive_decay["required_columns"] == ["n0", "decay_const", "t"]

    equilibrium = next(task for task in catalog.json()["tasks"] if task["name"] == "algo_equilibrium")
    assert all(method["status"] == "testing" for method in equilibrium["methods"])

    transport = next(task for task in catalog.json()["tasks"] if task["name"] == "algo_transport")
    fick_diffusion = next(method for method in transport["methods"] if method["name"] == "fick_diffusion")
    assert fick_diffusion["status"] == "verified"
    assert fick_diffusion["formula"] == "J = -D × dc_dx"
    assert fick_diffusion["required_columns"] == ["D", "dc_dx"]

    chromatography = next(method for method in transport["methods"] if method["name"] == "chromatography")
    assert chromatography["status"] == "verified"
    assert chromatography["formula"] == "N = (tR / sigma)²"
    assert chromatography["required_columns"] == ["tR", "sigma"]
    assert chromatography["input_columns"][1]["exclusive_minimum"] is True

    thermodynamic = next(
        task for task in catalog.json()["tasks"] if task["name"] == "algo_thermodynamic"
    )
    vanthoff = next(method for method in thermodynamic["methods"] if method["name"] == "vanthoff")
    assert vanthoff["status"] == "verified"
    assert vanthoff["formula"] == "K2 = K1 × exp[-dH / R × (1 / T2 - 1 / T1)]"
    assert vanthoff["required_columns"] == ["K1", "dH", "T1", "T2"]

    activity_coefficient = next(
        method for method in thermodynamic["methods"] if method["name"] == "activity_coefficient"
    )
    assert activity_coefficient["status"] == "verified"
    assert activity_coefficient["required_columns"] == ["z", "ionic_strength"]
    assert activity_coefficient["input_columns"][0]["data_type"] == "integer"


def test_run_and_download_first_order_result(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    response = post_first_order(client, make_kinetic_workbook())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["name"] == "first_order_results.xlsx"

    download = client.get(payload["artifacts"][0]["download_url"])
    assert download.status_code == 200
    result = pd.read_excel(BytesIO(download.content))
    assert list(result.columns) == ["c0", "k", "t", "C_t"]
    assert round(result.loc[0, "C_t"], 6) == 60.653066


def test_run_and_download_second_order_result_matches_reference_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    response = post_second_order(client, make_second_order_workbook())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["name"] == "second_order_results.xlsx"

    download = client.get(payload["artifacts"][0]["download_url"])
    assert download.status_code == 200
    result = pd.read_excel(BytesIO(download.content))
    assert list(result.columns) == ["c0", "k", "t", "C_t"]
    assert result["C_t"].tolist() == pytest.approx([16.66666666666667, 12.5, 25.0])


def test_run_and_download_radioactive_decay_result_matches_reference_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    response = post_radioactive_decay(client, make_radioactive_decay_workbook())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["name"] == "radioactive_decay_results.xlsx"

    download = client.get(payload["artifacts"][0]["download_url"])
    assert download.status_code == 200
    result = pd.read_excel(BytesIO(download.content))
    assert list(result.columns) == ["n0", "decay_const", "t", "N_t"]
    assert result["N_t"].tolist() == pytest.approx(
        [606.5306597126335, 303.2653298563167, 250.0, 0.0]
    )


def test_run_and_download_fick_diffusion_result_matches_reference_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    response = post_fick_diffusion(client, make_fick_diffusion_workbook())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["name"] == "fick_diffusion_results.xlsx"

    download = client.get(payload["artifacts"][0]["download_url"])
    assert download.status_code == 200
    result = pd.read_excel(BytesIO(download.content))
    assert list(result.columns) == ["D", "dc_dx", "J"]
    assert result["J"].tolist() == pytest.approx([1e-6, -1e-6, 0.0])


def test_run_and_download_chromatography_result_matches_reference_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    response = post_chromatography(client, make_chromatography_workbook())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["name"] == "chromatography_results.xlsx"

    download = client.get(payload["artifacts"][0]["download_url"])
    assert download.status_code == 200
    result = pd.read_excel(BytesIO(download.content))
    assert list(result.columns) == ["tR", "sigma", "N"]
    assert result["N"].tolist() == pytest.approx([400.0, 25.0, 0.0])


def test_run_and_download_vanthoff_result_matches_reference_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    response = post_vanthoff(client, make_vanthoff_workbook())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["name"] == "vanthoff_results.xlsx"

    download = client.get(payload["artifacts"][0]["download_url"])
    assert download.status_code == 200
    result = pd.read_excel(BytesIO(download.content))
    assert list(result.columns) == ["K1", "dH", "T1", "T2", "K2"]
    assert result["K2"].tolist() == pytest.approx(
        [198.49404942444545, 0.5037934401054369, 7.5, 3.0]
    )


def test_run_and_download_activity_coefficient_result_matches_reference_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    response = post_activity_coefficient(client, make_activity_coefficient_workbook())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["name"] == "activity_coefficient_results.xlsx"

    download = client.get(payload["artifacts"][0]["download_url"])
    assert download.status_code == 200
    result = pd.read_excel(BytesIO(download.content))
    assert list(result.columns) == ["z", "ionic_strength", "log_gamma"]
    assert result["log_gamma"].tolist() == pytest.approx(
        [-0.13898454656673898, -1.0636221788523552, 0.0, 0.0]
    )


def test_reject_non_xlsx_upload(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_kinetic",
            "method": "first_order",
            "element": "Any",
        },
        files={"dataset": ("data.csv", b"c0,k,t\n100,0.1,5", "text/csv")},
    )
    assert response.status_code == 422
    assert "Only .xlsx" in response.json()["detail"]


def test_reject_unreadable_xlsx(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_first_order(client, b"this is not an Excel workbook")
    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a readable .xlsx workbook"


def test_reject_workbook_with_missing_columns(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_first_order(client, make_workbook([{"c0": 100, "t": 5}]))
    assert response.status_code == 422
    assert response.json()["detail"] == "Missing required Excel columns: k"


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        ([{"c0": 0.0, "k": 0.01, "t": 5.0}], "Column 'c0' values must be greater than 0"),
        (
            [{"c0": 100.0, "k": -0.01, "t": 5.0}],
            "Column 'k' values must be greater than or equal to 0",
        ),
        (
            [{"c0": 100.0, "k": 0.01, "t": -1.0}],
            "Column 't' values must be greater than or equal to 0",
        ),
        (
            [{"c0": 100.0, "k": "not-a-number", "t": 5.0}],
            "Column 'k' must contain numeric values without empty cells",
        ),
    ],
)
def test_reject_invalid_second_order_values(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_second_order(client, make_workbook(rows))
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        (
            [{"n0": -1.0, "decay_const": 0.05, "t": 10.0}],
            "Column 'n0' values must be greater than or equal to 0",
        ),
        (
            [{"n0": 1000.0, "decay_const": -0.05, "t": 10.0}],
            "Column 'decay_const' values must be greater than or equal to 0",
        ),
        (
            [{"n0": 1000.0, "decay_const": 0.05, "t": -1.0}],
            "Column 't' values must be greater than or equal to 0",
        ),
        (
            [{"n0": 1000.0, "decay_const": "not-a-number", "t": 10.0}],
            "Column 'decay_const' must contain numeric values without empty cells",
        ),
    ],
)
def test_reject_invalid_radioactive_decay_values(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_radioactive_decay(client, make_workbook(rows))
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        (
            [{"D": -1e-9, "dc_dx": 1000.0}],
            "Column 'D' values must be greater than or equal to 0",
        ),
        (
            [{"D": "not-a-number", "dc_dx": 1000.0}],
            "Column 'D' must contain numeric values without empty cells",
        ),
        (
            [{"D": 1e-9, "dc_dx": "not-a-number"}],
            "Column 'dc_dx' must contain numeric values without empty cells",
        ),
        (
            [{"D": None, "dc_dx": 1000.0}],
            "Column 'D' must contain numeric values without empty cells",
        ),
    ],
)
def test_reject_invalid_fick_diffusion_values(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_fick_diffusion(client, make_workbook(rows))
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        (
            [{"tR": -1.0, "sigma": 0.5}],
            "Column 'tR' values must be greater than or equal to 0",
        ),
        (
            [{"tR": 10.0, "sigma": 0.0}],
            "Column 'sigma' values must be greater than 0",
        ),
        (
            [{"tR": 10.0, "sigma": -0.5}],
            "Column 'sigma' values must be greater than 0",
        ),
        (
            [{"tR": "not-a-number", "sigma": 0.5}],
            "Column 'tR' must contain numeric values without empty cells",
        ),
        (
            [{"tR": 10.0, "sigma": "not-a-number"}],
            "Column 'sigma' must contain numeric values without empty cells",
        ),
    ],
)
def test_reject_invalid_chromatography_values(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_chromatography(client, make_workbook(rows))
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        (
            [{"K1": 0.0, "dH": 50000.0, "T1": 298.15, "T2": 350.0}],
            "Column 'K1' values must be greater than 0",
        ),
        (
            [{"K1": 10.0, "dH": 50000.0, "T1": 0.0, "T2": 350.0}],
            "Column 'T1' values must be greater than 0",
        ),
        (
            [{"K1": 10.0, "dH": 50000.0, "T1": 298.15, "T2": -1.0}],
            "Column 'T2' values must be greater than 0",
        ),
        (
            [{"K1": "not-a-number", "dH": 50000.0, "T1": 298.15, "T2": 350.0}],
            "Column 'K1' must contain numeric values without empty cells",
        ),
        (
            [{"K1": 10.0, "dH": "not-a-number", "T1": 298.15, "T2": 350.0}],
            "Column 'dH' must contain numeric values without empty cells",
        ),
        (
            [{"K1": 10.0, "dH": -1e12, "T1": 1000.0, "T2": 1.0}],
            "van't Hoff parameters produce a result outside the supported numeric range",
        ),
    ],
)
def test_reject_invalid_vanthoff_values(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_vanthoff(client, make_workbook(rows))
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        (
            [{"z": 1.5, "ionic_strength": 0.1}],
            "Column 'z' must contain integer values",
        ),
        (
            [{"z": 1, "ionic_strength": -0.1}],
            "Column 'ionic_strength' values must be greater than or equal to 0",
        ),
        (
            [{"z": "not-a-number", "ionic_strength": 0.1}],
            "Column 'z' must contain numeric values without empty cells",
        ),
        (
            [{"z": 1, "ionic_strength": "not-a-number"}],
            "Column 'ionic_strength' must contain numeric values without empty cells",
        ),
    ],
)
def test_reject_invalid_activity_coefficient_values(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_activity_coefficient(client, make_workbook(rows))
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("task", "not_a_task", "Unknown task"),
        ("method", "not_a_method", "Unknown method"),
        ("element", "Hg", "Unknown element"),
    ],
)
def test_reject_invalid_algorithm_selection(tmp_path, field, value, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    data = {
        "task": "algo_kinetic",
        "method": "first_order",
        "element": "Any",
    }
    data[field] = value
    response = client.post(
        "/api/chemical-modeling/run",
        data=data,
        files={"dataset": ("kinetic.xlsx", make_kinetic_workbook())},
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


def test_reject_method_that_has_not_completed_online_verification(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_kinetic",
            "method": "adsorption_kinetics",
            "element": "Any",
        },
        files={"dataset": ("kinetic.xlsx", make_kinetic_workbook())},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Method 'adsorption_kinetics' has not completed Online verification"


def test_reject_empty_and_oversized_uploads(tmp_path):
    app = create_app(tmp_path / "runtime")
    app.state.online_service.max_upload_bytes = 8
    client = TestClient(app)

    empty = post_first_order(client, b"")
    assert empty.status_code == 422
    assert empty.json()["detail"] == "The uploaded file is empty"

    oversized = post_first_order(client, b"123456789")
    assert oversized.status_code == 413
    assert "exceeds 8 bytes" in oversized.json()["detail"]


def test_jobs_are_isolated_and_unknown_artifact_returns_404(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))

    first = post_first_order(client, make_kinetic_workbook()).json()
    second = post_first_order(client, make_kinetic_workbook()).json()

    assert first["job_id"] != second["job_id"]
    assert client.get(first["artifacts"][0]["download_url"]).status_code == 200
    assert client.get(second["artifacts"][0]["download_url"]).status_code == 200
    assert client.get(f"/api/jobs/{first['job_id']}/files/not-found.xlsx").status_code == 404
