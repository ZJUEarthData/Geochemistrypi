"""Integration tests for the minimal Online API."""

from dataclasses import replace
import asyncio
import json
from io import BytesIO
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from geochemistrypi.online.app import create_app
from geochemistrypi.online.limits import (
    MAX_CONCURRENT_TASKS,
    MAX_UPLOAD_BYTES,
    TASK_TIMEOUT_SECONDS,
)
from geochemistrypi.online.method_metadata import METHOD_METADATA
from geochemistrypi.online.task_runner import TaskCancelledError, TaskTimeoutError


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


def make_hybrid_rows() -> list[dict]:
    return [
        {
            "Pressure": 2.5,
            "T": 1773.0,
            "SiO2": 36.83,
            "TiO2": 10.56,
            "Al2O3": 9.94,
            "FeO": 18.57,
            "MgO": 9.24,
            "CaO": 12.89,
            "NiO": 0.43,
            "Na2O": 0.34,
            "K2O": 0.03,
            "H2O": 0.0,
            "Fe": 64.06,
            "Ni+Cu+Co": 0.0,
            "S": 34.19,
            "O": 1.64,
        },
        {
            "Pressure": 2.5,
            "T": 1773.0,
            "SiO2": 41.24,
            "TiO2": 4.84,
            "Al2O3": 12.44,
            "FeO": 15.94,
            "MgO": 7.58,
            "CaO": 13.37,
            "NiO": 0.01,
            "Na2O": 0.67,
            "K2O": 0.14,
            "H2O": 0.0,
            "Fe": 64.74,
            "Ni+Cu+Co": 0.0,
            "S": 33.81,
            "O": 1.49,
        },
    ]


def make_named_sheet_workbook(rows: list[dict], sheet_name: str) -> bytes:
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def post_method(
    client: TestClient,
    *,
    task: str,
    method: str,
    element: str,
    content: bytes,
    filename: str | None = None,
):
    upload_name = filename or f"{method}.xlsx"
    return client.post(
        "/api/chemical-modeling/run",
        data={"task": task, "method": method, "element": element},
        files={
            "dataset": (
                upload_name,
                content,
                (
                    "text/csv"
                    if upload_name.lower().endswith(".csv")
                    else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
            )
        },
    )


def post_first_order(
    client: TestClient,
    content: bytes,
    filename: str = "kinetic.xlsx",
    headers: dict[str, str] | None = None,
):
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
                (
                    "text/csv"
                    if filename.lower().endswith(".csv")
                    else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
            )
        },
        headers=headers,
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
    health_payload = health.json()
    assert health_payload["status"] == "ok"
    assert health_payload["service"] == "geochemistrypi-online"
    assert len(health_payload["instance_id"]) == 16
    assert health_payload["source_revision"]
    assert health_payload["build_id"]
    assert health_payload["max_upload_bytes"] == MAX_UPLOAD_BYTES
    assert health_payload["task_timeout_seconds"] == TASK_TIMEOUT_SECONDS
    assert health_payload["max_concurrent_tasks"] == MAX_CONCURRENT_TASKS

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
    assert all(method["status"] == "verified" for method in equilibrium["methods"])
    mass_action = next(method for method in equilibrium["methods"] if method["name"] == "mass_action")
    assert mass_action["required_columns"] == ["K", "stoich", "initial_concentrations"]
    assert mass_action["input_columns"][1]["data_type"] == "string"

    adsorption = next(method for method in kinetic["methods"] if method["name"] == "adsorption_kinetics")
    assert adsorption["status"] == "verified"
    assert adsorption["required_columns"] == ["model", "qe", "k", "t"]

    fractionation = next(
        task for task in catalog.json()["tasks"] if task["name"] == "algo_fractionation"
    )
    internal_standard = next(
        method for method in fractionation["methods"] if method["name"] == "internal_standard"
    )
    assert internal_standard["status"] == "verified"
    assert internal_standard["elements"] == ["Hg"]
    assert internal_standard["required_columns"] == [
        "Label",
        "202Hg",
        "202Hg/198Hg",
        "201Hg/198Hg",
        "200Hg/198Hg",
        "199Hg/198Hg",
    ]
    double_spike = next(
        method for method in fractionation["methods"] if method["name"] == "double_spike"
    )
    assert double_spike["status"] == "verified"
    assert double_spike["elements"] == ["Mo"]
    assert len(double_spike["required_columns"]) == 9

    solubility = next(task for task in catalog.json()["tasks"] if task["name"] == "algo_solubility")
    assert solubility["available"] is True
    rubie = next(method for method in solubility["methods"] if method["name"] == "rubie")
    assert rubie["status"] == "verified"
    assert rubie["required_columns"] == ["Pressure", "T"]
    assert "Laurenz et al. (2016)" in rubie["description"]
    ding = next(method for method in solubility["methods"] if method["name"] == "ding")
    assert ding["status"] == "verified"
    assert ding["required_columns"][-1] == "sulfide_Ni"
    blanchard = next(method for method in solubility["methods"] if method["name"] == "blanchard")
    assert blanchard["status"] == "verified"
    assert blanchard["required_columns"][-3:] == ["Fe", "Ni", "Cu"]
    assert [column["name"] for column in blanchard["input_columns"] if not column["required"]] == [
        "MnO",
        "P2O5",
        "Cr2O3",
    ]
    hybrid = next(method for method in solubility["methods"] if method["name"] == "hybrid")
    assert hybrid["status"] == "verified"
    assert hybrid["required_columns"][-4:] == ["Fe", "Ni+Cu+Co", "S", "O"]
    assert "SCSS" not in hybrid["required_columns"]

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
    advection = next(
        method for method in transport["methods"] if method["name"] == "advection_dispersion"
    )
    assert advection["status"] == "verified"
    assert advection["required_columns"] == ["C0", "v", "D", "x", "t"]

    thermodynamic = next(
        task for task in catalog.json()["tasks"] if task["name"] == "algo_thermodynamic"
    )
    gibbs = next(
        method for method in thermodynamic["methods"] if method["name"] == "gibbs_minimization"
    )
    assert gibbs["status"] == "verified"
    assert gibbs["required_columns"] == [
        "gibbs_energies",
        "stoichiometry",
        "component_totals",
    ]
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


def make_linear_regression_csv(row_count: int = 40) -> bytes:
    rows = ["X1,X2,Target"]
    for value in range(1, row_count + 1):
        second_feature = (value * value) % 11
        target = 3 * value - 2 * second_feature + 5
        rows.append(f"{value},{second_feature},{target}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def make_classification_csv(row_count: int = 40) -> bytes:
    rows = ["X1,X2,Class"]
    offset = row_count // 2
    for value in range(row_count):
        first_feature = value - offset
        second_feature = (value * 3) % 7
        label = "low" if first_feature < 0 else "high"
        rows.append(f"{first_feature},{second_feature},{label}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def make_clustering_csv() -> bytes:
    rows = ["X1,X2"]
    for center_x, center_y in [(-10.0, -10.0), (0.0, 0.0), (10.0, 10.0)]:
        for value in range(10):
            rows.append(
                f"{center_x + value * 0.04},{center_y + (value % 3) * 0.05}"
            )
    return ("\n".join(rows) + "\n").encode("utf-8")


def make_unsupervised_csv() -> bytes:
    rows = ["X1,X2,X3"]
    for value in range(40):
        rows.append(
            f"{(value % 10) * 0.1},{((value * 3) % 11) * 0.1},"
            f"{((value * 7) % 13) * 0.1}"
        )
    rows.extend(["20,20,20", "-20,-18,-22"])
    return ("\n".join(rows) + "\n").encode("utf-8")


def make_time_series_csv() -> bytes:
    rows = ["Age,AgeMax,Probability,Latitude,Longitude"]
    for index in range(30):
        age = index * 10
        probability = 0.2 if index < 15 else 0.8
        rows.append(
            f"{age},{age + 8},{probability},{-60 + index * 4},{-150 + index * 10}"
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def make_element_time_series_csv() -> bytes:
    rows = ["Age,MGO,SIO2"]
    for age in range(0, 400, 20):
        rows.append(f"{age},{6 + age * 0.01},{45 + age * 0.005}")
    rows.append("50,,48")
    return ("\n".join(rows) + "\n").encode("utf-8")


def make_predicted_time_series_csv() -> bytes:
    geochemical_columns = [
        "SIO2",
        "TIO2",
        "AL2O3",
        "MNO",
        "MGO",
        "CAO",
        "NA2O",
        "K2O",
        "P2O5",
        "CR",
        "NI",
        "RB",
        "SR",
        "Y",
        "ZR",
        "NB",
    ]
    rows = [
        ",".join(
            ["Sample", "Age", "AgeMax", "Latitude", "Longitude", *geochemical_columns]
        )
    ]
    for index in range(30):
        geochemistry = [
            48 + (index % 8),
            0.7 + (index % 5) * 0.1,
            14 + (index % 6) * 0.2,
            0.12,
            5.5 - (index % 4) * 0.3,
            8.5 - (index % 3) * 0.2,
            3.1 + (index % 5) * 0.1,
            1.2 + (index % 4) * 0.1,
            0.18,
            120 + index,
            80 + index,
            20 + index,
            350 + index * 2,
            25 + index * 0.2,
            110 + index,
            8 + index * 0.1,
        ]
        rows.append(
            ",".join(
                str(value)
                for value in [
                    f"S{index + 1}",
                    index * 10,
                    index * 10 + 8,
                    -60 + index * 4,
                    -150 + index * 10,
                    *geochemistry,
                ]
            )
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def test_data_mining_catalog_starts_with_verified_dataset_profile(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.get("/api/data-mining/catalog")
    assert response.status_code == 200
    features = response.json()["features"]
    assert [feature["name"] for feature in features] == [
        "dataset_profile",
        "data_preprocessing",
        "regression",
        "classification",
        "clustering",
        "dimensionality_reduction",
        "anomaly_detection",
        "time_series",
    ]
    assert features[0]["status"] == "verified"
    assert features[0]["input_formats"] == [".xlsx", ".csv"]
    assert features[1]["status"] == "verified"
    assert features[1]["outputs"] == [
        "处理结果预览",
        "CSV 处理数据",
        "JSON 处理记录",
    ]
    assert features[2]["status"] == "verified"
    assert features[2]["outputs"] == [
        "回归指标",
        "模型系数（线性模型）",
        "预测结果 CSV",
        "JSON 模型报告",
        "已训练 Pipeline",
        "Application Data 推理",
    ]
    assert [method["name"] for method in features[2]["methods"]] == [
        "linear_regression",
        "polynomial_regression",
        "lasso_regression",
        "elastic_net",
        "bayesian_ridge_regression",
        "ridge_regression",
        "decision_tree",
        "extra_trees",
        "gradient_boosting",
        "k_nearest_neighbors",
        "multi_layer_perceptron",
        "random_forest",
        "stochastic_gradient_descent",
        "support_vector_machine",
        "xgboost",
    ]
    assert all(method["status"] == "verified" for method in features[2]["methods"])
    assert all(feature["status"] == "verified" for feature in features)
    assert features[3]["outputs"] == [
        "分类指标",
        "混淆矩阵",
        "预测结果 CSV",
        "JSON 模型报告",
        "已训练 Pipeline",
        "Application Data 推理",
    ]
    assert [method["name"] for method in features[3]["methods"]] == [
        "logistic_regression",
        "support_vector_machine",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "multi_layer_perceptron",
        "gradient_boosting",
        "k_nearest_neighbors",
        "stochastic_gradient_descent",
        "adaboost",
        "xgboost",
    ]
    assert all(method["status"] == "verified" for method in features[3]["methods"])
    assert features[4]["outputs"] == [
        "聚类指标",
        "簇大小与中心",
        "聚类结果 CSV",
        "JSON 模型报告",
    ]
    assert [method["name"] for method in features[4]["methods"]] == [
        "kmeans",
        "dbscan",
        "agglomerative",
        "affinity_propagation",
        "mean_shift",
        "optics",
    ]
    assert [method["uses_cluster_count"] for method in features[4]["methods"]] == [
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert [method["name"] for method in features[5]["methods"]] == [
        "pca",
        "tsne",
        "mds",
    ]
    assert features[5]["outputs"] == [
        "低维坐标预览",
        "模型诊断指标",
        "降维结果 CSV",
        "JSON 模型报告",
    ]
    assert [method["name"] for method in features[6]["methods"]] == [
        "isolation_forest",
        "local_outlier_factor",
    ]
    assert features[6]["outputs"] == [
        "正常/异常样品统计",
        "异常分数与标签",
        "异常检测结果 CSV",
        "JSON 模型报告",
    ]
    assert features[7]["outputs"] == [
        "陆上玄武岩比例曲线",
        "年龄分箱结果表",
        "时间序列结果 CSV",
        "SVG 矢量图",
        "JSON 分析报告",
    ]


def test_profile_excel_dataset_returns_quality_summary_and_json_report(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    content = make_workbook(
        [
            {"Sample": "A", "Value": 1.0, "Fe": 10.0, "Constant": 1},
            {"Sample": "B", "Value": 2.0, "Fe": None, "Constant": 1},
            {"Sample": "B", "Value": 2.0, "Fe": None, "Constant": 1},
        ]
    )
    response = client.post(
        "/api/data-mining/profile",
        files={"dataset": ("samples.xlsx", content)},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["source_filename"] == "samples.xlsx"
    summary = payload["summary"]
    assert summary["rows"] == 3
    assert summary["columns"] == 4
    assert summary["total_cells"] == 12
    assert summary["missing_cells"] == 2
    assert summary["missing_rate"] == pytest.approx(1 / 6)
    assert summary["duplicate_rows"] == 1
    assert summary["numeric_columns"] == 3
    assert summary["text_columns"] == 1
    assert summary["datetime_columns"] == 0
    assert summary["boolean_columns"] == 0
    assert summary["infinite_cells"] == 0
    assert summary["memory_bytes"] > 0

    value_profile = next(column for column in payload["columns"] if column["name"] == "Value")
    assert value_profile["minimum"] == 1.0
    assert value_profile["maximum"] == 2.0
    assert value_profile["mean"] == pytest.approx(5 / 3)
    assert payload["preview"][1]["Fe"] is None
    assert any("30%" in warning for warning in payload["warnings"])
    assert any("1 行完全重复" in warning for warning in payload["warnings"])

    artifact = payload["artifacts"][0]
    download = client.get(artifact["download_url"])
    assert download.status_code == 200
    report = json.loads(download.content.decode("utf-8"))
    assert report["report_version"] == "dataset-profile-v1"
    assert report["summary"]["rows"] == 3
    assert report["source_filename"] == "samples.xlsx"


def test_profile_csv_dataset_is_supported(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/profile",
        files={
            "dataset": (
                "samples.csv",
                "Sample,Value\nA,1\nB,2\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["rows"] == 2
    assert payload["summary"]["columns"] == 2
    assert payload["summary"]["numeric_columns"] == 1
    assert payload["summary"]["text_columns"] == 1


def test_preprocess_dataset_selects_columns_fills_mean_and_downloads_results(
    tmp_path,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": json.dumps(["Sample", "Value"]),
            "missing_strategy": "fill_mean",
        },
        files={
            "dataset": (
                "samples.csv",
                "Sample,Value,Note\nA,1,x\nB,,x\nC,3,\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["source_filename"] == "samples.csv"
    assert payload["selected_columns"] == ["Sample", "Value"]
    assert payload["missing_strategy"] == "fill_mean"
    assert payload["preview"][1] == {"Sample": "B", "Value": 2.0}

    summary = payload["summary"]
    assert summary == {
        "original_rows": 3,
        "original_columns": 3,
        "processed_rows": 3,
        "processed_columns": 2,
        "removed_rows": 0,
        "removed_columns": 1,
        "original_missing_cells": 1,
        "processed_missing_cells": 0,
        "filled_cells": 1,
    }
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "processed_data.csv",
        "preprocessing_report.json",
    ]

    csv_download = client.get(payload["artifacts"][0]["download_url"])
    assert csv_download.status_code == 200
    processed = pd.read_csv(BytesIO(csv_download.content))
    assert list(processed.columns) == ["Sample", "Value"]
    assert processed["Value"].tolist() == [1.0, 2.0, 3.0]

    report_download = client.get(payload["artifacts"][1]["download_url"])
    assert report_download.status_code == 200
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["report_version"] == "data-preprocessing-v1"
    assert report["selected_columns"] == ["Sample", "Value"]
    assert report["summary"]["filled_cells"] == 1


def test_preprocess_dataset_can_drop_rows_with_missing_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": json.dumps(["Sample", "Value"]),
            "missing_strategy": "drop_rows",
        },
        files={
            "dataset": (
                "samples.csv",
                "Sample,Value\nA,1\nB,\nC,3\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["processed_rows"] == 2
    assert payload["summary"]["removed_rows"] == 1
    assert payload["summary"]["processed_missing_cells"] == 0
    assert [row["Sample"] for row in payload["preview"]] == ["A", "C"]


def test_preprocess_keep_preserves_missing_values_and_column_order(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": json.dumps(["Value", "Sample"]),
            "missing_strategy": "keep",
        },
        files={
            "dataset": (
                "samples.csv",
                "Sample,Value,Unused\nA,1,x\nB,,y\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["selected_columns"] == ["Value", "Sample"]
    assert list(payload["preview"][0]) == ["Value", "Sample"]
    assert payload["preview"][1]["Value"] is None
    assert payload["summary"]["processed_missing_cells"] == 1
    assert payload["summary"]["filled_cells"] == 0
    assert payload["summary"]["removed_columns"] == 1


def test_preprocess_fill_median_only_changes_numeric_missing_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": json.dumps(["Value", "Group"]),
            "missing_strategy": "fill_median",
        },
        files={
            "dataset": (
                "samples.csv",
                "Value,Group\n1,A\n,B\n100,\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["preview"][1]["Value"] == 50.5
    assert payload["preview"][2]["Group"] is None
    assert payload["summary"]["original_missing_cells"] == 2
    assert payload["summary"]["processed_missing_cells"] == 1
    assert payload["summary"]["filled_cells"] == 1
    assert any("非数值缺失单元格" in warning for warning in payload["warnings"])


def test_preprocess_fill_mode_changes_numeric_and_text_missing_values(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": json.dumps(["Value", "Group"]),
            "missing_strategy": "fill_mode",
        },
        files={
            "dataset": (
                "samples.csv",
                "Value,Group\n1,A\n,A\n1,\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["preview"][1] == {"Value": 1.0, "Group": "A"}
    assert payload["preview"][2] == {"Value": 1.0, "Group": "A"}
    assert payload["summary"]["processed_missing_cells"] == 0
    assert payload["summary"]["filled_cells"] == 2
    assert payload["warnings"] == ["处理结果中没有缺失单元格。"]


def test_preprocess_rejects_strategy_that_removes_every_row(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": json.dumps(["Sample", "Value"]),
            "missing_strategy": "drop_rows",
        },
        files={
            "dataset": (
                "samples.csv",
                "Sample,Value\nA,\n,2\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 422
    assert "removes all data rows" in response.json()["detail"]


def test_preprocess_rejects_upload_over_size_limit(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": json.dumps(["Sample"]),
            "missing_strategy": "keep",
        },
        files={
            "dataset": (
                "oversized.csv",
                b"Sample\n" + b"A" * (20 * 1024 * 1024 + 1),
                "text/csv",
            )
        },
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_default_upload_limit_is_20_mib(tmp_path):
    app = create_app(tmp_path / "runtime")

    expected_bytes = 20 * 1024 * 1024
    assert app.state.online_service.max_upload_bytes == expected_bytes
    assert app.state.data_mining_service.max_upload_bytes == expected_bytes


def test_timed_out_calculation_returns_504(tmp_path, monkeypatch):
    app = create_app(tmp_path / "runtime")

    async def reject_timeout(*args, **kwargs):
        raise TaskTimeoutError(
            "Calculation exceeded the 30-minute limit and was stopped."
        )

    monkeypatch.setattr(app.state.task_runner, "run", reject_timeout)
    response = post_first_order(TestClient(app), make_kinetic_workbook())

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "Calculation exceeded the 30-minute limit and was stopped."
    )


def test_task_status_and_cancel_api(tmp_path, monkeypatch):
    app = create_app(tmp_path / "runtime")
    task_id = "44444444-4444-4444-8444-444444444444"

    async def fake_run(operation, *, arguments, tracking_id, task_label):
        assert tracking_id == task_id
        app.state.task_runner._register(tracking_id, task_label)
        while True:
            status_payload = app.state.task_runner.get_status(task_id)
            if status_payload and status_payload["status"] == "cancelled":
                raise TaskCancelledError("Calculation was cancelled.")
            await asyncio.sleep(0.01)

    monkeypatch.setattr(app.state.task_runner, "run", fake_run)
    with TestClient(app) as client:
        response_holder = {}

        def send_request():
            response_holder["response"] = post_first_order(
                client,
                make_kinetic_workbook(),
                headers={"X-Task-ID": task_id},
            )

        request = threading.Thread(target=send_request)
        request.start()
        for _ in range(200):
            status_response = client.get(f"/api/tasks/{task_id}")
            if status_response.status_code == 200:
                break
            time.sleep(0.01)

        assert status_response.json()["status"] == "queued"
        cancelled = client.post(f"/api/tasks/{task_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        request.join(timeout=5)
        assert response_holder["response"].status_code == 409


def test_linear_regression_returns_metrics_coefficients_and_downloads(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/regression",
        data={
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
        },
        files={
            "dataset": (
                "regression.csv",
                make_linear_regression_csv(),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["model"] == "linear_regression"
    assert payload["target_column"] == "Target"
    assert payload["feature_columns"] == ["X1", "X2"]
    assert payload["test_size"] == 0.25
    assert payload["random_state"] == 42
    assert payload["summary"] == {
        "original_rows": 40,
        "usable_rows": 40,
        "dropped_rows": 0,
        "train_rows": 30,
        "test_rows": 10,
        "feature_count": 2,
    }
    assert payload["metrics"]["r2"] == pytest.approx(1.0)
    assert payload["metrics"]["mean_absolute_error"] == pytest.approx(
        0.0,
        abs=1e-10,
    )
    assert payload["metrics"]["root_mean_squared_error"] == pytest.approx(
        0.0,
        abs=1e-10,
    )
    coefficient_map = {
        item["feature"]: item["coefficient"]
        for item in payload["coefficients"]
    }
    assert coefficient_map == pytest.approx({"X1": 3.0, "X2": -2.0})
    assert payload["intercept"] == pytest.approx(5.0)
    assert payload["equation"].startswith("Target = 5")
    assert len(payload["preview"]) == 10
    assert set(payload["preview"][0]) == {
        "source_row",
        "actual",
        "predicted",
        "residual",
    }
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "regression_predictions.csv",
        "regression_report.json",
        "trained_pipeline.joblib",
    ]

    predictions_download = client.get(payload["artifacts"][0]["download_url"])
    assert predictions_download.status_code == 200
    predictions = pd.read_csv(BytesIO(predictions_download.content))
    assert list(predictions.columns) == [
        "source_row",
        "actual",
        "predicted",
        "residual",
    ]
    assert len(predictions) == 10

    report_download = client.get(payload["artifacts"][1]["download_url"])
    assert report_download.status_code == 200
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["report_version"] == "linear-regression-v1"
    assert report["random_state"] == 42
    assert report["metrics"]["r2"] == pytest.approx(1.0)
    assert report["pipeline_artifact"] == "trained_pipeline.joblib"
    pipeline_download = client.get(payload["artifacts"][2]["download_url"])
    assert pipeline_download.status_code == 200
    assert pipeline_download.content


def test_regression_accepts_hyperparameters_and_cross_validation(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/regression",
        data={
            "model": "ridge_regression",
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
            "hyperparameters": json.dumps(
                {"alpha": 0.5, "fit_intercept": True}
            ),
            "cross_validation_folds": "5",
        },
        files={"dataset": ("training.csv", make_linear_regression_csv(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["hyperparameters"] == {"alpha": 0.5, "fit_intercept": True}
    assert payload["cross_validation"]["folds"] == 5
    assert payload["cross_validation"]["strategy"] == "Shuffled K-Fold"
    assert {item["name"] for item in payload["cross_validation"]["metrics"]} == {
        "r2",
        "mean_absolute_error",
        "root_mean_squared_error",
    }


def test_rejects_unsupported_supervised_hyperparameter(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/regression",
        data={
            "model": "linear_regression",
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "hyperparameters": json.dumps({"unsafe_parameter": 1}),
        },
        files={"dataset": ("training.csv", make_linear_regression_csv(), "text/csv")},
    )
    assert response.status_code == 422
    assert "Unsupported hyperparameter" in response.json()["detail"]


def test_model_comparison_ranks_models_and_downloads_reports(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/model-comparison",
        data={
            "task_type": "regression",
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "models": json.dumps(["linear_regression", "ridge_regression"]),
            "hyperparameters": json.dumps(
                {"ridge_regression": {"alpha": 0.5}}
            ),
            "cross_validation_folds": "5",
        },
        files={"dataset": ("training.csv", make_linear_regression_csv(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task_type"] == "regression"
    assert payload["cross_validation_folds"] == 5
    assert payload["comparison_metric"] == "r2"
    assert payload["best_model"] == "linear_regression"
    assert [item["rank"] for item in payload["results"]] == [1, 2]
    assert all(item["status"] == "success" for item in payload["results"])
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "model_comparison.csv",
        "model_comparison_report.json",
    ]
    assert client.get(payload["artifacts"][0]["download_url"]).status_code == 200
    assert client.get(payload["artifacts"][1]["download_url"]).status_code == 200


def test_classification_cross_validation_rejects_too_few_rows_per_class(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/classification",
        data={
            "target_column": "Class",
            "feature_columns": json.dumps(["X1", "X2"]),
            "cross_validation_folds": "10",
        },
        files={"dataset": ("training.csv", make_classification_csv(12), "text/csv")},
    )
    assert response.status_code == 422
    assert "every class must contain at least 10 rows" in response.json()["detail"]


def test_saved_regression_pipeline_predicts_independent_application_data(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    training = client.post(
        "/api/data-mining/regression",
        data={
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
        },
        files={"dataset": ("training.csv", make_linear_regression_csv(), "text/csv")},
    )
    assert training.status_code == 200, training.text
    training_payload = training.json()

    application = client.post(
        "/api/data-mining/inference",
        data={"training_job_id": training_payload["job_id"]},
        files={
            "dataset": (
                "application.csv",
                b"Sample,X1,X2\nA,50,4\nB,51,\nC,,\n",
                "text/csv",
            )
        },
    )
    assert application.status_code == 200, application.text
    payload = application.json()
    assert payload["training_job_id"] == training_payload["job_id"]
    assert payload["task_type"] == "regression"
    assert payload["feature_columns"] == ["X1", "X2"]
    assert payload["prediction_column"] == "predicted_Target"
    assert payload["summary"] == {
        "original_rows": 3,
        "predicted_rows": 2,
        "excluded_rows": 1,
        "imputed_rows": 1,
        "feature_count": 2,
    }
    assert payload["preview"][0]["predicted_Target"] == pytest.approx(147.0)
    assert payload["preview"][0]["inference_status"] == "predicted"
    assert payload["preview"][1]["inference_status"] == "predicted_with_imputation"
    assert payload["preview"][2]["inference_status"] == "excluded_no_numeric_features"
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "application_predictions.csv",
        "application_inference_report.json",
    ]
    predictions_download = client.get(payload["artifacts"][0]["download_url"])
    assert predictions_download.status_code == 200
    predictions = pd.read_csv(BytesIO(predictions_download.content))
    assert list(predictions.columns) == [
        "source_row",
        "Sample",
        "X1",
        "X2",
        "predicted_Target",
        "inference_status",
    ]
    assert len(predictions) == 3


def test_saved_xgboost_classification_pipeline_restores_text_labels(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    training = client.post(
        "/api/data-mining/classification",
        data={
            "model": "xgboost",
            "target_column": "Class",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
        },
        files={"dataset": ("training.csv", make_classification_csv(), "text/csv")},
    )
    assert training.status_code == 200, training.text

    application = client.post(
        "/api/data-mining/inference",
        data={"training_job_id": training.json()["job_id"]},
        files={
            "dataset": (
                "application.csv",
                b"Sample,X1,X2\nA,-10,2\nB,10,2\n",
                "text/csv",
            )
        },
    )
    assert application.status_code == 200, application.text
    payload = application.json()
    assert payload["task_type"] == "classification"
    assert [row["predicted_Class"] for row in payload["preview"]] == [
        "low",
        "high",
    ]


def test_application_inference_rejects_missing_training_feature(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    training = client.post(
        "/api/data-mining/classification",
        data={
            "target_column": "Class",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
        },
        files={"dataset": ("training.csv", make_classification_csv(), "text/csv")},
    )
    assert training.status_code == 200, training.text
    response = client.post(
        "/api/data-mining/inference",
        data={"training_job_id": training.json()["job_id"]},
        files={"dataset": ("application.csv", b"X1\n1\n2\n", "text/csv")},
    )
    assert response.status_code == 422
    assert "missing required feature columns: X2" in response.json()["detail"]


@pytest.mark.parametrize(
    ("model_name", "expected_display_name", "minimum_coefficients"),
    [
        ("linear_regression", "Linear Regression", 2),
        ("polynomial_regression", "Polynomial Regression", 5),
        ("lasso_regression", "Lasso Regression", 2),
        ("elastic_net", "Elastic Net", 2),
        ("bayesian_ridge_regression", "Bayesian Ridge Regression", 2),
        ("ridge_regression", "Ridge Regression", 2),
        ("decision_tree", "Decision Tree", 0),
        ("extra_trees", "Extra-Trees", 0),
        ("gradient_boosting", "Gradient Boosting", 0),
        ("k_nearest_neighbors", "K-Nearest Neighbors", 0),
        ("multi_layer_perceptron", "Multi-layer Perceptron", 0),
        ("random_forest", "Random Forest", 0),
        ("stochastic_gradient_descent", "Stochastic Gradient Descent", 0),
        ("support_vector_machine", "Support Vector Machine", 0),
        ("xgboost", "XGBoost", 0),
    ],
)
def test_v080_regression_model_registry_runs_verified_models(
    tmp_path,
    model_name,
    expected_display_name,
    minimum_coefficients,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/regression",
        data={
            "model": model_name,
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
        },
        files={
            "dataset": (
                "regression.csv",
                make_linear_regression_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == model_name
    assert payload["model_display_name"] == expected_display_name
    assert len(payload["coefficients"]) >= minimum_coefficients
    assert payload["metrics"]["mean_absolute_error"] >= 0
    assert payload["metrics"]["root_mean_squared_error"] >= 0
    if minimum_coefficients:
        assert payload["equation"].startswith("Target =")
        assert payload["intercept"] is not None
    else:
        assert payload["equation"] is None
        assert payload["intercept"] is None

    report = client.get(payload["artifacts"][1]["download_url"])
    assert report.status_code == 200
    report_payload = json.loads(report.content.decode("utf-8"))
    assert report_payload["model"] == model_name
    assert report_payload["model_display_name"] == expected_display_name


def test_reject_unknown_regression_model(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/regression",
        data={
            "model": "unknown_regressor",
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.2",
        },
        files={
            "dataset": (
                "regression.csv",
                make_linear_regression_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 422
    assert "Unknown regression model" in response.json()["detail"]


def test_linear_regression_drops_incomplete_rows_before_split(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    content = make_linear_regression_csv(20).decode("utf-8")
    content = content.replace("5,3,14", "5,,14")
    content = content.replace("10,1,33", "10,1,")
    response = client.post(
        "/api/data-mining/regression",
        data={
            "target_column": "Target",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.2",
        },
        files={
            "dataset": (
                "regression.csv",
                content.encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["original_rows"] == 20
    assert payload["summary"]["usable_rows"] == 18
    assert payload["summary"]["dropped_rows"] == 2
    assert any("删除了 2 行" in warning for warning in payload["warnings"])


@pytest.mark.parametrize(
    ("target", "features", "test_size", "content", "expected_message"),
    [
        (
            "Unknown",
            ["X1"],
            0.2,
            make_linear_regression_csv(),
            "Unknown target column",
        ),
        (
            "Target",
            ["X1", "Target"],
            0.2,
            make_linear_regression_csv(),
            "cannot also be a feature",
        ),
        (
            "Target",
            ["Group"],
            0.2,
            ("Group,Target\n" + "\n".join(f"A,{value}" for value in range(12))).encode(
                "utf-8"
            ),
            "requires numeric columns",
        ),
        (
            "Target",
            ["X1"],
            0.2,
            make_linear_regression_csv(9),
            "at least 10 complete numeric rows",
        ),
        (
            "Target",
            ["X1"],
            0.2,
            (
                "X1,Target\n"
                + "\n".join(f"{value},1" for value in range(1, 13))
            ).encode("utf-8"),
            "at least two distinct values",
        ),
        (
            "Target",
            ["X1"],
            0.05,
            make_linear_regression_csv(),
            "between 0.1 and 0.5",
        ),
    ],
)
def test_reject_invalid_linear_regression_configuration(
    tmp_path,
    target,
    features,
    test_size,
    content,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/regression",
        data={
            "target_column": target,
            "feature_columns": json.dumps(features),
            "test_size": str(test_size),
        },
        files={
            "dataset": (
                "regression.csv",
                content,
                "text/csv",
            )
        },
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


def test_reject_malformed_regression_feature_list(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/regression",
        data={
            "target_column": "Target",
            "feature_columns": "not-json",
            "test_size": "0.2",
        },
        files={
            "dataset": (
                "regression.csv",
                make_linear_regression_csv(),
                "text/csv",
            )
        },
    )
    assert response.status_code == 422
    assert "valid JSON list" in response.json()["detail"]


def test_logistic_classification_returns_metrics_confusion_and_downloads(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/classification",
        data={
            "target_column": "Class",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
        },
        files={
            "dataset": (
                "classification.csv",
                make_classification_csv(),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["model"] == "logistic_regression"
    assert payload["target_column"] == "Class"
    assert payload["feature_columns"] == ["X1", "X2"]
    assert payload["test_size"] == 0.25
    assert payload["random_state"] == 42
    assert payload["classes"] == ["high", "low"]
    assert payload["summary"] == {
        "original_rows": 40,
        "usable_rows": 40,
        "dropped_rows": 0,
        "train_rows": 30,
        "test_rows": 10,
        "feature_count": 2,
        "class_count": 2,
    }
    assert payload["metrics"]["accuracy"] >= 0.9
    assert payload["metrics"]["precision_macro"] >= 0.9
    assert payload["metrics"]["recall_macro"] >= 0.9
    assert payload["metrics"]["f1_macro"] >= 0.9
    assert sum(item["count"] for item in payload["confusion_matrix"]) == 10
    assert len(payload["preview"]) == 10
    assert set(payload["preview"][0]) == {
        "source_row",
        "actual",
        "predicted",
        "correct",
    }
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "classification_predictions.csv",
        "classification_report.json",
        "trained_pipeline.joblib",
    ]

    predictions_download = client.get(payload["artifacts"][0]["download_url"])
    assert predictions_download.status_code == 200
    predictions = pd.read_csv(BytesIO(predictions_download.content))
    assert list(predictions.columns) == [
        "source_row",
        "actual",
        "predicted",
        "correct",
    ]
    assert len(predictions) == 10

    report_download = client.get(payload["artifacts"][1]["download_url"])
    assert report_download.status_code == 200
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["report_version"] == "logistic-classification-v1"
    assert report["random_state"] == 42
    assert report["metrics"]["accuracy"] >= 0.9
    assert report["pipeline_artifact"] == "trained_pipeline.joblib"
    pipeline_download = client.get(payload["artifacts"][2]["download_url"])
    assert pipeline_download.status_code == 200
    assert pipeline_download.content


@pytest.mark.parametrize(
    ("model_name", "expected_display_name"),
    [
        ("logistic_regression", "Logistic Regression"),
        ("support_vector_machine", "Support Vector Machine"),
        ("decision_tree", "Decision Tree"),
        ("random_forest", "Random Forest"),
        ("extra_trees", "Extra-Trees"),
        ("multi_layer_perceptron", "Multi-layer Perceptron"),
        ("gradient_boosting", "Gradient Boosting"),
        ("k_nearest_neighbors", "K-Nearest Neighbors"),
        ("stochastic_gradient_descent", "Stochastic Gradient Descent"),
        ("adaboost", "AdaBoost"),
        ("xgboost", "XGBoost"),
    ],
)
def test_v080_classification_registry_runs_verified_models(
    tmp_path,
    model_name,
    expected_display_name,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/classification",
        data={
            "model": model_name,
            "target_column": "Class",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.25",
        },
        files={
            "dataset": (
                "classification.csv",
                make_classification_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == model_name
    assert payload["model_display_name"] == expected_display_name
    assert payload["classes"] == ["high", "low"]
    assert payload["metrics"]["accuracy"] >= 0.5
    assert payload["metrics"]["f1_macro"] >= 0.5
    assert sum(item["count"] for item in payload["confusion_matrix"]) == 10

    report = client.get(payload["artifacts"][1]["download_url"])
    assert report.status_code == 200
    report_payload = json.loads(report.content.decode("utf-8"))
    assert report_payload["model"] == model_name
    assert report_payload["model_display_name"] == expected_display_name


def test_reject_unknown_classification_model(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/classification",
        data={
            "model": "unknown_classifier",
            "target_column": "Class",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.2",
        },
        files={
            "dataset": (
                "classification.csv",
                make_classification_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 422
    assert "Unknown classification model" in response.json()["detail"]


def test_classification_drops_incomplete_rows_before_split(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    content = make_classification_csv().decode("utf-8")
    content = content.replace("-15,1,low", "-15,,low")
    content = content.replace("15,0,high", "15,0,")
    response = client.post(
        "/api/data-mining/classification",
        data={
            "target_column": "Class",
            "feature_columns": json.dumps(["X1", "X2"]),
            "test_size": "0.2",
        },
        files={"dataset": ("classification.csv", content.encode("utf-8"))},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["original_rows"] == 40
    assert payload["summary"]["usable_rows"] == 38
    assert payload["summary"]["dropped_rows"] == 2
    assert any("删除了 2 行" in warning for warning in payload["warnings"])


@pytest.mark.parametrize(
    ("target", "features", "test_size", "content", "expected_message"),
    [
        (
            "Unknown",
            ["X1"],
            0.2,
            make_classification_csv(),
            "Unknown target column",
        ),
        (
            "Class",
            ["X1", "Class"],
            0.2,
            make_classification_csv(),
            "cannot also be a feature",
        ),
        (
            "Class",
            ["Group"],
            0.2,
            ("Group,Class\n" + "\n".join("A,low" for _ in range(12))).encode(
                "utf-8"
            ),
            "requires numeric feature columns",
        ),
        (
            "Class",
            ["X1"],
            0.2,
            make_classification_csv(10),
            "at least 12 complete rows",
        ),
        (
            "Class",
            ["X1"],
            0.2,
            ("X1,Class\n" + "\n".join(f"{value},one" for value in range(12))).encode(
                "utf-8"
            ),
            "at least two classes",
        ),
        (
            "Class",
            ["X1"],
            0.2,
            (
                "X1,Class\n"
                + "\n".join(
                    f"{value},{'rare' if value == 0 else 'common'}"
                    for value in range(12)
                )
            ).encode("utf-8"),
            "Each target class must contain at least two",
        ),
        (
            "Class",
            ["X1"],
            0.05,
            make_classification_csv(),
            "between 0.1 and 0.5",
        ),
    ],
)
def test_reject_invalid_classification_configuration(
    tmp_path,
    target,
    features,
    test_size,
    content,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/classification",
        data={
            "target_column": target,
            "feature_columns": json.dumps(features),
            "test_size": str(test_size),
        },
        files={"dataset": ("classification.csv", content, "text/csv")},
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


def test_kmeans_clustering_returns_metrics_centers_and_downloads(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/clustering",
        data={
            "feature_columns": json.dumps(["X1", "X2"]),
            "cluster_count": "3",
        },
        files={
            "dataset": (
                "clustering.csv",
                make_clustering_csv(),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["model"] == "kmeans"
    assert payload["model_display_name"] == "K-Means"
    assert payload["feature_columns"] == ["X1", "X2"]
    assert payload["cluster_count"] == 3
    assert payload["requested_cluster_count"] == 3
    assert payload["noise_rows"] == 0
    assert payload["random_state"] == 42
    assert payload["summary"] == {
        "original_rows": 30,
        "usable_rows": 30,
        "dropped_rows": 0,
        "feature_count": 2,
        "cluster_count": 3,
    }
    assert payload["metrics"]["silhouette_score"] > 0.9
    assert payload["metrics"]["davies_bouldin_score"] < 0.1
    assert payload["metrics"]["calinski_harabasz_score"] > 1000
    assert sum(item["rows"] for item in payload["cluster_sizes"]) == 30
    assert [item["rows"] for item in payload["cluster_sizes"]] == [10, 10, 10]
    center_x_values = sorted(
        item["values"]["X1"] for item in payload["cluster_centers"]
    )
    assert center_x_values == pytest.approx([-9.82, 0.18, 10.18])
    assert len(payload["preview"]) == 20
    assert set(payload["preview"][0]) == {"source_row", "X1", "X2", "cluster"}
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "clustering_assignments.csv",
        "clustering_report.json",
    ]

    assignments_download = client.get(payload["artifacts"][0]["download_url"])
    assert assignments_download.status_code == 200
    assignments = pd.read_csv(BytesIO(assignments_download.content))
    assert list(assignments.columns) == ["source_row", "X1", "X2", "cluster"]
    assert len(assignments) == 30

    report_download = client.get(payload["artifacts"][1]["download_url"])
    assert report_download.status_code == 200
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["report_version"] == "kmeans-clustering-v1"
    assert report["random_state"] == 42
    assert report["summary"]["cluster_count"] == 3


@pytest.mark.parametrize(
    ("model_name", "expected_display_name", "uses_cluster_count"),
    [
        ("kmeans", "K-Means", True),
        ("dbscan", "DBSCAN", False),
        ("agglomerative", "Agglomerative Clustering", True),
        ("affinity_propagation", "Affinity Propagation", False),
        ("mean_shift", "Mean Shift", False),
        ("optics", "OPTICS", False),
    ],
)
def test_v080_clustering_registry_runs_verified_models(
    tmp_path,
    model_name,
    expected_display_name,
    uses_cluster_count,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/clustering",
        data={
            "model": model_name,
            "feature_columns": json.dumps(["X1", "X2"]),
            "cluster_count": "3",
        },
        files={
            "dataset": (
                "clustering.csv",
                make_clustering_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == model_name
    assert payload["model_display_name"] == expected_display_name
    assert payload["cluster_count"] >= 2
    assert payload["requested_cluster_count"] == (3 if uses_cluster_count else None)
    assert sum(item["rows"] for item in payload["cluster_sizes"]) == 30
    assert len(payload["cluster_centers"]) == payload["cluster_count"]
    assert payload["metrics"]["silhouette_score"] > -1
    assert payload["metrics"]["davies_bouldin_score"] >= 0
    assert payload["metrics"]["calinski_harabasz_score"] >= 0
    if model_name == "optics":
        assert payload["noise_rows"] > 0
        assert any(item["cluster"] == -1 for item in payload["cluster_sizes"])

    report = client.get(payload["artifacts"][1]["download_url"])
    assert report.status_code == 200
    report_payload = json.loads(report.content.decode("utf-8"))
    assert report_payload["model"] == model_name
    assert report_payload["model_display_name"] == expected_display_name
    assert report_payload["noise_rows"] == payload["noise_rows"]


def test_reject_unknown_clustering_model(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/clustering",
        data={
            "model": "unknown_clusterer",
            "feature_columns": json.dumps(["X1", "X2"]),
            "cluster_count": "3",
        },
        files={
            "dataset": (
                "clustering.csv",
                make_clustering_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 422
    assert "Unknown clustering model" in response.json()["detail"]


def test_clustering_drops_incomplete_rows(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    content = make_clustering_csv().decode("utf-8")
    content = content.replace("-9.8,-9.9", "-9.8,")
    response = client.post(
        "/api/data-mining/clustering",
        data={
            "feature_columns": json.dumps(["X1", "X2"]),
            "cluster_count": "3",
        },
        files={"dataset": ("clustering.csv", content.encode("utf-8"))},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["original_rows"] == 30
    assert payload["summary"]["usable_rows"] == 29
    assert payload["summary"]["dropped_rows"] == 1
    assert any("删除了 1 行" in warning for warning in payload["warnings"])


@pytest.mark.parametrize(
    ("features", "cluster_count", "content", "expected_message"),
    [
        (
            ["X1"],
            1,
            make_clustering_csv(),
            "between 2 and 10",
        ),
        (
            ["Group"],
            2,
            ("Group\n" + "\n".join("A" for _ in range(12))).encode("utf-8"),
            "requires numeric feature columns",
        ),
        (
            ["X1", "X2"],
            3,
            b"X1,X2\n0,0\n1,1\n2,2\n3,3\n4,4\n5,5\n",
            "at least 10 complete numeric rows",
        ),
        (
            ["X1", "X2"],
            3,
            (
                "X1,X2\n"
                + "\n".join(
                    "0,0" if value % 2 == 0 else "1,1"
                    for value in range(12)
                )
            ).encode("utf-8"),
            "at least as many distinct feature rows",
        ),
    ],
)
def test_reject_invalid_clustering_configuration(
    tmp_path,
    features,
    cluster_count,
    content,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/clustering",
        data={
            "feature_columns": json.dumps(features),
            "cluster_count": str(cluster_count),
        },
        files={"dataset": ("clustering.csv", content, "text/csv")},
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


@pytest.mark.parametrize(
    ("model_name", "display_name", "metric_name"),
    [
        ("pca", "PCA", "total_explained_variance_ratio"),
        ("tsne", "T-SNE", "kl_divergence"),
        ("mds", "MDS", "stress"),
    ],
)
def test_v080_dimensionality_reduction_registry_runs_verified_models(
    tmp_path,
    model_name,
    display_name,
    metric_name,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/dimensionality-reduction",
        data={
            "model": model_name,
            "feature_columns": json.dumps(["X1", "X2", "X3"]),
            "component_count": "2",
        },
        files={
            "dataset": (
                "unsupervised.csv",
                make_unsupervised_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == model_name
    assert payload["model_display_name"] == display_name
    assert payload["component_count"] == 2
    assert payload["random_state"] == 42
    assert payload["summary"] == {
        "original_rows": 42,
        "usable_rows": 42,
        "dropped_rows": 0,
        "feature_count": 3,
        "component_count": 2,
    }
    assert payload["metrics"][metric_name] is not None
    assert len(payload["preview"]) == 20
    assert set(payload["preview"][0]) == {
        "source_row",
        "component_1",
        "component_2",
    }
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "dimensionality_reduction_coordinates.csv",
        "dimensionality_reduction_report.json",
    ]

    coordinates_download = client.get(payload["artifacts"][0]["download_url"])
    assert coordinates_download.status_code == 200
    coordinates = pd.read_csv(BytesIO(coordinates_download.content))
    assert list(coordinates.columns) == [
        "source_row",
        "X1",
        "X2",
        "X3",
        "component_1",
        "component_2",
    ]
    assert len(coordinates) == 42

    report_download = client.get(payload["artifacts"][1]["download_url"])
    assert report_download.status_code == 200
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["report_version"] == "v080-dimensionality-reduction-v1"
    assert report["model"] == model_name
    assert report["summary"]["component_count"] == 2


@pytest.mark.parametrize("model_name", ["pca", "tsne", "mds"])
def test_v080_dimensionality_reduction_supports_three_dimensions(
    tmp_path,
    model_name,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/dimensionality-reduction",
        data={
            "model": model_name,
            "feature_columns": json.dumps(["X1", "X2", "X3"]),
            "component_count": "3",
        },
        files={"dataset": ("dataset.csv", make_unsupervised_csv(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["component_count"] == 3
    assert set(payload["preview"][0]) == {
        "source_row",
        "component_1",
        "component_2",
        "component_3",
    }


@pytest.mark.parametrize(
    ("model_name", "display_name"),
    [
        ("isolation_forest", "Isolation Forest"),
        ("local_outlier_factor", "Local Outlier Factor"),
    ],
)
def test_v080_anomaly_detection_registry_runs_verified_models(
    tmp_path,
    model_name,
    display_name,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/anomaly-detection",
        data={
            "model": model_name,
            "feature_columns": json.dumps(["X1", "X2", "X3"]),
        },
        files={
            "dataset": (
                "unsupervised.csv",
                make_unsupervised_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == model_name
    assert payload["model_display_name"] == display_name
    assert payload["random_state"] == (
        42 if model_name == "isolation_forest" else None
    )
    assert payload["summary"]["original_rows"] == 42
    assert payload["summary"]["usable_rows"] == 42
    assert payload["summary"]["dropped_rows"] == 0
    assert payload["summary"]["feature_count"] == 3
    assert (
        payload["summary"]["normal_rows"] + payload["summary"]["anomaly_rows"]
        == 42
    )
    assert payload["summary"]["anomaly_rows"] >= 1
    assert payload["score_summary"]["maximum"] >= payload["score_summary"]["mean"]
    assert payload["score_summary"]["mean"] >= payload["score_summary"]["minimum"]
    assert len(payload["preview"]) == 20
    assert set(payload["preview"][0]) == {
        "source_row",
        "anomaly_label",
        "is_anomaly",
        "anomaly_score",
    }
    preview_scores = [row["anomaly_score"] for row in payload["preview"]]
    assert preview_scores == sorted(preview_scores, reverse=True)
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "anomaly_detection_results.csv",
        "anomaly_detection_report.json",
    ]

    results_download = client.get(payload["artifacts"][0]["download_url"])
    assert results_download.status_code == 200
    results = pd.read_csv(BytesIO(results_download.content))
    assert list(results.columns) == [
        "source_row",
        "X1",
        "X2",
        "X3",
        "anomaly_label",
        "is_anomaly",
        "anomaly_score",
    ]
    assert len(results) == 42

    report_download = client.get(payload["artifacts"][1]["download_url"])
    assert report_download.status_code == 200
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["report_version"] == "v080-anomaly-detection-v1"
    assert report["model"] == model_name
    assert report["summary"]["anomaly_rows"] == payload["summary"]["anomaly_rows"]


def test_v080_time_series_returns_bins_figure_and_downloads(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/time-series",
        data={
            "age_column": "Age",
            "age_max_column": "AgeMax",
            "probability_column": "Probability",
            "latitude_column": "Latitude",
            "longitude_column": "Longitude",
            "age_unit": "Ma",
            "bin_width": "50",
            "bootstrap_iterations": "20",
        },
        files={
            "dataset": (
                "time-series.csv",
                make_time_series_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["age_unit"] == "Ma"
    assert payload["bin_width"] == 50
    assert payload["bootstrap_iterations"] == 20
    assert payload["random_state"] == 2025
    assert payload["probability_source"] == "uploaded"
    assert payload["probability_model"] is None
    assert payload["summary"] == {
        "original_rows": 30,
        "usable_rows": 30,
        "dropped_rows": 0,
        "sampled_out_rows": 0,
        "bin_count": 6,
        "populated_bins": 6,
    }
    assert [item["age"] for item in payload["bins"]] == [
        25.0,
        75.0,
        125.0,
        175.0,
        225.0,
        275.0,
    ]
    assert all(item["mean_proportion"] is not None for item in payload["bins"])
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "subaerial_proportion.csv",
        "subaerial_proportion.svg",
        "time_series_report.json",
    ]

    csv_download = client.get(payload["artifacts"][0]["download_url"])
    assert csv_download.status_code == 200
    results = pd.read_csv(BytesIO(csv_download.content))
    assert list(results.columns) == [
        "age",
        "mean_proportion",
        "uncertainty_2sigma",
    ]
    assert len(results) == 6

    svg_download = client.get(payload["artifacts"][1]["download_url"])
    assert svg_download.status_code == 200
    assert b"<svg" in svg_download.content
    assert b"Estimated proportion of subaerial basalts" in svg_download.content

    report_download = client.get(payload["artifacts"][2]["download_url"])
    assert report_download.status_code == 200
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["report_version"] == "v080-time-series-v1"
    assert report["column_mapping"]["age"] == "Age"
    assert report["probability_source"] == "uploaded"
    assert report["summary"]["populated_bins"] == 6


def test_element_mean_time_series_returns_100_ma_bins_and_2sem(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/time-series/element-mean",
        data={
            "age_column": "Age",
            "value_column": "MGO",
            "age_unit": "Ma",
            "bin_width": "100",
            "value_unit": "wt%",
            "filter_column": "SIO2",
            "filter_min": "43",
            "filter_max": "51",
        },
        files={
            "dataset": (
                "element-series.csv",
                make_element_time_series_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["analysis_type"] == "element_mean"
    assert payload["value_column"] == "MGO"
    assert payload["value_unit"] == "wt%"
    assert payload["uncertainty_method"] == "2_sem"
    assert payload["bin_width"] == 100
    assert payload["bootstrap_iterations"] == 0
    assert payload["random_state"] is None
    assert payload["summary"] == {
        "original_rows": 21,
        "usable_rows": 20,
        "dropped_rows": 1,
        "sampled_out_rows": 0,
        "bin_count": 4,
        "populated_bins": 4,
    }
    assert [item["age"] for item in payload["bins"]] == [50, 150, 250, 350]
    assert [item["sample_count"] for item in payload["bins"]] == [5, 5, 5, 5]
    assert payload["bins"][0]["mean_proportion"] == pytest.approx(6.4)
    assert payload["bins"][0]["uncertainty_2sigma"] == pytest.approx(
        2 * pd.Series([6.0, 6.2, 6.4, 6.6, 6.8]).std() / 5**0.5
    )
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "element_mean_time_series.csv",
        "element_mean_time_series.svg",
        "element_mean_time_series_report.json",
    ]

    results = pd.read_csv(
        BytesIO(client.get(payload["artifacts"][0]["download_url"]).content)
    )
    assert list(results.columns) == [
        "age",
        "mean_value",
        "uncertainty_2sem",
        "sample_count",
    ]
    assert len(results) == 4
    svg = client.get(payload["artifacts"][1]["download_url"])
    assert b"MGO mean through time" in svg.content
    report = client.get(payload["artifacts"][2]["download_url"]).json()
    assert report["report_version"] == "element-mean-time-series-v1"
    assert report["filter"] == {
        "column": "SIO2",
        "minimum": 43.0,
        "maximum": 51.0,
    }


def test_model_predicted_time_series_is_versioned_and_auditable(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/time-series/predict",
        data={
            "age_column": "Age",
            "age_max_column": "AgeMax",
            "latitude_column": "Latitude",
            "longitude_column": "Longitude",
            "age_unit": "Ma",
            "bin_width": "50",
            "bootstrap_iterations": "20",
        },
        files={
            "dataset": (
                "raw-geochemistry.csv",
                make_predicted_time_series_csv(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["probability_source"] == "model_predicted"
    assert payload["probability_model"]["version"] == "liu-2024-surrogate-hgbr-v1"
    assert payload["probability_model"]["metrics"]["r2"] > 0.8
    assert payload["prediction_summary"] == {
        "predicted_rows": 30,
        "insufficient_feature_rows": 0,
        "eligible_time_series_rows": 30,
        "sampled_time_series_rows": 30,
        "minimum_features_per_row": 12,
    }
    assert payload["summary"]["sampled_out_rows"] == 0
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "subaerial_proportion.csv",
        "subaerial_proportion.svg",
        "time_series_report.json",
        "predicted_subaerial_probabilities.csv",
    ]

    prediction_download = client.get(payload["artifacts"][3]["download_url"])
    assert prediction_download.status_code == 200
    predictions = pd.read_csv(BytesIO(prediction_download.content))
    assert len(predictions) == 30
    assert predictions["predicted_subaerial_probability"].between(0, 1).all()
    assert predictions["model_version"].eq("liu-2024-surrogate-hgbr-v1").all()

    report_download = client.get(payload["artifacts"][2]["download_url"])
    report = json.loads(report_download.content.decode("utf-8"))
    assert report["probability_source"] == "model_predicted"
    assert report["probability_model"]["version"] == "liu-2024-surrogate-hgbr-v1"


@pytest.mark.parametrize(
    ("updates", "expected_message"),
    [
        ({"age_unit": "ka"}, "Age unit must be Ma or Ga"),
        ({"bin_width": "0"}, "Bin width must be a positive finite number"),
        (
            {"bin_width": "0.01"},
            "The selected bin width would create more than 5,000 age bins",
        ),
        (
            {"bootstrap_iterations": "5"},
            "Bootstrap iterations must be between 10 and 1,000",
        ),
        (
            {"probability_column": "Missing"},
            "Unknown selected columns: Missing",
        ),
    ],
)
def test_time_series_rejects_invalid_configuration(
    tmp_path,
    updates,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    data = {
        "age_column": "Age",
        "age_max_column": "AgeMax",
        "probability_column": "Probability",
        "latitude_column": "Latitude",
        "longitude_column": "Longitude",
        "age_unit": "Ma",
        "bin_width": "50",
        "bootstrap_iterations": "20",
        **updates,
    }
    response = client.post(
        "/api/data-mining/time-series",
        data=data,
        files={"dataset": ("time-series.csv", make_time_series_csv(), "text/csv")},
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


@pytest.mark.parametrize(
    ("endpoint", "data", "expected_message"),
    [
        (
            "/api/data-mining/dimensionality-reduction",
            {
                "model": "unknown_reducer",
                "feature_columns": json.dumps(["X1", "X2"]),
                "component_count": "2",
            },
            "Unknown dimensionality reduction model",
        ),
        (
            "/api/data-mining/dimensionality-reduction",
            {
                "model": "pca",
                "feature_columns": json.dumps(["X1", "X2", "X3"]),
                "component_count": "4",
            },
            "Component count must be 2 or 3",
        ),
        (
            "/api/data-mining/dimensionality-reduction",
            {
                "model": "pca",
                "feature_columns": json.dumps(["X1", "X2"]),
                "component_count": "3",
            },
            "Component count cannot exceed the number of selected features",
        ),
        (
            "/api/data-mining/anomaly-detection",
            {
                "model": "unknown_detector",
                "feature_columns": json.dumps(["X1", "X2"]),
            },
            "Unknown anomaly detection model",
        ),
    ],
)
def test_reject_invalid_v080_unsupervised_model_configuration(
    tmp_path,
    endpoint,
    data,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        endpoint,
        data=data,
        files={"dataset": ("dataset.csv", make_unsupervised_csv(), "text/csv")},
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/data-mining/classification",
        "/api/data-mining/clustering",
        "/api/data-mining/dimensionality-reduction",
        "/api/data-mining/anomaly-detection",
    ],
)
def test_reject_malformed_modeling_feature_list(tmp_path, endpoint):
    client = TestClient(create_app(tmp_path / "runtime"))
    data = {"feature_columns": "not-json"}
    if endpoint.endswith("classification"):
        data.update({"target_column": "Class", "test_size": "0.2"})
    elif endpoint.endswith("clustering"):
        data.update({"cluster_count": "3"})
    response = client.post(
        endpoint,
        data=data,
        files={"dataset": ("dataset.csv", make_classification_csv())},
    )
    assert response.status_code == 422
    assert "valid JSON list" in response.json()["detail"]


@pytest.mark.parametrize(
    ("selected_columns", "missing_strategy", "expected_message"),
    [
        ("not-json", "keep", "valid JSON list"),
        (json.dumps([]), "keep", "Select at least one column"),
        (json.dumps(["Unknown"]), "keep", "Unknown selected columns"),
        (json.dumps(["Sample"]), "not-a-strategy", "Unknown missing-value strategy"),
    ],
)
def test_reject_invalid_data_preprocessing_options(
    tmp_path,
    selected_columns,
    missing_strategy,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/preprocess",
        data={
            "selected_columns": selected_columns,
            "missing_strategy": missing_strategy,
        },
        files={
            "dataset": (
                "samples.csv",
                "Sample,Value\nA,1\nB,2\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


@pytest.mark.parametrize(
    ("filename", "content", "expected_message"),
    [
        ("samples.txt", b"a,b\n1,2\n", "supports .xlsx and .csv"),
        ("samples.csv", b"", "uploaded file is empty"),
        ("samples.csv", b"\xff\xfe\x00", "CSV files must use UTF-8 encoding"),
        ("samples.csv", b"a,b\n", "dataset contains no data rows"),
    ],
)
def test_reject_invalid_data_mining_uploads(
    tmp_path,
    filename,
    content,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/data-mining/profile",
        files={"dataset": (filename, content)},
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


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


def test_run_and_download_first_order_csv_result(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    content = (
        pd.DataFrame([{"c0": 100.0, "k": 0.1, "t": 5.0}])
        .to_csv(index=False)
        .encode("utf-8")
    )

    response = post_first_order(client, content, filename="kinetic.csv")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    result = pd.read_excel(
        BytesIO(client.get(payload["artifacts"][0]["download_url"]).content)
    )
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


def test_run_mass_balance_with_dynamic_species_columns(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    workbook = make_workbook(
        [
            {"total_mass": 0.2, "Na+": 0.1, "Cl-": 0.1},
            {"total_mass": 0.25, "Na+": 0.1, "Cl-": 0.1},
        ]
    )
    response = post_method(
        client,
        task="algo_equilibrium",
        method="mass_balance",
        element="Any",
        content=workbook,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["artifacts"][0]["name"] == "mass_balance_results.xlsx"
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["species_sum"].tolist() == pytest.approx([0.2, 0.2])
    assert result["mass_difference"].tolist() == pytest.approx([0.0, -0.05])
    assert result["is_balanced"].tolist() == [True, False]


def test_run_precipitation_dissolution_matches_log10_reference(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    workbook = make_workbook(
        [
            {"ion_activity_product": 1e-8, "ksp": 1e-9},
            {"ion_activity_product": 1e-9, "ksp": 1e-9},
            {"ion_activity_product": 1e-10, "ksp": 1e-9},
        ]
    )
    response = post_method(
        client,
        task="algo_equilibrium",
        method="precipitation_dissolution",
        element="Any",
        content=workbook,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["saturation_index"].tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert result["state"].tolist() == ["precipitation", "equilibrium", "dissolution"]


def test_run_ion_exchange_matches_gaines_thomas_reference(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    workbook = make_workbook(
        [
            {"eq_conc_a": 0.05, "eq_conc_b": 0.05, "selectivity": 1.2},
            {"eq_conc_a": 0.0, "eq_conc_b": 1.0, "selectivity": 2.0},
        ]
    )
    response = post_method(
        client,
        task="algo_equilibrium",
        method="ion_exchange",
        element="Any",
        content=workbook,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["exchange_ratio"].tolist() == pytest.approx([1.2, 0.0])


def test_run_mass_action_matches_analytic_a_to_b_solution(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    workbook = make_workbook(
        [
            {
                "K": 4.0,
                "stoich": '{"A":-1,"B":1}',
                "initial_concentrations": '{"A":1.0,"B":0.0}',
            },
            {
                "K": 1.0,
                "stoich": '{"A":-1,"B":1}',
                "initial_concentrations": '{"A":1.0,"B":1.0}',
            },
        ]
    )
    response = post_method(
        client,
        task="algo_equilibrium",
        method="mass_action",
        element="Any",
        content=workbook,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    first = json.loads(result.loc[0, "equilibrium_concentrations"])
    second = json.loads(result.loc[1, "equilibrium_concentrations"])
    assert first == pytest.approx({"A": 0.2, "B": 0.8}, abs=1e-10)
    assert second == pytest.approx({"A": 1.0, "B": 1.0}, abs=1e-10)


def test_run_both_adsorption_kinetics_models(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    workbook = make_workbook(
        [
            {"model": "first", "qe": 50.0, "k": 0.2, "t": 5.0},
            {"model": "second", "qe": 50.0, "k": 0.01, "t": 5.0},
        ]
    )
    response = post_method(
        client,
        task="algo_kinetic",
        method="adsorption_kinetics",
        element="Any",
        content=workbook,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["q_t"].tolist() == pytest.approx(
        [31.60602794142788, 35.714285714285715]
    )


def test_run_advection_dispersion_matches_independent_reference(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    workbook = make_workbook(
        [
            {"C0": 100.0, "v": 1.0, "D": 0.5, "x": 2.0, "t": 1.0},
            {"C0": 100.0, "v": 1.0, "D": 0.5, "x": 0.0, "t": 0.0},
        ]
    )
    response = post_method(
        client,
        task="algo_transport",
        method="advection_dispersion",
        element="Any",
        content=workbook,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["C_xt"].tolist() == pytest.approx([24.19707245191434, 0.0])


def test_run_hg_internal_standard_matches_bracketed_reference(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    workbook = make_workbook(
        [
            {
                "Label": 3133.0,
                "202Hg": 100.0,
                "202Hg/198Hg": 1.0,
                "201Hg/198Hg": 1.0,
                "200Hg/198Hg": 1.0,
                "199Hg/198Hg": 1.0,
            },
            {
                "Label": 1.5,
                "202Hg": 110.0,
                "202Hg/198Hg": 1.1,
                "201Hg/198Hg": 1.1,
                "200Hg/198Hg": 1.1,
                "199Hg/198Hg": 1.1,
            },
            {
                "Label": 3133.0,
                "202Hg": 100.0,
                "202Hg/198Hg": 1.0,
                "201Hg/198Hg": 1.0,
                "200Hg/198Hg": 1.0,
                "199Hg/198Hg": 1.0,
            },
        ]
    )
    response = post_method(
        client,
        task="algo_fractionation",
        method="internal_standard",
        element="Hg",
        content=workbook,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["artifacts"][0]["name"] == "Hg_results.xlsx"
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    sample = result.iloc[1]
    assert sample["THg(%)"] == pytest.approx(10.0)
    assert sample["d202(‰)"] == pytest.approx(100.0)
    assert sample["d201(‰)"] == pytest.approx(100.0)
    assert sample["d200(‰)"] == pytest.approx(100.0)
    assert sample["d199(‰)"] == pytest.approx(100.0)
    assert sample["D199"] == pytest.approx(74.8)
    assert sample["D200"] == pytest.approx(49.76)
    assert sample["D201"] == pytest.approx(24.8)


@pytest.mark.parametrize("source_format", ["xlsx", "csv"])
def test_run_mo_double_spike_recovers_known_parameters(tmp_path, source_format):
    client = TestClient(create_app(tmp_path / "runtime"))
    masses = (100, 98, 97)
    spike = (0.5, 2.0, 0.7)
    standard = (0.1, 1.5, 0.6)
    phi_ref, beta_sample, beta_mix = 0.35, 0.2, -0.15
    mixture = tuple(
        (
            phi_ref * spike_ratio
            + (1 - phi_ref) * standard_ratio * (95 / mass) ** beta_sample
        )
        / (95 / mass) ** beta_mix
        for mass, spike_ratio, standard_ratio in zip(masses, spike, standard)
    )
    rows = [
        {
            "R_100_sp": spike[0],
            "R_98_sp": spike[1],
            "R_97_sp": spike[2],
            "R_100_std": standard[0],
            "R_98_std": standard[1],
            "R_97_std": standard[2],
            "r_100_mix": mixture[0],
            "r_98_mix": mixture[1],
            "r_97_mix": mixture[2],
        }
    ]
    content = (
        make_named_sheet_workbook(rows, "3程序处理_输入常数")
        if source_format == "xlsx"
        else pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    )
    response = post_method(
        client,
        task="algo_fractionation",
        method="double_spike",
        element="Mo",
        content=content,
        filename=f"double-spike.{source_format}",
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["artifacts"][0]["name"] == "Mo_results.csv"
    result = pd.read_csv(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result.loc[0, "phi_ref"] == pytest.approx(phi_ref, abs=1e-8)
    assert result.loc[0, "beta_sple"] == pytest.approx(beta_sample, abs=1e-8)
    assert result.loc[0, "beta_mix"] == pytest.approx(beta_mix, abs=1e-8)


def test_run_laurenz_scss_matches_published_equation(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task="algo_solubility",
        method="rubie",
        element="S",
        content=make_workbook(
            [
                {"Pressure": 10.0, "T": 2500.0},
                {"Pressure": 7.0, "T": 2373.0},
            ]
        ),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["artifacts"][0]["name"] == "solubility_rubie_results.xlsx"
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["SCSS_pred"].tolist() == pytest.approx(
        [3909.6377436328207, 4596.444496580791]
    )


def test_run_gibbs_minimization_matches_analytic_component_balances(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task="algo_thermodynamic",
        method="gibbs_minimization",
        element="Any",
        content=make_workbook(
            [
                {
                    "gibbs_energies": '{"A":0,"B":-10}',
                    "stoichiometry": '{"A":{"X":1},"B":{"X":1}}',
                    "component_totals": '{"X":2.5}',
                },
                {
                    "gibbs_energies": '{"A":0,"B":0,"AB":-100}',
                    "stoichiometry": (
                        '{"A":{"X":1},"B":{"Y":1},"AB":{"X":1,"Y":1}}'
                    ),
                    "component_totals": '{"X":1.5,"Y":1}',
                },
            ]
        ),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))

    first_moles = json.loads(result.loc[0, "equilibrium_moles"])
    second_moles = json.loads(result.loc[1, "equilibrium_moles"])
    assert result["minimum_gibbs"].tolist() == pytest.approx([-25.0, -100.0])
    assert first_moles == pytest.approx({"A": 0.0, "B": 2.5})
    assert second_moles == pytest.approx({"A": 0.5, "B": 0.0, "AB": 1.0})
    assert result["max_balance_residual"].tolist() == pytest.approx([0.0, 0.0])


def test_run_ding_scss_applies_cation_fractions_and_ni_correction(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    base_row = {
        "Pressure": 1.5,
        "T": 1873.15,
        "SiO2": 43.8,
        "TiO2": 5.0,
        "Al2O3": 10.0,
        "FeO": 18.7,
        "MgO": 8.0,
        "CaO": 11.0,
        "Na2O": 2.0,
        "K2O": 0.5,
    }
    response = post_method(
        client,
        task="algo_solubility",
        method="ding",
        element="S",
        content=make_workbook(
            [
                {**base_row, "sulfide_Ni": 0.0},
                {**base_row, "sulfide_Ni": 30.0},
            ]
        ),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["SCSS_Ni_free"].tolist() == pytest.approx(
        [4844.478244059477, 4844.478244059477]
    )
    assert result["Ni_correction_factor"].tolist() == pytest.approx([1.0, 1.843])
    assert result["SCSS_pred"].tolist() == pytest.approx(
        [4844.478244059477, 2628.5828779487124]
    )


def test_run_blanchard_scss_matches_both_published_equations(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    base_row = {
        "Pressure": 10.0,
        "T": 2473.0,
        "SiO2": 45.8,
        "TiO2": 0.21,
        "Al2O3": 4.53,
        "FeO": 8.17,
        "MgO": 37.09,
        "CaO": 3.68,
        "Na2O": 0.0,
        "K2O": 0.0,
        "H2O": 0.01,
        "MnO": 0.14,
        "P2O5": 0.0,
        "Cr2O3": 0.37,
    }
    response = post_method(
        client,
        task="algo_solubility",
        method="blanchard",
        element="S",
        content=make_workbook(
            [
                {**base_row, "Fe": 60.0, "Ni": 0.0, "Cu": 0.0},
                {**base_row, "Fe": 45.0, "Ni": 10.0, "Cu": 5.0},
            ]
        ),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["sulfide_Fe_fraction"].tolist() == pytest.approx(
        [1.0, 0.7638931875313235]
    )
    assert result["SCSS_eq11_ppm"].tolist() == pytest.approx(
        [5014.681461113733, 3830.681005784403]
    )
    assert result["SCSS_eq12_ppm"].tolist() == pytest.approx(
        [3865.6382421113226, 2952.9347186093996]
    )
    assert result["SCSS_pred"].tolist() == pytest.approx(
        result["SCSS_eq11_ppm"].tolist()
    )


def test_run_hybrid_uses_versioned_model_for_independent_batch_predictions(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task="algo_solubility",
        method="hybrid",
        element="S",
        content=make_workbook(make_hybrid_rows()),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    result = pd.read_excel(BytesIO(client.get(payload["artifacts"][0]["download_url"]).content))
    assert result["hybrid_model_version"].tolist() == [
        "zhangzhou2024-hybrid-rf-v1",
        "zhangzhou2024-hybrid-rf-v1",
    ]
    assert result["RF_base_pred_ppm"].tolist() == pytest.approx(
        [4530.142584522737, 3681.79391210459]
    )
    assert result["PT_correction_factor"].tolist() == pytest.approx(
        [1.149252435749123, 1.149252435749123]
    )
    assert result["SCSS_pred"].tolist() == pytest.approx(
        [5206.277399553583, 4231.310621412493]
    )
    assert result.loc[0, "SCSS_pred"] != result.loc[1, "SCSS_pred"]


@pytest.mark.parametrize(
    ("updates", "expected_message"),
    [
        (
            {"Pressure": 24.1},
            "Column 'Pressure' values must be between 0.0001 and 24",
        ),
        (
            {"T": 1400.0},
            "Column 'T' values must be between 1423 and 2623",
        ),
        (
            {"SiO2": 20.0},
            "Column 'SiO2' values must be between 27.712 and 77.9",
        ),
        (
            {"Fe": 80.0, "Ni+Cu+Co": 10.0, "S": 20.0, "O": 0.0},
            "Hybrid sulfide Fe + Ni+Cu+Co + S + O totals must be between 75 and 105 wt.%",
        ),
    ],
)
def test_reject_hybrid_inputs_outside_training_domain(
    tmp_path,
    updates,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    row = {**make_hybrid_rows()[0], **updates}
    response = post_method(
        client,
        task="algo_solubility",
        method="hybrid",
        element="S",
        content=make_workbook([row]),
    )
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("task", "method", "rows", "expected_message"),
    [
        (
            "algo_equilibrium",
            "mass_balance",
            [{"total_mass": 1.0}],
            "Mass balance requires at least one species concentration column",
        ),
        (
            "algo_equilibrium",
            "mass_balance",
            [{"total_mass": 1.0, "A": -0.1}],
            "Species column 'A' values must be greater than or equal to 0",
        ),
        (
            "algo_equilibrium",
            "precipitation_dissolution",
            [{"ion_activity_product": 0.0, "ksp": 1e-9}],
            "Column 'ion_activity_product' values must be greater than 0",
        ),
        (
            "algo_equilibrium",
            "ion_exchange",
            [{"eq_conc_a": 0.1, "eq_conc_b": 0.0, "selectivity": 1.0}],
            "Column 'eq_conc_b' values must be greater than 0",
        ),
        (
            "algo_equilibrium",
            "mass_action",
            [{"K": 4.0, "stoich": "not-json", "initial_concentrations": '{"A":1,"B":0}'}],
            "Row 2:",
        ),
        (
            "algo_kinetic",
            "adsorption_kinetics",
            [{"model": "unknown", "qe": 50.0, "k": 0.2, "t": 5.0}],
            "Column 'model' must contain either 'first' or 'second'",
        ),
        (
            "algo_transport",
            "advection_dispersion",
            [{"C0": 100.0, "v": 1.0, "D": 0.0, "x": 1.0, "t": 1.0}],
            "Column 'D' values must be greater than 0",
        ),
    ],
)
def test_reject_newly_verified_method_boundaries(
    tmp_path,
    task,
    method,
    rows,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task=task,
        method=method,
        element="Any",
        content=make_workbook(rows),
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


def test_reject_unsupported_chemical_modeling_upload(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_kinetic",
            "method": "first_order",
            "element": "Any",
        },
        files={"dataset": ("data.txt", b"c0,k,t\n100,0.1,5", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Chemical Modeling supports only .xlsx and .csv files"


def test_reject_non_utf8_chemical_modeling_csv(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_first_order(client, b"\xff\xfe\x00", filename="data.csv")
    assert response.status_code == 422
    assert response.json()["detail"] == "CSV files must use UTF-8 encoding"


def test_reject_unreadable_xlsx(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_first_order(client, b"this is not an Excel workbook")
    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a readable .xlsx workbook"


def test_reject_workbook_with_missing_columns(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_first_order(client, make_workbook([{"c0": 100, "t": 5}]))
    assert response.status_code == 422
    assert response.json()["detail"] == "Missing required dataset columns: k"


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
    ("rows", "expected_message"),
    [
        (
            [
                {
                    "Label": "sample-1",
                    "202Hg": 100.0,
                    "202Hg/198Hg": 1.0,
                    "201Hg/198Hg": 1.0,
                    "200Hg/198Hg": 1.0,
                    "199Hg/198Hg": 1.0,
                }
            ],
            "Hg internal standard requires at least two rows with Label '3133'",
        ),
        (
            [
                {
                    "Label": "sample-before-standard",
                    "202Hg": 100.0,
                    "202Hg/198Hg": 1.0,
                    "201Hg/198Hg": 1.0,
                    "200Hg/198Hg": 1.0,
                    "199Hg/198Hg": 1.0,
                },
                {
                    "Label": "3133",
                    "202Hg": 100.0,
                    "202Hg/198Hg": 1.0,
                    "201Hg/198Hg": 1.0,
                    "200Hg/198Hg": 1.0,
                    "199Hg/198Hg": 1.0,
                },
                {
                    "Label": "3133",
                    "202Hg": 100.0,
                    "202Hg/198Hg": 1.0,
                    "201Hg/198Hg": 1.0,
                    "200Hg/198Hg": 1.0,
                    "199Hg/198Hg": 1.0,
                },
            ],
            "Row 2: each sample must be bracketed by Label '3133' rows",
        ),
    ],
)
def test_reject_invalid_hg_internal_standard_bracketing(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task="algo_fractionation",
        method="internal_standard",
        element="Hg",
        content=make_workbook(rows),
    )
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


def test_reject_mo_double_spike_without_required_sheet(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task="algo_fractionation",
        method="double_spike",
        element="Mo",
        content=make_workbook([{"R_100_sp": 0.5}]),
    )
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Mo double-spike requires worksheet '3程序处理_输入常数'"
    )


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        (
            [{"Pressure": 10.0, "T": 0.0}],
            "Column 'T' values must be greater than 0",
        ),
        (
            [{"Pressure": -1.0, "T": 2500.0}],
            "Column 'Pressure' values must be greater than or equal to 0",
        ),
    ],
)
def test_reject_invalid_laurenz_scss_inputs(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task="algo_solubility",
        method="rubie",
        element="S",
        content=make_workbook(rows),
    )
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("rows", "expected_message"),
    [
        (
            [
                {
                    "gibbs_energies": "not-json",
                    "stoichiometry": '{"A":{"X":1}}',
                    "component_totals": '{"X":1}',
                }
            ],
            "Row 2:",
        ),
        (
            [
                {
                    "gibbs_energies": '{"A":0,"B":-10}',
                    "stoichiometry": '{"A":{"X":1}}',
                    "component_totals": '{"X":1}',
                }
            ],
            "gibbs_energies and stoichiometry must contain the same species",
        ),
        (
            [
                {
                    "gibbs_energies": '{"A":0}',
                    "stoichiometry": '{"A":{"X":1,"Y":0}}',
                    "component_totals": '{"X":1,"Y":1}',
                }
            ],
            "Gibbs minimization failed:",
        ),
    ],
)
def test_reject_invalid_gibbs_minimization_inputs(tmp_path, rows, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    response = post_method(
        client,
        task="algo_thermodynamic",
        method="gibbs_minimization",
        element="Any",
        content=make_workbook(rows),
    )
    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


@pytest.mark.parametrize(
    ("updates", "expected_message"),
    [
        (
            {"T": 1400.0},
            "Column 'T' values must be between 1473.15 and 2073.15",
        ),
        (
            {"TiO2": 20.0},
            "Column 'TiO2' values must be between 0.01 and 15",
        ),
        (
            {"sulfide_Ni": 60.0},
            "Column 'sulfide_Ni' values must be between 0 and 50",
        ),
        (
            {"Na2O": 0.0, "K2O": 0.0, "SiO2": 33.0},
            "Ding oxide totals must be between 90 and 105 wt.%",
        ),
    ],
)
def test_reject_ding_inputs_outside_calibration_domain(tmp_path, updates, expected_message):
    client = TestClient(create_app(tmp_path / "runtime"))
    row = {
        "Pressure": 1.5,
        "T": 1873.15,
        "SiO2": 43.8,
        "TiO2": 5.0,
        "Al2O3": 10.0,
        "FeO": 18.7,
        "MgO": 8.0,
        "CaO": 11.0,
        "Na2O": 2.0,
        "K2O": 0.5,
        "sulfide_Ni": 30.0,
    }
    row.update(updates)
    response = post_method(
        client,
        task="algo_solubility",
        method="ding",
        element="S",
        content=make_workbook([row]),
    )
    assert response.status_code == 422
    assert response.json()["detail"] == expected_message


@pytest.mark.parametrize(
    ("updates", "expected_message"),
    [
        (
            {"T": 2700.0},
            "Column 'T' values must be between 1423 and 2623",
        ),
        (
            {"FeO": 0.4},
            "Column 'FeO' values must be between 0.5 and 40.1",
        ),
        (
            {"SiO2": 20.0},
            "Blanchard oxide totals must be between 90 and 110 wt.%",
        ),
        (
            {"Fe": 90.0, "Ni": 10.0, "Cu": 5.0},
            "Blanchard sulfide Fe + Ni + Cu must not exceed 100 wt.%",
        ),
    ],
)
def test_reject_blanchard_inputs_outside_calibration_domain(
    tmp_path,
    updates,
    expected_message,
):
    client = TestClient(create_app(tmp_path / "runtime"))
    row = {
        "Pressure": 10.0,
        "T": 2473.0,
        "SiO2": 45.8,
        "TiO2": 0.21,
        "Al2O3": 4.53,
        "FeO": 8.17,
        "MgO": 37.09,
        "CaO": 3.68,
        "Na2O": 0.0,
        "K2O": 0.0,
        "H2O": 0.01,
        "MnO": 0.14,
        "P2O5": 0.0,
        "Cr2O3": 0.37,
        "Fe": 60.0,
        "Ni": 0.0,
        "Cu": 0.0,
    }
    row.update(updates)
    response = post_method(
        client,
        task="algo_solubility",
        method="blanchard",
        element="S",
        content=make_workbook([row]),
    )
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


def test_reject_method_that_has_not_completed_online_verification(tmp_path, monkeypatch):
    key = ("algo_solubility", "hybrid")
    monkeypatch.setitem(
        METHOD_METADATA,
        key,
        replace(
            METHOD_METADATA[key],
            status="testing",
            status_message="Temporary test-only status",
        ),
    )
    client = TestClient(create_app(tmp_path / "runtime"))
    response = client.post(
        "/api/chemical-modeling/run",
        data={
            "task": "algo_solubility",
            "method": "hybrid",
            "element": "S",
        },
        files={"dataset": ("kinetic.xlsx", make_kinetic_workbook())},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Method 'hybrid' has not completed Online verification"


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
