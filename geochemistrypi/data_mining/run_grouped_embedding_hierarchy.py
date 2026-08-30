"""Grouped embedding followed by hierarchical clustering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _read_table(path: Path, sheet: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    selected_sheet = int(sheet) if sheet.isdigit() else sheet
    return pd.read_excel(path, sheet_name=selected_sheet)


def run_grouped_embedding_hierarchy(
    *,
    input_path: Path,
    output_root: Path,
    experiment_name: str,
    run_name: str,
    group_column: str,
    feature_columns: Sequence[str],
    sheet: str = "0",
    components: int = 3,
    perplexity: float = 35.0,
    iterations: int = 1000,
    early_exaggeration: float = 12.0,
    learning_rate: str = "auto",
    init: str = "pca",
    seed: int = 42,
    metric: str = "euclidean",
    linkage_method: str = "complete",
    metadata_column: str | None = None,
    cluster_count: int | None = None,
) -> Path:
    data = _read_table(input_path, sheet)
    required = [group_column, *feature_columns]
    if metadata_column is not None:
        required.append(metadata_column)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    selected = data.loc[:, required].copy()
    if selected[group_column].isna().any() or selected[list(feature_columns)].isna().any().any():
        raise ValueError("Group and feature columns must be complete before this workflow.")
    if selected[group_column].nunique() < 2:
        raise ValueError("At least two groups are required.")

    scaled = StandardScaler().fit_transform(selected[list(feature_columns)])
    rate: str | float = learning_rate
    if learning_rate != "auto":
        rate = float(learning_rate)
    embedding = TSNE(
        n_components=components,
        perplexity=perplexity,
        n_iter=iterations,
        early_exaggeration=early_exaggeration,
        learning_rate=rate,
        init=init,
        random_state=seed,
        metric=metric,
    ).fit_transform(scaled)

    coordinate_columns = [f"Dimension {index + 1}" for index in range(components)]
    coordinates = pd.DataFrame(embedding, columns=coordinate_columns)
    coordinates.insert(0, group_column, selected[group_column].to_numpy())
    group_means = coordinates.groupby(group_column, sort=True)[coordinate_columns].mean()
    linkage_matrix = linkage(group_means.to_numpy(), method=linkage_method, metric=metric)

    output_directory = output_root / experiment_name / run_name
    data_directory = output_directory / "artifacts" / "data"
    image_directory = output_directory / "artifacts" / "image" / "model_output"
    parameter_directory = output_directory / "parameters"
    summary_directory = output_directory / "summary"
    for directory in (data_directory, image_directory, parameter_directory, summary_directory):
        directory.mkdir(parents=True, exist_ok=True)

    coordinates.to_csv(data_directory / "Sample Embedding Coordinates.csv", index=False)
    group_means.reset_index().to_csv(data_directory / "Grouped Embedding Means.csv", index=False)
    linkage_table = pd.DataFrame(linkage_matrix, columns=["left", "right", "distance", "count"])
    linkage_table.to_csv(data_directory / "Hierarchical Linkage Matrix.csv", index=False)
    if cluster_count is not None:
        if not 2 <= cluster_count <= len(group_means):
            raise ValueError("Cluster count must be between 2 and the number of groups.")
        membership = pd.DataFrame(
            {group_column: group_means.index, "hierarchical_cluster": fcluster(linkage_matrix, cluster_count, criterion="maxclust")}
        )
        if metadata_column is not None:
            modes = selected.groupby(group_column)[metadata_column].agg(
                lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]
            )
            membership[metadata_column] = membership[group_column].map(modes)
        membership.to_csv(data_directory / "Grouped Hierarchical Membership.csv", index=False)

    figure, axis = plt.subplots(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=group_means.index.astype(str).tolist(), orientation="right", ax=axis)
    axis.set_xlabel("Euclidean distance")
    axis.set_ylabel(group_column)
    axis.set_title(f"Grouped t-SNE hierarchy ({linkage_method} linkage)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(image_directory / f"Grouped Embedding Dendrogram.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    parameters = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "row_count": int(len(selected)),
        "group_count": int(len(group_means)),
        "group_column": group_column,
        "feature_columns": list(feature_columns),
        "scaling": "standardization",
        "embedding": {
            "type": "t-SNE", "components": components, "perplexity": perplexity,
            "iterations": iterations, "early_exaggeration": early_exaggeration,
            "learning_rate": learning_rate, "init": init, "seed": seed, "metric": metric,
        },
        "aggregation": "arithmetic mean by group",
        "hierarchy": {"method": linkage_method, "metric": metric},
        "cluster_count": cluster_count,
        "metadata_column": metadata_column,
    }
    text = json.dumps(parameters, ensure_ascii=False, indent=2) + "\n"
    (parameter_directory / "Grouped Embedding Hierarchy Parameters.json").write_text(text, encoding="utf-8")
    (summary_directory / "Grouped Embedding Hierarchy Parameters.json").write_text(text, encoding="utf-8")
    return output_directory
