from pathlib import Path

import pandas as pd

from geochemistrypi.data_mining.run_grouped_embedding_hierarchy import run_grouped_embedding_hierarchy


def test_grouped_embedding_hierarchy_writes_auditable_outputs(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "Group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "F1": list(range(12)),
            "F2": [value * 2 for value in range(12)],
            "Clan": ["X"] * 4 + ["Y"] * 4 + ["Z"] * 4,
        }
    ).to_csv(source, index=False)
    output = run_grouped_embedding_hierarchy(
        input_path=source,
        output_root=tmp_path / "out",
        experiment_name="experiment",
        run_name="run",
        group_column="Group",
        feature_columns=("F1", "F2"),
        components=2,
        perplexity=3,
        iterations=250,
        learning_rate="50",
        init="random",
        seed=42,
        metadata_column="Clan",
        cluster_count=3,
    )
    grouped = pd.read_csv(output / "artifacts" / "data" / "Grouped Embedding Means.csv")
    linkage = pd.read_csv(output / "artifacts" / "data" / "Hierarchical Linkage Matrix.csv")
    assert grouped["Group"].tolist() == ["A", "B", "C"]
    assert linkage.shape == (2, 4)
    membership = pd.read_csv(output / "artifacts" / "data" / "Grouped Hierarchical Membership.csv")
    assert membership["Clan"].tolist() == ["X", "Y", "Z"]
    assert (output / "artifacts" / "image" / "model_output" / "Grouped Embedding Dendrogram.png").is_file()
