from unittest.mock import patch

import numpy as np
import pandas as pd

from geochemistrypi.data_mining.model.decomposition import DecompositionWorkflowBase, PCADecomposition


def test_pca_special_components_reuses_generated_principal_component_table() -> None:
    """The PCA plot must receive the component table generated for the fitted model."""

    features = pd.DataFrame(
        {
            "SiO2": [49.0, 52.0, 55.0, 58.0],
            "MgO": [9.0, 7.0, 5.0, 3.0],
            "FeO": [11.0, 10.0, 8.0, 7.0],
        }
    )
    workflow = PCADecomposition(n_components=2)
    reduced = workflow.model.fit_transform(features)
    DecompositionWorkflowBase.X = features

    with patch.object(workflow, "_biplot") as biplot:
        workflow.special_components(reduced_data=reduced, components_num=2)

    expected = pd.DataFrame(
        workflow.model.components_.T,
        index=features.columns,
        columns=["PC1", "PC2"],
    )
    pd.testing.assert_frame_equal(workflow.pc_data, expected)
    np.testing.assert_allclose(biplot.call_args.kwargs["pc_data"], expected)


def test_pca_component_selection_applies_to_scores_and_loadings() -> None:
    features = pd.DataFrame(
        {
            "SiO2": [49.0, 52.0, 55.0, 58.0],
            "MgO": [9.0, 7.0, 5.0, 3.0],
            "FeO": [11.0, 10.0, 8.0, 7.0],
        }
    )
    workflow = PCADecomposition(n_components=3)
    reduced = workflow.model.fit_transform(features)
    DecompositionWorkflowBase.X = features

    with (
        patch.object(
            workflow,
            "choose_dimension_data",
            return_value=([0, 2], pd.DataFrame()),
        ),
        patch.object(workflow, "_biplot") as biplot,
        patch.object(workflow, "_triplot"),
    ):
        workflow.special_components(reduced_data=reduced, components_num=3)

    assert biplot.call_args.kwargs["reduced_data"].columns.tolist() == [
        "Principal Axis 1",
        "Principal Axis 3",
    ]
    assert biplot.call_args.kwargs["pc_data"].columns.tolist() == ["PC1", "PC3"]
    assert biplot.call_args.kwargs["pc_data"].index.tolist() == features.columns.tolist()
