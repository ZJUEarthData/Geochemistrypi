# PR7 Decomposition Coverage

## Outcome

PR7 exposes all three decomposition families in the GeochemistryPi 0.8.0
public CLI through the local stdio MCP wrapper: PCA, T-SNE, and MDS. The
existing `geochemistrypi data-mining` command remains the only scientific
execution engine. MCP validates semantic inputs, compiles guarded interactive
answers, and indexes the CLI's original outputs.

`start_analysis` now accepts strict task-discriminated `classification`,
`regression`, `clustering`, and `decomposition` requests. Decomposition is
target-free and does not inherit supervised fields.

## Root CLI repairs

Real PCA characterization exposed two pre-existing defects in the public CLI.
First, the generated principal-component loading table was printed but never
stored in `self.pc_data`, so the mandatory PCA bi-plot received `None` and
failed. Second, PCA runs with three or more components selected columns from
the reduced sample coordinates and incorrectly passed those sample values as
principal-component loadings.

The core repair returns and stores the generated loading table, then applies
the same selected component indices separately to the reduced scores and the
loading columns. Focused regression tests cover both the two-component and
selected-component branches. Real two- and three-component public CLI runs
now complete and generate the expected PCA bi/tri-plots.

## Versioned model contract

The capability matrix is stored in
`tests/mcp_wrapper/parity/fixtures/decomposition_capability_matrix_v1.json`
and checked against `DECOMPOSITION_MODELS` in the public CLI source.

| MCP model ID | Existing public CLI model | Exposed parameters |
| --- | --- | --- |
| `pca` | PCA | component count, SVD solver |
| `tsne` | T-SNE | component count, perplexity, learning rate, iterations, early exaggeration |
| `mds` | MDS | component count, metric mode, initializations, maximum iterations |

Every accepted request explicitly names one model. The aggregate model branch,
target columns, supervised feature selection and splitting, AutoML,
application-data inference, unresolved missing values, and prior-experiment
attachment are not exposed.

## Validation and interaction coverage

Before starting a process, the wrapper checks that identifiers are unique and
non-missing; selected features are numeric and finite; missing-value handling
leaves at least two rows; and at least two final features remain for the CLI's
mandatory diagrams. It also enforces PCA's
`min(retained rows, final features)` component bound, ARPACK's strict bound,
and T-SNE's requirement that perplexity be smaller than the retained row
count.

The interaction compiler covers the real dimensional-reduction mode number,
all three model numbers, exact method-specific prompts, optional feature
engineering and scaling, transform-pipeline completion, and PCA's conditional
component choices. Two-component PCA asks no plot-selection questions;
three-component PCA asks for the two bi-plot axes; larger PCA requests also
select the three tri-plot axes.

## Original outputs and parity

The existing CLI continues to create and own:

- `X Reduced.xlsx`;
- the trained PCA, T-SNE, or MDS model;
- decomposition two-dimensional, heatmap, and contour outputs;
- PCA loading and reduced-data tables plus bi/tri-plots where applicable;
- hyperparameter files;
- transform-pipeline configuration and optional fitted pipeline;
- copied summary files and MLflow tracking data.

Real characterization completed successfully for PCA, T-SNE, and MDS. The PR7
parity scenario runs the same PCA request once by feeding the compiled answers
directly to the public CLI and once through a real stdio MCP server. It compares
the complete output inventory, input hash, transformed data, PCA loading table,
hyperparameters, plot presence, task/result semantics, and artifact count. The
MCP protocol remains healthy after execution.

## Verification

The completed local verification checkpoint on 2026-08-02 is recorded after
the full core, MCP, formatting, wheel, and clean-install checks in the roadmap
and implementation requirements. This document describes local uncommitted
work; it does not claim remote CI, a commit, a push, or a public release.
