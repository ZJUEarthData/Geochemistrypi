import GeoPiVerify.Predicates

namespace GeoPiVerify

def propositionCheck (checkId : String) (p : Prop) [Decidable p] : CheckResult :=
  { checkId := checkId, passed := decide p }

def checks (c : CaseTrace) : List CheckResult :=
  [ propositionCheck "D01.input_rows_identified"
      (InputRowsIdentified c.dataset)
  , propositionCheck "D02.split_is_disjoint_partition"
      (SplitIsDisjointPartition c.dataset)
  , propositionCheck "D03.supervised_views_row_aligned"
      (SupervisedViewsRowAligned c.dataset)
  , propositionCheck "D04.column_roles_guarded_and_disjoint"
      (ColumnRolesGuardedAndDisjoint c.dataset)
  , propositionCheck "D05.derived_feature_lineage_safe"
      (DerivedFeatureLineageSafe c.dataset)
  , propositionCheck "D06.filtered_rows_keep_lineage"
      (FilteredRowsKeepLineage c.dataset)
  , propositionCheck "P01.effective_schema_matches_training"
      (EffectiveSchemaMatchesTraining c.pipeline)
  , propositionCheck "P02.stateful_fit_uses_training_rows_only"
      (StatefulFitUsesTrainingRowsOnly c.pipeline)
  , propositionCheck "P03.fitted_state_reused_for_model_and_inference"
      (FittedStateReusedForModelAndInference c.pipeline)
  , propositionCheck "P04.model_input_schema_matches_pipeline_output"
      (ModelInputSchemaMatchesPipelineOutput c.pipeline)
  , propositionCheck "P05.declared_and_materialized_stage_order_equal"
      (DeclaredAndMaterializedStageOrderEqual c.pipeline)
  , propositionCheck "P06.observed_stage_outputs_finite"
      (ObservedStageOutputsFinite c.pipeline)
  , propositionCheck "L01.codec_total_and_injective"
      (CodecTotalAndInjective c.labels)
  , propositionCheck "L02.one_codec_fitted_once_for_all_splits"
      (OneCodecFittedOnceForAllSplits c.labels)
  , propositionCheck "L03.codec_persisted_and_predictions_decodable"
      (CodecPersistedAndPredictionsDecodable c.labels)
  , propositionCheck "A01.predictions_bound_to_source_rows"
      (PredictionsBoundToSourceRows c.prediction)
  , propositionCheck "A02.artifact_pairs_aligned_and_mismatch_rejected"
      (ArtifactPairsAlignedAndMismatchRejected c.prediction)
  , propositionCheck "A03.model_artifact_and_state_share_run"
      (ModelArtifactAndStateShareRun c.prediction c.execution)
  , propositionCheck "E01.selected_models_eligible_and_trained"
      (SelectedModelsEligibleAndTrained c.execution)
  , propositionCheck "E02.model_registry_immutable_during_run"
      (ModelRegistryImmutableDuringRun c.execution)
  ]

def accepted (c : CaseTrace) : Bool := conformsB c

def failedCheckIds (results : List CheckResult) : List String :=
  results.foldr (fun result acc => if result.passed then acc else result.checkId :: acc) []

def counterexampleIsolated (c : CaseTrace) (failed : List String) : Bool :=
  if c.caseKind == "production" then
    true
  else if c.caseKind == "baseline" then
    failed.isEmpty
  else
    failed == [c.targetCheckId]

def expectationMatches (c : CaseTrace) (isAccepted : Bool) : Bool :=
  if c.caseKind == "production" then true else isAccepted == c.expectedConformant

def publicCheckIds : List String :=
  [ "D01.input_rows_identified"
  , "D02.split_is_disjoint_partition"
  , "D03.supervised_views_row_aligned"
  , "D04.column_roles_guarded_and_disjoint"
  , "D05.derived_feature_lineage_safe"
  , "D06.filtered_rows_keep_lineage"
  , "P01.effective_schema_matches_training"
  , "P02.stateful_fit_uses_training_rows_only"
  , "P03.fitted_state_reused_for_model_and_inference"
  , "P04.model_input_schema_matches_pipeline_output"
  , "P05.declared_and_materialized_stage_order_equal"
  , "P06.observed_stage_outputs_finite"
  , "L01.codec_total_and_injective"
  , "L02.one_codec_fitted_once_for_all_splits"
  , "L03.codec_persisted_and_predictions_decodable"
  , "A01.predictions_bound_to_source_rows"
  , "A02.artifact_pairs_aligned_and_mismatch_rejected"
  , "A03.model_artifact_and_state_share_run"
  , "E01.selected_models_eligible_and_trained"
  , "E02.model_registry_immutable_during_run"
  ]

def counterexampleTargets (reports : List CaseReport) : List String :=
  (reports.filter (fun report => report.caseKind == "counterexample")).map (·.targetCheckId) |>.eraseDups

def counterexampleCoverageComplete (reports : List CaseReport) : Bool :=
  let targets := counterexampleTargets reports
  targets.length == publicCheckIds.length && publicCheckIds.all (fun checkId => targets.contains checkId)

def certify (c : CaseTrace) : Except String {t : CaseTrace // Conforms t} :=
  if h : Conforms c then .ok ⟨c, h⟩ else .error s!"case {c.caseId} does not conform"

def reportCase (c : CaseTrace) : CaseReport :=
  let isAccepted := accepted c
  let checkResults := checks c
  let failed := failedCheckIds checkResults
  { caseId := c.caseId
  , caseKind := c.caseKind
  , description := c.description
  , expectedConformant := c.expectedConformant
  , targetCheckId := c.targetCheckId
  , accepted := isAccepted
  , expectationMatched := expectationMatches c isAccepted
  , failedCheckIds := failed
  , isolationMatched := counterexampleIsolated c failed
  , checks := checkResults
  }

def reportBundle (bundle : TraceBundle) : BundleReport :=
  let reports := bundle.cases.map reportCase
  let acceptedCount := (reports.filter (·.accepted)).length
  let counterexamples := reports.filter (fun report => report.caseKind == "counterexample")
  let targets := counterexampleTargets reports
  { schemaVersion := bundle.schemaVersion
  , sourceCommit := bundle.sourceCommit
  , caseCount := reports.length
  , acceptedCount := acceptedCount
  , rejectedCount := reports.length - acceptedCount
  , counterexampleCount := counterexamples.length
  , coveredCheckCount := targets.length
  , counterexampleCoverageComplete := counterexampleCoverageComplete reports
  , allCounterexamplesIsolated := reports.all (·.isolationMatched)
  , allCasesAccepted := reports.all (·.accepted)
  , allExpectationsMatched := reports.all (·.expectationMatched)
  , cases := reports
  }

end GeoPiVerify
