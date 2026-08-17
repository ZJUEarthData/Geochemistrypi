import Lean

namespace GeoPiVerify

open Lean

/-- One derived feature together with the columns and rows used to compute it. -/
structure DerivedFeatureTrace where
  name : String
  sourceColumns : List String
  /-- Empty means row-local. Non-empty means an aggregate was fitted on these rows. -/
  aggregateFitRowIds : List Nat
deriving Repr, DecidableEq, FromJson, ToJson

/-- Dataset identity, filtering, split, role, and derived-column lineage facts. -/
structure DatasetTrace where
  rowIds : List Nat
  /-- One boolean per source row records whether its original business identity was non-empty. -/
  rowIdentityNonemptyMask : List Bool
  filterInputRowIds : List Nat
  filterOutputRowIds : List Nat
  filterXRowIds : List Nat
  filterTargetRowIds : List Nat
  filterNameRowIds : List Nat
  trainRowIds : List Nat
  testRowIds : List Nat
  xTrainRowIds : List Nat
  yTrainRowIds : List Nat
  nameTrainRowIds : List Nat
  xTestRowIds : List Nat
  yTestRowIds : List Nat
  nameTestRowIds : List Nat
  featureColumns : List String
  targetColumns : List String
  identifierColumns : List String
  /-- Role pairs for which the business path performed an explicit overlap check. -/
  roleValidationPairs : List String
  featureEngineeringEnabled : Bool
  /-- Columns that the feature constructor can read on the audited path. -/
  allowedDerivedSourceColumns : List String
  derivedFeatures : List DerivedFeatureTrace
deriving Repr, DecidableEq, FromJson, ToJson

/-- One logical stateful preprocessing stage and its observed fit and output facts. -/
structure StageTrace where
  stageId : String
  name : String
  fitRowIds : List Nat
  fitCount : Nat
  trainingStateDigest : String
  inferenceStateDigest : String
  outputValueCount : Nat
  outputNonFiniteCount : Nat
deriving Repr, DecidableEq, FromJson, ToJson

/-- Ordered schemas, model input identity, and stateful preprocessing lineage. -/
structure PipelineTrace where
  preprocessingEnabled : Bool
  trainRowIds : List Nat
  declaredStageIds : List String
  materializedStageIds : List String
  trainFeatureSchema : List String
  inferenceInputFeatureSchema : List String
  effectiveInferenceFeatureSchema : List String
  pipelineOutputFeatureSchema : List String
  modelTrainFeatureSchema : List String
  pipelineTrainOutputDigest : String
  modelTrainInputDigest : String
  stages : List StageTrace
deriving Repr, DecidableEq, FromJson, ToJson

structure LabelMapping where
  label : String
  code : Nat
deriving Repr, DecidableEq, FromJson, ToJson

/-- Runtime, split-specific, persisted, and prediction-side label-codec facts. -/
structure LabelTrace where
  codecEnabled : Bool
  sourceLabels : List String
  runtimeMappings : List LabelMapping
  fullMappings : List LabelMapping
  trainMappings : List LabelMapping
  testMappings : List LabelMapping
  persistedMappings : List LabelMapping
  codecFitCount : Nat
  predictedCodes : List Nat
  decodedPredictions : List String
deriving Repr, DecidableEq, FromJson, ToJson

/-- Prediction, sample, exported artifact, and run-lineage facts. -/
structure PredictionTrace where
  scope : String
  sourceRowIds : List Nat
  predictionValues : List String
  sampleRowIds : List Nat
  artifactRowIds : List Nat
  artifactPredictionValues : List String
  /-- The audited export policy when equally sized inputs carry different identities. -/
  artifactMismatchPolicy : String
  modelRunId : String
  artifactRunId : String
deriving Repr, DecidableEq, FromJson, ToJson

/-- Model selection, training completion, shared registry, and active-run facts. -/
structure ExecutionTrace where
  eligibleModels : List String
  selectedModelIds : List String
  trainedModelIds : List String
  trainedModelCount : Nat
  registryBefore : List String
  registryAfter : List String
  registryMutationOperations : List String
  activeRunId : String
  stateOwnerRunId : String
deriving Repr, DecidableEq, FromJson, ToJson

/-- A baseline, a single-node counterexample, or a production-code audit trace. -/
structure CaseTrace where
  caseId : String
  caseKind : String
  description : String
  /-- Oracle metadata is read only for baseline and counterexample cases. -/
  expectedConformant : Bool
  targetCheckId : String
  dataset : DatasetTrace
  pipeline : PipelineTrace
  labels : LabelTrace
  prediction : PredictionTrace
  execution : ExecutionTrace
deriving Repr, DecidableEq, FromJson, ToJson

structure TraceBundle where
  schemaVersion : Nat
  sourceCommit : String
  generatedAt : String
  cases : List CaseTrace
deriving Repr, DecidableEq, FromJson, ToJson

structure CheckResult where
  checkId : String
  passed : Bool
deriving Repr, DecidableEq, FromJson, ToJson

structure CaseReport where
  caseId : String
  caseKind : String
  description : String
  expectedConformant : Bool
  targetCheckId : String
  accepted : Bool
  expectationMatched : Bool
  failedCheckIds : List String
  isolationMatched : Bool
  checks : List CheckResult
deriving Repr, DecidableEq, FromJson, ToJson

structure BundleReport where
  schemaVersion : Nat
  sourceCommit : String
  caseCount : Nat
  acceptedCount : Nat
  rejectedCount : Nat
  counterexampleCount : Nat
  coveredCheckCount : Nat
  counterexampleCoverageComplete : Bool
  allCounterexamplesIsolated : Bool
  allCasesAccepted : Bool
  allExpectationsMatched : Bool
  cases : List CaseReport
deriving Repr, DecidableEq, FromJson, ToJson

end GeoPiVerify
