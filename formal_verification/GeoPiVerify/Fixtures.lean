import GeoPiVerify.Checker

namespace GeoPiVerify

def goodDataset : DatasetTrace :=
  { rowIds := [0, 1, 2, 3]
  , rowIdentityNonemptyMask := [true, true, true, true]
  , filterInputRowIds := [0, 1, 2, 3]
  , filterOutputRowIds := [0, 1, 2, 3]
  , filterXRowIds := [0, 1, 2, 3]
  , filterTargetRowIds := [0, 1, 2, 3]
  , filterNameRowIds := [0, 1, 2, 3]
  , trainRowIds := [0, 2, 3]
  , testRowIds := [1]
  , xTrainRowIds := [0, 2, 3]
  , yTrainRowIds := [0, 2, 3]
  , nameTrainRowIds := [0, 2, 3]
  , xTestRowIds := [1]
  , yTestRowIds := [1]
  , nameTestRowIds := [1]
  , featureColumns := ["SiO2", "MgO", "ratio"]
  , targetColumns := ["rock_type"]
  , identifierColumns := ["sample_id"]
  , roleValidationPairs := requiredRoleValidationPairs
  , featureEngineeringEnabled := true
  , allowedDerivedSourceColumns := ["SiO2", "MgO"]
  , derivedFeatures :=
      [{ name := "ratio", sourceColumns := ["SiO2", "MgO"], aggregateFitRowIds := [] }]
  }

def goodPipeline : PipelineTrace :=
  { preprocessingEnabled := true
  , trainRowIds := [0, 2, 3]
  , declaredStageIds := ["scale-1"]
  , materializedStageIds := ["scale-1"]
  , trainFeatureSchema := ["SiO2", "MgO", "ratio"]
  , inferenceInputFeatureSchema := ["sample_id", "SiO2", "MgO", "ratio", "unused"]
  , effectiveInferenceFeatureSchema := ["SiO2", "MgO", "ratio"]
  , pipelineOutputFeatureSchema := ["SiO2", "MgO", "ratio"]
  , modelTrainFeatureSchema := ["SiO2", "MgO", "ratio"]
  , pipelineTrainOutputDigest := "model-input-1"
  , modelTrainInputDigest := "model-input-1"
  , stages :=
      [{ stageId := "scale-1"
       , name := "StandardScaler"
       , fitRowIds := [0, 2, 3]
       , fitCount := 1
       , trainingStateDigest := "scaler-state-1"
       , inferenceStateDigest := "scaler-state-1"
       , outputValueCount := 9
       , outputNonFiniteCount := 0 }]
  }

def goodLabels : LabelTrace :=
  let mapping := [{ label := "basalt", code := 4 }, { label := "granite", code := 9 }]
  { codecEnabled := true
  , sourceLabels := ["basalt", "granite"]
  , runtimeMappings := mapping
  , fullMappings := mapping
  , trainMappings := mapping
  , testMappings := mapping
  , persistedMappings := mapping
  , codecFitCount := 1
  , predictedCodes := [9]
  , decodedPredictions := ["granite"]
  }

def goodPrediction : PredictionTrace :=
  { scope := "test"
  , sourceRowIds := [1]
  , predictionValues := ["granite"]
  , sampleRowIds := [1]
  , artifactRowIds := [1]
  , artifactPredictionValues := ["granite"]
  , artifactMismatchPolicy := "reject"
  , modelRunId := "run-1"
  , artifactRunId := "run-1"
  }

def goodExecution : ExecutionTrace :=
  { eligibleModels := ["Logistic Regression", "Decision Tree"]
  , selectedModelIds := ["Logistic Regression"]
  , trainedModelIds := ["Logistic Regression"]
  , trainedModelCount := 1
  , registryBefore := ["Logistic Regression", "Decision Tree"]
  , registryAfter := ["Logistic Regression", "Decision Tree"]
  , registryMutationOperations := []
  , activeRunId := "run-1"
  , stateOwnerRunId := "run-1"
  }

def goodCase : CaseTrace :=
  { caseId := "lean-baseline"
  , caseKind := "baseline"
  , description := "Closed positive baseline"
  , expectedConformant := true
  , targetCheckId := ""
  , dataset := goodDataset
  , pipeline := goodPipeline
  , labels := goodLabels
  , prediction := goodPrediction
  , execution := goodExecution
  }

def roleGuardMutant : CaseTrace :=
  { goodCase with
    caseId := "mutant-role-guard-missing"
    caseKind := "counterexample"
    expectedConformant := false
    targetCheckId := "D04.column_roles_guarded_and_disjoint"
    dataset := { goodDataset with roleValidationPairs := ["feature_target"] } }

def fitLeakMutant : CaseTrace :=
  { goodCase with
    caseId := "mutant-test-row-in-fit"
    caseKind := "counterexample"
    expectedConformant := false
    targetCheckId := "P02.stateful_fit_uses_training_rows_only"
    pipeline :=
      { goodPipeline with
        stages :=
          [{ stageId := "scale-1"
           , name := "StandardScaler"
           , fitRowIds := [0, 1, 2, 3]
           , fitCount := 1
           , trainingStateDigest := "scaler-state-1"
           , inferenceStateDigest := "scaler-state-1"
           , outputValueCount := 9
           , outputNonFiniteCount := 0 }] } }

def codecPersistenceMutant : CaseTrace :=
  { goodCase with
    caseId := "mutant-codec-not-persisted"
    caseKind := "counterexample"
    expectedConformant := false
    targetCheckId := "L03.codec_persisted_and_predictions_decodable"
    labels := { goodLabels with persistedMappings := [] } }

def artifactPolicyMutant : CaseTrace :=
  { goodCase with
    caseId := "mutant-positional-artifact-fallback"
    caseKind := "counterexample"
    expectedConformant := false
    targetCheckId := "A02.artifact_pairs_aligned_and_mismatch_rejected"
    prediction := { goodPrediction with artifactMismatchPolicy := "positional_fallback" } }

theorem good_fixture_conforms : Conforms goodCase := by decide +kernel
theorem role_guard_mutant_rejected : ¬ Conforms roleGuardMutant := by decide +kernel
theorem fit_leak_mutant_rejected : ¬ Conforms fitLeakMutant := by decide +kernel
theorem codec_persistence_mutant_rejected : ¬ Conforms codecPersistenceMutant := by decide +kernel
theorem artifact_policy_mutant_rejected : ¬ Conforms artifactPolicyMutant := by decide +kernel

#print axioms good_fixture_conforms
#print axioms role_guard_mutant_rejected
#print axioms fit_leak_mutant_rejected
#print axioms codec_persistence_mutant_rejected
#print axioms artifact_policy_mutant_rejected

end GeoPiVerify
