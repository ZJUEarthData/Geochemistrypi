import GeoPiVerify.Types

namespace GeoPiVerify

def sameMembersB [BEq α] (xs ys : List α) : Bool :=
  xs.all (fun x => ys.contains x) && ys.all (fun y => xs.contains y)

def natLeB (left right : Nat) : Bool :=
  decide (left ≤ right)

def splitAlternating : List Nat → List Nat × List Nat
  | [] => ([], [])
  | [item] => ([item], [])
  | first :: second :: rest =>
      let parts := splitAlternating rest
      (first :: parts.1, second :: parts.2)

def mergeNatsAux : Nat → List Nat → List Nat → List Nat
  | 0, left, right => left ++ right
  | _ + 1, [], right => right
  | _ + 1, left, [] => left
  | fuel + 1, leftHead :: leftTail, rightHead :: rightTail =>
      if natLeB leftHead rightHead then
        leftHead :: mergeNatsAux fuel leftTail (rightHead :: rightTail)
      else
        rightHead :: mergeNatsAux fuel (leftHead :: leftTail) rightTail

def mergeNats (left right : List Nat) : List Nat :=
  mergeNatsAux (left.length + right.length) left right

def sortNatsAux : Nat → List Nat → List Nat
  | 0, items => items
  | _ + 1, [] => []
  | _ + 1, [item] => [item]
  | fuel + 1, first :: second :: rest =>
      let parts := splitAlternating (first :: second :: rest)
      mergeNats (sortNatsAux fuel parts.1) (sortNatsAux fuel parts.2)

def sortNats (items : List Nat) : List Nat :=
  sortNatsAux items.length items

def adjacentNatsDistinctB : List Nat → Bool
  | [] => true
  | first :: rest => loop first rest
where
  loop (previous : Nat) : List Nat → Bool
    | [] => true
    | current :: remaining =>
        previous != current && loop current remaining

def natNodupB (items : List Nat) : Bool :=
  adjacentNatsDistinctB (sortNats items)

def sameNatMembersB (xs ys : List Nat) : Bool :=
  sortNats xs == sortNats ys

def listDisjointB [BEq α] (xs ys : List α) : Bool :=
  xs.all (fun x => !(ys.contains x))

def listSubsetB [BEq α] (xs ys : List α) : Bool :=
  xs == ys || xs.all (fun x => ys.contains x)

def MappingLabels (mappings : List LabelMapping) : List String :=
  mappings.map (·.label)

def MappingCodes (mappings : List LabelMapping) : List Nat :=
  mappings.map (·.code)

def encode? (mappings : List LabelMapping) (label : String) : Option Nat :=
  (mappings.find? (fun item => item.label == label)).map (·.code)

def decode? (mappings : List LabelMapping) (code : Nat) : Option String :=
  (mappings.find? (fun item => item.code == code)).map (·.label)

def decodeAll? (mappings : List LabelMapping) : List Nat → Option (List String)
  | [] => some []
  | code :: rest => do
      let label ← decode? mappings code
      let labels ← decodeAll? mappings rest
      pure (label :: labels)

/-- D01. Every source row has a non-empty, unique business identity. -/
abbrev InputRowsIdentified (d : DatasetTrace) : Prop :=
  d.rowIds ≠ [] ∧
  natNodupB d.rowIds = true ∧
  d.rowIdentityNonemptyMask.length = d.rowIds.length ∧
  d.rowIdentityNonemptyMask.all id = true

/-- D02. Train and test rows are non-empty, disjoint, and cover the source rows. -/
abbrev SplitIsDisjointPartition (d : DatasetTrace) : Prop :=
  d.trainRowIds ≠ [] ∧
  d.testRowIds ≠ [] ∧
  natNodupB (d.trainRowIds ++ d.testRowIds) = true ∧
  sameNatMembersB (d.trainRowIds ++ d.testRowIds) d.rowIds = true

/-- D03. X, target, and identifier views carry the same ordered rows on both sides. -/
abbrev SupervisedViewsRowAligned (d : DatasetTrace) : Prop :=
  d.xTrainRowIds = d.yTrainRowIds ∧
  d.yTrainRowIds = d.nameTrainRowIds ∧
  d.xTrainRowIds = d.trainRowIds ∧
  d.xTestRowIds = d.yTestRowIds ∧
  d.yTestRowIds = d.nameTestRowIds ∧
  d.xTestRowIds = d.testRowIds

def requiredRoleValidationPairs : List String :=
  ["feature_target", "feature_identifier", "target_identifier"]

/-- D04. Column roles are disjoint and the workflow explicitly validates every pair. -/
abbrev ColumnRolesGuardedAndDisjoint (d : DatasetTrace) : Prop :=
  d.featureColumns ≠ [] ∧
  d.targetColumns ≠ [] ∧
  d.identifierColumns ≠ [] ∧
  d.featureColumns.Nodup ∧
  d.targetColumns.Nodup ∧
  d.identifierColumns.Nodup ∧
  listDisjointB d.featureColumns d.targetColumns = true ∧
  listDisjointB d.featureColumns d.identifierColumns = true ∧
  listDisjointB d.targetColumns d.identifierColumns = true ∧
  requiredRoleValidationPairs.all (fun pair => d.roleValidationPairs.contains pair) = true

abbrev DerivedFeatureSourcesSafe (d : DatasetTrace) : Prop :=
  d.allowedDerivedSourceColumns.Nodup ∧
  listDisjointB d.allowedDerivedSourceColumns d.targetColumns = true ∧
  listDisjointB d.allowedDerivedSourceColumns d.identifierColumns = true ∧
  d.derivedFeatures.all (fun feature =>
    !feature.name.isEmpty &&
    !feature.sourceColumns.isEmpty &&
    feature.sourceColumns.all (fun column => d.allowedDerivedSourceColumns.contains column)) = true

abbrev DerivedAggregatesUseTrainingRows (d : DatasetTrace) : Prop :=
  d.derivedFeatures.all (fun feature =>
    feature.aggregateFitRowIds.isEmpty ||
      (natNodupB feature.aggregateFitRowIds &&
       listSubsetB feature.aggregateFitRowIds d.trainRowIds)) = true

/-- D05. Enabled feature engineering can only read permitted features and train-scope aggregates. -/
abbrev DerivedFeatureLineageSafe (d : DatasetTrace) : Prop :=
  if d.featureEngineeringEnabled then
    d.derivedFeatures ≠ [] ∧
    d.allowedDerivedSourceColumns ≠ [] ∧
    DerivedFeatureSourcesSafe d ∧
    DerivedAggregatesUseTrainingRows d
  else
    d.derivedFeatures = []

/-- D06. Filtering may remove rows, but every remaining X, target, and name row stays paired. -/
abbrev FilteredRowsKeepLineage (d : DatasetTrace) : Prop :=
  d.filterInputRowIds ≠ [] ∧
  natNodupB d.filterInputRowIds = true ∧
  d.filterOutputRowIds ≠ [] ∧
  natNodupB d.filterOutputRowIds = true ∧
  listSubsetB d.filterOutputRowIds d.filterInputRowIds = true ∧
  d.filterXRowIds = d.filterOutputRowIds ∧
  d.filterTargetRowIds = d.filterOutputRowIds ∧
  d.filterNameRowIds = d.filterOutputRowIds

/-- P01. Extra application columns are allowed, while the effective ordered schema equals training. -/
abbrev EffectiveSchemaMatchesTraining (p : PipelineTrace) : Prop :=
  p.trainFeatureSchema ≠ [] ∧
  p.trainFeatureSchema.Nodup ∧
  p.inferenceInputFeatureSchema.Nodup ∧
  p.effectiveInferenceFeatureSchema.Nodup ∧
  p.effectiveInferenceFeatureSchema = p.trainFeatureSchema ∧
  listSubsetB p.effectiveInferenceFeatureSchema p.inferenceInputFeatureSchema = true

/-- P02. A stateful fit may use a training subset but may never consume a test row. -/
abbrev StatefulFitUsesTrainingRowsOnly (p : PipelineTrace) : Prop :=
  if p.preprocessingEnabled then
    p.trainRowIds ≠ [] ∧
    natNodupB p.trainRowIds = true ∧
    p.stages ≠ [] ∧
    p.stages.all (fun stage =>
      !stage.fitRowIds.isEmpty &&
      natNodupB stage.fitRowIds &&
      listSubsetB stage.fitRowIds p.trainRowIds) = true
  else
    p.stages = []

/-- P03. One fitted state produces both model training input and inference transformation. -/
abbrev FittedStateReusedForModelAndInference (p : PipelineTrace) : Prop :=
  if p.preprocessingEnabled then
    p.stages.all (fun stage =>
      stage.fitCount == 1 &&
      !stage.trainingStateDigest.isEmpty &&
      stage.trainingStateDigest == stage.inferenceStateDigest) = true ∧
    p.pipelineTrainOutputDigest ≠ "" ∧
    p.pipelineTrainOutputDigest = p.modelTrainInputDigest
  else
    True

/-- P04. The model receives the exact feature schema emitted by the training pipeline. -/
abbrev ModelInputSchemaMatchesPipelineOutput (p : PipelineTrace) : Prop :=
  p.pipelineOutputFeatureSchema ≠ [] ∧
  p.pipelineOutputFeatureSchema.Nodup ∧
  p.pipelineOutputFeatureSchema = p.modelTrainFeatureSchema

/-- P05. Declared, materialized, and observed preprocessing order are identical. -/
abbrev DeclaredAndMaterializedStageOrderEqual (p : PipelineTrace) : Prop :=
  if p.preprocessingEnabled then
    p.declaredStageIds ≠ [] ∧
    p.declaredStageIds.Nodup ∧
    p.materializedStageIds.Nodup ∧
    p.declaredStageIds = p.materializedStageIds ∧
    p.materializedStageIds = p.stages.map (·.stageId)
  else
    p.declaredStageIds = [] ∧ p.materializedStageIds = [] ∧ p.stages = []

/-- P06. Every scalar observed at a stage output is represented and finite. -/
abbrev ObservedStageOutputsFinite (p : PipelineTrace) : Prop :=
  if p.preprocessingEnabled then
    p.stages.all (fun stage =>
      stage.outputValueCount > 0 &&
      stage.outputNonFiniteCount == 0) = true
  else
    True

abbrev CodecDomainCovered (l : LabelTrace) : Prop :=
  sameMembersB (MappingLabels l.runtimeMappings) l.sourceLabels = true

abbrev CodecInjective (l : LabelTrace) : Prop :=
  (MappingLabels l.runtimeMappings).Nodup ∧
  (MappingCodes l.runtimeMappings).Nodup

abbrev CodecRoundTrips (l : LabelTrace) : Prop :=
  l.sourceLabels.all (fun label =>
    ((encode? l.runtimeMappings label).bind (decode? l.runtimeMappings)) == some label) = true

/-- L01. Codes need not be contiguous, but the mapping must be total and injective. -/
abbrev CodecTotalAndInjective (l : LabelTrace) : Prop :=
  if l.codecEnabled then
    l.sourceLabels ≠ [] ∧
    l.sourceLabels.Nodup ∧
    l.runtimeMappings ≠ [] ∧
    CodecDomainCovered l ∧
    CodecInjective l ∧
    CodecRoundTrips l
  else
    l.runtimeMappings = []

/-- L02. One codec is fitted once and reused for full, training, and test targets. -/
abbrev OneCodecFittedOnceForAllSplits (l : LabelTrace) : Prop :=
  if l.codecEnabled then
    l.codecFitCount = 1 ∧
    l.runtimeMappings = l.fullMappings ∧
    l.runtimeMappings = l.trainMappings ∧
    l.runtimeMappings = l.testMappings
  else
    l.codecFitCount = 0

abbrev PredictionsDecode (l : LabelTrace) : Prop :=
  decodeAll? l.runtimeMappings l.predictedCodes = some l.decodedPredictions ∧
  l.decodedPredictions.all (fun label => l.sourceLabels.contains label) = true

/-- L03. A transformed label domain remains available after persistence and at prediction time. -/
abbrev CodecPersistedAndPredictionsDecodable (l : LabelTrace) : Prop :=
  if l.codecEnabled then
    l.persistedMappings ≠ [] ∧
    l.persistedMappings = l.runtimeMappings ∧
    PredictionsDecode l
  else
    l.persistedMappings = [] ∧ l.predictedCodes = []

/-- A01. Prediction values are in one-to-one ordered correspondence with source samples. -/
abbrev PredictionsBoundToSourceRows (p : PredictionTrace) : Prop :=
  (p.scope = "test" ∨ p.scope = "application") ∧
  p.sourceRowIds ≠ [] ∧
  p.predictionValues ≠ [] ∧
  p.sourceRowIds = p.sampleRowIds ∧
  natNodupB p.sampleRowIds = true ∧
  p.predictionValues.length = p.sampleRowIds.length

/-- A02. Exported pairs stay exact and identity mismatches are rejected instead of aligned by position. -/
abbrev ArtifactPairsAlignedAndMismatchRejected (p : PredictionTrace) : Prop :=
  p.sampleRowIds = p.artifactRowIds ∧
  p.predictionValues = p.artifactPredictionValues ∧
  p.artifactMismatchPolicy = "reject"

/-- A03. The model, exported artifact, and active run share one non-empty run identity. -/
abbrev ModelArtifactAndStateShareRun (p : PredictionTrace) (e : ExecutionTrace) : Prop :=
  p.modelRunId ≠ "" ∧
  p.modelRunId = p.artifactRunId ∧
  p.modelRunId = e.activeRunId

/-- E01. Every selected model is eligible and appears exactly once in the completed set. -/
abbrev SelectedModelsEligibleAndTrained (e : ExecutionTrace) : Prop :=
  e.eligibleModels ≠ [] ∧
  e.eligibleModels.Nodup ∧
  e.selectedModelIds ≠ [] ∧
  e.selectedModelIds.Nodup ∧
  e.trainedModelIds.Nodup ∧
  listSubsetB e.selectedModelIds e.eligibleModels = true ∧
  sameMembersB e.selectedModelIds e.trainedModelIds = true ∧
  e.trainedModelCount = e.trainedModelIds.length

/-- E02. UI-only options do not mutate the shared model registry. -/
abbrev ModelRegistryImmutableDuringRun (e : ExecutionTrace) : Prop :=
  e.registryBefore ≠ [] ∧
  e.registryBefore.Nodup ∧
  e.registryMutationOperations = [] ∧
  e.registryBefore = e.registryAfter

/-- The public verification surface contains exactly twenty independent business contracts. -/
structure PublicConforms (c : CaseTrace) : Prop where
  d01 : InputRowsIdentified c.dataset
  d02 : SplitIsDisjointPartition c.dataset
  d03 : SupervisedViewsRowAligned c.dataset
  d04 : ColumnRolesGuardedAndDisjoint c.dataset
  d05 : DerivedFeatureLineageSafe c.dataset
  d06 : FilteredRowsKeepLineage c.dataset
  p01 : EffectiveSchemaMatchesTraining c.pipeline
  p02 : StatefulFitUsesTrainingRowsOnly c.pipeline
  p03 : FittedStateReusedForModelAndInference c.pipeline
  p04 : ModelInputSchemaMatchesPipelineOutput c.pipeline
  p05 : DeclaredAndMaterializedStageOrderEqual c.pipeline
  p06 : ObservedStageOutputsFinite c.pipeline
  l01 : CodecTotalAndInjective c.labels
  l02 : OneCodecFittedOnceForAllSplits c.labels
  l03 : CodecPersistedAndPredictionsDecodable c.labels
  a01 : PredictionsBoundToSourceRows c.prediction
  a02 : ArtifactPairsAlignedAndMismatchRejected c.prediction
  a03 : ModelArtifactAndStateShareRun c.prediction c.execution
  e01 : SelectedModelsEligibleAndTrained c.execution
  e02 : ModelRegistryImmutableDuringRun c.execution

instance publicConformsDecidable (c : CaseTrace) : Decidable (PublicConforms c) :=
  if h01 : InputRowsIdentified c.dataset then
    if h02 : SplitIsDisjointPartition c.dataset then
      if h03 : SupervisedViewsRowAligned c.dataset then
        if h04 : ColumnRolesGuardedAndDisjoint c.dataset then
          if h05 : DerivedFeatureLineageSafe c.dataset then
            if h06 : FilteredRowsKeepLineage c.dataset then
              if h07 : EffectiveSchemaMatchesTraining c.pipeline then
                if h08 : StatefulFitUsesTrainingRowsOnly c.pipeline then
                  if h09 : FittedStateReusedForModelAndInference c.pipeline then
                    if h10 : ModelInputSchemaMatchesPipelineOutput c.pipeline then
                      if h11 : DeclaredAndMaterializedStageOrderEqual c.pipeline then
                        if h12 : ObservedStageOutputsFinite c.pipeline then
                          if h13 : CodecTotalAndInjective c.labels then
                            if h14 : OneCodecFittedOnceForAllSplits c.labels then
                              if h15 : CodecPersistedAndPredictionsDecodable c.labels then
                                if h16 : PredictionsBoundToSourceRows c.prediction then
                                  if h17 : ArtifactPairsAlignedAndMismatchRejected c.prediction then
                                    if h18 : ModelArtifactAndStateShareRun c.prediction c.execution then
                                      if h19 : SelectedModelsEligibleAndTrained c.execution then
                                        if h20 : ModelRegistryImmutableDuringRun c.execution then
                                          isTrue
                                            { d01 := h01, d02 := h02, d03 := h03, d04 := h04
                                            , d05 := h05, d06 := h06, p01 := h07, p02 := h08
                                            , p03 := h09, p04 := h10, p05 := h11, p06 := h12
                                            , l01 := h13, l02 := h14, l03 := h15, a01 := h16
                                            , a02 := h17, a03 := h18, e01 := h19, e02 := h20 }
                                        else isFalse (fun h => h20 h.e02)
                                      else isFalse (fun h => h19 h.e01)
                                    else isFalse (fun h => h18 h.a03)
                                  else isFalse (fun h => h17 h.a02)
                                else isFalse (fun h => h16 h.a01)
                              else isFalse (fun h => h15 h.l03)
                            else isFalse (fun h => h14 h.l02)
                          else isFalse (fun h => h13 h.l01)
                        else isFalse (fun h => h12 h.p06)
                      else isFalse (fun h => h11 h.p05)
                    else isFalse (fun h => h10 h.p04)
                  else isFalse (fun h => h09 h.p03)
                else isFalse (fun h => h08 h.p02)
              else isFalse (fun h => h07 h.p01)
            else isFalse (fun h => h06 h.d06)
          else isFalse (fun h => h05 h.d05)
        else isFalse (fun h => h04 h.d04)
      else isFalse (fun h => h03 h.d03)
    else isFalse (fun h => h02 h.d02)
  else isFalse (fun h => h01 h.d01)

def conformsB (c : CaseTrace) : Bool := decide (PublicConforms c)

abbrev Conforms (c : CaseTrace) : Prop := conformsB c = true

end GeoPiVerify
