import GeoPiVerify.Checker

namespace GeoPiVerify

theorem accepted_iff_conforms (c : CaseTrace) :
    accepted c = true ↔ Conforms c := by
  rfl

theorem accepted_implies_role_boundary (c : CaseTrace)
    (h : accepted c = true) : ColumnRolesGuardedAndDisjoint c.dataset := by
  change decide (PublicConforms c) = true at h
  exact (of_decide_eq_true h).d04

theorem accepted_implies_fit_scope (c : CaseTrace)
    (h : accepted c = true) : StatefulFitUsesTrainingRowsOnly c.pipeline := by
  change decide (PublicConforms c) = true at h
  exact (of_decide_eq_true h).p02

theorem accepted_implies_codec_persistence (c : CaseTrace)
    (h : accepted c = true) : CodecPersistedAndPredictionsDecodable c.labels := by
  change decide (PublicConforms c) = true at h
  exact (of_decide_eq_true h).l03

theorem accepted_implies_artifact_alignment (c : CaseTrace)
    (h : accepted c = true) : ArtifactPairsAlignedAndMismatchRejected c.prediction := by
  change decide (PublicConforms c) = true at h
  exact (of_decide_eq_true h).a02

#print axioms accepted_iff_conforms
#print axioms accepted_implies_role_boundary
#print axioms accepted_implies_fit_scope
#print axioms accepted_implies_codec_persistence
#print axioms accepted_implies_artifact_alignment

end GeoPiVerify
