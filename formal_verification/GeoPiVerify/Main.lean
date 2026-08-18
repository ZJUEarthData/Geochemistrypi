import GeoPiVerify

open Lean
open GeoPiVerify

private def failWith (message : String) : IO UInt32 := do
  IO.eprintln s!"trace checker error: {message}"
  return 2

private def parseBundle (path : String) : IO (Except String TraceBundle) := do
  try
    let raw ← IO.FS.readFile path
    match Json.parse raw with
    | .error message => return .error s!"invalid JSON: {message}"
    | .ok json =>
        match fromJson? json with
        | .error message => return .error s!"schema decode failed: {message}"
        | .ok bundle => return .ok bundle
  catch error =>
    return .error s!"cannot read '{path}': {error}"

private def validateEnvelope (bundle : TraceBundle) : Except String Unit := do
  if bundle.schemaVersion != 2 then
    throw s!"unsupported schemaVersion {bundle.schemaVersion}; expected 2"
  if bundle.sourceCommit.isEmpty then
    throw "sourceCommit must not be empty"
  if bundle.cases.isEmpty then
    throw "cases must not be empty"
  if !(decide (bundle.cases.map (·.caseId)).Nodup) then
    throw "caseId values must be unique"
  if bundle.cases.any (·.caseId.isEmpty) then
    throw "caseId must not be empty"
  if bundle.cases.any (fun c =>
      c.caseKind != "baseline" && c.caseKind != "counterexample" && c.caseKind != "production") then
    throw "caseKind must be baseline, counterexample, or production"
  if bundle.cases.any (fun c => c.caseKind == "baseline" && (!c.expectedConformant || !c.targetCheckId.isEmpty)) then
    throw "a baseline case must expect conformance and must not target a check"
  if bundle.cases.any (fun c => c.caseKind == "counterexample" && (c.expectedConformant || c.targetCheckId.isEmpty)) then
    throw "each counterexample must expect rejection and target one check"
  if bundle.cases.any (fun c => c.caseKind == "production" && !c.targetCheckId.isEmpty) then
    throw "a production case must not carry a counterexample target"
  if bundle.cases.any (fun c => c.caseKind == "counterexample" && !(publicCheckIds.contains c.targetCheckId)) then
    throw "counterexample targetCheckId is not a public check"

def main (args : List String) : IO UInt32 := do
  let path ← match args with
    | [path] => pure path
    | _ => return ← failWith "usage: geopi-tracecheck TRACE.json"
  let parsed ← parseBundle path
  let bundle ← match parsed with
    | .error message => return ← failWith message
    | .ok bundle => pure bundle
  match validateEnvelope bundle with
  | .error message => failWith message
  | .ok () =>
      let report := reportBundle bundle
      IO.println (toJson report).pretty
      /- Fixture expectations are test metadata only; they can never make a rejected
         production trace exit successfully. -/
      return if report.allCasesAccepted then 0 else 1
