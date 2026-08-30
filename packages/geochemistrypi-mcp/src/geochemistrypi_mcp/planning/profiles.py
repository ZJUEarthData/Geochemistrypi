"""Configuration-only scientific benchmark profiles compiled into public MCP requests."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator
from ruamel.yaml import YAML

from ..api.schemas import AnalysisRequestValue, DatasetReference, EnvironmentContract, EnvironmentProfileContract, ReproducibilityContract, StrictModel
from .artifact_mapping import unavailable_artifact_mapping
from .interaction_plan import INTERACTION_PLAN_VERSION, InteractionPlan

_MAX_PROFILE_BYTES = 1024 * 1024
_REQUEST_ADAPTER = TypeAdapter(AnalysisRequestValue)


class BenchmarkProfileError(ValueError):
    """Raised when a benchmark profile is unsafe, invalid, or internally inconsistent."""


class BenchmarkProfileNotReadyError(BenchmarkProfileError):
    """Raised when a valid configuration template is deliberately non-executable."""


class BenchmarkIdentity(StrictModel):
    """Human and machine identity; never used for execution branching."""

    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$", max_length=120)
    title: str = Field(min_length=1, max_length=255)
    citation: str | None = Field(None, min_length=1, max_length=1_000)


class ProfileState(StrictModel):
    """Evidence gate kept separate from strict executable request schemas."""

    execution_ready: bool = True
    comparison_ready: bool = False
    claim_ready: bool = False
    blocker_category: Literal[
        "READY",
        "MCP_CONTRACT_GAP",
        "CLI_CAPABILITY_GAP",
        "DATA_OR_PARAMETER_GAP",
        "SCIENTIFIC_UNCERTAIN",
    ] = "READY"
    evidence_level: str = Field("unspecified", min_length=1, max_length=120)
    unknown_fields: tuple[str, ...] = Field(default=(), max_length=256)
    notes: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("evidence_level", "unknown_fields", "notes")
    @classmethod
    def validate_text(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or "\n" in normalized or "\r" in normalized:
                raise ValueError("profile-state text must be non-blank and single-line")
            return normalized
        normalized_values = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized_values):
            raise ValueError("profile-state lists must contain non-blank single-line values")
        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("profile-state lists must not contain duplicates")
        return normalized_values

    @model_validator(mode="after")
    def validate_gate(self) -> "ProfileState":
        if self.execution_ready and self.blocker_category != "READY":
            raise ValueError("an execution-ready profile must use blocker_category='READY'")
        if self.execution_ready and self.unknown_fields:
            raise ValueError("an execution-ready profile cannot declare unknown fields")
        if not self.execution_ready and self.blocker_category == "READY":
            raise ValueError("a blocked profile must identify a non-READY blocker category")
        if self.claim_ready and not (self.execution_ready and self.comparison_ready):
            raise ValueError("claim_ready requires execution_ready and comparison_ready")
        return self


class ProfileStage(StrictModel):
    """One paper-agnostic node in a declared scientific workflow graph."""

    stage_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$", max_length=120)
    family: str = Field(min_length=1, max_length=120)
    method: str = Field(min_length=1, max_length=120)
    inputs: tuple[str, ...] = Field(default=(), max_length=128)
    outputs: tuple[str, ...] = Field(default=(), max_length=128)
    execution_phase: Literal[
        "core_scientific",
        "post_run_comparison",
        "derived_supporting_evidence",
    ] = "core_scientific"

    @field_validator("family", "method", "inputs", "outputs")
    @classmethod
    def validate_stage_text(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or "\n" in normalized or "\r" in normalized:
                raise ValueError("stage names must be non-blank single-line values")
            return normalized
        normalized_values = tuple(item.strip() for item in value)
        if any(not item or "\n" in item or "\r" in item for item in normalized_values):
            raise ValueError("stage ports must contain non-blank single-line values")
        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("stage ports must not contain duplicates")
        return normalized_values


class ProfileWorkflow(StrictModel):
    """Paper-agnostic workflow identity expected after plan compilation."""

    family: Literal[
        "time_series",
        "supervised_learning",
        "dimension_reduction",
        "clustering",
        "anomaly_detection",
    ]
    mode: str = Field(min_length=1, max_length=80)
    method: str | None = Field(None, min_length=1, max_length=120)
    stages: tuple[ProfileStage, ...] = Field(default=(), max_length=64)

    @field_validator("mode", "method")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("workflow values must be non-blank single-line names")
        return normalized

    @model_validator(mode="after")
    def validate_stage_graph(self) -> "ProfileWorkflow":
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("workflow stage ids must be unique")
        producers: dict[str, str] = {}
        for stage in self.stages:
            for output in stage.outputs:
                if output in producers:
                    raise ValueError(f"workflow output {output!r} has multiple producers")
                producers[output] = stage.stage_id
        dependencies = {stage.stage_id: {producers[item] for item in stage.inputs if item in producers} for stage in self.stages}
        remaining = {stage_id: set(required) for stage_id, required in dependencies.items()}
        resolved: set[str] = set()
        while remaining:
            available = {stage_id for stage_id, required in remaining.items() if required <= resolved}
            if not available:
                raise ValueError("workflow stages contain a dependency cycle")
            resolved.update(available)
            for stage_id in available:
                del remaining[stage_id]
        return self


class ProfileAcceptanceRules(StrictModel):
    """Evaluation and evidence conditions carried into the MCP request."""

    evaluation: dict[str, Any] = Field(default_factory=dict, max_length=64)
    require_execution_ready: bool = False
    required_artifact_ids: tuple[str, ...] = Field(default=(), max_length=128)


class BenchmarkProfile(StrictModel):
    """Versioned configuration that expands into an ordinary strict analysis request."""

    profile_version: Literal[1] = 1
    benchmark: BenchmarkIdentity
    profile_state: ProfileState = Field(default_factory=ProfileState)
    workflow: ProfileWorkflow
    dataset: DatasetReference | dict[str, Any]
    environment: EnvironmentContract = Field(default_factory=EnvironmentContract)
    environment_profile: EnvironmentProfileContract | None = None
    reproducibility: ReproducibilityContract = Field(default_factory=ReproducibilityContract)
    parameters: dict[str, Any] = Field(default_factory=dict, max_length=256)
    expected_artifacts: tuple[dict[str, Any], ...] = Field(default=(), max_length=128)
    acceptance_rules: ProfileAcceptanceRules = Field(default_factory=ProfileAcceptanceRules)

    @model_validator(mode="after")
    def validate_reserved_parameters(self) -> "BenchmarkProfile":
        reserved = {
            "task",
            "training_dataset",
            "training_dataset_path",
            "reproducibility",
            "artifact_requirements",
            "evaluation",
        }
        conflicts = sorted(reserved & set(self.parameters))
        if conflicts:
            raise ValueError(f"profile parameters cannot override contract-owned fields: {conflicts}")
        environment_specified = any(value not in (None, {}, (), []) for value in self.environment.model_dump(mode="json").values())
        if self.environment_profile is not None and environment_specified:
            raise ValueError("environment_profile replaces the legacy inline environment contract")
        if self._unadapted_core_stages and self.profile_state.execution_ready:
            raise ValueError("multi-stage workflow profiles remain blocked until a pipeline adapter is available")
        detected_unknowns = _unknown_paths(self.model_dump(mode="json", exclude={"profile_state"}))
        if self.profile_state.execution_ready and detected_unknowns:
            raise ValueError(f"execution-ready profiles cannot contain UNKNOWN values: {detected_unknowns}")
        return self

    @property
    def _unadapted_core_stages(self) -> tuple[ProfileStage, ...]:
        """Return declared core stages that need a pipeline adapter before execution."""
        return tuple(stage for stage in self.workflow.stages if stage.execution_phase == "core_scientific")

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        """Return declared and detected uncertainty without interpreting it as a value."""
        detected = _unknown_paths(self.model_dump(mode="json", exclude={"profile_state"}))
        return tuple(dict.fromkeys((*self.profile_state.unknown_fields, *detected)))

    @property
    def blocking_issues(self) -> tuple[str, ...]:
        issues = [f"Profile readiness is {self.profile_state.blocker_category}." for _ in range(1 if not self.profile_state.execution_ready else 0)]
        issues.extend(f"Unresolved profile field: {field}." for field in self.unresolved_fields)
        issues.extend(self.profile_state.notes)
        if self._unadapted_core_stages:
            issues.append("The declared multi-stage workflow has no public CLI pipeline adapter.")
        return tuple(dict.fromkeys(issues))

    def to_analysis_request(self) -> AnalysisRequestValue:
        """Compile configuration only; all scientific validation remains in the request schemas."""
        if not self.profile_state.execution_ready or self.unresolved_fields or self._unadapted_core_stages:
            details = "; ".join(self.blocking_issues) or "profile evidence is incomplete"
            raise BenchmarkProfileNotReadyError(f"Benchmark profile is a non-executable template: {details}")
        task, workflow_values = _request_workflow(self.workflow)
        environment_value = self.environment.model_dump(mode="json", exclude_none=True)
        if task == "time_series":
            unsupported_environment_fields = sorted(set(environment_value) - {"expected_identity_sha256", "dependency_versions"})
            if unsupported_environment_fields:
                raise BenchmarkProfileError("Time Series profiles must freeze the full runtime with expected_identity_sha256; " f"unsupported individual fields: {unsupported_environment_fields}")
        dataset_value = self.dataset.model_dump(mode="json") if hasattr(self.dataset, "model_dump") else self.dataset
        reproducibility_value = self.reproducibility.model_dump(mode="json", exclude_none=True)
        reproducibility_value["environment"] = environment_value
        if task == "time_series":
            requested_seed = self.reproducibility.model_seed
            if requested_seed is not None and requested_seed != self.parameters.get("seed"):
                raise BenchmarkProfileError("Time Series profile model_seed must match the top-level request seed")
            reproducibility_value = {"environment": environment_value}
        request_value = {
            **self.parameters,
            **workflow_values,
            "task": task,
            "training_dataset": dataset_value,
            "evaluation": self.acceptance_rules.evaluation,
            "reproducibility": reproducibility_value,
            "environment_profile": (self.environment_profile.model_dump(mode="json") if self.environment_profile is not None else None),
            "artifact_requirements": list(self.expected_artifacts),
        }
        try:
            request = _REQUEST_ADAPTER.validate_python(request_value)
        except ValueError as exc:
            raise BenchmarkProfileError(f"Benchmark profile does not compile into a valid MCP analysis request: {exc}") from exc
        artifact_ids = {item.requirement_id for item in request.artifact_requirements}
        missing = sorted(set(self.acceptance_rules.required_artifact_ids) - artifact_ids)
        if missing:
            raise BenchmarkProfileError(f"Acceptance rules reference undeclared artifact ids: {missing}")
        return request

    def compatibility_plan(self) -> InteractionPlan:
        """Compile an unresolved template into a diagnostic plan that cannot run."""
        if self.profile_state.execution_ready and not self.unresolved_fields and not self._unadapted_core_stages:
            raise BenchmarkProfileError("execution-ready profiles require the ordinary analysis-plan compiler")
        issues = self.blocking_issues
        if not issues:
            issues = ("The profile is not executable.",)
        artifact_mappings = tuple(
            unavailable_artifact_mapping(
                scientific_type=str(requirement.get("scientific_type", "scientific_artifact")),
                output_role=str(requirement.get("output_role") or requirement.get("requirement_id") or "scientific.output"),
                reason="The profile is blocked before a public CLI adapter can be selected.",
            )
            for requirement in self.expected_artifacts
        )
        environment_profile_id = self.environment_profile.profile_id if self.environment_profile is not None else None
        environment_profile_identity = (
            hashlib.sha256(
                json.dumps(
                    self.environment_profile.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if self.environment_profile is not None
            else None
        )
        method = self.workflow.method or self.workflow.mode
        return InteractionPlan(
            schema_version=INTERACTION_PLAN_VERSION,
            name=f"profile-template-{self.workflow.family}-{self.workflow.mode}",
            public_command=(),
            steps=(),
            workflow_family=self.workflow.family,
            workflow_mode=self.workflow.mode,
            method=method,
            scientific_contract_id=(f"scientific-contract-v2/profile-template/" f"{self.workflow.family}/{self.workflow.mode}/{method}"),
            adapter_id=None,
            adapter_version=None,
            environment_profile="profile.environment_profile",
            environment_profile_id=environment_profile_id,
            environment_profile_identity_sha256=environment_profile_identity,
            artifact_contract=tuple(str(requirement.get("requirement_id", requirement.get("scientific_type", "scientific_artifact"))) for requirement in self.expected_artifacts),
            artifact_mappings=artifact_mappings,
            adapter_status=("unavailable" if self.profile_state.blocker_category == "CLI_CAPABILITY_GAP" or self._unadapted_core_stages else "requirements_unmet"),
            execution_ready=False,
            blocking_issues=issues,
        )


def _unknown_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_unknown_paths(item, child))
        return tuple(paths)
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            paths.extend(_unknown_paths(item, child))
        return tuple(paths)
    if isinstance(value, str) and value.strip().upper() == "UNKNOWN":
        return (prefix or "<root>",)
    return ()


def _request_workflow(workflow: ProfileWorkflow) -> tuple[str, dict[str, str]]:
    if workflow.family == "time_series":
        if workflow.mode not in {"subaerial_proportion", "element_mean"}:
            raise BenchmarkProfileError(f"Unsupported Time Series profile mode: {workflow.mode!r}")
        return "time_series", {"mode": workflow.mode}
    if workflow.family == "supervised_learning":
        if workflow.mode not in {"classification", "regression"}:
            raise BenchmarkProfileError(f"Unsupported supervised-learning profile mode: {workflow.mode!r}")
        return workflow.mode, {}
    if workflow.family == "dimension_reduction":
        return "decomposition", {}
    if workflow.family in {"clustering", "anomaly_detection"}:
        return workflow.family, {}
    raise BenchmarkProfileError(f"Unsupported workflow family: {workflow.family!r}")


def load_benchmark_profile(path: Path, expected_sha256: str | None = None) -> tuple[BenchmarkProfile, str]:
    """Load one bounded YAML profile and return its content hash for provenance."""
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise BenchmarkProfileError("Benchmark profiles must be YAML files.")
    try:
        size_bytes = resolved.stat().st_size
        payload = resolved.read_bytes()
    except OSError as exc:
        raise BenchmarkProfileError(f"Benchmark profile is unavailable: {resolved}") from exc
    if not resolved.is_file() or size_bytes > _MAX_PROFILE_BYTES:
        raise BenchmarkProfileError("Benchmark profile is not a bounded regular file.")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise BenchmarkProfileError("Benchmark profile does not match expected_sha256.")
    yaml = YAML(typ="safe")
    try:
        value = yaml.load(payload.decode("utf-8"))
        profile = BenchmarkProfile.model_validate(value)
    except (UnicodeError, ValueError) as exc:
        raise BenchmarkProfileError(f"Benchmark profile is invalid: {exc}") from exc
    return profile, actual_sha256


def attest_profile_plan(profile: BenchmarkProfile, plan: Any) -> tuple[str, ...]:
    """Return configuration mismatches without changing or dispatching the plan."""
    mismatches = []
    if plan.workflow_family != profile.workflow.family:
        mismatches.append(f"compiled workflow family {plan.workflow_family!r} does not match the profile")
    if plan.workflow_mode != profile.workflow.mode:
        mismatches.append(f"compiled workflow mode {plan.workflow_mode!r} does not match the profile")
    if profile.workflow.method is not None and plan.method != profile.workflow.method:
        mismatches.append(f"compiled workflow method {plan.method!r} does not match the profile")
    if profile.environment_profile is not None and plan.environment_profile_id != profile.environment_profile.profile_id:
        mismatches.append("compiled environment profile identity does not match the benchmark profile")
    if profile.acceptance_rules.require_execution_ready and not plan.execution_ready:
        mismatches.append("the profile requires an execution-ready adapter but the compiled plan is blocked")
    return tuple(mismatches)
