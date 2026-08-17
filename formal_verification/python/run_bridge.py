#!/usr/bin/env python3
"""Run counterexample self-checking first, then the mutation-free production audit."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

FORMAL = Path(__file__).resolve().parents[1]
ROOT = FORMAL.parent
RESULTS = FORMAL / "results"
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("GEOPI_BRIDGE_TIMEOUT_SECONDS", "600"))


def pinned_toolchain() -> str:
    """Read the project's single source of truth for the Lean toolchain."""
    toolchain = (FORMAL / "lean-toolchain").read_text(encoding="utf-8").strip()
    if not toolchain:
        raise RuntimeError("formal_verification/lean-toolchain must not be empty")
    return toolchain


def portable_log(value: str) -> str:
    """Remove machine paths and normalize console output for stable logs."""
    replacements = [
        (str(Path(sys.prefix).resolve()), "<python-env>"),
        (str(ROOT.resolve()), "<repo>"),
        (str(Path.home().resolve()), "<home>"),
    ]
    for prefix, replacement in replacements:
        value = value.replace(prefix, replacement)
    normalized = "\n".join(line.rstrip() for line in value.splitlines())
    return normalized + ("\n" if value.endswith(("\n", "\r")) else "")


def run(name: str, command: list[str], *, cwd: Path, env: dict[str, str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seconds = timeout or DEFAULT_TIMEOUT_SECONDS
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=seconds)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        (RESULTS / f"{name}.stdout.txt").write_text(portable_log(stdout), encoding="utf-8")
        timeout_log = stderr + f"\n{name} timed out after {seconds} seconds\n"
        (RESULTS / f"{name}.stderr.txt").write_text(portable_log(timeout_log), encoding="utf-8")
        raise RuntimeError(f"{name} timed out after {seconds} seconds") from error
    (RESULTS / f"{name}.stdout.txt").write_text(portable_log(completed.stdout), encoding="utf-8")
    (RESULTS / f"{name}.stderr.txt").write_text(portable_log(completed.stderr), encoding="utf-8")
    return completed


def require_code(result: subprocess.CompletedProcess[str], allowed: set[int], name: str) -> None:
    if result.returncode not in allowed:
        raise RuntimeError(f"{name} exited {result.returncode}. See formal_verification/results/{name}.stderr.txt")


def json_report(result: subprocess.CompletedProcess[str], output: Path) -> dict[str, Any]:
    report = json.loads(result.stdout)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toolchain",
        help="override formal_verification/lean-toolchain for one diagnostic run",
    )
    args = parser.parse_args()
    toolchain = args.toolchain or pinned_toolchain()

    RESULTS.mkdir(parents=True, exist_ok=True)
    counter_trace = RESULTS / "counterexample_trace.json"
    counter_observations = RESULTS / "counterexample_observations.json"
    production_trace = RESULTS / "production_trace.json"
    production_observations = RESULTS / "production_observations.json"
    generated = FORMAL / "GeoPiVerify" / "Generated" / "CurrentRun.lean"
    counter_python_path = RESULTS / "counterexample_python_report.json"
    production_python_path = RESULTS / "production_python_report.json"

    env = dict(os.environ)
    env["ELAN_TOOLCHAIN"] = toolchain
    # Elan installs extensionless shims on macOS and .exe shims on Windows.
    # Let the operating system resolve the correct filename through PATH.
    elan_home = Path(env.get("ELAN_HOME", Path.home() / ".elan"))
    elan_bin = elan_home / "bin"
    env["PATH"] = f"{elan_bin}{os.pathsep}{env.get('PATH', '')}"

    commands: dict[str, dict[str, Any]] = {}
    try:
        version = run("lean_version", ["lean", "--version"], cwd=FORMAL, env=env, timeout=60)
        require_code(version, {0}, "lean_version")
        commands["leanVersion"] = {"exitCode": 0, "stdout": version.stdout.strip()}

        counter_probe = run(
            "counterexample_probe",
            [sys.executable, str(FORMAL / "python" / "generate_counterexamples.py"), "--trace", str(counter_trace), "--observations", str(counter_observations)],
            cwd=ROOT,
            env=env,
        )
        require_code(counter_probe, {0}, "counterexample_probe")
        commands["counterexampleProbe"] = {"exitCode": 0}

        generation = run(
            "generate_counterexample_run",
            [sys.executable, str(FORMAL / "python" / "generate_current_run.py"), "--counterexamples", str(counter_trace), "--output", str(generated)],
            cwd=ROOT,
            env=env,
        )
        require_code(generation, {0}, "generate_counterexample_run")
        commands["generateCounterexampleRun"] = {"exitCode": 0}

        counter_build = run("lean_build_counterexamples", ["lake", "build", "--wfail"], cwd=FORMAL, env=env)
        require_code(counter_build, {0}, "lean_build_counterexamples")
        commands["leanBuildCounterexamples"] = {"exitCode": 0}

        counter_python = run(
            "counterexample_python_check",
            [sys.executable, str(FORMAL / "python" / "check_trace.py"), str(counter_trace), "--output", str(counter_python_path)],
            cwd=ROOT,
            env=env,
        )
        require_code(counter_python, {1}, "counterexample_python_check")
        counter_python_report = json.loads(counter_python_path.read_text(encoding="utf-8"))
        commands["counterexamplePythonCheck"] = {"exitCode": counter_python.returncode}

        counter_lean = run("counterexample_lean_check", ["lake", "exe", "geopi-tracecheck", str(counter_trace)], cwd=FORMAL, env=env)
        require_code(counter_lean, {1}, "counterexample_lean_check")
        counter_lean_report = json_report(counter_lean, RESULTS / "counterexample_lean_report.json")
        commands["counterexampleLeanCheck"] = {"exitCode": counter_lean.returncode}

        counter_reports_equal = counter_python_report == counter_lean_report
        counter_suite_passed = all(
            [
                counter_python_report["counterexampleCoverageComplete"],
                counter_python_report["allCounterexamplesIsolated"],
                counter_python_report["allExpectationsMatched"],
                counter_python_report["cases"][0]["accepted"],
            ]
        )
        if not counter_reports_equal or not counter_suite_passed:
            raise RuntimeError("counterexample self-check did not pass before production audit")

        production_probe = run(
            "runtime_probe",
            [sys.executable, str(FORMAL / "python" / "runtime_probe.py"), "--trace", str(production_trace), "--observations", str(production_observations)],
            cwd=ROOT,
            env=env,
        )
        require_code(production_probe, {0}, "runtime_probe")
        commands["runtimeProbe"] = {"exitCode": 0}

        final_generation = run(
            "generate_current_run",
            [sys.executable, str(FORMAL / "python" / "generate_current_run.py"), "--counterexamples", str(counter_trace), "--production", str(production_trace), "--output", str(generated)],
            cwd=ROOT,
            env=env,
        )
        require_code(final_generation, {0}, "generate_current_run")
        commands["generateCurrentRun"] = {"exitCode": 0}

        final_build = run("lean_build_final", ["lake", "build", "--wfail"], cwd=FORMAL, env=env)
        require_code(final_build, {0}, "lean_build_final")
        commands["leanBuildFinal"] = {"exitCode": 0}

        production_python = run(
            "production_python_check",
            [sys.executable, str(FORMAL / "python" / "check_trace.py"), str(production_trace), "--output", str(production_python_path)],
            cwd=ROOT,
            env=env,
        )
        require_code(production_python, {0, 1}, "production_python_check")
        production_python_report = json.loads(production_python_path.read_text(encoding="utf-8"))
        commands["productionPythonCheck"] = {"exitCode": production_python.returncode}

        production_lean = run("production_lean_check", ["lake", "exe", "geopi-tracecheck", str(production_trace)], cwd=FORMAL, env=env)
        require_code(production_lean, {0, 1}, "production_lean_check")
        production_lean_report = json_report(production_lean, RESULTS / "production_lean_report.json")
        commands["productionLeanCheck"] = {"exitCode": production_lean.returncode}

        kernel = run("leanchecker", ["lake", "env", "leanchecker", "--fresh", "GeoPiVerify"], cwd=FORMAL, env=env)
        require_code(kernel, {0}, "leanchecker")
        commands["leanchecker"] = {"exitCode": 0}

        tests = run("pytest", [sys.executable, "-m", "pytest", "-q", str(FORMAL / "tests")], cwd=ROOT, env=env)
        require_code(tests, {0}, "pytest")
        match = re.search(r"(\d+) passed", tests.stdout)
        commands["pytest"] = {"exitCode": 0, "passed": int(match.group(1)) if match else None}

        production_reports_equal = production_python_report == production_lean_report
        production_case = production_python_report["cases"][0]
        production_conforms = bool(production_case["accepted"])
        failed_ids = list(production_case["failedCheckIds"])
        passed_count = sum(item["passed"] for item in production_case["checks"])
        expected_counter_exit = 1
        expected_production_exit = 0 if production_conforms else 1
        checker_exit_codes_match = (
            counter_python.returncode == counter_lean.returncode == expected_counter_exit and production_python.returncode == production_lean.returncode == expected_production_exit
        )
        bridge_passed = all(
            [
                counter_reports_equal,
                production_reports_equal,
                counter_suite_passed,
                checker_exit_codes_match,
                counter_build.returncode == 0,
                final_build.returncode == 0,
                kernel.returncode == 0,
                tests.returncode == 0,
            ]
        )
        summary = {
            "bridgePassed": bridge_passed,
            "productionConforms": production_conforms,
            "counterexampleSuitePassed": counter_suite_passed,
            "counterexampleReportsExactlyEqual": counter_reports_equal,
            "productionReportsExactlyEqual": production_reports_equal,
            "checkerExitCodesMatch": checker_exit_codes_match,
            "publicCheckCount": len(production_case["checks"]),
            "productionPassedCheckCount": passed_count,
            "productionFailedCheckCount": len(failed_ids),
            "productionFailedCheckIds": failed_ids,
            "counterexampleCaseCount": counter_python_report["caseCount"],
            "counterexampleCount": counter_python_report["counterexampleCount"],
            "coveredCheckCount": counter_python_report["coveredCheckCount"],
            "counterexampleCoverageComplete": counter_python_report["counterexampleCoverageComplete"],
            "allCounterexamplesIsolated": counter_python_report["allCounterexamplesIsolated"],
            "commands": commands,
        }
        (RESULTS / "bridge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if bridge_passed else 1
    except Exception as error:
        failure = {"bridgePassed": False, "error": f"{type(error).__name__}: {error}", "commands": commands}
        (RESULTS / "bridge_summary.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(failure["error"], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
