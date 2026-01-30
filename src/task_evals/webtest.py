"""
Evaluation functions for webtest task.

This module provides evaluation functions for webtest workspace directories,
running playwright-based tests and calculating scores.
"""

import json
import shutil
import subprocess
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import dspy
from dotenv import dotenv_values

class ShortenMessageSignature(dspy.Signature):
    """Shorten an error message to a single line within a max length."""
    message = dspy.InputField(desc="Error or failure message to shorten.")
    max_length = dspy.InputField(desc="Maximum length in characters.")
    shortened = dspy.OutputField(desc="Shortened single-line message within max_length.")


class ShortenMessage(dspy.Module):
    """DSPy module to shorten error messages."""
    def __init__(self):
        super().__init__()
        self.shorten = dspy.Predict(ShortenMessageSignature)

    def forward(self, message: str, max_length: int):
        return self.shorten(message=message, max_length=max_length)


# ---------------------------------------------------------------------------
# V8 coverage conftest – injected into the workspace before running tests
# ---------------------------------------------------------------------------
_COVERAGE_CONFTEST = "" #open(Path(__file__).parent / "conftest.py", "r", encoding="utf-8").read()

def _prepare_coverage_conftest(workspace_path: Path) -> Optional[str]:
    """
    Write a coverage-instrumented conftest.py directly into the workspace.

    If the workspace already has a root-level conftest.py, the coverage code is
    prepended so both coexist.  Returns the original file content (so the caller
    can restore it later) or ``None`` if there was no pre-existing conftest.
    """
    conftest_file = workspace_path / "conftest.py"

    if conftest_file.exists():
        original = conftest_file.read_text(encoding="utf-8")
        content = _COVERAGE_CONFTEST + "\n\n# --- original conftest.py ---\n\n" + original
    else:
        original = None
        content = _COVERAGE_CONFTEST

    conftest_file.write_text(content, encoding="utf-8")
    return original


def _restore_conftest(workspace_path: Path, original: Optional[str]) -> None:
    """Undo what ``_prepare_coverage_conftest`` did."""
    conftest_file = workspace_path / "conftest.py"
    if original is not None:
        conftest_file.write_text(original, encoding="utf-8")
    else:
        conftest_file.unlink(missing_ok=True)


def _compute_coverage_metrics(coverage_dir: Path) -> Dict[str, Any]:
    """
    Parse V8 precise-coverage JSON files and return aggregate metrics.

    Returns a dict with:
        scripts            – number of distinct script URLs
        functions_total    – total JS functions seen
        functions_covered  – functions executed at least once
        bytes_total        – total source bytes across all functions
        bytes_covered      – source bytes covered (block-level when available)
        function_coverage  – ratio (0-1)
        byte_coverage      – ratio (0-1)
        per_script         – {url: {same keys minus per_script}} breakdown
    """
    empty: Dict[str, Any] = {
        "scripts": 0,
        "functions_total": 0,
        "functions_covered": 0,
        "bytes_total": 0,
        "bytes_covered": 0,
        "function_coverage": 0.0,
        "byte_coverage": 0.0,
        "per_script": {},
    }

    if not coverage_dir.is_dir():
        return empty

    json_files = sorted(coverage_dir.glob("*.precise.json"))
    if not json_files:
        return empty

    # Merge coverage entries across files, keyed by script URL
    scripts: Dict[str, list] = {}
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("result", []):
            url = entry.get("url", "")
            # keep only application scripts (http(s) / file)
            if not url or not url.startswith(("http://", "https://", "file://")):
                continue
            scripts.setdefault(url, []).extend(entry.get("functions", []))

    if not scripts:
        return empty

    total_funcs = 0
    covered_funcs = 0
    total_bytes = 0
    covered_bytes = 0
    per_script: Dict[str, Any] = {}

    for url, functions in scripts.items():
        sf = sc = sb = scb = 0
        for func in functions:
            ranges = func.get("ranges", [])
            if not ranges:
                continue
            root = ranges[0]
            func_size = root["endOffset"] - root["startOffset"]
            if func_size <= 0:
                continue

            sf += 1
            if root["count"] > 0:
                sc += 1

            sb += func_size
            if func.get("isBlockCoverage", False) and len(ranges) > 1:
                # Sweep-line: boundaries split into segments; innermost
                # range (last match, since V8 lists outer-to-inner) wins.
                boundaries = sorted(
                    {r["startOffset"] for r in ranges}
                    | {r["endOffset"] for r in ranges}
                )
                seg_covered = 0
                for i in range(len(boundaries) - 1):
                    pos = boundaries[i]
                    seg_len = boundaries[i + 1] - pos
                    count = 0
                    for r in ranges:
                        if r["startOffset"] <= pos < r["endOffset"]:
                            count = r["count"]
                    if count > 0:
                        seg_covered += seg_len
                scb += seg_covered
            else:
                # No block detail – fall back to function-level
                if root["count"] > 0:
                    scb += func_size

        total_funcs += sf
        covered_funcs += sc
        total_bytes += sb
        covered_bytes += scb
        per_script[url] = {
            "functions_total": sf,
            "functions_covered": sc,
            "bytes_total": sb,
            "bytes_covered": scb,
            "function_coverage": round(sc / sf, 4) if sf else 0.0,
            "byte_coverage": round(scb / sb, 4) if sb else 0.0,
        }

    return {
        "scripts": len(scripts),
        "functions_total": total_funcs,
        "functions_covered": covered_funcs,
        "bytes_total": total_bytes,
        "bytes_covered": covered_bytes,
        "function_coverage": round(covered_funcs / total_funcs, 4) if total_funcs else 0.0,
        "byte_coverage": round(covered_bytes / total_bytes, 4) if total_bytes else 0.0,
        "per_script": per_script,
    }


def _count_test_functions(test_files: List[Path], test_function_pattern: re.Pattern) -> int:
    """Count total test functions across all files."""
    total = 0
    for test_file in test_files:
        with open(test_file, 'r') as f:
            content = f.read()
            test_functions = test_function_pattern.findall(content)
            total += len(test_functions)
    return total


def _calculate_score(tests_passed: int, tests_total: int, expected_tests: int) -> Tuple[float, float]:
    """
    Calculate score and pass rate based on test results.

    Returns:
        Tuple of (score, pass_rate)
    """
    pass_rate = tests_passed / tests_total if tests_total > 0 else 0.0

    if tests_total >= expected_tests:
        score = pass_rate
    else:
        score = tests_passed / expected_tests if expected_tests > 0 else 0.0

    score = min(score, 1.0)
    return score, pass_rate

def _shorten_message(
    lm: dspy.LM,
    failure_list: List[str],
    max_length: int = 160,
) -> str:
    """Return a single-line, shortened message with LM."""
    combined = "\n".join(failure_list).strip()
    if len(combined) <= max_length:
        return combined
    with dspy.context(lm=lm):
        predictor = ShortenMessage()
        shortened = predictor(message=combined, max_length=max_length).shortened
    return shortened

def _extract_failure_reason(
    file_results: List[Dict[str, Any]],
    lm: dspy.LM,
) -> Optional[str]:
    """Extract a short failure reason from test output."""
    failure_list = []
    for result in file_results:
        if result.get("status") in {"failed", "error", "timeout"}:
            failure_list.append(result.get("stderr", "").strip())
            failure_list.append(result.get("stdout", "").strip())
    if failure_list:
        return _shorten_message(lm, failure_list=failure_list)
    return ""

def _build_feedback(
    lm: dspy.LM,
    error_msg: Optional[str],
    tests_passed: int,
    tests_failed: int,
    tests_total: int,
    expected_tests: int,
    score: float,
    file_results: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Build a natural language feedback string based on results."""
    if error_msg:
        return error_msg

    if score >= 1.0:
        return "All tests passed successfully!"

    if tests_total > 0:
        base = (
            f"{tests_total} tests detected, {tests_failed} failed "
            f"({tests_passed}/{tests_total} passed)."
        )
    else:
        base = f"{tests_total} tests detected (0/{tests_total} passed)."

    reason_parts = []
    if tests_total < expected_tests:
        reason_parts.append(
            f"Expected {expected_tests} tests, found {tests_total}."
        )
    if tests_failed > 0 and file_results is not None:
        failure_reason = _extract_failure_reason(file_results, lm)
        if failure_reason:
            reason_parts.append(f"Reason: {failure_reason}")

    if reason_parts:
        return f"{base} " + " ".join(reason_parts)
    return base


def _create_error_result(workspace_dir: str, test_files: List[str], error_msg: str) -> Dict[str, Any]:
    """Create a standardized error result dictionary."""
    return {
        "workspace_dir": str(workspace_dir),
        "test_files": test_files,
        "error": error_msg,
        "feedback": error_msg,
        "success": False,
        "score": 0.0,
        "pass_rate": 0.0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
    }


def _run_file_with_pytest(
    test_file: Path,
    workspace_path: Path,
    timeout: int = 60,
) -> Optional[Tuple[int, int, subprocess.CompletedProcess]]:
    """
    Run a single test file with pytest inside Docker container.

    Args:
        test_file: Path to the test file
        workspace_path: Path to the workspace directory
        timeout: Timeout in seconds for this individual file (default: 60)

    Returns:
        Tuple of (tests_passed, tests_failed, result) if successful, None otherwise.
    """
    print(f"    Running pytest in Docker for {test_file}...")

    all_envs = os.environ.copy()
    env_config = dotenv_values(".env")
    forward_env = [key for key in env_config.keys() if key in all_envs]

    docker_volumes = []
    docker_volumes.append(f"{workspace_path.absolute()}:/workspace/project")
    docker_volumes.append(f"{os.path.abspath('.vertex-ai.json')}:/workspace/.vertex-ai.json:ro")

    # Build Docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "/workspace/.venv/bin/python",
        "-w", "/workspace/project",
        "-e", "GOOGLE_APPLICATION_CREDENTIALS=/workspace/.vertex-ai.json",
    ]
    for volume in docker_volumes:
        docker_cmd.extend(["-v", volume])
    for key in forward_env:
        docker_cmd.extend(["-e", f"{key}={all_envs[key]}"])
    docker_cmd.extend([
        "migration-analysis:latest",
        "-m", "pytest",
        "-v", "--tb=short",
        str(test_file)
    ])

    # print(f"    Docker command: {' '.join(docker_cmd)}")

    pytest_result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=timeout*2,  # Extra buffer for Docker overhead
    )

    print(f"    Docker pytest stdout:\n{pytest_result.stdout}")
    print(f"    Docker pytest return code: {pytest_result.returncode}")

    # Parse output to count test results (same logic as before)
    combined_output = f"{pytest_result.stdout}\n{pytest_result.stderr}"
    passed_pattern = re.compile(r"::(\w*test\w*)\s+PASSED", re.IGNORECASE)
    failed_pattern = re.compile(r"::(\w*test\w*)\s+FAILED", re.IGNORECASE)

    tests_passed = len(passed_pattern.findall(combined_output))
    tests_failed = len(failed_pattern.findall(combined_output))

    # If pytest detected tests, return the results
    if tests_passed + tests_failed > 0:
        return (tests_passed, tests_failed, pytest_result)

    return None

def _run_tests_batch(
    lm: dspy.LM,
    workspace_path: Path,
    test_files: List[Path],
    test_file_names: List[str],
    test_function_pattern: re.Pattern,
    main_block_pattern: re.Pattern,
    expected_tests: int,
    timeout_per_file: int = 60,
) -> Dict[str, Any]:
    """
    Run multiple test files, trying pytest first for each, then falling back to python3.

    Args:
        workspace_path: Path to the workspace directory
        test_files: List of test file paths
        test_file_names: List of test file names (for error reporting)
        test_function_pattern: Regex pattern to find test functions
        main_block_pattern: Regex pattern to find __main__ blocks
        expected_tests: Expected number of test functions
        timeout_per_file: Timeout in seconds for each individual test file (default: 60)

    Returns:
        Evaluation result dictionary.
    """
    tests_passed = 0
    tests_failed = 0
    file_results = []

    # Check if Docker is available once
    docker_available = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True,
    ).returncode == 0

    if not docker_available:
        raise RuntimeError("Docker is not available. Please install Docker to run test evaluation.")

    # Check if image exists
    image_check = subprocess.run(
        ["docker", "images", "-q", "migration-analysis:latest"],
        capture_output=True,
        text=True,
    )
    if not image_check.stdout.strip():
        raise RuntimeError("Docker image 'migration-analysis:latest' not found. Please build it with: docker build -t migration-analysis:latest .")

    pytest_available = True  # Always true in Docker

    # Set up V8 coverage collection – write conftest.py into workspace
    original_conftest = _prepare_coverage_conftest(workspace_path)
    print(f"  V8 coverage enabled (output: {workspace_path / 'coverage-raw'})")

    for test_file in test_files:
        relative_path = test_file.relative_to(workspace_path)
        print(f"  Running {relative_path}...")

        execution_method = "pytest"
        result = None
        status = None
        test_functions_in_file = 0

        try:
            # Count test functions in this file
            with open(test_file, 'r') as f:
                content = f.read()
                test_function_names = test_function_pattern.findall(content)
                test_functions_in_file = len(test_function_names)
                has_main_block = main_block_pattern.search(content)

            # Try pytest first if available
            if pytest_available:
                pytest_result = _run_file_with_pytest(
                    relative_path, workspace_path, timeout_per_file,
                )
                if pytest_result is None:
                    raise RuntimeError("Failed to parse pytest output or no tests detected.")
                file_passed, file_failed, result = pytest_result
                tests_passed += file_passed
                tests_failed += file_failed
                status = "passed" if file_failed == 0 else "failed"
                                    
        except subprocess.TimeoutExpired:
            # Handle timeout for this specific file
            print(f"    ⚠️  Test file timed out after {timeout_per_file}s")
            tests_failed += test_functions_in_file
            status = "timeout"
            result = type('obj', (object,), {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test execution timed out after {timeout_per_file}s'
            })()

        except Exception as e:
            # Handle any other error for this specific file
            print(f"    ⚠️  Test file failed with error: {str(e)}")
            tests_failed += test_functions_in_file
            status = "error"
            result = type('obj', (object,), {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test execution failed: {str(e)}'
            })()

        file_results.append({
            "file": str(relative_path),
            "test_functions": test_functions_in_file,
            "returncode": result.returncode if result else -1,
            "stdout": result.stdout if result else '',
            "stderr": result.stderr if result else '',
            "status": status,
            "execution_method": execution_method,
        })

    tests_total = tests_passed + tests_failed
    score, pass_rate = _calculate_score(tests_passed, tests_total, expected_tests)
    success = (tests_failed == 0 and tests_total > 0)
    feedback = _build_feedback(
        lm,
        None,
        tests_passed,
        tests_failed,
        tests_total,
        expected_tests,
        score,
        file_results,
    )

    print(f"Tests completed: {tests_passed}/{tests_total} test functions passed")
    print(f"Pass rate: {pass_rate:.1%}")
    print(f"Score: {score:.2f} (expected {expected_tests} tests, found {tests_total})")

    # Collect V8 coverage files and compute metrics
    cov_dir = workspace_path / "coverage-raw"
    coverage_metrics = _compute_coverage_metrics(cov_dir)
    if coverage_metrics["scripts"] > 0:
        print(
            f"  Coverage: {coverage_metrics['byte_coverage']:.1%} bytes, "
            f"{coverage_metrics['function_coverage']:.1%} functions "
            f"({coverage_metrics['scripts']} scripts)"
        )

    # Clean up coverage-raw directory
    if cov_dir.is_dir():
        shutil.rmtree(cov_dir, ignore_errors=True)

    # Restore original conftest (or remove the one we injected)
    _restore_conftest(workspace_path, original_conftest)

    return {
        "feedback": feedback,
        "workspace_dir": str(workspace_path),
        "test_files": test_file_names,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_total": tests_total,
        "score": score,
        "pass_rate": pass_rate,
        "success": success,
        "test_output": file_results,
        "coverage_metrics": coverage_metrics,
    }


def run_single_instance_eval(
    lm: dspy.LM,
    workspace_dir: str,
    example: Optional[dict] = None,
    expected_tests: int = 5,
    score_per_test: float = 0.2,
    timeout_per_file: int = 60,
) -> Dict[str, Any]:
    """
    Run test files in a workspace directory and calculate scores.

    This function automatically discovers all test files (test_*.py) in the workspace
    directory and runs them. Scoring is based on two components:
    1. Test count: Whether the agent wrote the expected number of tests
    2. Pass rate: Percentage of tests that pass

    Score calculation:
    - If tests_total >= expected_tests: score = pass_rate (tests_passed / tests_total)
    - If tests_total < expected_tests: score = tests_passed / expected_tests
    - Maximum score is always clipped at 1.0

    Args:
        workspace_dir: Path to workspace directory containing test files
        example: Optional example dictionary (not used, kept for compatibility)
        expected_tests: Expected number of test FUNCTIONS (default: 5)
        score_per_test: Deprecated, kept for backwards compatibility
        timeout_per_file: Timeout in seconds for each individual test file (default: 60)
        lm: DSPy LM for feedback message shortening

    Returns:
        Dictionary containing evaluation results with the following keys:
            - workspace_dir: Path to the workspace directory
            - test_files: List of test files found
            - tests_passed: Number of test functions that passed
            - tests_failed: Number of test functions that failed
            - tests_total: Total number of test functions run
            - score: Calculated score based on pass rate and test count, clipped at 1.0
            - pass_rate: Percentage of tests that passed (0.0 to 1.0)
            - success: Whether all tests passed
            - feedback: Natural language feedback summary
            - test_output: Full test execution output
            - error: Error message if execution failed
            - coverage_metrics: Dictionary containing coverage metrics (byte_coverage, function_coverage, scripts)

    Example:
        >>> result = run_single_instance_eval(
        ...     workspace_dir="results/webtest/gemini-2.5-flash_SKILL_agentic_workspace/example2_rollout0",
        ...     timeout_per_file=30
        ... )
        >>> print(f"Score: {result['score']:.2f} (Pass rate: {result['pass_rate']:.1%})")
    """
    workspace_path = Path(workspace_dir)

    print(f"🐳 Running tests in Docker (image: migration-analysis:latest)")

    # Validate workspace exists
    if not workspace_path.exists():
        return _create_error_result(
            workspace_dir,
            [],
            f"Workspace directory does not exist!"
        )
    
    # Check trace file presence
    trace_files = list(workspace_path.glob("trace_*.md"))
    if not trace_files:
        return _create_error_result(
            workspace_dir,
            [],
            f"No trace_*.md files found. The execution was incomplete."
        )

    # Discover test files
    test_files = list(workspace_path.glob("**/*test*.py"))
    if not test_files:
        return _create_error_result(
            workspace_dir,
            [],
            f"No test files found. The execution was incomplete."
        )

    test_file_names = [str(f.relative_to(workspace_path)) for f in test_files]
    print(f"Found {len(test_files)} test file(s): {', '.join(test_file_names)}")
    print(f"Working directory: {workspace_path}")

    # Setup regex patterns
    test_function_pattern = re.compile(r"^\s*def\s+(\w*test\w*)\s*\(", re.MULTILINE | re.IGNORECASE)
    main_block_pattern = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']', re.MULTILINE)

    try:
        # Count test functions and check for missing __main__ blocks
        total_test_functions = _count_test_functions(test_files, test_function_pattern)
        print(f"Found {total_test_functions} test function(s) across {len(test_files)} file(s)")
    except Exception as e:
        # Handle errors during file scanning/parsing (before running tests)
        return _create_error_result(
            workspace_dir,
            test_file_names,
            f"Failed to analyze test files: {str(e)}"
        )

    # Run all test files (tries pytest per file, falls back to python3)
    # Errors and timeouts are now handled per file within _run_tests_batch
    return _run_tests_batch(
        lm,
        workspace_path,
        test_files,
        test_file_names,
        test_function_pattern,
        main_block_pattern,
        expected_tests,
        timeout_per_file,
    )
