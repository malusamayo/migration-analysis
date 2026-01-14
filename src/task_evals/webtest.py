"""
Evaluation functions for webtest task.

This module provides evaluation functions for webtest workspace directories,
running playwright-based tests and calculating scores.
"""

import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

def _count_test_functions(test_files: List[Path], test_function_pattern: re.Pattern) -> int:
    """Count total test functions across all files."""
    total = 0
    for test_file in test_files:
        with open(test_file, 'r') as f:
            content = f.read()
            test_functions = test_function_pattern.findall(content)
            total += len(test_functions)
    return total


def _check_main_blocks(test_files: List[Path], test_function_pattern: re.Pattern, main_block_pattern: re.Pattern) -> None:
    """Warn about test files that don't have __main__ blocks."""
    for test_file in test_files:
        with open(test_file, 'r') as f:
            content = f.read()
            test_functions = test_function_pattern.findall(content)
            if test_functions and not main_block_pattern.search(content):
                print(f"  Warning: {test_file.name} has test functions but no __main__ block to execute them")


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


def _create_error_result(workspace_dir: str, test_files: List[str], error_msg: str) -> Dict[str, Any]:
    """Create a standardized error result dictionary."""
    return {
        "workspace_dir": str(workspace_dir),
        "test_files": test_files,
        "error": error_msg,
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
    timeout: int = 60
) -> Optional[Tuple[int, int, subprocess.CompletedProcess]]:
    """
    Run a single test file with pytest.

    Args:
        test_file: Path to the test file
        workspace_path: Path to the workspace directory
        timeout: Timeout in seconds for this individual file (default: 60)

    Returns:
        Tuple of (tests_passed, tests_failed, result) if successful, None otherwise.
    """
    try:
        print(f"    Trying pytest for {test_file}...")
        pytest_result = subprocess.run(
            ["python3", "-m", "pytest", "-v", "--tb=short", "--timeout=10", str(test_file)],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        print(f"    Pytest return code: {pytest_result.returncode}")

        # Parse output to count test results
        combined_output = f"{pytest_result.stdout}\n{pytest_result.stderr}"
        passed_pattern = re.compile(r"::(\w*test\w*)\s+PASSED", re.IGNORECASE)
        failed_pattern = re.compile(r"::(\w*test\w*)\s+FAILED", re.IGNORECASE)

        tests_passed = len(passed_pattern.findall(combined_output))
        tests_failed = len(failed_pattern.findall(combined_output))

        # If pytest detected tests, return the results
        if tests_passed + tests_failed > 0:
            return (tests_passed, tests_failed, pytest_result)

        return None
    except subprocess.TimeoutExpired as e:
        print(f"    Pytest execution timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"    Pytest execution failed: {e}")
        return None

def _run_file_with_python(
    test_file: Path,
    workspace_path: Path,
    test_function_names: List[str],
    has_main_block: bool,
    timeout: int = 60
) -> subprocess.CompletedProcess:
    """
    Run a single test file directly using Python.

    If the file has no __main__ block, creates a wrapper script to call test functions.

    Args:
        test_file: Path to the test file
        workspace_path: Path to the workspace directory
        test_function_names: List of test function names found in the file
        has_main_block: Whether the file has a __main__ block
        timeout: Timeout in seconds for this individual file (default: 60)

    Returns:
        CompletedProcess result from subprocess.run
    """
    relative_path = test_file.relative_to(workspace_path)

    if len(test_function_names) > 0 and not has_main_block:
        # Create a wrapper script that imports and calls the test functions
        wrapper_script = f"""
import sys
sys.path.insert(0, '{workspace_path}')
from {test_file.stem} import {', '.join(test_function_names)}

if __name__ == '__main__':
    {'; '.join([f'{name}()' for name in test_function_names])}
"""
        return subprocess.run(
            ["python3", "-c", wrapper_script],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        # Run the file directly
        return subprocess.run(
            ["python3", str(relative_path)],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def _run_tests_batch(
    workspace_path: Path,
    test_files: List[Path],
    test_file_names: List[str],
    test_function_pattern: re.Pattern,
    main_block_pattern: re.Pattern,
    expected_tests: int,
    timeout_per_file: int = 60
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

    # Check if pytest is available once
    pytest_available = subprocess.run(
        ["python3", "-m", "pytest", "--version"],
        capture_output=True,
        text=True,
    ).returncode == 0

    for test_file in test_files:
        relative_path = test_file.relative_to(workspace_path)
        print(f"  Running {relative_path}...")

        execution_method = None
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
                pytest_result = _run_file_with_pytest(relative_path, workspace_path, timeout_per_file)
                if pytest_result is not None:
                    file_passed, file_failed, result = pytest_result
                    execution_method = "pytest"
                    tests_passed += file_passed
                    tests_failed += file_failed
                    status = "passed" if file_failed == 0 else "failed"

            # Fallback to direct python3 execution
            if execution_method is None:
                execution_method = "python3"
                result = _run_file_with_python(test_file, workspace_path, test_function_names, has_main_block, timeout_per_file)

                # All functions in file pass or fail together
                if result.returncode == 0:
                    tests_passed += test_functions_in_file
                    status = "passed"
                else:
                    tests_failed += test_functions_in_file
                    status = "failed"

        except subprocess.TimeoutExpired:
            # Handle timeout for this specific file
            print(f"    ⚠️  Test file timed out after {timeout_per_file}s")
            tests_failed += test_functions_in_file
            status = "timeout"
            execution_method = execution_method or "python3"
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
            execution_method = execution_method or "unknown"
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

    print(f"Tests completed: {tests_passed}/{tests_total} test functions passed")
    print(f"Pass rate: {pass_rate:.1%}")
    print(f"Score: {score:.2f} (expected {expected_tests} tests, found {tests_total})")

    return {
        "workspace_dir": str(workspace_path),
        "test_files": test_file_names,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_total": tests_total,
        "score": score,
        "pass_rate": pass_rate,
        "success": success,
        "test_output": file_results,
    }


def run_single_instance_eval(
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
            - test_output: Full test execution output
            - error: Error message if execution failed

    Example:
        >>> result = run_single_instance_eval(
        ...     workspace_dir="results/webtest/gemini-2.5-flash_SKILL_agentic_workspace/example2_rollout0",
        ...     timeout_per_file=30
        ... )
        >>> print(f"Score: {result['score']:.2f} (Pass rate: {result['pass_rate']:.1%})")
    """
    workspace_path = Path(workspace_dir)

    # Validate workspace exists
    if not workspace_path.exists():
        return _create_error_result(
            workspace_dir,
            [],
            f"Workspace directory does not exist: {workspace_dir}"
        )

    # Discover test files
    test_files = list(workspace_path.glob("**/*test*.py"))
    if not test_files:
        return _create_error_result(
            workspace_dir,
            [],
            f"No test files (test_*.py) found in: {workspace_dir}"
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
        _check_main_blocks(test_files, test_function_pattern, main_block_pattern)
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
        workspace_path,
        test_files,
        test_file_names,
        test_function_pattern,
        main_block_pattern,
        expected_tests,
        timeout_per_file
    )
