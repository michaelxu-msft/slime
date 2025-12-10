"""
SandboxFusion code evaluation module for slime.

This module provides sandboxed code execution for code generation tasks using
containerized or isolated execution environments. It integrates with SandboxFusion
or similar sandbox services for secure code execution.

The evaluator can operate in two modes:
1. Remote API mode: Connects to a SandboxFusion API endpoint
2. Local sandbox mode: Uses local containerized execution (Docker/Podman)
"""

import asyncio
import json
from typing import Any, Optional, Union

import aiohttp


async def execute_code_remote(
    code: str,
    test_cases: dict,
    sandbox_url: str,
    timeout: int = 30,
    language: str = "python"
) -> dict:
    """
    Execute code on a remote SandboxFusion service.
    
    Args:
        code: The code to execute
        test_cases: Dictionary with 'inputs' and 'outputs'
        sandbox_url: URL of the SandboxFusion API endpoint
        timeout: Maximum execution time in seconds
        language: Programming language (default: python)
    
    Returns:
        Dictionary with execution results containing 'success', 'results', and optional 'error'
    """
    # Extract test cases
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    
    if not inputs or not outputs:
        return {
            "success": False,
            "error": "Missing test cases (inputs or outputs)",
            "results": [],
        }
    
    # Run each test case independently
    results = []
    
    async with aiohttp.ClientSession() as session:
        for inp, expected_out in zip(inputs, outputs):
            # Convert input to stdin format
            if isinstance(inp, list):
                stdin = "\n".join(str(x) for x in inp)
            else:
                stdin = str(inp)
            
            # Prepare payload according to SandboxFusion API spec
            payload = {
                "compile_timeout": timeout,
                "run_timeout": timeout,
                "code": code,
                "stdin": stdin,
                "language": language,
                "files": {},
                "fetch_files": []
            }
            
            try:
                async with session.post(
                    sandbox_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout + 5)
                ) as resp:
                    result = await resp.json()
                    
                    # Handle validation errors (422)
                    if resp.status == 422:
                        error_details = result.get("detail", [])
                        error_msg = "; ".join([f"{e.get('msg', 'Unknown error')}" for e in error_details])
                        return {
                            "success": False,
                            "error": f"Validation error: {error_msg}",
                            "results": results,
                        }
                    
                    # Raise for other HTTP errors
                    resp.raise_for_status()
                    
                    # Check if execution was successful (status: "Success" or "Failed")
                    status = result.get("status", "").lower()
                    if status == "success":
                        # Extract stdout from run_result
                        run_result = result.get("run_result", {})
                        compile_result = result.get("compile_result", {})
                        
                        # Check for compilation errors
                        if compile_result and compile_result.get("exit_code") != 0:
                            results.append(False)
                            continue
                        
                        # Get stdout and stderr
                        stdout = run_result.get("stdout", "").strip()
                        stderr = run_result.get("stderr", "").strip()
                        return_code = run_result.get("return_code", run_result.get("exit_code", -1))
                        
                        # If there's a non-zero return code or stderr with errors, consider it a failure
                        if return_code != 0 or (stderr and ("error" in stderr.lower() or "exception" in stderr.lower() or "traceback" in stderr.lower())):
                            results.append(False)
                            continue
                        
                        # Compare output with expected
                        expected_raw = expected_out
                        if isinstance(expected_out, list) and len(expected_out) == 1:
                            expected_out = expected_out[0]

                        expected_str = str(expected_out).strip()
                        expected_candidates = [expected_str]

                        if isinstance(expected_raw, list) and len(expected_raw) > 1:
                            joined_newline = "\n".join(str(x) for x in expected_raw).strip()
                            expected_candidates.append(joined_newline)
                            joined_space = " ".join(str(x) for x in expected_raw).strip()
                            expected_candidates.append(joined_space)

                        if stdout in expected_candidates:
                            results.append(True)
                        else:
                            # Try parsing as JSON if direct comparison fails
                            try:
                                stdout_parsed = json.loads(stdout)
                                expected_parsed = json.loads(expected_str) if isinstance(expected_out, str) else expected_out
                                results.append(stdout_parsed == expected_parsed)
                            except:
                                # Try numeric comparison
                                try:
                                    stdout_num = float(stdout)
                                    expected_num = float(expected_str)
                                    results.append(abs(stdout_num - expected_num) < 1e-6)
                                except:
                                    results.append(False)
                    elif status == "failed":
                        # Execution failed - extract error details from run_result if available
                        run_result = result.get("run_result", {})
                        if run_result:
                            stderr = run_result.get("stderr", "")
                            return_code = run_result.get("return_code", run_result.get("exit_code", -1))
                            # This is a failed test case, mark as failure and continue
                            results.append(False)
                        else:
                            # No run_result, this is a more serious error
                            error_msg = result.get("message", "Unknown error")
                            results.append(False)
                    else:
                        # Unknown status
                        results.append(False)
                        
            except aiohttp.ClientError as e:
                return {
                    "success": False,
                    "error": f"API request failed: {str(e)}",
                    "results": results,
                }
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": "Request timeout",
                    "results": results,
                }
    
    return {
        "success": True,
        "results": results,
        "total": len(results),
        "passed": sum(results),
    }


def extract_code_from_markdown(completion: str) -> str:
    """
    Extract code from markdown code blocks.
    
    Args:
        completion: Raw completion string potentially containing markdown
    
    Returns:
        Extracted code
    """
    # Handle markdown code blocks
    if "```python" in completion:
        code = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        code = completion.split("```")[1].split("```")[0]
        # Remove language identifier if present
        lines = code.strip().split("\n")
        if lines and lines[0].strip() in ["python", "py", "cpp", "c++", "java", "javascript", "js"]:
            code = "\n".join(lines[1:])
    else:
        code = completion
    
    return code.strip()


async def compute_sandboxfusion_reward(
    response: str,
    label: Union[str, dict],
    metadata: Optional[dict] = None,
    timeout: int = 30,
    sandbox_url: Optional[str] = None,
    all_tests_must_pass: bool = True,
) -> float:
    """
    Compute reward for code generation using SandboxFusion.
    
    This function integrates with slime's reward manager hub and provides
    secure code execution through SandboxFusion or compatible sandbox services.
    
    Args:
        response: Generated code response from the model
        label: Test cases as dict or JSON string with 'inputs' and 'outputs'
        metadata: Optional metadata (can contain sandbox_url, timeout, language, etc.)
        timeout: Maximum execution time in seconds
        sandbox_url: URL of the SandboxFusion API (can be overridden in metadata)
        all_tests_must_pass: If True, return 1.0 only if all tests pass;
                           if False, return fraction of tests passed
    
    Returns:
        Reward score (1.0 for success, 0.0 for failure, or fraction)
    """
    # Extract code from markdown if present
    solution = extract_code_from_markdown(response)
    
    # Parse test cases
    if isinstance(label, str):
        try:
            test_cases = json.loads(label)
        except json.JSONDecodeError:
            return 0.0
    elif isinstance(label, dict):
        test_cases = label
    else:
        return 0.0
    
    # Handle nested structure
    if "ground_truth" in test_cases:
        test_cases = test_cases["ground_truth"]
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                return 0.0
    
    # Validate test case structure
    if not isinstance(test_cases, dict) or "inputs" not in test_cases or "outputs" not in test_cases:
        return 0.0
    
    # Get configuration from metadata
    if metadata:
        timeout = metadata.get("timeout", timeout)
        sandbox_url = metadata.get("sandbox_url", sandbox_url)
        language = metadata.get("language", "python")
    else:
        language = "python"
    
    # Validate sandbox URL
    if not sandbox_url:
        # Fallback to environment variable or default
        import os
        sandbox_url = os.environ.get("SANDBOXFUSION_URL")
        if not sandbox_url:
            if metadata is not None:
                metadata["eval_error"] = "SandboxFusion URL not configured"
            return 0.0
    
    # Ensure URL has the /run_code endpoint
    if not sandbox_url.endswith("/run_code"):
        sandbox_url = sandbox_url.rstrip("/") + "/run_code"
    
    try:
        # Execute code through SandboxFusion
        result = await execute_code_remote(
            code=solution,
            test_cases=test_cases,
            sandbox_url=sandbox_url,
            timeout=timeout,
            language=language,
        )
        
        # Check if execution was successful
        if not result.get("success", False):
            if metadata is not None:
                metadata["eval_error"] = result.get("error", "Unknown error")
            return 0.0
        
        # Get test results
        results = result.get("results", [])
        if not results:
            return 0.0
        
        # Compute score
        if all_tests_must_pass:
            # Binary score: all tests must pass
            score = 1.0 if all(r for r in results) else 0.0
        else:
            # Fractional score: proportion of tests passed
            score = sum(1.0 for r in results if r) / len(results)
        
        return score
        
    except Exception as e:
        # Log error if metadata is provided
        if metadata is not None and isinstance(metadata, dict):
            metadata["eval_error"] = str(e)
        return 0.0


# Async wrappers for use with custom_rm_path
async def async_rm(args, sample, **kwargs):
    """
    Async wrapper for SandboxFusion evaluation to be used with custom_rm_path.
    
    This function can be loaded via load_function() and used as args.custom_rm_path.
    
    Args:
        args: Training arguments (should contain sandbox_url)
        sample: Sample object with response, label, and metadata
        **kwargs: Additional keyword arguments
    
    Returns:
        Reward score (float)
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    response = sample.response
    label = sample.label
    
    # Get sandbox URL from args or metadata
    sandbox_url = getattr(args, "sandbox_url", None) or metadata.get("sandbox_url")
    timeout = getattr(args, "sandbox_timeout", 30)
    
    # Determine scoring mode from metadata
    rm_type = metadata.get("rm_type", "sandboxfusion")
    all_tests_must_pass = rm_type in ["prime_code", "livecodebench"]
    
    return await compute_sandboxfusion_reward(
        response=response,
        label=label,
        metadata=metadata,
        timeout=timeout,
        sandbox_url=sandbox_url,
        all_tests_must_pass=all_tests_must_pass,
    )


async def batched_async_rm(args, samples: list, **kwargs) -> list[Union[int, float]]:
    """
    Batched async wrapper for SandboxFusion evaluation.
    
    This function can be loaded via load_function() and used as args.custom_rm_path
    for batched evaluation.
    
    Args:
        args: Training arguments
        samples: List of Sample objects
        **kwargs: Additional keyword arguments
    
    Returns:
        List of reward scores
    """
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
