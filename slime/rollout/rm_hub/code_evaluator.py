"""
Code evaluation module for slime.

This module provides code execution and evaluation functionality for reinforcement learning
on code generation tasks. It safely executes generated code against test cases in isolated
processes with timeout protection.

Based on the architecture from FlowRL/verl but adapted for slime's reward manager hub.
"""

import json
import multiprocessing
import signal
import sys
from io import StringIO
from typing import Any, Optional, Union


def _execute_code(in_outs: dict, generation: str, result: list, metadata_list: list, timeout: int):
    """
    Execute code with test cases in an isolated context.
    
    Args:
        in_outs: Dictionary containing 'inputs' and 'outputs' test cases
        generation: The generated code string to execute
        result: Shared list to store execution results
        metadata_list: Shared list to store execution metadata
        timeout: Maximum execution time per test case
    """
    metadata = {"timeout": False, "error": None}
    
    try:
        # Parse test cases
        inputs = in_outs.get("inputs", [])
        outputs = in_outs.get("outputs", [])
        
        if not inputs or not outputs:
            result.append([False] * max(len(inputs), len(outputs), 1))
            metadata["error"] = "Missing test cases"
            metadata_list.append(metadata)
            return
        
        # Execute code to define functions
        exec_globals = {}
        try:
            exec(generation, exec_globals)
        except Exception as e:
            result.append([False] * len(inputs))
            metadata["error"] = f"Compilation error: {str(e)}"
            metadata_list.append(metadata)
            return
        
        # Find the main function (assume it's the first user-defined function)
        # or look for common entry point names
        main_func = None
        for name, obj in exec_globals.items():
            if callable(obj) and not name.startswith("_") and name not in ["print", "input", "len"]:
                main_func = obj
                break
        
        if main_func is None:
            result.append([False] * len(inputs))
            metadata["error"] = "No callable function found"
            metadata_list.append(metadata)
            return
        
        # Run test cases
        test_results = []
        for inp, expected_out in zip(inputs, outputs):
            try:
                # Prepare input arguments
                if isinstance(inp, list):
                    args = inp
                else:
                    args = [inp]
                
                # Execute with timeout
                def handler(signum, frame):
                    raise TimeoutError("Execution timeout")
                
                old_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout)
                
                try:
                    # Capture stdout for print-based solutions
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()
                    
                    actual_out = main_func(*args)
                    
                    # Restore stdout
                    sys.stdout = old_stdout
                    printed_output = captured_output.getvalue().strip()
                    
                    # Compare output
                    # Handle both return values and printed output
                    if actual_out is not None:
                        output_to_check = actual_out
                    elif printed_output:
                        output_to_check = printed_output
                    else:
                        output_to_check = None
                    
                    # Normalize expected output
                    if isinstance(expected_out, list) and len(expected_out) == 1:
                        expected_out = expected_out[0]
                    
                    # Convert to comparable format
                    if isinstance(output_to_check, (int, float)):
                        output_to_check = str(output_to_check)
                    if isinstance(expected_out, (int, float)):
                        expected_out = str(expected_out)
                    
                    is_correct = str(output_to_check).strip() == str(expected_out).strip()
                    test_results.append(is_correct)
                    
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    sys.stdout = old_stdout
                    
            except TimeoutError:
                test_results.append(False)
                metadata["timeout"] = True
            except Exception as e:
                test_results.append(False)
                if metadata["error"] is None:
                    metadata["error"] = f"Runtime error: {str(e)}"
        
        result.append(test_results)
        metadata_list.append(metadata)
        
    except Exception as e:
        result.append([False] * len(in_outs.get("inputs", [])))
        metadata["error"] = f"Execution error: {str(e)}"
        metadata_list.append(metadata)


def check_correctness(in_outs: dict, generation: str, timeout: int = 6) -> tuple[list[bool], dict]:
    """
    Execute code with test cases in isolated process with timeout.
    
    Args:
        in_outs: Dictionary with 'inputs' and 'outputs' lists
        generation: Generated code string
        timeout: Maximum execution time in seconds
    
    Returns:
        Tuple of (test_results, metadata)
        test_results: List of boolean results for each test case
        metadata: Dictionary with execution information
    """
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    
    p = multiprocessing.Process(
        target=_execute_code,
        args=(in_outs, generation, result, metadata_list, timeout)
    )
    p.start()
    p.join(timeout=timeout + 1)
    
    if p.is_alive():
        p.kill()
        p.join()
        num_tests = len(in_outs.get("inputs", []))
        result = [[False] * num_tests]
        metadata_list = [{"timeout": True, "error": "Process timeout"}]
    
    if not result:
        num_tests = len(in_outs.get("inputs", []))
        return [False] * num_tests, {"timeout": False, "error": "No result"}
    
    metadata = metadata_list[0] if metadata_list else {"timeout": False, "error": None}
    return result[0], metadata


def extract_code_from_markdown(completion: str) -> str:
    """
    Extract Python code from markdown code blocks.
    
    Args:
        completion: Raw completion string potentially containing markdown
    
    Returns:
        Extracted Python code
    """
    # Handle markdown code blocks
    if "```python" in completion:
        code = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        code = completion.split("```")[1].split("```")[0]
        # Remove language identifier if present
        lines = code.strip().split("\n")
        if lines and lines[0].strip() in ["python", "py"]:
            code = "\n".join(lines[1:])
    else:
        code = completion
    
    return code.strip()


def compute_code_reward(
    response: str,
    label: Union[str, dict],
    metadata: Optional[dict] = None,
    timeout: int = 6,
    all_tests_must_pass: bool = True
) -> float:
    """
    Compute reward for code generation based on test case execution.
    
    This is the main reward function that integrates with slime's reward manager hub.
    
    Args:
        response: Generated code response from the model
        label: Test cases as dict or JSON string with 'inputs' and 'outputs'
        metadata: Optional metadata (can contain rm_type, timeout, etc.)
        timeout: Maximum execution time in seconds
        all_tests_must_pass: If True, return 1.0 only if all tests pass; 
                           if False, return fraction of tests passed
    
    Returns:
        Reward score (1.0 for success, 0.0 for failure, or fraction if all_tests_must_pass=False)
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
    
    # Handle nested structure (e.g., {"ground_truth": {"inputs": ..., "outputs": ...}})
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
    
    # Get timeout from metadata if provided
    if metadata and "timeout" in metadata:
        timeout = metadata["timeout"]
    
    try:
        # Run tests
        results, exec_metadata = check_correctness(
            in_outs=test_cases,
            generation=solution,
            timeout=timeout
        )
        
        # Compute score
        if not results:
            return 0.0
        
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


def compute_prime_code_reward(response: str, label: Any, metadata: Optional[dict] = None) -> float:
    """
    Compute reward for PRIME code benchmark.
    
    Alias for compute_code_reward with binary scoring (all tests must pass).
    """
    return compute_code_reward(response, label, metadata, all_tests_must_pass=True)


def compute_livecodebench_reward(response: str, label: Any, metadata: Optional[dict] = None) -> float:
    """
    Compute reward for LiveCodeBench.
    
    Alias for compute_code_reward with binary scoring (all tests must pass).
    """
    return compute_code_reward(response, label, metadata, all_tests_must_pass=True)


# Async wrappers for use with custom_rm_path
async def async_rm(args, sample, **kwargs):
    """
    Async wrapper for code evaluation to be used with custom_rm_path.
    
    This function can be loaded via load_function() and used as args.custom_rm_path.
    Supports different code evaluation modes via metadata['rm_type']:
    - 'code': Pass@k scoring (returns fraction of tests passed)
    - 'prime_code': Binary scoring (all tests must pass)
    - 'livecodebench': Binary scoring (all tests must pass)
    
    Args:
        args: Training arguments (can contain timeout settings)
        sample: Sample object with response, label, and metadata
        **kwargs: Additional keyword arguments
    
    Returns:
        Reward score (float)
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "code").strip()
    response = sample.response
    label = sample.label
    
    # Handle boxed extraction if needed
    if rm_type.startswith("boxed_"):
        from .math_utils import extract_answer as extract_boxed_answer
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]
    
    # Determine scoring mode
    if rm_type == "prime_code":
        return compute_prime_code_reward(response, label, metadata)
    elif rm_type == "livecodebench":
        return compute_livecodebench_reward(response, label, metadata)
    else:  # Default to "code" mode with pass@k scoring
        return compute_code_reward(response, label, metadata, all_tests_must_pass=False)


async def batched_async_rm(args, samples: list, **kwargs) -> list[Union[int, float]]:
    """
    Batched async wrapper for code evaluation to be used with custom_rm_path.
    
    This function can be loaded via load_function() and used as args.custom_rm_path
    for batched evaluation.
    
    Args:
        args: Training arguments
        samples: List of Sample objects
        **kwargs: Additional keyword arguments
    
    Returns:
        List of reward scores
    """
    import asyncio
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
