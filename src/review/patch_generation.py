"""
Module for generating patches to improve task descriptions.

This module provides DSPy-based components and functions for generating
patches based on trajectory comparison analysis.
"""

import json
import dspy
from pathlib import Path
from typing import List, Dict, Any

from ..utils import LM_DICT


def format_patch_file(
    patch_name: str,
    patch_type: str,
    patch_content: str,
    patch_rationale: str = "",
    generation_context: str = ""
) -> str:
    """
    Format a patch file with proper metadata header.

    Args:
        patch_name: Name of the patch
        patch_type: Type of patch (e.g., 'clarification', 'universal', 'constraint')
        patch_content: The actual patch content (unified diff)
        patch_rationale: Rationale for the patch (can be multi-line)
        generation_context: Optional context about how the patch was generated

    Returns:
        Formatted patch file content as a string
    """
    # Format rationale as multi-line comment
    rationale_lines = []
    if patch_rationale:
        for line in patch_rationale.strip().split('\n'):
            rationale_lines.append(f"# {line}" if line else "#")

    # Build header
    header_parts = [
        f"# Patch: {patch_name}",
        f"# Type: {patch_type}",
        "#",
        "# Rationale:",
    ]
    header_parts.extend(rationale_lines)

    if generation_context:
        header_parts.extend([
            "#",
            f"# {generation_context}",
        ])

    # Combine header and content
    header_parts.extend([
        "",
        "---BEGIN PATCH---",
        patch_content.strip(),
        "---END PATCH---"
    ])

    return "\n".join(header_parts) + "\n"


class PatchGeneration(dspy.Signature):
    """Generate patches to improve task descriptions based on trajectory comparison analysis."""

    task_description = dspy.InputField(
        desc="The original task/request that the agent was trying to complete"
    )
    behavioral_differences = dspy.InputField(
        desc="List of behavioral differences identified from comparing multiple trajectory rollouts"
    )
    patches = dspy.OutputField(
        desc="A JSON list of patch objects. Each patch should have: 'patch_name' (descriptive name like 'add_error_handling_guidance.patch'), 'rationale' (why this patch improves the task), 'patch_type' ('clarification', 'constraint', 'example', or 'error_handling'), and 'patch_content' (the actual text to add/modify in the task description)"
    )

class PatchRefinement(dspy.Signature):
    """Refine an existing patch based on user feedback. The refined patch should be a complete, standalone patch that can be applied directly to the base prompt."""

    original_patch_content = dspy.InputField(
        desc="The original patch content (a unified diff format patch)"
    )
    original_rationale = dspy.InputField(
        desc="The original rationale for why this patch was created"
    )
    user_feedback = dspy.InputField(
        desc="User's feedback on what should be changed or improved in the patch"
    )
    base_prompt_context = dspy.InputField(
        desc="Relevant sections of the base prompt that this patch modifies (for context)"
    )
    refined_patch = dspy.OutputField(
        desc="A complete, standalone unified diff patch that incorporates the user's feedback and can be applied directly to the base prompt. This should be a fresh diff, not a modification of the original patch."
    )
    updated_rationale = dspy.OutputField(
        desc="Updated rationale explaining the changes made based on feedback"
    )


class GenerateTaskPatches(dspy.Module):
    """Module for generating patches to task descriptions."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(PatchGeneration)

    def forward(self, task_description: str, behavioral_differences: List[str]):
        """Generate patches based on comparison analysis."""
        return self.generate(
            task_description=task_description,
            behavioral_differences=behavioral_differences
        )


def generate_patches_from_comparison(
    comparison_result: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Generate patches to improve task descriptions based on trajectory comparison.

    Args:
        comparison_result: Result dictionary from compare_rollout_trajectories
        model_name: Model to use for patch generation (default: "gemini-2.5-flash")
        output_dir: Directory to save patch files (default: same dir as comparison)

    Returns:
        Dictionary containing patch generation results
    """
    # Extract behavioral differences and task description
    task_description = comparison_result.get("task_description", "")
    behavioral_differences = comparison_result.get("comparison_analysis", [])

    if not behavioral_differences:
        raise ValueError("No behavioral differences found in comparison result")

    # Get LLM for patch generation
    lm = LM_DICT[model_name]

    print(f"Generating patches using {model_name}...")
    with dspy.context(lm=lm):
        patch_generator = GenerateTaskPatches()
        result = patch_generator(
            task_description=task_description,
            behavioral_differences=behavioral_differences
        )

    # Parse patches (handle both JSON string and list)
    patches_data = result.patches
    if isinstance(patches_data, str):
        try:
            patches = json.loads(patches_data)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract patches from text
            print("Warning: Could not parse patches as JSON, treating as text")
            patches = [{
                "patch_name": "generated_patch.patch",
                "rationale": "Generated from comparison analysis",
                "patch_type": "general",
                "patch_content": patches_data
            }]
    else:
        patches = patches_data

    # Ensure patches is a list
    if not isinstance(patches, list):
        patches = [patches]

    # Set output directory
    if output_dir is None:
        output_dir = Path("patches")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each patch as a separate file
    saved_patches = []
    for i, patch in enumerate(patches):
        if isinstance(patch, dict):
            patch_name = patch.get("patch_name", f"patch_{i+1}.patch")
            patch_content = patch.get("patch_content", "")
            patch_rationale = patch.get("rationale", "")
            patch_type = patch.get("patch_type", "general")
        else:
            # Handle non-dict patches
            patch_name = f"patch_{i+1}.patch"
            patch_content = str(patch)
            patch_rationale = ""
            patch_type = "general"

        # Create patch file with metadata header
        patch_file_content = format_patch_file(
            patch_name=patch_name,
            patch_type=patch_type,
            patch_content=patch_content,
            patch_rationale=patch_rationale,
            generation_context="This patch was automatically generated based on behavioral differences observed in multiple trajectory rollouts."
        )

        patch_path = output_dir / patch_name
        with open(patch_path, 'w', encoding='utf-8') as f:
            f.write(patch_file_content)

        saved_patches.append({
            "patch_name": patch_name,
            "patch_path": str(patch_path),
            "patch_type": patch_type,
            "rationale": patch_rationale
        })
        print(f"✅ Saved patch: {patch_path}")

    # Save summary
    summary = {
        "model": model_name,
        "num_patches": len(saved_patches),
        "patches": saved_patches,
        "task_description": task_description,
        "behavioral_differences_analyzed": len(behavioral_differences)
    }

    summary_path = output_dir / "patch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Patch summary saved to: {summary_path}")

    return summary


def generate_patches_batch(
    task_id: str,
    model_name: str,
    prompt_name: str = "default",
    num_examples: int = 10,
    patch_model: str = "gemini-2.5-flash",
    comparison_dir: Path = None,
    output_base_dir: Path = None
) -> List[Dict[str, Any]]:
    """
    Generate patches for multiple examples based on their comparison results.

    Args:
        task_id: Task identifier (e.g., "webgen", "webtest")
        model_name: Model used for generation
        prompt_name: Prompt template name (default: "default")
        num_examples: Number of examples to process (default: 10)
        patch_model: Model to use for patch generation (default: "gemini-2.5-flash")
        comparison_dir: Directory containing comparison results (default: results/{task_id}/comparisons/{model_name}_{prompt_name})
        output_base_dir: Base directory to save patches (default: results/{task_id}/patches/{model_name}_{prompt_name})

    Returns:
        List of patch generation results for each example
    """
    # Set default directories
    if comparison_dir is None:
        comparison_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}/comparisons")
    else:
        comparison_dir = Path(comparison_dir)

    if output_base_dir is None:
        output_base_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}/patches")
    else:
        output_base_dir = Path(output_base_dir)

    if not comparison_dir.exists():
        raise ValueError(f"Comparison directory does not exist: {comparison_dir}")

    output_base_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for example_id in range(num_examples):
        print(f"\n{'='*80}")
        print(f"Generating patches for Example {example_id}")
        print(f"{'='*80}")

        # Look for comparison file
        comparison_file = comparison_dir / f"example{example_id}_comparison.md"

        if not comparison_file.exists():
            print(f"⚠️  No comparison file found: {comparison_file}")
            continue

        # Load comparison result
        # First, check if there's a JSON version
        comparison_json = comparison_dir / "batch_comparison_summary.json"
        comparison_result = None

        if comparison_json.exists():
            with open(comparison_json, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                for result_item in summary.get("results", []):
                    if result_item.get("example_id") == example_id:
                        comparison_result = result_item.get("comparison")
                        break

        if comparison_result is None:
            print(f"⚠️  Could not load comparison result for example {example_id}")
            continue

        # Generate patches
        output_dir = output_base_dir / f"example{example_id}"

        try:
            patch_result = generate_patches_from_comparison(
                comparison_result=comparison_result,
                model_name=patch_model,
                output_dir=output_dir
            )
            results.append({
                "example_id": example_id,
                "num_patches": patch_result["num_patches"],
                "patches": patch_result["patches"],
                "output_dir": str(output_dir)
            })
            print(f"✅ Generated {patch_result['num_patches']} patch(es) for example {example_id}")
        except Exception as e:
            print(f"❌ Error generating patches for example {example_id}: {e}")
            results.append({
                "example_id": example_id,
                "error": str(e)
            })

    # Save batch summary
    batch_summary_path = output_base_dir / "batch_patch_summary.json"
    with open(batch_summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "task_id": task_id,
            "model_name": model_name,
            "prompt_name": prompt_name,
            "patch_model": patch_model,
            "num_examples_processed": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Batch patch generation completed!")
    print(f"Processed {len(results)} examples")
    print(f"Batch summary saved to: {batch_summary_path}")
    print(f"{'='*80}")

    return results


def generate_universal_patches(
    aggregated_analysis: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    DEPRECATED: This function is deprecated as aggregation is no longer part of the workflow.

    The new workflow should generate patches directly from individual comparison results,
    not from aggregated analyses.

    Old description (for reference):
    Generate universal patches from aggregated analysis (common patterns and improvements).

    Args:
        aggregated_analysis: DEPRECATED - Result from old aggregate_comparison_analyses()
        model_name: Model to use for patch generation (default: "gemini-2.5-flash")
        output_dir: Directory to save patch files

    Returns:
        Dictionary containing patch generation results
    """
    task_description = aggregated_analysis.get("task_description", "")
    common_patterns = aggregated_analysis.get("common_patterns", [])
    recommended_improvements = aggregated_analysis.get("recommended_improvements", [])

    if not common_patterns:
        raise ValueError("No common patterns found in aggregated analysis")

    # Combine patterns and improvements for patch generation
    behavioral_differences = []

    # Handle common_patterns as either list or string
    if isinstance(common_patterns, list):
        for pattern in common_patterns:
            behavioral_differences.append(f"PATTERN: {pattern}")
    else:
        behavioral_differences.append(f"PATTERNS:\n{common_patterns}")

    # Handle recommended_improvements as either list or string
    if isinstance(recommended_improvements, list):
        for improvement in recommended_improvements:
            behavioral_differences.append(f"IMPROVEMENT: {improvement}")
    else:
        behavioral_differences.append(f"IMPROVEMENTS:\n{recommended_improvements}")

    # Get LLM for patch generation
    lm = LM_DICT[model_name]

    print(f"Generating universal patches using {model_name}...")

    with dspy.context(lm=lm):
        patch_generator = GenerateTaskPatches()
        result = patch_generator(
            task_description=task_description,
            behavioral_differences=behavioral_differences
        )

    # Parse patches
    patches_data = result.patches.replace('```json', '').replace('```', '').strip()
    if isinstance(patches_data, str):
        try:
            patches = json.loads(patches_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse patches as JSON: {e}")
    else:
        patches = patches_data

    if not isinstance(patches, list):
        patches = [patches]

    # Set output directory
    if output_dir is None:
        output_dir = Path("patches")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each patch with numbered prefix
    saved_patches = []
    for i, patch in enumerate(patches, 1):
        if isinstance(patch, dict):
            patch_name = patch.get("patch_name", f"{i:02d}_patch.patch")
            # Ensure patch has numbered prefix
            if not patch_name[:2].isdigit():
                patch_name = f"{i:02d}_{patch_name}"
            patch_content = patch.get("patch_content", "")
            patch_rationale = patch.get("rationale", "")
            patch_type = patch.get("patch_type", "universal")
        else:
            patch_name = f"{i:02d}_patch.patch"
            patch_content = str(patch)
            patch_rationale = ""
            patch_type = "universal"

        # Create patch file with metadata
        patch_file_content = format_patch_file(
            patch_name=patch_name,
            patch_type=patch_type,
            patch_content=patch_content,
            patch_rationale=patch_rationale,
            generation_context="This universal patch was generated from aggregated analysis of multiple trajectory rollouts, addressing common patterns observed across examples."
        )

        patch_path = output_dir / patch_name
        with open(patch_path, 'w', encoding='utf-8') as f:
            f.write(patch_file_content)

        saved_patches.append({
            "patch_name": patch_name,
            "patch_path": str(patch_path),
            "patch_type": patch_type,
            "rationale": patch_rationale
        })
        print(f"✅ Saved universal patch: {patch_path}")

    # Save metadata
    # Calculate common_patterns_addressed count
    if isinstance(common_patterns, list):
        patterns_count = len(common_patterns)
    else:
        patterns_count = 1  # String-formatted patterns count as 1 aggregated set

    metadata = {
        "model": model_name,
        "num_patches": len(saved_patches),
        "patches": saved_patches,
        "num_examples_analyzed": aggregated_analysis.get("num_examples_analyzed", 0),
        "example_ids": aggregated_analysis.get("example_ids", []),
        "common_patterns_addressed": patterns_count
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Patch metadata saved to: {metadata_path}")

    return metadata


def edit_patch_with_feedback(
    patch_path: Path,
    user_feedback: str,
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None,
    base_prompt_path: Path = None
) -> Dict[str, Any]:
    """
    Edit an existing patch based on user feedback.

    This function takes a patch file and user feedback, then uses an LLM to
    generate a refined version of the patch that incorporates the feedback.
    The refined patch is a complete, standalone patch that can be applied to the base prompt.

    Args:
        patch_path: Path to the existing patch file
        user_feedback: User's feedback on how to improve the patch
        model_name: Model to use for patch refinement (default: "gemini-2.5-flash")
        output_path: Path to save the refined patch (default: overwrites original)
        base_prompt_path: Path to the base prompt file (optional, for better context)

    Returns:
        Dictionary containing the refinement results

    Example:
        >>> result = edit_patch_with_feedback(
        ...     patch_path=Path("patches/01_clarify.patch"),
        ...     user_feedback="Make the language more specific and add concrete examples",
        ...     base_prompt_path=Path("data/webgen/prompts/default.md")
        ... )
    """
    patch_path = Path(patch_path)

    if not patch_path.exists():
        raise ValueError(f"Patch file not found: {patch_path}")

    # Read the existing patch
    from .patch_utils import read_patch_file

    try:
        patch_data = read_patch_file(patch_path)
        original_content = patch_data["content"]
        original_rationale = patch_data["rationale"]
        patch_name = patch_data["name"]
        patch_type = patch_data["type"]
    except Exception as e:
        raise ValueError(f"Failed to read patch file: {e}")

    # Get base prompt context
    base_prompt_context = ""
    if base_prompt_path and Path(base_prompt_path).exists():
        with open(base_prompt_path, 'r', encoding='utf-8') as f:
            base_prompt_context = f.read()

    # Get LLM for patch refinement
    lm = LM_DICT[model_name]

    print(f"Refining patch '{patch_name}' using {model_name}...")
    print(f"User feedback: {user_feedback}")

    # Use DSPy to refine the patch
    with dspy.context(lm=lm):
        refiner = dspy.ChainOfThought(PatchRefinement)
        result = refiner(
            original_patch_content=original_content,
            original_rationale=original_rationale,
            user_feedback=user_feedback,
            base_prompt_context=base_prompt_context
        )

    refined_content = result.refined_patch
    updated_rationale = result.updated_rationale

    # Determine output path - overwrite original by default
    if output_path is None:
        output_path = patch_path
    else:
        output_path = Path(output_path)

    # Create refined patch file with updated metadata
    patch_file_content = format_patch_file(
        patch_name=patch_name,
        patch_type=patch_type,
        patch_content=refined_content,
        patch_rationale=updated_rationale,
        generation_context="This patch was refined based on user feedback."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(patch_file_content)

    print(f"✅ Refined patch saved to: {output_path}")

    # Create refinement history record
    history = {
        "original_patch": str(patch_path),
        "refined_patch": str(output_path),
        "model": model_name,
        "user_feedback": user_feedback,
        "original_rationale": original_rationale,
        "updated_rationale": updated_rationale,
        "patch_name": patch_name,
        "patch_type": patch_type
    }

    # Save refinement history in the same directory
    history_file = output_path.parent / "refinement_history.json"
    history_data = []

    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)

    history_data.append(history)

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)

    print(f"📋 Refinement history updated: {history_file}")

    return {
        "original_patch": str(patch_path),
        "refined_patch": str(output_path),
        "user_feedback": user_feedback,
        "original_rationale": original_rationale,
        "updated_rationale": updated_rationale,
        "patch_name": patch_name,
        "patch_type": patch_type,
        "model": model_name
    }




def apply_patches(
    base_prompt_path: Path,
    patch_dir: Path,
    output_path: Path,
    patch_files: List[str] = None,
    version: str = None
) -> Dict[str, Any]:
    """
    Apply patches to a base prompt to create an improved version.

    This unified function handles both:
    1. Applying all patches from a directory
    2. Applying specific patches in sequence

    Args:
        base_prompt_path: Path to the original prompt file
        patch_dir: Directory containing patch files
        output_path: Path to save the improved prompt
        patch_files: Optional list of specific patch filenames to apply (e.g., ["01_clarify.patch"]).
                    If None, applies all patches in the directory.
        version: Optional version identifier for tracking (e.g., "v1", "v2")

    Returns:
        Dictionary containing application results
    """
    base_prompt_path = Path(base_prompt_path)
    patch_dir = Path(patch_dir)
    output_path = Path(output_path)

    if not base_prompt_path.exists():
        raise ValueError(f"Base prompt file not found: {base_prompt_path}")

    if not patch_dir.exists():
        raise ValueError(f"Patch directory does not exist: {patch_dir}")

    print(f"Base prompt: {base_prompt_path}")
    print(f"Patch directory: {patch_dir}")
    print(f"Output file: {output_path}")

    # Determine which patches to apply
    from .patch_utils import read_patch_file, apply_unified_diff

    if patch_files:
        # Use specific patches provided by user
        print(f"Applying specific patches: {', '.join(patch_files)}")
        patches_to_apply = [patch_dir / pf for pf in patch_files]

        # Verify all exist
        for patch_path in patches_to_apply:
            if not patch_path.exists():
                raise ValueError(f"Patch file not found: {patch_path}")
    else:
        # Apply all patches from directory
        print(f"Applying all patches from directory")
        patches_to_apply = sorted(patch_dir.glob("*.patch"))

        if not patches_to_apply:
            print(f"⚠️  No patch files found in {patch_dir}")
            return {
                "base_prompt": str(base_prompt_path),
                "output_file": str(output_path),
                "patches_applied": 0,
                "patches": [],
                "version": version,
                "versions_file": None
            }

        print(f"Found {len(patches_to_apply)} patch file(s)")

    # Read base prompt
    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply each patch sequentially
    applied_patches = []
    for patch_path in patches_to_apply:
        print(f"  Applying: {patch_path.name}...")

        try:
            patch_data = read_patch_file(patch_path)
            content = apply_unified_diff(content, patch_data["content"])
            applied_patches.append({
                "patch_file": patch_path.name,
                "patch_name": patch_data["name"],
                "patch_type": patch_data["type"],
                "rationale": patch_data["rationale"]
            })
            print(f"    ✅ Applied: {patch_data['name']}")
        except Exception as e:
            print(f"    ❌ Failed to apply {patch_path.name}: {e}")
            raise

    # Write improved prompt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    num_patches = len(applied_patches)

    print(f"\n✅ Patch application completed!")
    print(f"Applied {num_patches} patch(es)")
    print(f"Improved prompt saved to: {output_path}")

    # Create/update versions.json if version is specified
    versions_file = None
    if version:
        versions_file = output_path.parent / "versions.json"
        versions_data = {}

        if versions_file.exists():
            with open(versions_file, 'r', encoding='utf-8') as f:
                versions_data = json.load(f)

        versions_data[version] = {
            "output_file": str(output_path),
            "base_prompt": str(base_prompt_path),
            "patches_applied": applied_patches,
            "num_patches": num_patches
        }

        with open(versions_file, 'w', encoding='utf-8') as f:
            json.dump(versions_data, f, indent=2, ensure_ascii=False)

        print(f"📋 Version tracking updated: {versions_file}")

    return {
        "base_prompt": str(base_prompt_path),
        "output_file": str(output_path),
        "patches_applied": num_patches,
        "patches": applied_patches,
        "version": version,
        "versions_file": str(versions_file) if versions_file else None
    }