"""
Module for generating skill files from trajectory comparison analysis.

This module provides DSPy-based components for:
1. Proposing raw skills from trajectory comparisons
2. Merging skills into a consolidated set
3. Generating final skill files with proper format
"""

import json
import tqdm
import yaml
import shutil
import dspy
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..utils import LM_DICT, batch_inference


# =============================================================================
# Core DSPy Modules for Skill Generation
# =============================================================================

class Skill(BaseModel):
    """Pydantic model for a skill with proper format."""
    skill_name: str
    skill_description: str
    skill_trigger: str
    skill_body: str
    duplicate_count: Optional[int] = 1  # Number of similar skills merged into this one

class SkillGeneration(dspy.Signature):
    """Generate skill files based on trajectory comparison analysis.

    Each skill should have:
    - 'skill_name' (short identifier like 'error-recovery')
    - 'skill_description' (clear description of what the skill does)
    - 'skill_trigger' (a natural language description of when the skill should be triggered -
      this should be clear and specific enough that another agent can decide when to use this
      skill by comparing the current execution state against this trigger description)
    - 'skill_body' (markdown content with detailed instructions, examples, and guidance)
    """

    task_description = dspy.InputField(
        desc="The original task/request that agents were trying to complete"
    )
    comparison_analysis = dspy.InputField(
        desc="Analysis of behavioral differences between trajectories"
    )
    insight_extracted = dspy.InputField(
        desc="Short, actionable insights extracted from the comparison"
    )
    skills: List[Skill] = dspy.OutputField(
        desc="A list of skills. Each skill MUST include: skill_name, skill_description, skill_trigger (when to use the skill), and skill_body."
    )


class GenerateSkills(dspy.Module):
    """Module for generating structured skills from trajectory comparisons."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SkillGeneration)

    def forward(self, task_description: str, comparison_analysis: str, insight_extracted: str, config=None):
        """Generate skills from comparison analysis.

        Args:
            task_description: The original task description
            comparison_analysis: The comparison analysis from CompareTrajectories
            insight_extracted: Actionable insights extracted from the comparison
            config: Optional configuration dict

        Returns:
            DSPy prediction with structured skills
        """
        if config is None:
            config = {}
        return self.generate(
            task_description=task_description,
            comparison_analysis=comparison_analysis,
            insight_extracted=insight_extracted,
            config=config
        )


# =============================================================================
# Helper Functions for Skill Generation Workflow
# =============================================================================

def generate_skills_from_comparison(
    comparison_result: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Generate structured skills based on a trajectory comparison result.

    Args:
        comparison_result: Result from compare_rollout_trajectories containing
                          comparison_analysis, insight_extracted, and task_description
        model_name: Model to use for skill generation
        output_path: Optional path to save skills
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing generated skills
    """
    comparison_analysis = comparison_result["comparison_analysis"]
    insight_extracted = comparison_result["insight_extracted"]
    task_description = comparison_result.get("task_description", "")

    # Get LLM for skill generation
    lm = LM_DICT[model_name]

    print(f"Generating skills using {model_name} (seed={random_seed})...")
    with dspy.context(lm=lm):
        generator = GenerateSkills()
        result = generator(
            task_description=task_description,
            comparison_analysis=comparison_analysis,
            insight_extracted=insight_extracted,
            config={"rollout_id": random_seed},
        )

    # Convert skills to serializable format
    skills_data = [
        {
            "skill_name": skill.skill_name,
            "skill_description": skill.skill_description,
            "skill_trigger": skill.skill_trigger,
            "skill_body": skill.skill_body,
        }
        for skill in result.skills
    ]

    # Prepare output
    skill_result = {
        "model": model_name,
        "task_description": task_description,
        "comparison_analysis": comparison_analysis,
        "insight_extracted": insight_extracted,
        "skills": skills_data,
        "num_skills": len(skills_data),
    }

    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(skill_result, f, indent=2, ensure_ascii=False)
        elif output_path.suffix in [".yaml", ".yml"]:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(skill_result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            # Markdown format
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Skills Generated\n\n")
                f.write(f"**Model Used:** {model_name}\n")
                f.write(f"**Number of Skills:** {len(skills_data)}\n\n")
                f.write(f"## Task Description\n\n{task_description}\n\n")
                f.write(f"## Comparison Analysis\n\n{comparison_analysis}\n\n")
                f.write(f"## Insights Extracted\n\n{insight_extracted}\n\n")
                f.write(f"## Generated Skills\n\n")
                for skill in skills_data:
                    f.write(f"### {skill['skill_name']}\n\n")
                    f.write(f"**Description:** {skill['skill_description']}\n\n")
                    f.write(f"**Trigger:** {skill['skill_trigger']}\n\n")
                    f.write(f"**Body:**\n\n{skill['skill_body']}\n\n---\n\n")

        print(f"Skills saved to: {output_path}")

    return skill_result


def generate_skills_batch(
    comparison_results: List[Dict[str, Any]],
    model_name: str = "gemini-2.5-flash",
    output_dir: Path = None,
    random_seed: int = 0,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Generate skills from multiple comparison results in parallel.

    Args:
        comparison_results: List of comparison results, each containing
                          comparison_analysis, insight_extracted, task_description, and example_id
        model_name: Model to use for skill generation
        output_dir: Directory to save individual skill results
        random_seed: Random seed for reproducibility
        max_workers: Maximum number of parallel workers

    Returns:
        List of skill generation results
    """
    print(f"Generating skills from {len(comparison_results)} comparisons in parallel...")

    # Prepare arguments for batch inference
    args_list = []
    for comp_result in comparison_results:
        output_path = None
        if output_dir:
            example_id = comp_result.get("example_id", 0)
            output_path = output_dir / f"skills_example{example_id}.yaml"

        args_list.append({
            "comparison_result": comp_result,
            "model_name": model_name,
            "output_path": output_path,
            "random_seed": random_seed,
        })

    # Use batch_inference for parallel processing
    skill_results = batch_inference(
        program=generate_skills_from_comparison,
        args_list=args_list,
        use_process=False,
        max_workers=max_workers
    )

    # Print status for each result
    for comp_result, skill_result in zip(comparison_results, skill_results):
        example_id = comp_result.get("example_id", "?")
        if skill_result and skill_result.get("num_skills", 0) > 0:
            print(f"  ✅ Example {example_id}: {skill_result['num_skills']} skills generated")
        else:
            print(f"  ⚠️  Example {example_id}: Skill generation failed")

    return skill_results


def build_skills(
    comparison_results: List[Dict[str, Any]],
    output_dir: Path,
    model_name: str = "gemini-2.5-flash",
    embedder_model: str = "openai/text-embedding-3-small",
    similarity_threshold: float = 0.85,
    random_seed: int = 0,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Build skills from comparison results with automatic deduplication.

    This function:
    1. Generates skills from all comparison results using generate_skills_batch
    2. Creates a SkillManager
    3. Iteratively adds each skill to the manager (with embedding-based duplicate detection)
    4. Writes the final deduplicated skills to disk

    Args:
        comparison_results: List of comparison results
        output_dir: Directory to save final skills
        model_name: Model for skill generation and duplicate detection
        embedder_model: Model for embeddings
        similarity_threshold: Threshold for similarity detection
        random_seed: Random seed for reproducibility
        max_workers: Maximum parallel workers for skill generation

    Returns:
        Dictionary containing generation info
    """
    from .skill_manager import SkillManager

    # Step 1: Generate skills from all comparisons
    print("\n" + "="*80)
    print("STEP 1: Generating skills from comparisons")
    print("="*80)
    skill_results = generate_skills_batch(
        comparison_results=comparison_results,
        model_name=model_name,
        random_seed=random_seed,
        max_workers=max_workers,
    )

    # Filter out failed results
    valid_skill_results = [r for r in skill_results if r and r.get("num_skills", 0) > 0]
    print(f"\n✅ Generated skills from {len(valid_skill_results)}/{len(skill_results)} comparisons")

    # Flatten skill data for easier iteration
    all_skill_data = [
        skill_data
        for skill_result in valid_skill_results
        for skill_data in skill_result.get("skills", [])
    ]

    # Step 2: Create SkillManager and add skills iteratively
    print("\n" + "="*80)
    print("STEP 2: Adding skills to manager with duplicate detection")
    print("="*80)
    manager = SkillManager(
        embedder_model=embedder_model,
        similarity_threshold=similarity_threshold,
        lm_name=model_name,
    )

    total_skills_generated = 0
    for skill_data in tqdm.tqdm(all_skill_data):
        skill = Skill(
            skill_name=skill_data["skill_name"],
            skill_description=skill_data["skill_description"],
            skill_trigger=skill_data["skill_trigger"],
            skill_body=skill_data["skill_body"],
        )
        manager.add_skill(skill)
        total_skills_generated += 1

    print(f"\n📊 Skill Manager Stats:")
    stats = manager.get_stats()
    print(f"  Total skills generated: {total_skills_generated}")
    print(f"  Unique skills after deduplication: {stats['num_skills']}")
    print(f"  Duplicates merged: {stats['total_duplicates_merged']}")

    # Step 3: Write skills to disk
    print("\n" + "="*80)
    print("STEP 3: Writing skills to disk")
    print("="*80)
    generation_info = manager.save_skills(output_dir)

    print("\n" + "="*80)
    print("SKILL BUILDING COMPLETE")
    print("="*80)
    print(f"Skills directory: {output_dir}")
    print(f"Unique skills: {stats['num_skills']}")
    print(f"Total duplicates merged: {stats['total_duplicates_merged']}")
    print("="*80)

    return generation_info


def copy_existing_skills(
    existing_skills_folder: Path,
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Copy existing skills to the output directory.

    Args:
        existing_skills_folder: Path to folder containing existing skills
        output_dir: Directory to copy skills to

    Returns:
        List of skill metadata dictionaries
    """
    if not existing_skills_folder.exists():
        print(f"⚠️  Warning: Existing skills folder not found: {existing_skills_folder}")
        return []

    existing_skills = []

    # Find all SKILL.md files in the existing folder
    skill_files = list(existing_skills_folder.rglob("SKILL.md"))

    for skill_file in skill_files:
        # Get the skill folder name (parent directory)
        skill_folder = skill_file.parent
        skill_name = skill_folder.name

        # Copy the entire skill folder to output directory
        dest_folder = output_dir / skill_name
        if dest_folder.exists():
            shutil.rmtree(dest_folder)
        shutil.copytree(skill_folder, dest_folder)

        # Read the skill file to extract metadata
        try:
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    skill_description = frontmatter.get('description', '')
                    skill_trigger = frontmatter.get('trigger', None)

                    skill_metadata = {
                        'name': skill_name,
                        'description': skill_description,
                        'path': f"{skill_name}/SKILL.md",
                        'utility_score': None  # Existing skills don't have utility scores yet
                    }
                    if skill_trigger:
                        skill_metadata['trigger'] = skill_trigger

                    existing_skills.append(skill_metadata)
        except Exception as e:
            print(f"⚠️  Warning: Could not read skill file {skill_file}: {e}")
            continue

    print(f"✅ Copied {len(existing_skills)} existing skill(s)")
    return existing_skills