"""
Module for generating skill files from trajectory comparison analysis.

This module provides DSPy-based components for:
1. Proposing raw skills from trajectory comparisons
2. Merging skills into a consolidated set
3. Generating final skill files with proper format
"""

import json
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


class SkillDeduplication(dspy.Signature):
    """Deduplicate and refine skills from multiple comparisons by removing redundancy.

    Analyze all skills and:
    1) Identify duplicate or highly overlapping skills
    2) Keep the best version of each unique skill concept
    3) Preserve atomic, focused skills (do NOT merge distinct skills into larger ones)
    4) Eliminate true redundancies while maintaining skill diversity
    5) Track how many input skills contributed to each output skill (duplicate_count)

    IMPORTANT: Keep skills small and atomic. Only remove skills that are truly redundant.
    If two skills address different aspects or concerns, keep both even if they seem related.

    For each output skill, set 'duplicate_count' to the number of input skills it represents
    (e.g., if 5 similar skills were consolidated into one, duplicate_count = 5).

    Each output skill MUST have:
    - 'skill_name' (short identifier like 'error-recovery')
    - 'skill_description' (clear description of what the skill does)
    - 'skill_trigger' (when to use the skill - clear and specific)
    - 'skill_body' (markdown content with detailed instructions and examples)
    - 'duplicate_count' (integer count of how many input skills this represents)
    """

    task_description = dspy.InputField(
        desc="The original task/request that agents were trying to complete"
    )
    all_skills: List[Skill] = dspy.InputField(
        desc="All skills from multiple comparisons that need to be deduplicated"
    )
    deduplicated_skills: List[Skill] = dspy.OutputField(
        desc="Deduplicated list of atomic skills with redundancies removed. Each MUST include: skill_name, skill_description, skill_trigger, skill_body, and duplicate_count (number of input skills consolidated). Keep skills focused and atomic."
    )


class DeduplicateSkills(dspy.Module):
    """Module for deduplicating skills from multiple comparisons by removing redundancy."""

    def __init__(self):
        super().__init__()
        self.deduplicate = dspy.ChainOfThought(SkillDeduplication)

    def forward(self, task_description: str, all_skills: List[Skill], config=None):
        """Deduplicate skills by removing redundancies.

        Args:
            task_description: The original task description
            all_skills: All skills from multiple comparisons (list of Skill objects)
            config: Optional configuration dict

        Returns:
            DSPy prediction with deduplicated skills
        """
        if config is None:
            config = {}
        return self.deduplicate(
            task_description=task_description,
            all_skills=all_skills,
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


def merge_skills(
    skill_results: List[Dict[str, Any]],
    task_description: str,
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Deduplicate skills from multiple generation results by removing redundancy.

    This function does NOT merge distinct skills into larger ones. Instead, it:
    - Identifies and removes duplicate skills
    - Keeps the best version of each unique skill
    - Preserves atomic, focused skills

    Args:
        skill_results: List of skill results from generate_skills_from_comparison
        task_description: The task description
        model_name: Model to use for deduplication
        output_path: Optional path to save deduplicated skills
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing deduplicated skills (atomic Skill objects with redundancy removed)
    """
    # Combine all skills into a list of Skill objects
    all_skills = []
    for result in skill_results:
        skills = result.get("skills", [])
        for skill_data in skills:
            all_skills.append(Skill(
                skill_name=skill_data['skill_name'],
                skill_description=skill_data['skill_description'],
                skill_trigger=skill_data['skill_trigger'],
                skill_body=skill_data['skill_body']
            ))

    # Get LLM for deduplication
    lm = LM_DICT[model_name]

    print(f"Deduplicating {len(all_skills)} skills from {len(skill_results)} skill sets using {model_name} (seed={random_seed})...")
    with dspy.context(lm=lm):
        deduplicator = DeduplicateSkills()
        result = deduplicator(
            task_description=task_description,
            all_skills=all_skills,
            config={"rollout_id": random_seed},
        )

    # Convert deduplicated skills to serializable format
    deduplicated_skills_data = [
        {
            "skill_name": skill.skill_name,
            "skill_description": skill.skill_description,
            "skill_trigger": skill.skill_trigger,
            "skill_body": skill.skill_body,
            "duplicate_count": skill.duplicate_count,
        }
        for skill in result.deduplicated_skills
    ]

    # Prepare output (keep field names as merged_skills for backward compatibility)
    merged_result = {
        "model": model_name,
        "task_description": task_description,
        "num_skill_sets": len(skill_results),
        "merged_skills": deduplicated_skills_data,  # Keep this name for compatibility
        "num_merged_skills": len(deduplicated_skills_data),
    }

    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_result, f, indent=2, ensure_ascii=False)
        elif output_path.suffix in [".yaml", ".yml"]:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(merged_result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
           assert False, "Output path must be .json or .yaml/.yml for merged skills."

        print(f"Deduplicated skills saved to: {output_path}")

    return merged_result


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


def write_skills_to_files(
    merged_result: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Write deduplicated atomic skills to disk as skill files.

    Args:
        merged_result: Result from merge_skills containing deduplicated atomic skills
        output_dir: Directory to save skill files (each skill in its own subdirectory)

    Returns:
        Dictionary containing generation results with skill metadata
    """
    task_description = merged_result["task_description"]
    merged_skills = merged_result["merged_skills"]  # List of skill dicts

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(merged_skills)} skill files to {output_dir}...")

    # Save each skill to its own directory
    saved_skills = []
    for skill_data in merged_skills:
        skill_name = skill_data["skill_name"]
        skill_description = skill_data["skill_description"]
        skill_trigger = skill_data["skill_trigger"]
        skill_body = skill_data["skill_body"]
        duplicate_count = skill_data.get("duplicate_count", 1)

        # Create skill directory
        skill_dir = output_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Create skill file using format_skill_file
        skill_file_content = format_skill_file(
            skill_name=skill_name,
            skill_description=skill_description,
            skill_body=skill_body,
            skill_trigger=skill_trigger
        )

        skill_path = skill_dir / "SKILL.md"
        with open(skill_path, 'w', encoding='utf-8') as f:
            f.write(skill_file_content)

        skill_metadata = {
            "name": skill_name,
            "description": skill_description,
            "path": f"{skill_name}/SKILL.md",
            "duplicate_count": duplicate_count,
        }
        if skill_trigger:
            skill_metadata["trigger"] = skill_trigger

        saved_skills.append(skill_metadata)
        print(f"✅ Saved skill: {skill_path}")

    # sort saved skills by count
    saved_skills.sort(key=lambda x: x.get("duplicate_count", 1), reverse=True)

    # Save metadata.yaml
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'skills': saved_skills
        }, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"📋 Skill metadata saved to: {metadata_path}")

    # Save detailed generation info as JSON
    generation_info = {
        "output_dir": str(output_dir),
        "num_skills": len(saved_skills),
        "skills": saved_skills,
        "task_description": task_description,
    }

    generation_info_path = output_dir / "generation_info.json"
    with open(generation_info_path, 'w', encoding='utf-8') as f:
        json.dump(generation_info, f, indent=2, ensure_ascii=False)

    print(f"📋 Generation info saved to: {generation_info_path}")

    return generation_info


# =============================================================================
# Skill File Formatting and Management
# =============================================================================

def format_skill_file(
    skill_name: str,
    skill_description: str,
    skill_body: str,
    skill_trigger: str = None
) -> str:
    """
    Format a skill file with proper YAML frontmatter and markdown body.

    Args:
        skill_name: Name of the skill (used in frontmatter)
        skill_description: Description of what the skill does and when to use it
        skill_body: The markdown body with instructions and guidance
        skill_trigger: Natural language description of when the skill should be triggered

    Returns:
        Formatted skill file content as a string
    """
    # Build YAML frontmatter
    frontmatter = {
        'name': skill_name,
        'description': skill_description
    }

    if skill_trigger:
        frontmatter['trigger'] = skill_trigger

    # Combine frontmatter and body
    skill_content = "---\n"
    skill_content += yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    skill_content += "---\n\n"
    skill_content += skill_body.strip() + "\n"

    return skill_content

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