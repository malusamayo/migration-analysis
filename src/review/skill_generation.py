"""
Module for generating skill files from aggregated analysis.

This module provides DSPy-based components and functions for generating
skill files based on trajectory comparison analysis.
"""

import json
import yaml
import shutil
import dspy
from pathlib import Path
from typing import List, Dict, Any

from ..utils import LM_DICT


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

from pydantic import BaseModel
class Skill(BaseModel):
    skill_name: str
    skill_description: str
    skill_trigger: str
    skill_body: str
    

class SkillGeneration(dspy.Signature):
    """Generate skill files based on aggregated analysis of trajectory comparisons.
Each skill should have:
- 'skill_name' (short identifier like 'error-recovery')
- 'skill_description' (clear description of what the skill does)
- 'skill_trigger' (a natural language description of when the skill should be triggered - this should be clear and specific enough that another agent can decide when to use this skill by comparing the current execution state against this trigger description)
- 'skill_body' (markdown content with detailed instructions, examples, and guidance)"""

    task_description = dspy.InputField(
        desc="The original task/request that agents were trying to complete"
    )
    common_patterns = dspy.InputField(
        desc="Common behavioral patterns identified across multiple trajectory rollouts"
    )
    recommended_improvements = dspy.InputField(
        desc="Recommended improvements based on the analysis"
    )
    skills: List[Skill] = dspy.OutputField(
        desc="A list of skills. Each skill MUST include: skill_name, skill_description, skill_trigger (when to use the skill), and skill_body."
    )


class GenerateTaskSkills(dspy.Module):
    """Module for generating skills from task analysis."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SkillGeneration)

    def forward(self, task_description: str, common_patterns: str, recommended_improvements: str):
        """Generate skills based on aggregated analysis."""
        return self.generate(
            task_description=task_description,
            common_patterns=common_patterns,
            recommended_improvements=recommended_improvements
        )


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


def generate_universal_skills(
    aggregated_analysis: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
    output_dir: Path = None,
    latest_version_dir: Path = None,
) -> Dict[str, Any]:
    """
    Generate universal skills from aggregated analysis.

    Instead of generating patches, this creates skill files that can be used
    to improve agent behavior based on common patterns identified across examples.

    Args:
        aggregated_analysis: Result from aggregate_comparison_analyses()
        model_name: Model to use for skill generation (default: "gemini-2.5-flash")
        output_dir: Base directory to save skill files (will auto-version as v1, v2, etc.)

    Returns:
        Dictionary containing skill generation results
    """
    task_description = aggregated_analysis.get("task_description", "")
    common_patterns = aggregated_analysis.get("common_patterns", [])
    recommended_improvements = aggregated_analysis.get("recommended_improvements", [])

    if not common_patterns:
        raise ValueError("No common patterns found in aggregated analysis")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing skills from the latest version if it exists
    existing_skills_metadata = []
    if latest_version_dir and latest_version_dir.exists():
        print(f"📋 Copying existing skills from {latest_version_dir}...")
        existing_skills_metadata = copy_existing_skills(latest_version_dir, output_dir)
    else:
        print(f"📋 No existing skills to copy.")

    # Format patterns and improvements for skill generation
    patterns_text = ""
    if isinstance(common_patterns, list):
        patterns_text = "\n".join([f"- {p}" for p in common_patterns])
    else:
        patterns_text = str(common_patterns)

    improvements_text = ""
    if isinstance(recommended_improvements, list):
        improvements_text = "\n".join([f"- {i}" for i in recommended_improvements])
    else:
        improvements_text = str(recommended_improvements)

    # Get LLM for skill generation
    lm = LM_DICT[model_name]

    print(f"Generating skills using {model_name}...")

    with dspy.context(lm=lm):
        skill_generator = GenerateTaskSkills()
        skills = skill_generator(
            task_description=task_description,
            common_patterns=patterns_text,
            recommended_improvements=improvements_text
        ).skills

    # Save each skill
    saved_skills = []
    for i, skill in enumerate(skills, 1):
        skill_name = skill.skill_name
        skill_description = skill.skill_description
        skill_trigger = skill.skill_trigger
        skill_body = skill.skill_body

        # Create skill directory
        skill_dir = output_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Create skill file
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
            "utility_score": None  # New skills don't have utility scores yet
        }
        if skill_trigger:
            skill_metadata["trigger"] = skill_trigger

        saved_skills.append(skill_metadata)
        print(f"✅ Saved skill: {skill_path}")

    # Combine all skills metadata (existing + new)
    all_skills_metadata = existing_skills_metadata + saved_skills

    # Save metadata.yaml
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'skills': all_skills_metadata
        }, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n📋 Skill metadata saved to: {metadata_path}")

    # Also save detailed generation info as JSON
    generation_info = {
        "model": model_name,
        "version": output_dir.name,
        "output_dir": str(output_dir),
        "previous_version": latest_version_dir.name if latest_version_dir else None,
        "num_new_skills": len(saved_skills),
        "num_existing_skills": len(existing_skills_metadata),
        "total_skills": len(all_skills_metadata),
        "new_skills": saved_skills,
        "existing_skills": existing_skills_metadata,
        "num_examples_analyzed": aggregated_analysis.get("num_examples_analyzed", 0),
        "example_ids": aggregated_analysis.get("example_ids", []),
    }

    generation_info_path = output_dir / "generation_info.json"
    with open(generation_info_path, 'w', encoding='utf-8') as f:
        json.dump(generation_info, f, indent=2, ensure_ascii=False)

    print(f"📋 Generation info saved to: {generation_info_path}")

    return generation_info
