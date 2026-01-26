"""
Skill management with embedding-based duplicate detection.

This module provides a SkillManager class for managing skills with:
- In-memory skill storage with embedding cache
- Embedding-based similarity search
- LLM-based duplicate detection and merging
- Batch deduplication using existing DeduplicateSkills module
- Persistence to disk in standard format
"""

import json
import yaml
import dspy
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .skill_generation import (
    Skill,
)
from ..utils import LM_DICT


# =============================================================================
# DSPy Modules for Duplicate Detection
# =============================================================================

class AnalyzeSkillDuplicate(dspy.Signature):
    """Determine if a new skill is a duplicate of any existing skills.

    Analyze the new skill against all similar existing skills and determine:
    1) Whether the new skill is a duplicate of any existing skill
    2) If duplicate, which existing skill it duplicates and how to merge them
    3) If not duplicate, explain why it's distinct

    Consider skills duplicate if they:
    - Target the same trigger conditions
    - Solve the same underlying problem
    - Have substantially overlapping instructions

    Consider skills different if they:
    - Address different aspects of a problem
    - Have different trigger conditions
    - Provide complementary vs redundant guidance
    """

    new_skill: Skill = dspy.InputField(
        desc="The new skill being added"
    )
    similar_existing_skills: List[Skill] = dspy.InputField(
        desc="List of existing skills that may be similar (ordered by similarity score, highest first)"
    )

    is_duplicate: bool = dspy.OutputField(
        desc="Boolean: True if new skill is a duplicate of any existing skill, False if distinct"
    )
    duplicate_skill_index: int = dspy.OutputField(
        desc="If is_duplicate=True, the index (0-based) of the similar_existing_skills list for the skill it duplicates. If False, -1."
    )
    merged_skill: Skill = dspy.OutputField(
        desc="If is_duplicate=True, return merged skill combining the best aspects of both. If False, return new_skill unchanged."
    )


class AnalyzeSkillDuplicateModule(dspy.Module):
    """Module for analyzing if a new skill is a duplicate of any existing skills."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeSkillDuplicate)

    def forward(self, new_skill: Skill, similar_skills: List[Tuple[Skill, float]], config=None):
        """Analyze if new skill is a duplicate of any similar existing skills.

        Args:
            new_skill: The new skill being added
            similar_skills: List of (existing_skill, similarity_score) tuples, sorted by similarity (descending)
            config: Optional configuration dict

        Returns:
            DSPy prediction with is_duplicate, duplicate_skill_index, and merged_skill (as Skill object)
        """
        if config is None:
            config = {}

        # Extract just the Skill objects (similarity scores are implicit in the ordering)
        similar_skills_list = [skill for skill, _ in similar_skills]

        return self.analyze(
            new_skill=new_skill,
            similar_existing_skills=similar_skills_list,
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


# =============================================================================
# SkillManager Class
# =============================================================================

class SkillManager:
    """Manage a collection of skills with embedding-based duplicate detection.

    This class provides:
    - In-memory skill storage with embedding cache
    - Embedding-based similarity search using OpenAI embeddings
    - LLM-based duplicate detection
    - Skill merging when duplicates are found
    - Batch deduplication using DeduplicateSkills module
    - Persistence to disk in standard format
    """

    def __init__(
        self,
        embedder_model: str = "openai/text-embedding-3-small",
        similarity_threshold: float = 0.85,
        lm_name: str = "gemini-2.5-flash",
        batch_size: int = 100,
    ):
        """Initialize the SkillManager.

        Args:
            embedder_model: Model to use for embeddings (default: openai/text-embedding-3-small)
            similarity_threshold: Threshold for considering skills similar (default: 0.85)
            lm_name: Language model name from LM_DICT for duplicate analysis
            batch_size: Batch size for embedding computation
        """
        self.skills: List[Skill] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (num_skills, embedding_dim)
        self.embedder = dspy.Embedder(embedder_model, batch_size=batch_size)
        self.similarity_threshold = similarity_threshold
        self.lm = LM_DICT[lm_name]
        self.duplicate_analyzer = AnalyzeSkillDuplicateModule()

    def _compute_skill_embedding_text(self, skill: Skill) -> str:
        """Create text representation of skill for embedding.

        Combines name, description, and trigger for rich semantic representation.

        Args:
            skill: The skill to create text for

        Returns:
            Combined text string for embedding
        """
        parts = [
            f"Name: {skill.skill_name}",
            f"Description: {skill.skill_description}",
            f"Trigger: {skill.skill_trigger}",
        ]
        return " | ".join(parts)

    def _compute_embeddings(self, skills: List[Skill]) -> np.ndarray:
        """Compute embeddings for a list of skills.

        Args:
            skills: List of Skill objects

        Returns:
            Numpy array of embeddings, shape (len(skills), embedding_dim)
        """
        texts = [self._compute_skill_embedding_text(skill) for skill in skills]
        embeddings = self.embedder(texts)

        # Ensure it's a 2D numpy array
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm)

        return float(similarity)

    def _find_similar_skills(self, skill_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find the top-k most similar skills to a given embedding.

        Args:
            skill_embedding: Embedding vector for the new skill
            top_k: Number of similar skills to return

        Returns:
            List of (skill_index, similarity_score) tuples, sorted by similarity (descending)
        """
        if self.embeddings is None or len(self.skills) == 0:
            return []

        # Compute similarities with all existing skills
        similarities = []
        for idx in range(len(self.skills)):
            sim = self._cosine_similarity(skill_embedding, self.embeddings[idx])
            similarities.append((idx, sim))

        # Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and limit to top_k
        filtered = [(idx, score) for idx, score in similarities if score >= self.similarity_threshold]
        return filtered[:top_k]

    def add_skill(self, new_skill: Skill) -> bool:
        """Add a skill to the manager with duplicate detection.

        Flow:
        1. Compute embedding for new skill
        2. Find top-5 most similar skills (cosine similarity >= threshold)
        3. If similar skills found, use LLM to batch analyze all similar skills for duplication
        4. If duplicate found, merge into existing skill
        5. If not duplicate, add to pool

        Args:
            new_skill: The Skill object to add

        Returns:
            True if skill was added as new, False if merged into existing
        """
        # Compute embedding for new skill
        new_embedding = self._compute_embeddings([new_skill])[0]

        # Find similar skills
        similar_skills = self._find_similar_skills(new_embedding, top_k=5)

        if not similar_skills:
            # No similar skills, add directly
            self.skills.append(new_skill)
            if self.embeddings is None:
                self.embeddings = new_embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, new_embedding])
            return True

        # Batch analyze all similar skills with LLM
        with dspy.context(lm=self.lm):
            # Prepare list of (skill, similarity_score) tuples
            similar_skills_with_objects = [(self.skills[idx], score) for idx, score in similar_skills]

            # Use LLM to determine if duplicate in batch
            result = self.duplicate_analyzer(
                new_skill=new_skill,
                similar_skills=similar_skills_with_objects
            )

            if result.is_duplicate:
                # Merge: get duplicate_skill_index and merged_skill from LLM output
                try:
                    # Get the index of the skill to merge with
                    duplicate_idx = int(result.duplicate_skill_index)

                    if duplicate_idx < 0 or duplicate_idx >= len(similar_skills):
                        raise ValueError(f"Invalid duplicate_skill_index: {duplicate_idx}")

                    # Get the actual skill index in self.skills
                    skill_idx = similar_skills[duplicate_idx][0]
                    existing_skill = self.skills[skill_idx]
                    similarity_score = similar_skills[duplicate_idx][1]

                    # Get merged skill from result (now a Skill object)
                    merged_skill_result = result.merged_skill

                    # Update duplicate count
                    merged_skill = Skill(
                        skill_name=merged_skill_result.skill_name,
                        skill_description=merged_skill_result.skill_description,
                        skill_trigger=merged_skill_result.skill_trigger,
                        skill_body=merged_skill_result.skill_body,
                        duplicate_count=existing_skill.duplicate_count + 1
                    )

                    # Replace existing skill
                    self.skills[skill_idx] = merged_skill

                    # Update embedding for the merged skill
                    merged_embedding = self._compute_embeddings([merged_skill])[0]
                    self.embeddings[skill_idx] = merged_embedding

                    print(f"✅ Merged '{new_skill.skill_name}' into '{existing_skill.skill_name}' (similarity: {similarity_score:.3f})")
                    return False  # Skill was merged, not added

                except Exception as e:
                    print(f"⚠️  Warning: Failed to merge skill: {e}")
                    # Fall through to add as new skill

        # No duplicates found, add as new skill
        self.skills.append(new_skill)
        if self.embeddings is None:
            self.embeddings = new_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])

        print(f"✅ Added new skill: '{new_skill.skill_name}'")
        return True

    def deduplicate_skills(self, task_description: str = "") -> List[Skill]:
        """Deduplicate all skills using the existing DeduplicateSkills module.

        This performs a batch deduplication across all skills, which may catch
        duplicates that weren't found during incremental add_skill() calls.

        Args:
            task_description: Optional task description for context

        Returns:
            List of deduplicated Skill objects
        """
        if not self.skills:
            return []

        print(f"Deduplicating {len(self.skills)} skills...")

        with dspy.context(lm=self.lm):
            deduplicator = DeduplicateSkills()
            deduplicated_skills = deduplicator(
                task_description=task_description,
                all_skills=self.skills
            ).deduplicated_skills

        # Update internal state with deduplicated skills
        self.skills = list(deduplicated_skills)

        # Recompute embeddings for deduplicated skills
        if self.skills:
            self.embeddings = self._compute_embeddings(self.skills)
        else:
            self.embeddings = None

        print(f"✅ Deduplication complete: {len(self.skills)} skills remaining")
        return self.skills

    def save_skills(self, output_dir: Path, task_description: str = "Skills managed by SkillManager") -> Dict[str, Any]:
        """Save all skills to disk using the standard format.

        Args:
            output_dir: Directory to save skills to
            task_description: Optional task description for metadata

        Returns:
            Dictionary containing generation info
        """
        if not self.skills:
            print("⚠️  No skills to save")
            return {"num_skills": 0, "skills": [], "output_dir": str(output_dir), "task_description": task_description}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing {len(self.skills)} skill files to {output_dir}...")

        # Save each skill to its own directory
        saved_skills = []
        for skill in self.skills:
            skill_name = skill.skill_name
            skill_description = skill.skill_description
            skill_trigger = skill.skill_trigger
            skill_body = skill.skill_body
            duplicate_count = skill.duplicate_count or 1

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

        # Sort saved skills by duplicate count
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

    def load_skills(self, skill_dir: Path) -> None:
        """Load skills from disk into memory.

        Reads all SKILL.md files from subdirectories and loads them into
        the manager, computing embeddings for each. Reads duplicate_count
        from metadata.yaml.

        Args:
            skill_dir: Directory containing skill subdirectories
        """
        skill_dir = Path(skill_dir)

        if not skill_dir.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        # Load metadata.yaml to get duplicate_count for each skill
        metadata_path = skill_dir / "metadata.yaml"
        duplicate_counts = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
                for skill_meta in metadata.get('skills', []):
                    skill_name = skill_meta.get('name')
                    if skill_name:
                        duplicate_counts[skill_name] = skill_meta.get('duplicate_count', 1)

        # Find all SKILL.md files
        skill_files = list(skill_dir.rglob("SKILL.md"))

        loaded_skills = []
        for skill_file in skill_files:
            try:
                with open(skill_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse frontmatter
                if not content.startswith('---'):
                    print(f"⚠️  Skipping {skill_file}: No YAML frontmatter")
                    continue

                parts = content.split('---', 2)
                if len(parts) < 3:
                    print(f"⚠️  Skipping {skill_file}: Invalid format")
                    continue

                frontmatter = yaml.safe_load(parts[1])
                skill_body = parts[2].strip()

                # Get skill name and look up duplicate_count from metadata
                skill_name = frontmatter.get('name', skill_file.parent.name)
                duplicate_count = duplicate_counts.get(skill_name, 1)

                # Create Skill object
                skill = Skill(
                    skill_name=skill_name,
                    skill_description=frontmatter.get('description', ''),
                    skill_trigger=frontmatter.get('trigger', ''),
                    skill_body=skill_body,
                    duplicate_count=duplicate_count
                )

                loaded_skills.append(skill)

            except Exception as e:
                print(f"⚠️  Warning: Failed to load {skill_file}: {e}")
                continue

        # Update internal state
        self.skills = loaded_skills
        self.skills.sort(key=lambda x: x.duplicate_count, reverse=True)

        # Compute embeddings for all loaded skills
        if self.skills:
            self.embeddings = self._compute_embeddings(self.skills)
        else:
            self.embeddings = None

        print(f"✅ Loaded {len(self.skills)} skills from {skill_dir}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the skill collection.

        Returns:
            Dictionary with counts and metadata
        """
        return {
            "num_skills": len(self.skills),
            "total_duplicates_merged": sum(
                (skill.duplicate_count or 1) - 1 for skill in self.skills
            ),
            "skill_names": [skill.skill_name for skill in self.skills],
        }

    def get_top_k_skills(self, k: int) -> List[Skill]:
        """Get top-k skills by duplicate count.

        Args:
            k: Number of top skills to return

        Returns:
            List of top-k Skill objects sorted by duplicate_count (descending)
        """
        if not self.skills:
            return []

        # Sort by duplicate_count (descending) and take top k
        sorted_skills = sorted(
            self.skills,
            key=lambda s: s.duplicate_count or 1,
            reverse=True
        )
        return sorted_skills[:k]

    def get_random_skills(self, k: int, seed: Optional[int] = None) -> List[Skill]:
        """Get k random skills from the collection.

        Args:
            k: Number of random skills to return
            seed: Optional random seed for reproducibility

        Returns:
            List of k randomly selected Skill objects
        """
        if not self.skills:
            return []

        if seed is not None:
            random.seed(seed)

        # Return min(k, num_skills) to handle case where k > len(self.skills)
        sample_size = min(k, len(self.skills))
        return random.sample(self.skills, sample_size)

    def get_skills_by_names(self, skill_names: List[str]) -> List[Skill]:
        """Get skills by their names.

        Args:
            skill_names: List of skill names to retrieve

        Returns:
            List of Skill objects matching the given names (in order)
        """
        name_to_skill = {skill.skill_name: skill for skill in self.skills}
        return [name_to_skill[name] for name in skill_names if name in name_to_skill]