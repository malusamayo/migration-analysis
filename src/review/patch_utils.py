"""
Utilities for working with unified diff patches.

This module provides functions for parsing, applying, and manipulating
unified diff format patches.
"""

import re
from pathlib import Path
from typing import List, Dict, Any


HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def extract_patch_body(patch_lines: List[str]) -> str:
    """Return the raw unified diff between BEGIN/END markers."""
    in_patch = False
    body_lines: List[str] = []
    for line in patch_lines:
        if line.strip() == "---BEGIN PATCH---":
            in_patch = True
            continue
        if line.strip() == "---END PATCH---":
            break
        if in_patch:
            body_lines.append(line)

    patch_body = "\n".join(body_lines).strip("\n")
    if not patch_body:
        raise ValueError("Patch content missing between ---BEGIN PATCH--- and ---END PATCH---")
    return patch_body


def parse_unified_diff(diff_text: str) -> List[Dict[str, Any]]:
    """Parse unified diff text into a structured representation."""
    lines = diff_text.splitlines()
    file_patches: List[Dict[str, Any]] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line.startswith(("diff ", "index ", "new file mode", "deleted file mode")):
            i += 1
            continue
        if not line.startswith("--- "):
            i += 1
            continue

        old_file = line[4:].strip()
        i += 1
        if i >= len(lines) or not lines[i].startswith("+++ "):
            raise ValueError("Malformed patch: missing +++ header")
        new_file = lines[i][4:].strip()
        i += 1

        hunks: List[Dict[str, Any]] = []
        while i < len(lines) and lines[i].startswith("@@"):
            header = lines[i]
            match = HUNK_HEADER_RE.match(header)
            if not match:
                raise ValueError(f"Malformed hunk header: {header}")
            start_old = int(match.group(1))
            len_old = int(match.group(2) or "1")
            start_new = int(match.group(3))
            len_new = int(match.group(4) or "1")
            i += 1

            hunk_lines: List[str] = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                hunk_lines.append(lines[i])
                i += 1

            hunks.append({
                "start_old": start_old,
                "len_old": len_old,
                "start_new": start_new,
                "len_new": len_new,
                "lines": hunk_lines
            })

        if not hunks:
            raise ValueError("Malformed patch: no hunks found after headers")

        file_patches.append({
            "old_file": old_file,
            "new_file": new_file,
            "hunks": hunks
        })

    if not file_patches:
        raise ValueError("No diff hunks found in patch content")

    return file_patches


def normalize_line(line: str) -> str:
    """Normalize a line for fuzzy matching by removing extra whitespace."""
    return " ".join(line.split())


def calculate_line_similarity(line1: str, line2: str) -> float:
    """Calculate similarity between two lines (0.0 to 1.0)."""
    norm1 = normalize_line(line1)
    norm2 = normalize_line(line2)
    
    if norm1 == norm2:
        return 1.0
    
    # Check if one is a substring of the other (common with minor edits)
    if norm1 in norm2 or norm2 in norm1:
        return 0.9
    
    # Calculate simple character overlap ratio
    if not norm1 or not norm2:
        return 0.0
    
    shorter = min(len(norm1), len(norm2))
    longer = max(len(norm1), len(norm2))
    
    # Count matching characters at the start
    matches = sum(1 for a, b in zip(norm1, norm2) if a == b)
    
    return matches / longer


def calculate_hunk_similarity(lines: List[str], expected_lines: List[str], start_idx: int) -> float:
    """Calculate overall similarity of a hunk match."""
    if start_idx + len(expected_lines) > len(lines):
        return 0.0
    
    actual_lines = lines[start_idx:start_idx + len(expected_lines)]
    similarities = [
        calculate_line_similarity(actual, expected)
        for actual, expected in zip(actual_lines, expected_lines)
    ]
    
    return sum(similarities) / len(similarities) if similarities else 0.0


def locate_hunk_start(lines: List[str], expected_lines: List[str], hint_index: int, fuzzy: bool = True) -> int:
    """Find where the current hunk should apply within the file.
    
    Args:
        lines: The lines of the target file
        expected_lines: The expected context lines from the patch
        hint_index: The suggested starting position
        fuzzy: If True, use fuzzy matching when exact match fails
    """
    if not expected_lines:
        return max(0, min(hint_index, len(lines)))

    max_start = len(lines) - len(expected_lines)
    if max_start < 0:
        raise ValueError("Patch expects more lines than available in target file")

    hint_index = max(0, min(hint_index, max_start))
    candidate_indices = sorted(
        range(max_start + 1),
        key=lambda idx: (abs(idx - hint_index), idx)
    )

    # First try exact match
    for idx in candidate_indices:
        if lines[idx:idx + len(expected_lines)] == expected_lines:
            return idx

    # If fuzzy matching is enabled and exact match failed, try fuzzy match
    if fuzzy:
        best_match = None
        best_similarity = 0.0
        threshold = 0.75  # Require at least 75% similarity
        
        for idx in candidate_indices:
            similarity = calculate_hunk_similarity(lines, expected_lines, idx)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = idx
        
        if best_match is not None and best_similarity >= threshold:
            return best_match

    raise ValueError("Patch hunk does not match target content")


def apply_unified_diff(original_content: str, diff_text: str) -> str:
    """Apply a unified diff to the given text content."""
    file_patches = parse_unified_diff(diff_text)
    if len(file_patches) != 1:
        raise ValueError("Patch files should modify exactly one file")

    lines = original_content.splitlines()
    ends_with_newline = original_content.endswith("\n")
    offset = 0
    final_newline = ends_with_newline

    for hunk in file_patches[0]["hunks"]:
        start_old = hunk["start_old"]
        hunk_lines = hunk["lines"]

        if start_old == 0:
            hint_index = 0
        else:
            hint_index = start_old - 1 + offset
            if hint_index < 0:
                raise ValueError("Calculated patch start is before beginning of file")

        expected_lines: List[str] = []
        replacement_lines: List[str] = []
        prev_prefix = ""

        for raw_line in hunk_lines:
            if raw_line.startswith("\\ No newline at end of file"):
                if prev_prefix == "+":
                    final_newline = False
                continue

            if not raw_line:
                raise ValueError("Unexpected blank line inside hunk")

            prefix = raw_line[0]
            content = raw_line[1:]
            prev_prefix = prefix

            if prefix == " ":
                expected_lines.append(content)
                replacement_lines.append(content)
            elif prefix == "-":
                expected_lines.append(content)
            elif prefix == "+":
                replacement_lines.append(content)
            else:
                raise ValueError(f"Unexpected hunk line prefix: {prefix}")

        start_index = locate_hunk_start(lines, expected_lines, hint_index)

        lines[start_index:start_index + len(expected_lines)] = replacement_lines
        offset += (start_index - hint_index) + len(replacement_lines) - len(expected_lines)

    new_content = "\n".join(lines)
    if final_newline:
        new_content += "\n"

    return new_content


def read_patch_file(patch_file: Path) -> Dict[str, str]:
    """
    Read a patch file and extract its metadata and content.

    Args:
        patch_file: Path to the patch file

    Returns:
        Dictionary with 'name', 'type', 'rationale', and 'content' keys
    """
    with open(patch_file, 'r', encoding='utf-8') as f:
        patch_content = f.read()

    patch_lines = patch_content.split('\n')
    patch_name = patch_file.name
    patch_type = ""
    patch_rationale = ""

    for line in patch_lines:
        if line.startswith("# Patch:"):
            patch_name = line.replace("# Patch:", "").strip()
        elif line.startswith("# Type:"):
            patch_type = line.replace("# Type:", "").strip()
        elif line.startswith("# Rationale:"):
            patch_rationale = line.replace("# Rationale:", "").strip()
        elif line.strip() == "---BEGIN PATCH---":
            break

    patch_diff_text = extract_patch_body(patch_lines)

    return {
        "name": patch_name,
        "type": patch_type,
        "rationale": patch_rationale,
        "content": patch_diff_text
    }


def apply_patch_to_file(
    target_file: Path,
    patch_file: Path,
    output_file: Path = None,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Apply a single patch file to a target file.

    Args:
        target_file: Path to the file to patch
        patch_file: Path to the patch file
        output_file: Path to save the result (default: overwrite target_file)
        create_backup: Whether to create a backup of the original file

    Returns:
        Dictionary containing application results
    """
    if not target_file.exists():
        raise ValueError(f"Target file not found: {target_file}")

    if not patch_file.exists():
        raise ValueError(f"Patch file not found: {patch_file}")

    # Read original content
    with open(target_file, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Read and parse patch
    patch_data = read_patch_file(patch_file)

    # Apply patch
    try:
        modified_content = apply_unified_diff(original_content, patch_data["content"])
    except ValueError as err:
        raise ValueError(f"Failed to apply patch '{patch_file}': {err}") from err

    # Create backup if requested
    if create_backup:
        backup_file = target_file.with_suffix(target_file.suffix + '.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"📦 Backup created: {backup_file}")

    # Write modified content
    if output_file is None:
        output_file = target_file

    output_file = Path(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print(f"✅ Patch applied: {patch_data['name']}")
    print(f"   Output: {output_file}")

    return {
        "target_file": str(target_file),
        "patch_file": str(patch_file),
        "output_file": str(output_file),
        "patch_name": patch_data["name"],
        "patch_type": patch_data["type"],
        "patch_rationale": patch_data["rationale"],
        "backup_created": create_backup
    }


def apply_patches_to_file(
    target_file: Path,
    patch_dir: Path,
    output_file: Path = None,
    create_backup: bool = True,
    patch_pattern: str = "*.patch"
) -> Dict[str, Any]:
    """
    Apply multiple patches from a directory to a target file.

    Args:
        target_file: Path to the file to patch
        patch_dir: Directory containing patch files
        output_file: Path to save the result (default: overwrite target_file)
        create_backup: Whether to create a backup of the original file
        patch_pattern: Glob pattern for patch files (default: "*.patch")

    Returns:
        Dictionary containing application results
    """
    if not target_file.exists():
        raise ValueError(f"Target file not found: {target_file}")

    if not patch_dir.exists():
        raise ValueError(f"Patch directory not found: {patch_dir}")

    # Find all patch files
    patch_files = sorted(patch_dir.glob(patch_pattern))

    if not patch_files:
        print(f"⚠️  No patch files found matching '{patch_pattern}' in {patch_dir}")
        return {
            "target_file": str(target_file),
            "patches_applied": 0,
            "patches": []
        }

    print(f"Found {len(patch_files)} patch file(s) in {patch_dir}")

    # Read original content
    with open(target_file, 'r', encoding='utf-8') as f:
        modified_content = f.read()

    # Create backup if requested
    if create_backup:
        backup_file = target_file.with_suffix(target_file.suffix + '.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"📦 Backup created: {backup_file}")

    # Apply each patch sequentially
    applied_patches = []
    for patch_file in patch_files:
        print(f"\nApplying: {patch_file.name}")

        patch_data = read_patch_file(patch_file)

        try:
            modified_content = apply_unified_diff(modified_content, patch_data["content"])
        except ValueError as err:
            raise ValueError(f"Failed to apply patch '{patch_file}': {err}") from err

        applied_patches.append({
            "patch_file": str(patch_file),
            "patch_name": patch_data["name"],
            "patch_type": patch_data["type"],
            "patch_rationale": patch_data["rationale"]
        })

        print(f"  ✅ {patch_data['name']} ({patch_data['type']})")
        print(f"     Rationale: {patch_data['rationale']}")

    # Write modified content
    if output_file is None:
        output_file = target_file

    output_file = Path(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print(f"\n✅ Applied {len(applied_patches)} patch(es) to {output_file}")

    return {
        "target_file": str(target_file),
        "output_file": str(output_file),
        "patches_applied": len(applied_patches),
        "patches": applied_patches,
        "backup_created": create_backup
    }
