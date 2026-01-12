"""
Compare different trajectory rollouts using LLM-based analysis.

This script provides a command-line interface for batch comparison of trajectories,
patch generation, and patch application workflows.
"""

import json
import argparse
from pathlib import Path

from .trajectory_comparison import compare_examples_batch, aggregate_comparison_analyses
from .patch_generation import (
    generate_patches_batch,
    apply_patches,
    generate_universal_patches,
    edit_patch_with_feedback
)


def main():
    """Command-line interface for trajectory comparison."""
    parser = argparse.ArgumentParser(
        description="Compare agent trajectory rollouts using LLM analysis"
    )

    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Comparison mode")

    # Unified analyze mode (batch comparison + aggregation)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze trajectories: compare examples and aggregate patterns")

    analyze_parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task identifier (e.g., 'webgen', 'webtest')"
    )

    analyze_parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name used for generation"
    )

    analyze_parser.add_argument(
        "--prompt_name",
        type=str,
        default="default",
        help="Prompt template name (default: 'default')"
    )

    analyze_parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to analyze (default: 10)"
    )

    analyze_parser.add_argument(
        "--comparison_model",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use for comparison and aggregation (default: 'gemini-2.5-flash')"
    )

    analyze_parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save analysis results (default: results/{task_id}/{model_name}_{prompt_name}/comparisons)"
    )

    analyze_parser.add_argument(
        "--agentic",
        action="store_true",
        default=True,
        help="Whether the runs used agentic execution (default: True)"
    )

    analyze_parser.add_argument(
        "--non-agentic",
        action="store_true",
        help="Set if the runs used non-agentic execution"
    )

    analyze_parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers for batch processing (default: 8)"
    )

    analyze_parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregation step (only perform batch comparison)"
    )

    # Patch generation mode (unified)
    patch_parser = subparsers.add_parser("generate-patches", help="Generate patches from comparison results")

    patch_parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task identifier (e.g., 'webgen', 'webtest')"
    )

    patch_parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name used for generation"
    )

    patch_parser.add_argument(
        "--prompt_name",
        type=str,
        default="default",
        help="Prompt template name (default: 'default')"
    )

    patch_parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to process (default: 10)"
    )

    patch_parser.add_argument(
        "--patch_model",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use for patch generation (default: 'gemini-2.5-flash')"
    )

    patch_parser.add_argument(
        "--comparison_dir",
        type=str,
        help="Directory containing comparison results (default: results/{task_id}/comparisons/{model_name}_{prompt_name})"
    )

    patch_parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save patch files (default: results/{task_id}/patches/{model_name}_{prompt_name})"
    )

    patch_parser.add_argument(
        "--universal",
        action="store_true",
        help="Generate universal patches from aggregated analysis instead of per-example patches"
    )

    patch_parser.add_argument(
        "--aggregated_file",
        type=str,
        help="Path to aggregated analysis JSON file (only for --universal mode, default: results/{task_id}/{model_name}_{prompt_name}/comparisons/aggregated.json)"
    )

    # Apply patches mode
    apply_parser = subparsers.add_parser("apply-patches", help="Apply patches to create improved prompt")

    apply_parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task identifier (e.g., 'webgen', 'webtest')"
    )

    apply_parser.add_argument(
        "--model_name",
        type=str,
        help="Model name used for generation (required unless using --patch_dir)"
    )

    apply_parser.add_argument(
        "--prompt_name",
        type=str,
        default="default",
        help="Prompt template name (default: 'default')"
    )

    apply_parser.add_argument(
        "--patch_dir",
        type=str,
        help="Directory containing patch files (default: results/{task_id}/{model_name}_{prompt_name}/patches)"
    )

    apply_parser.add_argument(
        "--patches",
        type=str,
        help="Comma-separated list of specific patch files to apply (e.g., '01_clarify.patch,02_errors.patch'). If not specified, all patches in patch_dir are applied."
    )

    apply_parser.add_argument(
        "--output",
        type=str,
        help="Path to save improved prompt (default: results/{task_id}/{model_name}_{prompt_name}/improved_prompts/{prompt_name}_improved.md)"
    )

    # Edit patch mode (NEW)
    edit_parser = subparsers.add_parser("edit-patch", help="Edit a patch based on user feedback")

    edit_parser.add_argument(
        "--patch",
        type=str,
        required=True,
        help="Path to the patch file to edit"
    )

    edit_parser.add_argument(
        "--feedback",
        type=str,
        required=True,
        help="User feedback describing how to improve the patch"
    )

    edit_parser.add_argument(
        "--task_id",
        type=str,
        help="Task identifier (e.g., 'webgen', 'webtest') - required if using --prompt"
    )

    edit_parser.add_argument(
        "--prompt_name",
        type=str,
        help="Prompt name (e.g., 'default') - will be resolved to data/{task_id}/prompts/{prompt}.md"
    )

    edit_parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use for patch refinement (default: 'gemini-2.5-flash')"
    )

    edit_parser.add_argument(
        "--output",
        type=str,
        help="Path to save refined patch (default: overwrites original)"
    )

    args = parser.parse_args()

    # Handle no mode specified
    if not args.mode:
        parser.print_help()
        return

    # Unified analyze mode (batch comparison + aggregation)
    if args.mode == "analyze":

        agentic = not args.non_agentic

        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f"results/{args.task_id}/{args.model_name}_{args.prompt_name}/comparisons")

        # Step 1: Batch comparison
        print(f"\n{'='*80}")
        print("STEP 1: BATCH COMPARISON")
        print(f"{'='*80}")

        results = compare_examples_batch(
            task_id=args.task_id,
            model_name=args.model_name,
            prompt_name=args.prompt_name,
            num_examples=args.num_examples,
            comparison_model=args.comparison_model,
            output_dir=output_dir,
            agentic=agentic,
            max_workers=args.max_workers
        )

        # Print comparison summary
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        print(f"\nComparison Summary:")
        print(f"  Total examples: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed/Skipped: {len(failed)}")

        # Step 2: Aggregation (unless skipped)
        if not args.skip_aggregate and successful:
            print(f"\n{'='*80}")
            print("STEP 2: AGGREGATION")
            print(f"{'='*80}")

            output_path = output_dir / "aggregated.md"
            output_json = output_dir / "aggregated.json"

            aggregation_result = aggregate_comparison_analyses(
                comparison_dir=output_dir,
                model_name=args.comparison_model,
                output_path=output_path
            )

            # Save JSON version
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(aggregation_result, f, indent=2, ensure_ascii=False)

            print(f"\nAggregation Summary:")
            print(f"  Examples analyzed: {aggregation_result['num_examples_analyzed']}")

            # Handle patterns as either list or string
            patterns = aggregation_result['common_patterns']
            if isinstance(patterns, list):
                print(f"  Common patterns: {len(patterns)}")
            else:
                print(f"  Common patterns: Found (see output file)")

            # Handle improvements as either list or string
            improvements = aggregation_result['recommended_improvements']
            if isinstance(improvements, list):
                print(f"  Recommended improvements: {len(improvements)}")
            else:
                print(f"  Recommended improvements: Found (see output file)")

            print(f"  Output (markdown): {output_path}")
            print(f"  Output (JSON): {output_json}")

        # Final summary
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results directory: {output_dir}")
        if failed:
            print(f"⚠️  Failed/Skipped examples: {[r['example_id'] for r in failed]}")
        print(f"{'='*80}")

    # Patch generation mode
    elif args.mode == "generate-patches":
        if args.universal:
            # Universal patch generation

            # Set default paths
            if args.aggregated_file:
                aggregated_file = Path(args.aggregated_file)
            else:
                aggregated_file = Path(f"results/{args.task_id}/{args.model_name}_{args.prompt_name}/comparisons/aggregated.json")

            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = Path(f"results/{args.task_id}/{args.model_name}_{args.prompt_name}/patches")

            if not aggregated_file.exists():
                print(f"❌ Error: Aggregated analysis file not found: {aggregated_file}")
                print(f"   Run 'analyze' command first to generate aggregated analysis!")
                return

            # Load aggregated analysis
            with open(aggregated_file, 'r', encoding='utf-8') as f:
                aggregated_analysis = json.load(f)

            result = generate_universal_patches(
                aggregated_analysis=aggregated_analysis,
                model_name=args.patch_model,
                output_dir=output_dir
            )

            print(f"\n{'='*80}")
            print("UNIVERSAL PATCH GENERATION SUMMARY")
            print(f"{'='*80}")
            print(f"Examples analyzed: {result['num_examples_analyzed']}")
            print(f"Common patterns addressed: {result['common_patterns_addressed']}")
            print(f"Universal patches generated: {result['num_patches']}")
            print(f"Output directory: {output_dir}")
            print(f"{'='*80}")
        else:
            # Per-example patch generation
            results = generate_patches_batch(
                task_id=args.task_id,
                model_name=args.model_name,
                prompt_name=args.prompt_name,
                num_examples=args.num_examples,
                patch_model=args.patch_model,
                comparison_dir=Path(args.comparison_dir) if args.comparison_dir else None,
                output_base_dir=Path(args.output_dir) if args.output_dir else None
            )

            # Print summary
            print(f"\n{'='*80}")
            print("PATCH GENERATION SUMMARY")
            print(f"{'='*80}")
            successful = [r for r in results if "error" not in r]
            failed = [r for r in results if "error" in r]
            total_patches = sum(r.get("num_patches", 0) for r in successful)
            print(f"Total examples processed: {len(results)}")
            print(f"Successful generations: {len(successful)}")
            print(f"Failed generations: {len(failed)}")
            print(f"Total patches generated: {total_patches}")
            if failed:
                print(f"\nFailed examples: {[r['example_id'] for r in failed]}")
            print(f"{'='*80}")

    # Edit patch mode
    elif args.mode == "edit-patch":
        # Construct base prompt path if task_id and prompt are provided
        base_prompt = None
        if args.prompt_name:
            if not args.task_id:
                print("❌ Error: --task_id is required when using --prompt_name")
                return
            base_prompt = Path(f"data/{args.task_id}/prompts/{args.prompt_name}.md")
            if not base_prompt.exists():
                print(f"⚠️  Warning: Base prompt file not found: {base_prompt}")
                print("   Continuing without base prompt context...")
                base_prompt = None

        result = edit_patch_with_feedback(
            patch_path=Path(args.patch),
            user_feedback=args.feedback,
            model_name=args.model_name,
            output_path=Path(args.output) if args.output else None,
            base_prompt_path=base_prompt
        )

        print(f"\n{'='*80}")
        print("PATCH EDIT SUMMARY")
        print(f"{'='*80}")
        print(f"Original patch: {result['original_patch']}")
        print(f"Refined patch: {result['refined_patch']}")
        print(f"Model used: {result['model']}")
        if base_prompt:
            print(f"Base prompt: {base_prompt}")
        print(f"\nUser feedback:")
        print(f"  {result['user_feedback']}")
        print(f"\nOriginal rationale:")
        print(f"  {result['original_rationale']}")
        print(f"\nUpdated rationale:")
        print(f"  {result['updated_rationale']}")
        print(f"{'='*80}")

    # Apply patches mode
    elif args.mode == "apply-patches":
        # Set base prompt path
        base_prompt = Path(f"data/{args.task_id}/prompts/{args.prompt_name}.md")

        # Determine patch directory
        if args.patch_dir:
            patch_dir = Path(args.patch_dir)
        elif args.model_name:
            patch_dir = Path(f"results/{args.task_id}/{args.model_name}_{args.prompt_name}/patches")
        else:
            print(f"❌ Error: Either --patch_dir or --model_name must be specified")
            return

        # Auto-infer version number by checking existing files
        prompts_dir = Path(f"data/{args.task_id}/prompts")
        version = "v1"

        if prompts_dir.exists():
            existing_versions = []
            for file in prompts_dir.glob(f"{args.prompt_name}_v*.md"):
                # Extract version number (e.g., "default_v2.md" -> 2)
                try:
                    version_str = file.stem.split('_v')[-1]
                    existing_versions.append(int(version_str))
                except (ValueError, IndexError):
                    pass

            if existing_versions:
                next_version = max(existing_versions) + 1
                version = f"v{next_version}"

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Save versioned prompt to data directory
            output_path = Path(f"data/{args.task_id}/prompts/{args.prompt_name}_{version}.md")

        # Apply patches using the unified function
        patch_files = None
        if args.patches:
            patch_files = [p.strip() for p in args.patches.split(',')]

        result = apply_patches(
            base_prompt_path=base_prompt,
            patch_dir=patch_dir,
            output_path=output_path,
            patch_files=patch_files,
            version=version
        )

        print(f"\n{'='*80}")
        print("PATCH APPLICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Version: {version}")
        print(f"Base prompt: {result['base_prompt']}")
        print(f"Total patches applied: {result['patches_applied']}")
        print(f"Improved prompt saved to: {result['output_file']}")
        if result['patches']:
            print(f"\nApplied patches:")
            for patch in result['patches']:
                if isinstance(patch, dict):
                    print(f"  - {patch.get('patch_file', 'N/A')}: {patch.get('patch_name', 'N/A')}")
                else:
                    print(f"  - {patch}")
        if result['versions_file']:
            print(f"Version tracking: {result['versions_file']}")
        print(f"\n✅ Prompt saved to data directory and can be used with --prompt {args.prompt_name}_{version}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
