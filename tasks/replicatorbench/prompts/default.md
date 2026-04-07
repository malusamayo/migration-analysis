# Scientific Replication Task

You are reproducing a focal claim from a research paper. Work through four stages in order, writing one JSON file per stage.

## Available Inputs

- `initial_details.txt` — contains the focal claim statement and hypotheses; use these verbatim in Stage 1
- `original_paper.pdf` — full paper text; primary source for all fields except claim/hypotheses
- `replication_data/` — data files and (depending on mode) native analysis code

## Required Outputs

Write all four files as valid JSON only (no markdown fences).

---

## Stage 1: Extraction → `post_registration.json`

Extract structured information about the original study. For `claim.statement` and `claim.hypotheses`, copy directly from `initial_details.txt`. For all other fields, extract from `original_paper.pdf`. Use `"not stated"` if a field is absent from the designated source.

```json
{
  "original_study": {
    "claim": {
      "hypotheses": "Testable hypothesis from initial_details.txt",
      "hypotheses_location": "Where the hypothesis appears in the paper",
      "statement": "Exact claim statement from initial_details.txt",
      "statement_location": "Where the claim is stated in the paper",
      "study_type": "Experimental | Observational | Meta-Analysis"
    },
    "data": {
      "source": "Data source name (e.g., survey, database)",
      "wave_or_subset": "Specific waves or subsets if applicable",
      "sample_size": "Sample size",
      "unit_of_analysis": "Unit of analysis (e.g., individual, household)",
      "access_details": "Access restrictions or request process",
      "notes": "Caveats: encoding issues, nested structure, missing metadata, etc."
    },
    "method": {
      "description": "Narrative summary of how the study was conducted",
      "steps": "Ordered procedural steps to reproduce the study",
      "models": "Statistical model or approach (e.g., OLS regression)",
      "outcome_variable": "Dependent variable",
      "independent_variables": "Primary predictors",
      "control_variables": "Variables controlled for",
      "tools_software": "Software or packages mentioned"
    },
    "results": {
      "summary": "Narrative summary of main findings",
      "numerical_results": [
        {
          "outcome_name": "Label for this result",
          "value": 0.0,
          "unit": "unit of measurement or 'not stated'",
          "effect_size": "Cohen's d / odds ratio / etc., or 'not stated'",
          "confidence_interval": { "lower": 0.0, "upper": 0.0, "level": 0.95 },
          "p_value": "e.g. '<0.001' or 'not stated'",
          "statistical_significance": true,
          "direction": "positive | negative | null"
        }
      ]
    },
    "metadata": {
      "original_paper_id": "DOI or identifier",
      "original_paper_title": "Full title",
      "original_paper_code": "Code repository link or 'not stated'",
      "original_paper_data": "Data repository link or 'not stated'"
    }
  }
}
```

---

## Stage 2: Design → `replication_info.json`

Write a preregistration-style replication plan based on the extracted information and the files in `replication_data/`. Inspect the data files and any available code before filling this out.

```json
{
  "replication_study": {
    "hypothesis": "Focal hypothesis phrased as a testable statement",
    "study_type": "Experimental | Observational | Meta-Analysis | Other",
    "data_plan": {
      "dataset_identifier": "Name and version of the replication dataset",
      "source_type": "Data source type",
      "wave_or_subset": "Waves or subsets to use",
      "sample_size": "Expected sample size",
      "unit_of_analysis": "Unit of analysis",
      "access_details": "Access details",
      "qualification": {
        "explanation": "Why this dataset is appropriate for replication",
        "similarity_to_original": "How it matches the original data",
        "deviation_from_original": "How it differs from the original data"
      },
      "notes": "Caveats about the data"
    },
    "planned_method": {
      "steps": "Ordered procedural steps for the replication",
      "models": "Statistical model to use",
      "outcome_variable": "Dependent variable",
      "independent_variables": "Primary predictors",
      "control_variables": "Variables to control for",
      "tools_software": "Software and packages to use",
      "planned_estimation_and_test": {
        "estimation": "Target of estimation (e.g., coefficient)",
        "test": "Statistical test (e.g., t-test)"
      },
      "missing_data_handling": "How to handle missing values",
      "multiple_testing_policy": "Correction method if applicable",
      "inference_criteria": "Rules for judging support (significance threshold, direction)"
    },
    "codebase": {
      "files": {
        "<filename>": "Description of what this file does and how to run it"
      },
      "notes": "Overall notes on code, dependencies, runtime"
    },
    "docker_specs": {
      "base_image": "e.g. python:3.11",
      "packages": {
        "python": ["package>=version"],
        "other": ["system packages"]
      },
      "hardware": {
        "gpu_support": false,
        "min_gpu_memory_gb": 0,
        "min_ram_gb": 8
      },
      "volumes": ["./replication_data:/app/data"]
    },
    "analysis": {
      "instructions": "Steps or code logic to run the analysis",
      "comparison_metrics": "How to compare original vs replication results"
    }
  }
}
```

---

## Stage 3: Execution → `execution_results.json`

Execute the replication using the files in `replication_data/` following the plan in `replication_info.json`. If native code is available, prefer running it with minimal fixes. If code is withheld or absent, reconstruct the analysis from the paper.

```json
{
  "execution_summary": "Narrative overview: steps taken, overall success, alignment with preregistration",
  "code_executed": [
    {
      "command": "command that was run",
      "status": "Success | Partial Success | Failure",
      "logs": "Key log output, warnings, errors",
      "environment": "Runtime environment description"
    }
  ],
  "results": {
    "hypothesis_tested": "Restatement of the focal hypothesis",
    "findings_summary": [
      {
        "outcome_name": "Name of statistic",
        "value": 0.0,
        "standard_error": 0.0,
        "confidence_interval": "[lower, upper]",
        "p_value": "e.g. 0.007",
        "statistical_significance": "p < 0.01",
        "direction": "positive | negative | null",
        "effect_size": "if applicable"
      }
    ],
    "tables": [
      { "table_id": "e.g. Table 1", "table_description": "brief description", "table_file": "path if saved" }
    ],
    "figures": [
      { "figure_id": "e.g. Figure 1", "figure_description": "brief description", "figure_file": "path if saved" }
    ]
  }
}
```

---

## Stage 4: Interpretation → `interpret_results.json`

Compare your execution results to the original paper's findings and assess whether the replication supports the focal claim.

```json
{
  "interpretation_summary": "Narrative overview: key comparisons, fidelity, overall outcome",
  "execute_status": "Success | Partial Success | Failure",
  "fidelity_assessment": {
    "method_alignment": "How well executed methods matched the preregistration",
    "deviations": [
      { "issue_description": "Description of deviation", "impact": "Impact on results" }
    ]
  },
  "results_comparison": {
    "hypothesis_tested": "Restatement of the focal hypothesis",
    "original_results": "Summary of original paper findings with numerical values",
    "replication_results": "Summary of replication findings mirroring original structure",
    "overall_answer": "Do replication results satisfy preregistered criteria?"
  },
  "replication_report": "Short summary: did replication succeed, key numbers, direction",
  "failure_handling": [
    {
      "failure_type": "Data-Related Failures | Code/Execution Failures | Method/Alignment Failures | Results/Output Failures",
      "suggestions": "Actionable recommendations"
    }
  ],
  "notes": "Additional caveats or suggestions"
}
```

---

## Rules

- All output files must be valid JSON only — no markdown, no comments.
- Use `"not stated"` for fields absent from the designated source.
- Keep all outputs focused on the single focal claim from `initial_details.txt`.
- `direction` must be `"positive"`, `"negative"`, or `"null"`.
- `statistical_significance` in numerical results must be a boolean.
