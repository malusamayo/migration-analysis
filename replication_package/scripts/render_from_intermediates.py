import subprocess
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"


def run_script(script_name: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True)


def main() -> None:
    for script_name in [
        "normalize_manual_labels.py",
        "build_triplet_coverage.py",
        "render_table.py",
        "plot_cost_performance_quadrants.py",
        "plot_intelligence_performance.py",
        "plot_trajectory_diversity_score.py",
    ]:
        run_script(script_name)


if __name__ == "__main__":
    main()
