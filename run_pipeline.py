import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_classical_baselines.ipynb",
    "notebooks/03_modern_ml.ipynb",
    "notebooks/04_foundation_model.ipynb",
]

root = Path(__file__).parent

for nb in NOTEBOOKS:
    nb_path = root / nb
    print(f"\n{'='*60}")
    print(f"Running: {nb}")
    print('='*60)
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=3600",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAILED: {nb}")
        print(result.stderr)
        sys.exit(1)
    print(f"Done: {nb}")

print("\nPipeline complete.")
