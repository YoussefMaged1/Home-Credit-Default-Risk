"""
Home Credit Default Risk — Pipeline Entrypoint
================================================
Usage:
    uv run main.py --mode train
    uv run main.py --mode evaluate
    uv run main.py --mode inference
    uv run main.py --mode serve
    uv run main.py --mode all          # download → process → train → evaluate → inference
    uv run main.py --mode pipeline     # dvc repro (full DVC-managed pipeline)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ── Constants ─────────────────────────────────────────────────────────────────
SRC_TRAINING = Path("src") / "training"
SRC_SERVING = Path("src") / "serving"

STAGE_SCRIPTS = {
    "download": SRC_TRAINING / "download_data.py",
    "process": SRC_TRAINING / "process_data.py",
    "train": SRC_TRAINING / "train.py",
    "evaluate": SRC_TRAINING / "evaluate.py",
    "inference": SRC_SERVING / "inference.py",
    "serve": SRC_SERVING / "serve.py",
}

FULL_PIPELINE = ["download", "process", "train", "evaluate", "inference"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_script(stage: str) -> None:
    """Run a single pipeline stage via uv."""
    script = STAGE_SCRIPTS.get(stage)
    if script is None:
        logger.error(f"Unknown stage: '{stage}'")
        sys.exit(1)

    if not script.exists():
        logger.error(f"Script not found: {script}")
        sys.exit(1)

    logger.info(f"▶ Starting stage: {stage.upper()}")
    result = subprocess.run(
        ["uv", "run", str(script)],
        check=False,
    )

    if result.returncode != 0:
        logger.error(f"✗ Stage '{stage}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    logger.info(f"✓ Stage '{stage}' completed successfully")


def run_dvc_pipeline() -> None:
    """Run the full DVC-managed pipeline (dvc repro)."""
    logger.info("▶ Running full DVC pipeline (dvc repro)...")
    result = subprocess.run(["dvc", "repro"], check=False)

    if result.returncode != 0:
        logger.error(f"✗ DVC pipeline failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    logger.info("✓ DVC pipeline completed successfully")


def run_full_pipeline() -> None:
    """Run all stages sequentially without DVC."""
    logger.info("▶ Running full pipeline: " + " → ".join(FULL_PIPELINE))
    for stage in FULL_PIPELINE:
        run_script(stage)
    logger.info("✓ Full pipeline completed successfully")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Home Credit Default Risk — MLOps Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[*STAGE_SCRIPTS.keys(), "all", "pipeline"],
        help=("Pipeline mode to run. 'all' runs every stage sequentially. 'pipeline' uses DVC (dvc repro)."),
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    args = parse_args()

    logger.info(f"Home Credit Default Risk Pipeline | mode={args.mode}")

    if args.mode == "pipeline":
        run_dvc_pipeline()
    elif args.mode == "all":
        run_full_pipeline()
    else:
        run_script(args.mode)


if __name__ == "__main__":
    main()
