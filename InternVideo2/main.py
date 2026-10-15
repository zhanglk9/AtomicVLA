"""Batch annotation CLI. Heavy dependencies are loaded after argument parsing."""

import argparse
import math
import os
from pathlib import Path
import sys


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch robot trajectory annotation and compact training JSON export."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path, help="Dataset root containing data/ and meta/.")
    source.add_argument("--data-dir", type=Path, help="Recursively scan this parquet directory.")
    source.add_argument("--parquet", type=Path, nargs="+", help="One or more episode parquet files.")
    parser.add_argument("--task-json", type=Path, help="Override meta/tasks.jsonl.")
    parser.add_argument("--output-dir", type=Path, help="Run directory; default: output/batch.")
    parser.add_argument("--output", type=Path, help="Optional additional copy of legacy report.json.")
    parser.add_argument("--model-path", help="Trusted local model path or Hugging Face model ID.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index within CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Only process these task_index values.")
    parser.add_argument("--episodes-per-task", type=positive_integer, help="First N episodes per task; default: all.")
    parser.add_argument("--limit", type=positive_integer, help="Limit total episodes after task filtering.")
    parser.add_argument("--resume", action="store_true", help="Skip completed episodes and retry failures in the same run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/select inputs without loading the model or writing outputs.")
    parser.add_argument("--coarse-only", action="store_true", help="Kinematics only; no model and no training labels.")
    parser.add_argument("--offline", action="store_true", help="Disable Hugging Face network access; requires local/cached weights.")
    parser.add_argument("--keep-videos", action="store_true", help="Keep generated episode MP4 files; otherwise remove after processing.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; each episode uses seed + episode_index.")
    parser.add_argument("--export-every", type=positive_integer, default=10, help="Refresh aggregate JSON every N processed episodes.")
    parser.add_argument("--grip-thresh", type=positive_float, default=0.01)
    parser.add_argument("--translation-thresh", type=positive_float, default=0.03, help="Meters.")
    parser.add_argument("--rotation-thresh", type=positive_float, default=0.05, help="Radians.")
    parser.add_argument("--motion-window", type=positive_integer, default=5, help="At least 2 consecutive frames.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.device < 0 or args.seed < 0 or args.motion_window < 2:
        parser.error("device and seed must be nonnegative; motion-window must be at least 2")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output_dir = args.output_dir or Path(os.environ.get("INTERNVIDEO_OUTPUT_DIR", "output/batch"))
    from config import get_model_path
    args.model_path = args.model_path or get_model_path()
    if Path(args.model_path).exists():
        args.model_path = str(Path(args.model_path).resolve())
    try:
        from batch_annotator import run_batch
        return run_batch(args)
    except KeyboardInterrupt:
        print("Interrupted. Use the same command with --resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
