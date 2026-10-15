"""Rebuild training JSON from a batch run without loading a model."""

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_annotator import output_lock
from training_export import export_results, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional additional copy of train_annotations.json.")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not (directory / "run.json").is_file():
        parser.error("run-dir must contain run.json from the batch CLI")
    if args.output and args.output.resolve().is_relative_to(directory):
        if args.output.resolve() != directory / "train_annotations.json":
            parser.error("Within run-dir, output must be train_annotations.json")
    with output_lock(directory):
        manifest = json.loads((directory / "run.json").read_text())
        summary = export_results(directory, manifest)
        if args.output:
            training = json.loads((directory / "train_annotations.json").read_text())
            write_json(args.output, training)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
