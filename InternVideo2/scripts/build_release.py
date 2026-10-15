"""Build a source-only ZIP using an explicit allowlist."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


RELEASE_FILES = (
    "README.md", "requirements.txt", ".gitignore", "main.py",
    "batch_annotator.py", "training_export.py", "action_segmenter.py",
    "config.py", "task_decomposer.py", "video_processor.py",
    "video_segmenter.py", "utils.py", "scripts/export_training_json.py",
    "scripts/build_release.py", "examples/train_annotations.example.json",
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist" / "InternVideo2-batch-release.zip")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error(f"Refusing to overwrite existing archive: {target}")
    for name in RELEASE_FILES:
        if not (root / name).is_file():
            parser.error(f"Missing release file: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name in RELEASE_FILES:
            archive.write(root / name, arcname=f"InternVideo2/{name}")
    print(target)


if __name__ == "__main__":
    main()
