"""
Centralized configuration with environment variable overrides.

Environment variables:
- INTERNVIDEO_MODEL_PATH: model path or HuggingFace model ID
- INTERNVIDEO_TASK_JSON: task metadata path (tasks.jsonl)
- INTERNVIDEO_OUTPUT_DIR: output directory
- INTERNVIDEO_VIDEO_CACHE_DIR: video cache directory (parquet to mp4)
"""

import os
from pathlib import Path


def get_model_path() -> str:
    """Return model path, defaulting to the HuggingFace model ID."""
    return os.environ.get(
        "INTERNVIDEO_MODEL_PATH",
        "OpenGVLab/InternVideo2_5_Chat_8B"
    )


def get_task_json_path(data_root: str = None) -> str:
    """
    Return tasks metadata file path.
    If data_root is None, read from INTERNVIDEO_TASK_JSON.
    Otherwise infer {data_root}/meta/tasks.jsonl.
    """
    if data_root:
        return str(Path(data_root) / "meta" / "tasks.jsonl")
    return os.environ.get(
        "INTERNVIDEO_TASK_JSON",
        ""
    )


def get_output_dir() -> str:
    """Return output directory."""
    return os.environ.get(
        "INTERNVIDEO_OUTPUT_DIR",
        "./output"
    )


def get_video_cache_dir() -> str:
    """Return video cache directory."""
    default = str(Path(get_output_dir()) / "video_cache")
    return os.environ.get(
        "INTERNVIDEO_VIDEO_CACHE_DIR",
        default
    )


# Optional mapping from task instruction to sub-steps for analyze_action_segment3.
DEFAULT_TASK_MAPPING = {
    "open the top drawer and put the bowl inside": [
        "open the top drawer",
        "pick up the bowl",
        "place the bowl in the drawer",
    ],
    "turn on the stove": ["turn on the stove"],
    "open the middle drawer of the cabinet": [
        "open the middle drawer of the cabinet",
    ],
    "push the plate to the front of the stove": [
        "push the plate to the front of the stove",
    ],
    "put the wine bottle on top of the cabinet": [
        "pick up the wine bottle",
        "place the wine bottle on the cabinet",
    ],
}


def derive_data_root_from_parquet(parquet_path: str) -> str:
    """
    Infer dataset root from a parquet path.
    Example:
    .../franka_long_resize/data/chunk-000/episode_xxx.parquet
    -> .../franka_long_resize
    """
    p = Path(parquet_path).resolve()
    # Parent of data/chunk-xxx is treated as data root.
    if "data" in p.parts:
        idx = p.parts.index("data")
        return str(Path(*p.parts[:idx]))
    return str(p.parent.parent)
