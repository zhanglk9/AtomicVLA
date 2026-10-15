"""
Utility module for logging, configuration, and file operations.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    """Simple logger wrapper."""

    def __init__(self, name: str = "internvideo", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)


class ConfigManager:
    """Configuration manager."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config = self._load_default_config()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        return {
            "video_processor": {
                "model_path": "OpenGVLab/InternVideo2_5_Chat_8B",
                "input_size": 448,
                "num_segments": 16,
            },
            "video_segmenter": {"use_ffmpeg": True, "output_format": "mp4"},
        }

    def load_config(self, config_path: str):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            self._config = self._merge_configs(self._config, file_config)
        except Exception as e:
            Logger().warning(f"Failed to load config file {config_path}: {e}")

    def _merge_configs(self, default: Dict, override: Dict) -> Dict:
        result = default.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save_config(self, config_path: str):
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            Logger().error(f"Failed to save config file {config_path}: {e}")


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes = total_seconds // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def sanitize_filename(filename: str, max_length: int = 80) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename.strip()


def ensure_dir(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_unique_output_path(base_path: str, prefix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(base_path).parent
    base_name = Path(base_path).stem
    extension = Path(base_path).suffix
    unique_name = f"{base_name}_{prefix}_{timestamp}{extension}" if prefix else f"{base_name}_{timestamp}{extension}"
    return str(base_dir / unique_name)


def load_json(file_path: str) -> Dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        Logger().error(f"Failed to load JSON file {file_path}: {e}")
        return {}


def save_json(data: Dict, file_path: str):
    try:
        ensure_dir(str(Path(file_path).parent))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        Logger().error(f"Failed to save JSON file {file_path}: {e}")


def get_video_info(video_path: str) -> Dict:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info = {
            "path": video_path,
            "name": Path(video_path).name,
            "fps": fps,
            "frame_count": frame_count,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": frame_count / fps if fps > 0 else 0,
        }
        cap.release()
        return info
    except Exception as e:
        Logger().error(f"Failed to read video info for {video_path}: {e}")
        return {}
