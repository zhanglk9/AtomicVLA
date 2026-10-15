"""
Video segmenter that slices clips by action type from analysis reports.

Supports both FFmpeg and OpenCV backends.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2


class ActionVideoSegmenter:
    """Video segmenter with action-type grouped outputs."""

    def __init__(self, use_ffmpeg: bool = True):
        self.use_ffmpeg = use_ffmpeg
        self._check_dependencies()

    def _check_dependencies(self):
        if self.use_ffmpeg:
            try:
                subprocess.run(
                    ["ffmpeg", "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.use_ffmpeg = False

    def segment_video_from_analysis(
        self, video_path: str, analysis_report: Dict, output_dir: str
    ) -> Dict:
        task_breakdown = analysis_report.get(
            "segments", analysis_report.get("task_breakdown", [])
        )
        motion = analysis_report.get("motion_analysis", {})
        fps = motion.get("fps")
        if fps is None or fps <= 0:
            try:
                from utils import get_video_info

                info = get_video_info(video_path)
                fps = info.get("fps", 30) or 30
            except Exception:
                fps = 30

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        main_project_dir = os.path.join(
            output_dir, f"{video_name}_action_segments_{timestamp}"
        )
        os.makedirs(main_project_dir, exist_ok=True)

        action_groups = self._group_by_action_type(task_breakdown, fps)
        video_ext = Path(video_path).suffix or ".mp4"

        segmentation_results = {
            "original_video": video_path,
            "main_directory": main_project_dir,
            "fps": fps,
            "action_groups": {},
            "total_segments": len(task_breakdown),
            "successful_segments": 0,
            "failed_segments": 0,
        }

        for action_type, segments in action_groups.items():
            action_dir = os.path.join(main_project_dir, f"{action_type}_actions")
            os.makedirs(action_dir, exist_ok=True)
            action_results = {
                "action_type": action_type,
                "directory": action_dir,
                "segments": [],
                "successful_count": 0,
                "failed_count": 0,
            }

            for segment_info in segments:
                success = self._process_single_segment(
                    video_path, segment_info, action_dir, video_ext, action_results
                )
                if success:
                    segmentation_results["successful_segments"] += 1
                    action_results["successful_count"] += 1
                else:
                    segmentation_results["failed_segments"] += 1
                    action_results["failed_count"] += 1

            segmentation_results["action_groups"][action_type] = action_results

        self._save_simple_readme(segmentation_results, main_project_dir)
        return segmentation_results

    def _group_by_action_type(
        self, task_breakdown: List[dict], fps: float
    ) -> Dict[str, List[dict]]:
        action_groups = {}
        for i, action_info in enumerate(task_breakdown):
            action_verb = action_info.get("primary_action_verb", "manipulate")
            if not action_verb or action_verb == "manipulate":
                action_verb = self._extract_action_from_name(
                    action_info.get("action_name", "")
                )
            if action_verb not in action_groups:
                action_groups[action_verb] = []

            start_frame = action_info.get("start_frame")
            end_frame = action_info.get("end_frame")
            if start_frame is not None and end_frame is not None:
                start_time = start_frame / fps if fps > 0 else 0
                end_time = end_frame / fps if fps > 0 else 0
            else:
                start_time = action_info.get("start_time", 0)
                end_time = action_info.get("end_time", 0)
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)

            segment_info = {
                "step_number": i + 1,
                "action_name": action_info.get("action_name", f"Action_{i + 1}"),
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "timestamp_formatted": action_info.get("timestamp_formatted", ""),
                "chain_of_thought": action_info.get("chain_of_thought", ""),
            }
            action_groups[action_verb].append(segment_info)
        return action_groups

    def _extract_action_from_name(self, action_name: str) -> str:
        action_verbs = [
            "pick", "place", "move", "scoop", "grasp", "transfer", "pour",
            "push", "pull", "rotate", "adjust", "insert", "close", "open",
            "make", "lift", "drop", "hold", "release", "slide", "twist",
            "pinch", "squeeze", "press", "tap", "touch", "brush", "wipe",
            "spread", "fold", "unfold", "roll", "unroll", "wrap", "unwrap",
            "assemble", "disassemble", "attach", "detach", "connect", "disconnect",
            "mount", "unmount", "install", "remove", "extract",
            "approach", "navigate", "position", "align", "orient", "center",
            "fill", "empty", "load", "unload", "pack", "unpack", "seal", "unseal",
            "mix", "stir", "blend", "knead", "whisk", "flip", "season", "serve",
        ]
        action_name_lower = action_name.lower()
        for verb in action_verbs:
            if verb in action_name_lower:
                return verb
        return "manipulate"

    def _process_single_segment(
        self,
        video_path: str,
        segment_info: dict,
        action_dir: str,
        video_ext: str,
        action_results: dict,
    ) -> bool:
        step_num = segment_info["step_number"]
        action_name = segment_info["action_name"]
        start_time = segment_info["start_time"]
        end_time = segment_info["end_time"]
        clean_filename = self._create_filename_from_description(action_name, step_num)
        segment_filename = f"{clean_filename}{video_ext}"
        segment_path = os.path.join(action_dir, segment_filename)

        if self.use_ffmpeg:
            success = self._segment_video_ffmpeg(
                video_path, start_time, end_time, segment_path
            )
        else:
            success = self._segment_video_opencv(
                video_path, start_time, end_time, segment_path
            )

        action_results["segments"].append({
            "step_number": step_num,
            "original_action": action_name,
            "filename": segment_filename,
            "file_path": segment_path,
            "start_frame": segment_info["start_frame"],
            "end_frame": segment_info["end_frame"],
            "timestamp": segment_info["timestamp_formatted"],
            "success": success,
        })
        return success

    def _create_filename_from_description(
        self, action_description: str, step_num: int
    ) -> str:
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by",
        }
        words = action_description.strip().split()
        clean_words = [
            "".join(c for c in w if c.isalnum()).lower()
            for w in words
            if "".join(c for c in w if c.isalnum()).lower() not in stop_words
            and len("".join(c for c in w if c.isalnum())) > 1
        ]
        selected = clean_words[:8]
        if not selected:
            return f"step_{step_num:02d}_action"
        filename = f"step_{step_num:02d}_{'_'.join(selected)}"
        return filename[:80] if len(filename) > 80 else filename

    def _save_simple_readme(self, results: dict, main_dir: str):
        readme_file = os.path.join(main_dir, "README.md")
        content = [
            "# Robotic Action Video Segmentation Result",
            "",
            f"**Original Video**: {os.path.basename(results['original_video'])}",
            f"**FPS**: {results['fps']} FPS",
            f"**Total Segments**: {results['total_segments']}",
            f"**Successful Segments**: {results['successful_segments']}",
            f"**Failed Segments**: {results['failed_segments']}",
            "",
            "## Action Group Details",
            "",
        ]
        for action_type, group_info in results["action_groups"].items():
            content.append(f"### {action_type.upper()} Actions")
            content.append(f"**Folder**: `{action_type}_actions/`")
            content.append("")
            for segment in group_info["segments"]:
                if segment["success"]:
                    content.append(f"- `{segment['filename']}`")
                    content.append(f"  - **Action**: {segment['original_action']}")
                    content.append(f"  - **Timestamp**: {segment['timestamp']}")
                    content.append(
                        f"  - **Frames**: {segment['start_frame']} - {segment['end_frame']}"
                    )
                    content.append("")
        content.append("\n---\n*Automatically generated by robotic action segmentation pipeline*")
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def _segment_video_ffmpeg(
        self,
        input_video: str,
        start_time: float,
        end_time: float,
        output_path: str,
    ) -> bool:
        try:
            duration = end_time - start_time
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-i", input_video,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts", "-preset", "medium", "-crf", "18",
                output_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _segment_video_opencv(
        self,
        input_video: str,
        start_time: float,
        end_time: float,
        output_path: str,
    ) -> bool:
        try:
            cap = cv2.VideoCapture(input_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = max(0, round(start_time * fps))
            end_frame = min(total_frames, round(end_time * fps))
            if end_frame <= start_frame:
                end_frame = start_frame + max(1, int(0.5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                cap.release()
                return False
            frame_count = 0
            target_frames = end_frame - start_frame
            while frame_count < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1
            cap.release()
            out.release()
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception:
            return False
