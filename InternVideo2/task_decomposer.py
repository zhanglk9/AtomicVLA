"""
Task decomposer that coordinates segmentation and VLM analysis,
then exports a skill-library friendly report.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from action_segmenter import ActionSegmenter
from config import (
    DEFAULT_TASK_MAPPING,
    derive_data_root_from_parquet,
    get_model_path,
    get_task_json_path,
)
from video_processor import VideoProcessor


class ActionBasedRoboticDecomposer:
    """Action-based robotic task decomposer."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device_id: int = 0,
        target_fps: float = 10.0,
        task_json_path: Optional[str] = None,
        task_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        model_path = model_path or get_model_path()
        self.video_processor = VideoProcessor(
            model_path=model_path,
            device_id=device_id,
            task_mapping=task_mapping or DEFAULT_TASK_MAPPING,
        )
        self._task_json_path = task_json_path
        self.motion_analyzer = ActionSegmenter(task_json_path=task_json_path)
        self.target_fps = target_fps

    def decompose_robotic_task(
        self,
        parquet_path: str,
        output_txt: Optional[str] = None,
        video_output_path: Optional[str] = None,
        video_cache_dir: Optional[str] = None,
    ) -> Dict:
        """
        Decompose a robotic task from parquet input.

        Args:
            parquet_path: LeRobot parquet file path.
            output_txt: Optional text output path for action segments.
            video_output_path: Optional mp4 output path converted from parquet.
            video_cache_dir: Optional cache directory for generated videos.

        Returns:
            Analysis report dictionary.
        """
        task_json = self._task_json_path
        if not task_json:
            data_root = derive_data_root_from_parquet(parquet_path)
            task_json = get_task_json_path(data_root)
        if task_json and not os.path.exists(task_json):
            task_json = None

        if video_output_path is None and video_cache_dir:
            stem = Path(parquet_path).stem
            video_output_path = str(Path(video_cache_dir) / f"{stem}.mp4")
        elif video_output_path is None:
            stem = Path(parquet_path).stem
            cache_dir = Path(parquet_path).parent / ".video_cache"
            cache_dir.mkdir(exist_ok=True)
            video_output_path = str(cache_dir / f"{stem}.mp4")

        action_segments, task_instruct, video_path = self.motion_analyzer.run(
            parquet_path,
            output_txt=output_txt,
            video_output_path=video_output_path,
            task_json_path=task_json,
        )

        if not action_segments:
            return {"error": "No action segments detected"}

        # Full-clip analysis (Task Chain)
        last_act, last_s, last_e = action_segments[-1]
        pixel_values, num_patches_list = self.video_processor.load_video_segment(
            video_path, 0, last_e, num_segments=48
        )
        if pixel_values is None:
            return {"error": "Failed to load video for analysis"}

        segment_info_all = {
            "start_frame": 0,
            "end_frame": last_e,
            "step_number": 1,
        }
        all_analysis = self.video_processor.analyze_action_segment2(
            pixel_values, num_patches_list, segment_info_all, task_instruct
        )

        # Segment-by-segment analysis
        task_breakdown = []
        previous_state = "Without the previous step, the current one is the beginning."

        for i, (act, s, e) in enumerate(action_segments):
            pixel_values, num_patches_list = self.video_processor.load_video_segment(
                video_path, s, e, num_segments=48
            )
            if pixel_values is None:
                continue

            segment_info = {
                "start_frame": s,
                "end_frame": e,
                "step_number": i + 1,
                "all_step": len(action_segments),
            }
            action_last = f"{i + 1}/{len(action_segments)}"
            action_info, previous_state = self.video_processor.analyze_action_segment3(
                pixel_values,
                num_patches_list,
                action_last,
                segment_info,
                all_analysis,
                previous_state,
                task_instruct,
            )
            if action_info:
                task_breakdown.append(action_info)

        for i in range(len(task_breakdown) - 1):
            task_breakdown[i]["update_chain_of_thought"] = (
                "Start next step: " + task_breakdown[i + 1]["chain_of_thought"]
            )

        return self._generate_final_report(task_breakdown, parquet_path)

    def _generate_final_report(
        self, task_breakdown: List[dict], video_path: str
    ) -> Dict:
        skill_sequence = []
        scene_context = []

        for i, step in enumerate(task_breakdown):
            if step:
                skill_info = {
                    "step_number": i + 1,
                    "timestamp": step["timestamp_formatted"],
                    "duration": step["end_frame"] - step["start_frame"],
                    "primary_action": step.get("primary_action_verb", "manipulate"),
                    "action_description": step["action_name"],
                    "skill_parameters": step.get("skill_parameters", ""),
                    "scene_context": step.get("scene_description", ""),
                    "objects_detected": step.get("objects_detected", []),
                    "spatial_relationships": step.get("spatial_relationships", []),
                }
                skill_sequence.append(skill_info)
                scene_context.append({
                    "timestamp": step["timestamp_formatted"],
                    "scene": step.get("scene_description", ""),
                    "objects": step.get("objects_detected", []),
                })

        video_duration = None
        if task_breakdown:
            video_duration = max(
                step.get("end_frame", 0) for step in task_breakdown if step
            )

        return {
            "video_path": video_path,
            "analysis_type": "action_based_robotic_decomposition",
            "total_duration": video_duration,
            "video_duration": video_duration,
            "total_steps": len(skill_sequence),
            "segments": task_breakdown,
            "skill_sequence": skill_sequence,
            "scene_context": scene_context,
            "available_skills": self._generate_skill_library_description(skill_sequence),
            "skill_library_format": self._format_for_skill_library(skill_sequence),
        }

    def _generate_skill_library_description(
        self, skill_sequence: List[dict]
    ) -> dict:
        unique_actions = set()
        skill_definitions = {}
        for skill in skill_sequence:
            action = skill["primary_action"]
            unique_actions.add(action)
            if action not in skill_definitions:
                skill_definitions[action] = {
                    "name": action,
                    "description": f"Robotic {action} manipulation",
                    "parameters": [],
                    "examples": [],
                }
            skill_definitions[action]["examples"].append(skill["action_description"])
        return {
            "total_unique_skills": len(unique_actions),
            "skill_definitions": skill_definitions,
            "execution_order": [s["primary_action"] for s in skill_sequence],
        }

    def _format_for_skill_library(self, skill_sequence: List[dict]) -> str:
        lines = [
            "# Atomic Skill Sequence for VLM-MoE-LoRA-Diff Framework",
            "",
            "## Scene Analysis",
        ]
        for skill in skill_sequence:
            if skill["scene_context"]:
                lines.append(
                    f"- **Step {skill['step_number']}** "
                    f"({skill['timestamp']}): {skill['scene_context']}"
                )
        lines.extend(["", "## Skill Execution Plan"])
        for skill in skill_sequence:
            lines.append(f"**Step {skill['step_number']}**: `{skill['primary_action']}`")
            lines.append(f"- Description: {skill['action_description']}")
            if skill["skill_parameters"]:
                lines.append(f"- Parameters: {skill['skill_parameters']}")
            if skill["objects_detected"]:
                lines.append(f"- Objects: {', '.join(skill['objects_detected'])}")
            lines.append(f"- Duration: {skill['duration']}")
            lines.append("")
        return "\n".join(lines)

    def save_report(
        self,
        report: Dict,
        output_path: str,
        auto_segment: bool = False,
        segment_output_dir: Optional[str] = None,
    ) -> str:
        video_name = Path(report["video_path"]).stem
        output_dir = Path(output_path).parent
        main_output_path = output_dir / f"{video_name}_analysis.json"

        simplified = {
            "total_steps": report["total_steps"],
            "segments": report["segments"],
        }
        with open(main_output_path, "w", encoding="utf-8") as f:
            json.dump(simplified, f, ensure_ascii=False, indent=2)

        return str(main_output_path)
