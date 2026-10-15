"""
Action segmenter based on gripper changes and end-effector motion.

LIBERO uses seven-dimensional actions with gripper at index 6 and
state[:, :6] containing position in meters and an axis-angle rotation vector.
Legacy franka_long_resize actions retain gripper index 7.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

Segment = Tuple[str, int, int]  # (action, start, end)


class ActionSegmenter:
    def __init__(
        self,
        grip_thresh: float = 0.01,
        task_json_path: Optional[str] = None,
        translation_thresh: float = 0.03,
        rotation_thresh: float = 0.05,
        motion_window: int = 5,
    ):
        if not np.isfinite(translation_thresh) or translation_thresh <= 0:
            raise ValueError("translation_thresh must be a positive finite distance in meters")
        if not np.isfinite(rotation_thresh) or rotation_thresh <= 0:
            raise ValueError("rotation_thresh must be a positive finite angle in radians")
        if not isinstance(motion_window, int) or motion_window < 2:
            raise ValueError("motion_window must be an integer of at least 2 frames")
        self.grip_thresh = grip_thresh
        self.translation_thresh = translation_thresh
        self.rotation_thresh = rotation_thresh
        self.motion_window = motion_window
        self.pos_thresh = 0.1
        self.rotation_only_thresh = 0.02
        self.pick_distance_thresh = 0.1
        self.place_distance_thresh = 0.1
        self.consecutive_window = 50
        self.task_id = {}
        if task_json_path and os.path.exists(task_json_path):
            self.task_id = self._read_task_json(task_json_path)

    def _read_task_json(self, task_json_path: str) -> dict:
        data = []
        with open(task_json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return {item["task_index"]: item["task"] for item in data}

    def _detect_changes(
        self,
        actions: np.ndarray,
        task_instruct: str,
        eef_states: Optional[np.ndarray] = None,
    ):
        """Return frame indices and actions where state changes happen."""
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError("actions must have shape (frames, 7) or (frames, 8+)")
        gripper_index = 6 if actions.shape[1] == 7 else 7
        grippers = actions[:, gripper_index]
        if not np.isfinite(grippers).all():
            raise ValueError("Gripper values must be finite")
        grip_diff = np.diff(grippers)
        all_changes = []

        for i, d in enumerate(grip_diff):
            if abs(d) > self.grip_thresh:
                action = "Pick" if d > 0 else "Place"
                all_changes.append((i, action))

        filtered_changes = self._get_keep_indices(all_changes)
        filtered_changes = self._get_keep_indices2(filtered_changes)

        if "open" in task_instruct.lower() and filtered_changes:
            filtered_changes = self._add_open_action(filtered_changes, task_instruct)

        if eef_states is not None:
            motion_changes = self._detect_motion_changes(
                eef_states, grippers, filtered_changes
            )
            if filtered_changes:
                if "open" in task_instruct.lower():
                    motion_changes = [
                        change for change in motion_changes
                        if change[0] > filtered_changes[0][0]
                    ]
                if "close" in task_instruct.lower():
                    motion_changes = [
                        change for change in motion_changes
                        if change[0] < filtered_changes[-1][0]
                    ]
            existing_frames = {frame for frame, action in filtered_changes}
            filtered_changes.extend(
                (frame, action) for frame, action in motion_changes
                if frame not in existing_frames
            )
            filtered_changes.sort(key=lambda change: change[0])

        filtered_changes = self._filter_adjacent_duplicates(filtered_changes)

        change_idx = [c[0] for c in filtered_changes]
        change_act = [c[1] for c in filtered_changes]
        return change_idx, change_act

    def _window_rotation_angles(self, rotation_vectors: np.ndarray) -> np.ndarray:
        """Measure relative SO(3) angles without subtracting rotation vectors."""
        angles = np.linalg.norm(rotation_vectors, axis=1)
        scales = 0.5 * np.sinc(angles / (2 * np.pi))
        quaternions = np.column_stack((
            rotation_vectors * scales[:, None], np.cos(angles / 2)
        ))
        offset = self.motion_window - 1
        dots = np.sum(quaternions[offset:] * quaternions[:-offset], axis=1)
        return 2 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))

    def _detect_motion_changes(
        self,
        eef_states: np.ndarray,
        grippers: np.ndarray,
        existing_changes: list,
    ) -> list:
        """Add coarse motion events, keeping gripper events authoritative."""
        eef_states = np.asarray(eef_states, dtype=float)
        if (
            eef_states.ndim != 2
            or eef_states.shape[0] != len(grippers)
            or eef_states.shape[1] < 6
        ):
            raise ValueError("eef_states must have shape (frames, 6+) aligned with actions")
        if not np.isfinite(eef_states[:, :6]).all():
            raise ValueError("End-effector positions and rotation vectors must be finite")
        if len(eef_states) < self.motion_window:
            return []

        offset = self.motion_window - 1
        translation = eef_states[offset:, :3] - eef_states[:-offset, :3]
        translation_magnitudes = np.max(np.abs(translation), axis=1)
        rotation_angles = self._window_rotation_angles(eef_states[:, 3:6])
        motion_labels = []
        for window_index, (distance, angle) in enumerate(
            zip(translation_magnitudes, rotation_angles)
        ):
            window_grippers = grippers[window_index:window_index + self.motion_window]
            if np.any(np.abs(np.diff(window_grippers)) > self.grip_thresh):
                motion_labels.append(None)
            elif distance >= self.translation_thresh:
                motion_labels.append("Move")
            elif angle >= self.rotation_thresh:
                motion_labels.append("Turn" if window_grippers[-1] > 0 else "Adjust")
            else:
                motion_labels.append(None)

        has_gripper_events = any(
            action in ("Pick", "Place") for frame, action in existing_changes
        )
        changes = []
        run_start = 0
        while run_start < len(motion_labels):
            label = motion_labels[run_start]
            run_end = run_start + 1
            while run_end < len(motion_labels) and motion_labels[run_end] == label:
                run_end += 1
            if label is not None:
                last_frame = run_end - 1 + offset
                covered_by_gripper = has_gripper_events and label in ("Move", "Adjust")
                sustained_motion = last_frame - run_start + 1 >= max(
                    self.motion_window, 15
                )
                if sustained_motion and not covered_by_gripper:
                    changes.append((min(last_frame - 1, len(grippers) - 2), label))
            run_start = run_end
        return changes

    def _add_open_action(self, changes: list, task_instruct: str):
        """Insert an Open event at 2/3 before the first detected change."""
        if not changes:
            return changes
        first_change_idx = changes[0][0]
        open_idx = int(first_change_idx * 2 / 3)
        return [(open_idx, "Open")] + changes

    def _get_keep_indices(self, all_changes: list):
        """Keep only the last element in each consecutive Pick sequence."""
        if not all_changes:
            return all_changes
        last_changes = []
        while len(last_changes) != len(all_changes):
            keep_indices = set(range(len(all_changes)))
            i = 0
            while i < len(all_changes):
                current_idx, current_action = all_changes[i]
                if current_action == "Pick":
                    sequence_start = i
                    sequence_end = i
                    j = i + 1
                    while j < len(all_changes):
                        next_idx, next_action = all_changes[j]
                        if next_idx - current_idx <= self.consecutive_window:
                            if next_action == "Pick":
                                sequence_end = j
                            j += 1
                        else:
                            break
                    picks_in_sequence = [
                        k for k in range(sequence_start, sequence_end + 1)
                        if all_changes[k][1] == "Pick"
                    ]
                    if len(picks_in_sequence) >= 2:
                        for k in range(sequence_start, sequence_end + 1):
                            if k != picks_in_sequence[-1]:
                                keep_indices.discard(k)
                    i = sequence_end + 1
                else:
                    i += 1
            last_changes = all_changes
            all_changes = [last_changes[i] for i in range(len(last_changes)) if i in keep_indices]
        return all_changes

    def _get_keep_indices2(self, all_changes: list):
        """Keep only the last element in each consecutive Place sequence."""
        if not all_changes:
            return all_changes
        last_changes = []
        while len(last_changes) != len(all_changes):
            keep_indices = set(range(len(all_changes)))
            i = 0
            while i < len(all_changes):
                current_idx, current_action = all_changes[i]
                if current_action == "Place":
                    sequence_start = i
                    sequence_end = i
                    j = i + 1
                    while j < len(all_changes):
                        next_idx, next_action = all_changes[j]
                        if next_idx - current_idx <= self.consecutive_window:
                            if next_action == "Place":
                                sequence_end = j
                            j += 1
                        else:
                            break
                    places_in_sequence = [
                        k for k in range(sequence_start, sequence_end + 1)
                        if all_changes[k][1] == "Place"
                    ]
                    if len(places_in_sequence) >= 2:
                        for k in range(sequence_start, sequence_end + 1):
                            if k != places_in_sequence[-1]:
                                keep_indices.discard(k)
                    i = sequence_end + 1
                else:
                    i += 1
            last_changes = all_changes
            all_changes = [last_changes[i] for i in range(len(last_changes)) if i in keep_indices]
        return all_changes

    def _filter_adjacent_duplicates(self, changes: list):
        """Collapse adjacent duplicate actions and keep the last one."""
        if not changes:
            return changes
        result = []
        i = 0
        while i < len(changes):
            current_action = changes[i][1]
            j = i
            while j < len(changes) and changes[j][1] == current_action:
                j += 1
            result.append(changes[j - 1])
            i = j
        return result

    def _halfsplit_segments2(
        self,
        change_idx: List[int],
        change_act: List[str],
        total_len: int,
        instruct: str = "",
        min_len: int = 15,
    ) -> List[Segment]:
        """Handle pick tasks with a single detected change point."""
        if not change_idx or len(change_act) != 1:
            return [("Hold", 0, total_len)]
        change_point = change_idx[0] + 1
        split_point = min(change_point + 10, total_len)
        return [
            (change_act[0], 0, split_point),
            (change_act[0], split_point, total_len),
        ]

    def _halfsplit_segments(
        self,
        change_idx: List[int],
        change_act: List[str],
        total_len: int,
        instruct: str = "",
        min_len: int = 15,
    ) -> List[Segment]:
        """Convert change points to action segments, including close handling."""
        if not change_idx:
            return [("Hold", 0, total_len)]
        boundaries = [i + 1 for i in change_idx]
        if len(boundaries) == 1:
            return [(change_act[0], 0, total_len)]

        mids = []
        for j in range(len(boundaries) - 1):
            left, right = boundaries[j], boundaries[j + 1]
            mids.append(left + (right - left) // 3)

        segments = [(change_act[0], 0, mids[0])]
        for j in range(1, len(mids)):
            segments.append((change_act[j], mids[j - 1] + 1, mids[j]))

        has_close = "close" in instruct.lower()
        if has_close:
            last_change_pos = boundaries[-1]
            delay_pos = min(last_change_pos + 15, total_len)
            segments.append(("Place", mids[-1] + 1, delay_pos - 1))
            segments.append(("Close", delay_pos, total_len))
        else:
            segments.append((change_act[-1], mids[-1] + 1, total_len))

        merged = []
        for act, s, e in segments:
            if e < s:
                continue
            if merged and merged[-1][0] == act and merged[-1][2] + 1 == s:
                merged[-1] = (merged[-1][0], merged[-1][1], e)
            else:
                merged.append((act, s, e))

        final = []
        for act, s, e in merged:
            if final and (e - s + 1) < min_len:
                final[-1] = (final[-1][0], final[-1][1], e)
            else:
                final.append((act, s, e))

        for i in range(1, len(final)):
            if final[i][1] != final[i - 1][2] + 1:
                final[i] = (final[i][0], final[i - 1][2] + 1, final[i][2])
        return final

    def process_actions(
        self,
        actions: np.ndarray,
        task_instruct: str,
        eef_states: Optional[np.ndarray] = None,
    ) -> List[Segment]:
        change_idx, change_act = self._detect_changes(actions, task_instruct, eef_states)
        if "pick" in task_instruct.lower() and len(change_act) == 1:
            return self._halfsplit_segments2(
                change_idx, change_act, len(actions), task_instruct
            )
        return self._halfsplit_segments(
            change_idx, change_act, len(actions), task_instruct
        )

    def save_video(self, df: pd.DataFrame, output_path: str) -> str:
        """Write image sequence from parquet to mp4 and return output path."""
        first_img_bytes = df.iloc[0]["image"]["bytes"]
        first_img = Image.open(io.BytesIO(first_img_bytes)).convert("RGB")
        width, height = first_img.size

        os.makedirs(Path(output_path).parent, exist_ok=True)
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for _, row in df.iterrows():
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame = np.array(img)[:, :, ::-1]
            out.write(frame)

        out.release()
        return output_path

    def run(
        self,
        parquet_path: str,
        output_txt: Optional[str] = None,
        video_output_path: Optional[str] = None,
        task_json_path: Optional[str] = None,
        save_atomic_videos: bool = False,
        atomic_videos_dir: Optional[str] = None,
    ) -> Tuple[List[Segment], str, str]:
        """
        Read parquet and produce atomic actions.

        Returns:
            (atomic_actions, task_instruct, video_path)
        """
        df = pd.read_parquet(parquet_path)
        task_index = df["task_index"][0]

        if task_json_path and os.path.exists(task_json_path):
            task_id = self._read_task_json(task_json_path)
            task_instruct = task_id.get(task_index, "")
        elif self.task_id:
            task_instruct = self.task_id.get(task_index, "")
        else:
            task_instruct = ""

        if video_output_path is None:
            stem = Path(parquet_path).stem
            cache_dir = Path(parquet_path).parent / ".video_cache"
            cache_dir.mkdir(exist_ok=True)
            video_output_path = str(cache_dir / f"{stem}.mp4")

        self.save_video(df, video_output_path)
        actions = np.array(df["actions"].tolist())
        eef_states = None
        if actions.ndim == 2 and actions.shape[1] == 7:
            if "state" not in df.columns:
                raise ValueError("LIBERO motion detection requires the end-effector state column")
            eef_states = np.array(df["state"].tolist())
        atomic_actions = self.process_actions(actions, task_instruct, eef_states)

        if save_atomic_videos and atomic_videos_dir and video_output_path:
            self._process_and_save_actions(
                atomic_actions, video_output_path, atomic_videos_dir
            )

        if output_txt:
            with open(output_txt, "w", encoding="utf-8") as f:
                for act, s, e in atomic_actions:
                    f.write(f"{act} {s} {e}\n")

        return atomic_actions, task_instruct, video_output_path

    def _save_video_segment(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        output_path: str,
    ):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, fps,
            (int(cap.get(3)), int(cap.get(4)))
        )
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()

    def _process_and_save_actions(
        self,
        atomic_actions: List[Segment],
        video_path: str,
        output_dir: str,
    ):
        os.makedirs(output_dir, exist_ok=True)
        for idx, (action, start_frame, end_frame) in enumerate(atomic_actions):
            output_path = os.path.join(
                output_dir,
                f"action_{idx + 1}_{action}_{start_frame}-{end_frame}.mp4",
            )
            self._save_video_segment(
                video_path, start_frame, end_frame, output_path
            )
