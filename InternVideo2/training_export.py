"""Audit existing pipeline output and export the training-loader-compatible schema."""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def audit_annotation(record, task_mapping):
    raw = record.get("raw_task_chain") or ""
    match = re.search(r"\[(.*?)\]", raw)
    parsed = [item.strip() for item in match.group(1).split(",")] if match else []
    preset = task_mapping.get(record["task"])
    effective = preset if preset is not None else parsed
    count = len(record["coarse_segments"])
    if not effective:
        mode = "whole_instruction_fallback_unparsed_chain"
    elif len(effective) == count:
        mode = "positional_assignment"
    elif len(effective) == 2 and count == 4:
        mode = "repeat_two_steps_to_four"
    else:
        mode = "whole_instruction_fallback_count_mismatch"
    return {
        "assignment_mode": mode,
        "model_chain_steps": parsed,
        "effective_chain_steps": effective,
        "preset_override": preset is not None,
        "manually_verified": False,
    }


def training_entry(record):
    reasons = []
    audit = record.get("audit", {})
    if audit.get("assignment_mode") != "positional_assignment":
        reasons.append(audit.get("assignment_mode", "missing_assignment_audit"))
    annotation = record.get("annotation", {})
    segments = annotation.get("segments", [])
    coarse = record.get("coarse_segments", [])
    total_frames = record["source_frames"]
    if not segments or len(segments) != len(coarse) or annotation.get("total_steps") != len(segments):
        reasons.append("missing_or_inconsistent_segments")
    if annotation.get("total_duration") != total_frames:
        reasons.append("incomplete_episode")
    compact = []
    previous_end = -1
    for index, segment in enumerate(segments):
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= total_frames:
            reasons.append("invalid_frame_interval")
            continue
        if start != previous_end + 1:
            reasons.append("overlapping_or_missing_frames")
        previous_end = end
        skill = segment.get("primary_action_verb", "")
        instruction = segment.get("action_name", "")
        if not isinstance(skill, str) or not skill.strip() or not isinstance(instruction, str) or not instruction.strip():
            reasons.append("empty_skill_or_instruction")
        if index < len(coarse) and (start, end) != (coarse[index]["start_frame"], coarse[index]["end_frame"]):
            reasons.append("segment_alignment_changed")
        item = {
            "action_name": instruction,
            "primary_action_verb": skill,
            "start_frame": start,
            "end_frame": end,
        }
        if index < len(segments) - 1:
            update = segment.get("update_chain_of_thought", "")
            words = update.split() if isinstance(update, str) else []
            if not words or words[-1] != segments[index + 1].get("primary_action_verb"):
                reasons.append("invalid_next_skill")
            else:
                item["update_chain_of_thought"] = "the atomic skill is " + words[-1]
        compact.append(item)
    if previous_end != total_frames:
        reasons.append("incomplete_episode")
    if reasons:
        return None, sorted(set(reasons))
    return {"all_frames": total_frames, "total_steps": len(compact), "segments": compact}, []


def export_results(output_dir, manifest, legacy_output=None):
    output_dir = Path(output_dir)
    training, reports, coarse, excluded, failed = {}, {}, {}, {}, {}
    status_counts = Counter()
    for episode in manifest["episodes"]:
        episode_id = str(episode["episode_index"])
        path = output_dir / "episodes" / f"{episode_id}.json"
        if not path.exists():
            status_counts["pending"] += 1
            continue
        record = json.loads(path.read_text())
        if record.get("episode") != episode or record.get("run_id") != manifest["run_id"]:
            raise ValueError(f"Checkpoint does not match this run: {path}")
        status = record["status"]
        status_counts[status] += 1
        if "coarse_segments" in record:
            coarse[episode_id] = {
                "task_index": record["task_index"], "task": record["task"],
                "all_frames": record["source_frames"], "segments": record["coarse_segments"],
            }
        if status == "failed":
            failed[episode_id] = {"task_index": record["task_index"], "error": record["error"]}
        elif status == "completed":
            annotation = record["annotation"]
            reports[episode_id] = {
                "all_frames": annotation["total_duration"],
                "total_steps": annotation["total_steps"], "segments": annotation["segments"],
            }
            item, reasons = training_entry(record)
            if item is not None:
                training[episode_id] = item
            else:
                excluded[episode_id] = {"task_index": record["task_index"], "reasons": reasons}
    summary = {
        "updated_at": utc_now(), "selected_episodes": len(manifest["episodes"]),
        "status_counts": dict(status_counts), "training_episodes": len(training),
        "training_segments": sum(item["total_steps"] for item in training.values()),
        "excluded_episodes": len(excluded), "failed_episodes": len(failed),
        "training_episode_ids": [int(key) for key in training],
        "labels_manually_verified": False,
    }
    for name, value in (
        ("train_annotations.json", training), ("report.json", reports),
        ("coarse_segments.json", coarse), ("excluded.json", excluded),
        ("failed.json", failed), ("summary.json", summary),
    ):
        write_json(output_dir / name, value)
    if legacy_output is not None:
        write_json(legacy_output, reports)
    return summary
