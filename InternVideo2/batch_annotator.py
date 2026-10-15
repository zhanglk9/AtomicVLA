"""Batch orchestration around the unchanged trajectory/VLM annotation pipeline."""

from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import time
import traceback
import uuid

from training_export import audit_annotation, export_results, utc_now, write_json


@contextmanager
def output_lock(output_dir):
    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another process is writing {output_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_tasks(path):
    tasks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        task_id, instruction = entry["task_index"], entry["task"]
        if not isinstance(task_id, int) or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Invalid task entry in {path}")
        if task_id in tasks:
            raise ValueError(f"Duplicate task_index {task_id} in {path}")
        tasks[task_id] = instruction
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks


def discover_episodes(args):
    import pyarrow.parquet as pq

    if args.parquet:
        paths = sorted({path.resolve() for path in args.parquet})
    else:
        directory = args.data_root / "data" if args.data_root else args.data_dir
        if not directory.is_dir():
            raise ValueError(f"Data directory does not exist: {directory}")
        paths = sorted(directory.resolve().rglob("*.parquet"))
    if not paths:
        raise ValueError("No parquet files found")
    task_path = args.task_json or os.environ.get("INTERNVIDEO_TASK_JSON")
    if task_path:
        task_path = Path(task_path).resolve()
    elif args.data_root:
        task_path = args.data_root.resolve() / "meta" / "tasks.jsonl"
    else:
        candidates = {
            next((parent / "meta" / "tasks.jsonl" for parent in path.parents
                  if (parent / "meta" / "tasks.jsonl").is_file()), None)
            for path in paths
        }
        if None in candidates or len(candidates) != 1:
            raise ValueError("Cannot identify one tasks.jsonl; supply --task-json explicitly")
        task_path = candidates.pop()
    tasks = read_tasks(task_path)
    requested = set(args.task_ids) if args.task_ids is not None else None
    if requested is not None and requested - set(tasks):
        raise ValueError(f"Unknown task IDs: {sorted(requested - set(tasks))}")
    episodes, seen = [], set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        required = {"actions", "task_index"}
        if not args.coarse_only:
            required.add("image")
        if not required <= columns:
            raise ValueError(f"Missing columns {sorted(required - columns)}: {path}")
        selected = [name for name in ("task_index", "episode_index", "frame_index") if name in columns]
        table = parquet.read(columns=selected)
        if table.num_rows == 0:
            raise ValueError(f"Empty episode: {path}")
        task_values = set(table["task_index"].to_pylist())
        if len(task_values) != 1:
            raise ValueError(f"Each parquet must contain one task: {path}")
        task_id = task_values.pop()
        if task_id not in tasks:
            raise ValueError(f"Task {task_id} missing from tasks.jsonl: {path}")
        if "episode_index" in columns:
            identifiers = set(table["episode_index"].to_pylist())
            if len(identifiers) != 1:
                raise ValueError(f"Each parquet must contain one episode: {path}")
            episode_id = identifiers.pop()
        else:
            match = re.fullmatch(r"episode_(\d+)", path.stem)
            if match is None:
                raise ValueError(f"Need episode_index column or episode_NNN.parquet filename: {path}")
            episode_id = int(match.group(1))
        if not isinstance(episode_id, int) or episode_id < 0 or episode_id in seen:
            raise ValueError(f"Invalid or duplicate episode ID {episode_id}: {path}")
        seen.add(episode_id)
        if "frame_index" in columns and table["frame_index"].to_pylist() != list(range(table.num_rows)):
            raise ValueError(f"frame_index must be contiguous from zero: {path}")
        if requested is not None and task_id not in requested:
            continue
        stat = path.stat()
        episodes.append({
            "episode_index": episode_id, "task_index": task_id, "task": tasks[task_id],
            "path": str(path), "source_frames": table.num_rows,
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        })
    if requested is not None and requested - {entry["task_index"] for entry in episodes}:
        raise ValueError("Some requested tasks have no episodes in the selected files")
    episodes.sort(key=lambda entry: (entry["task_index"], entry["episode_index"]))
    if args.episodes_per_task:
        counts = Counter()
        limited = []
        for episode in episodes:
            if counts[episode["task_index"]] < args.episodes_per_task:
                limited.append(episode)
                counts[episode["task_index"]] += 1
        episodes = limited
    if args.limit:
        episodes = episodes[:args.limit]
    if not episodes:
        raise ValueError("No episodes remain after filtering")
    return episodes, task_path


def run_signature(args, episodes, task_path):
    root = Path(__file__).resolve().parent
    source_names = (
        "batch_annotator.py", "training_export.py", "action_segmenter.py",
        "video_processor.py", "task_decomposer.py", "config.py",
    )
    model = Path(args.model_path)
    model_identity = {}
    if not args.coarse_only and model.is_dir():
        for path in sorted(model.iterdir()):
            if path.is_file() and path.suffix in (".json", ".py", ".model", ".safetensors", ".bin"):
                stat = path.stat()
                model_identity[path.name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                if stat.st_size < 2_000_000:
                    model_identity[path.name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "format_version": 1,
        "mode": "coarse_only" if args.coarse_only else "full",
        "model_path": None if args.coarse_only else args.model_path,
        "model_identity": model_identity,
        "task_json": str(task_path),
        "task_json_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "motion": {"grip_thresh": args.grip_thresh, "translation_thresh": args.translation_thresh,
                   "rotation_thresh": args.rotation_thresh, "motion_window": args.motion_window},
        "seed": args.seed,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names},
        "episodes": episodes,
    }


def log_event(output_dir, event, **values):
    entry = {"at": utc_now(), "event": event, **values}
    line = json.dumps(entry, ensure_ascii=False)
    print(line, flush=True)
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def load_engine(args, task_path):
    import numpy as np
    from action_segmenter import ActionSegmenter

    detector = ActionSegmenter(
        task_json_path=str(task_path), grip_thresh=args.grip_thresh,
        translation_thresh=args.translation_thresh, rotation_thresh=args.rotation_thresh,
        motion_window=args.motion_window,
    )
    if args.coarse_only:
        return detector, None, {}
    import torch
    from task_decomposer import ActionBasedRoboticDecomposer

    seed = args.seed % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    engine = ActionBasedRoboticDecomposer(model_path=args.model_path, device_id=args.device, task_json_path=str(task_path))
    engine.motion_analyzer = detector
    capture = {}
    original_run = detector.run
    original_analyze = engine.video_processor.analyze_action_segment2

    def capture_segments(*call_args, **call_kwargs):
        segments, instruction, video = original_run(*call_args, **call_kwargs)
        capture["coarse_segments"] = [
            {"label": label, "start_frame": start, "end_frame": end}
            for label, start, end in segments
        ]
        return segments, instruction, video

    def capture_analysis(*call_args, **call_kwargs):
        response = original_analyze(*call_args, **call_kwargs)
        capture["raw_task_chain"] = response
        capture["sampled_video_frames"] = len(call_args[1])
        return response

    detector.run = capture_segments
    engine.video_processor.analyze_action_segment2 = capture_analysis
    return detector, engine, capture


def annotate_episode(args, episode, detector, engine, capture, video_path):
    capture.clear()
    current = Path(episode["path"]).stat()
    if current.st_size != episode["size_bytes"] or current.st_mtime_ns != episode["mtime_ns"]:
        raise ValueError("Episode file changed after preflight; start a new run with the updated data")
    if args.coarse_only:
        import numpy as np
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(episode["path"])
        columns = ["actions"] + (["state"] if "state" in parquet.schema_arrow.names else [])
        table = parquet.read(columns=columns)
        actions = np.asarray(table["actions"].to_pylist())
        states = None
        if actions.ndim == 2 and actions.shape[1] == 7:
            if "state" not in columns:
                raise ValueError("Seven-dimensional LIBERO actions require the state column")
            states = np.asarray(table["state"].to_pylist())
        segments = detector.process_actions(actions, episode["task"], states)
        return {"status": "coarse_only", "coarse_segments": [
            {"label": label, "start_frame": start, "end_frame": end} for label, start, end in segments
        ]}
    import numpy as np
    import torch

    seed = (args.seed + episode["episode_index"]) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    annotation = engine.decompose_robotic_task(episode["path"], video_output_path=str(video_path))
    if "error" in annotation:
        raise RuntimeError(annotation["error"])
    result = {"status": "completed", "annotation": annotation, **capture}
    result["audit"] = audit_annotation({"task": episode["task"], **result}, engine.video_processor.task_mapping)
    return result


def run_batch(args):
    episodes, task_path = discover_episodes(args)
    print(json.dumps({"selected_episodes": len(episodes), "selected_tasks": len({entry["task_index"] for entry in episodes}),
                      "task_json": str(task_path), "dry_run": args.dry_run}, ensure_ascii=False), flush=True)
    if args.dry_run:
        for entry in episodes:
            print(f"task={entry['task_index']} episode={entry['episode_index']} frames={entry['source_frames']} {entry['path']}")
        return 0
    output_dir = args.output_dir.resolve()
    legacy_output = args.output.resolve() if args.output else None
    if legacy_output is not None:
        if legacy_output.is_relative_to(output_dir) and legacy_output != output_dir / "report.json":
            raise ValueError("--output must be outside output-dir or exactly its report.json")
        reserved = {Path(entry["path"]) for entry in episodes} | {task_path}
        reserved.update(Path(__file__).resolve().parent.glob("*.py"))
        if legacy_output in reserved:
            raise ValueError("--output would overwrite an input or source file")
    signature = run_signature(args, episodes, task_path)
    with output_lock(output_dir):
        manifest_path = output_dir / "run.json"
        if args.resume:
            if not manifest_path.is_file():
                raise ValueError("--resume requires an existing run.json")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("signature") != signature:
                raise ValueError("Resume refused: input selection, data, model, code or parameters changed; use a new output directory")
        else:
            if any(path.name != ".run.lock" for path in output_dir.iterdir()):
                raise ValueError("Output directory is not empty; use --resume or a new output directory")
            versions = {}
            for name in ("numpy", "pandas", "pyarrow", "torch", "torchvision", "transformers", "decord", "sentencepiece"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            manifest = {"run_id": uuid.uuid4().hex, "created_at": utc_now(), "signature": signature,
                        "episodes": episodes, "runtime_versions": versions}
            write_json(manifest_path, manifest)
        pending = []
        for episode in episodes:
            path = output_dir / "episodes" / f"{episode['episode_index']}.json"
            if path.exists():
                checkpoint = json.loads(path.read_text())
                if checkpoint.get("episode") != episode or checkpoint.get("run_id") != manifest["run_id"]:
                    raise ValueError(f"Checkpoint mismatch: {path}")
                if checkpoint["status"] in ("completed", "coarse_only"):
                    continue
            pending.append(episode)
        export_results(output_dir, manifest, legacy_output)
        log_event(output_dir, "batch_start", pending=len(pending), skipped=len(episodes) - len(pending))
        if not pending:
            manifest["status"] = "finished"
            manifest["updated_at"] = utc_now()
            manifest["summary"] = export_results(output_dir, manifest, legacy_output)
            write_json(manifest_path, manifest)
            log_event(output_dir, "nothing_to_run")
            return 0
        manifest["status"] = "running"
        manifest.pop("last_error", None)
        write_json(manifest_path, manifest)
        try:
            detector, engine, capture = load_engine(args, task_path)
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            write_json(manifest_path, manifest)
            raise
        except Exception:
            manifest["status"] = "setup_failed"
            manifest["last_error"] = traceback.format_exc()
            write_json(manifest_path, manifest)
            raise
        video_dir = output_dir / "video_cache"
        if not args.coarse_only:
            video_dir.mkdir(exist_ok=True)
        interrupted = False
        try:
            for ordinal, episode in enumerate(pending, 1):
                started = time.monotonic()
                episode_id = str(episode["episode_index"])
                record = {"run_id": manifest["run_id"], "episode": episode,
                          "task_index": episode["task_index"], "task": episode["task"],
                          "source_frames": episode["source_frames"]}
                video_path = video_dir / f"episode_{episode_id}.mp4"
                log_event(output_dir, "episode_start", episode_index=int(episode_id), ordinal=ordinal, total=len(pending))
                try:
                    record.update(annotate_episode(args, episode, detector, engine, capture, video_path))
                except Exception:
                    record.update(capture)
                    record.update(status="failed", error=traceback.format_exc())
                finally:
                    if not args.keep_videos and video_path.exists():
                        video_path.unlink()
                    if not args.coarse_only:
                        import torch
                        torch.cuda.empty_cache()
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                record["finished_at"] = utc_now()
                write_json(output_dir / "episodes" / f"{episode_id}.json", record)
                log_event(output_dir, "episode_end", episode_index=int(episode_id), status=record["status"], seconds=record["elapsed_seconds"])
                if ordinal % args.export_every == 0:
                    export_results(output_dir, manifest, legacy_output)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            summary = export_results(output_dir, manifest, legacy_output)
            manifest["status"] = "interrupted" if interrupted else (
                "incomplete" if summary["status_counts"].get("pending") else "finished"
            )
            manifest["updated_at"] = utc_now()
            manifest["summary"] = summary
            write_json(manifest_path, manifest)
        log_event(output_dir, manifest["status"], **summary)
        print(f"Training JSON: {output_dir / 'train_annotations.json'}", flush=True)
        return 130 if interrupted else (1 if summary["failed_episodes"] else 0)
