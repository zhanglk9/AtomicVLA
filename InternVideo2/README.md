# InternVideo2 Batch Annotation

Batch atomic-skill annotation for LIBERO trajectories in the supported LeRobot Parquet format.
Gripper changes and end-effector motion define segment boundaries; InternVideo2.5 generates a task chain and assigns steps to segments in order.

## Setup

Use Linux, Python 3.11, and a CUDA GPU with BF16 support.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU inference, install a FlashAttention build compatible with your Torch/CUDA environment. The tested setup uses:

```bash
pip install flash-attn==2.7.1.post4 --no-build-isolation
```

Prepare `OpenGVLab/InternVideo2_5_Chat_8B` locally. 

## Data

```text
dataset/
├── meta/tasks.jsonl
└── data/chunk-*/episode_*.parquet
```

Each Parquet file contains one complete episode with `image`, `actions`, `state`, and `task_index`.


```json
{"task_index": 0, "task": "pick up the cup and place it on the plate"}
```

## Annotation

Run from the project root. Replace the example paths with your dataset and checkpoint locations.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --data-root /path/to/dataset \
  --model-path /path/to/InternVideo2_5_Chat_8B \
  --output-dir output/full \
  --offline
```

The default processes all episodes and loads the model once. Useful options:

- `--episodes-per-task 1`: test the lowest-index episode from each task.
- `--task-ids 0 3`: process only the selected tasks.
- `--dry-run`: check and list inputs without loading the model or writing results.


## Output

Use **`output/full/train_annotations.json`** for training. A format example is provided in `examples/train_annotations.example.json`.

`report.json` keeps the full results; `excluded.json` lists filtered annotations, and `failed.json` records processing errors.
Per-episode checkpoints support interrupted runs. To rebuild the training JSON without loading the model:

```bash
python scripts/export_training_json.py --run-dir output/full
```
