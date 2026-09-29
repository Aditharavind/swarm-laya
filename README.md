# Swarm-Laya

Fine-tunes [Laya](https://github.com/NandhaKishorM/laya) — a non-autoregressive
"System 1" decision model that answers typed questions (`choice` / `score` /
`noul`) over a state in a single forward pass — to make per-robot decisions
in a synthetic swarm-robotics setting, per the project abstract: separate a
lightweight vision/perception layer (out of scope here; this project starts
from structured state, not images) from a specialized decision model,
trained on labeled data from an automated synthetic-data pipeline instead of
costly real-world collection.

## Pipeline

```
swarm_env/            PyBullet swarm simulator, expert policy, state encoder, action schema
data_generation/      runs the simulator + expert policy, writes labeled Laya-format JSONL
training/             preprocess.py (tokenize) + train.py (single-GPU RLCD fine-tuning)
eval/                 decision accuracy, latency, collision rate, task completion, generalization
```

### 1. Generate the synthetic dataset

```bash
python data_generation/generate_dataset.py --out ./data --episodes 400 --ood-episodes 60
```

Each `reset()` draws a fresh scenario — 2-8 robots (9-16 for the held-out
`ood` family), 3-14 obstacles (16-28 for `ood`), a shared target, per-robot
battery, comm dropout and sensor noise. Robots are driven in closed loop by
`swarm_env/expert_policy.py`, a potential-field controller that scores five
actions (`move_to_target`, `avoid_obstacle`, `return_to_base`,
`request_swarm_help`, `hold_position`) and turns the scores into a soft
label distribution — this is the "expert planning/control policy" from the
abstract, and the distribution (not just the arg-max) becomes Laya's RLCD
training target. Output rows match the schema Laya's own fine-tuning
notebook expects: `{"state": ..., "questions": ..., "gold": ...}` written to
`data/{train,val,test}.jsonl` (in-distribution, split by scenario so a
robot's whole trajectory stays in one split) and
`data/ood_generalization.jsonl` (wider ranges, held out entirely from
training — used only in evaluation to measure generalization).

### 2. Preprocess (tokenize)

```bash
python training/preprocess.py --data-dir ./data --out-dir ./data/preprocessed
```

Downloads the base `convaiinnovations/laya` checkpoint's tokenizer/config
and tokenizes every split into `*_items.pt`, mirroring cell 6 of Laya's
[fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).

### 3. Fine-tune

```bash
python training/train.py --data-dir ./data/preprocessed --output-dir ./checkpoints/swarm_laya
```

This is the notebook's `train_ddp.py` (RLCD: GRPO-style policy gradient over
proper scoring rules + soft cross-entropy, gradient checkpointing, fp16)
with `torch.distributed`/DDP removed for a single GPU. Defaults
(`--micro-batch 2 --grad-accum 16`, effective batch 32) are sized for a 6 GB
card — **this machine has a 6 GB RTX 3060, not the 12 GB the abstract
assumed**; raise `--micro-batch` if you run it on a bigger GPU. Plain fp32
AdamW on ModernBERT-large's 421M parameters alone (weights + gradients +
optimizer state) exceeds 6 GB before a single activation is allocated, so
the optimizer is [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)'
8-bit AdamW instead, falling back to plain `torch.optim.AdamW` if
bitsandbytes isn't installed. `--max-len 512 --head-max-len 128` also
shrink the notebook's 1024/256 budget, since swarm states are short JSON
blobs, not multi-paragraph tickets — this roughly halves activation memory
for free. Calibration temperatures (one per question type) are fit on the
`val` split after the last epoch, exactly as the notebook does on its
held-out calibration slice.

### 4. Evaluate

```bash
python eval/evaluate.py --checkpoint ./checkpoints/swarm_laya
```

Reports the abstract's metrics:
- **decision accuracy** and **per-class accuracy** against expert labels, on
  `test.jsonl` (in-distribution) and `ood_generalization.jsonl` (unseen
  scenarios)
- **response latency** (p50/p95, ms) per decision
- **collision rate** and **task completion rate** from closed-loop
  simulator rollouts where the fine-tuned model (instead of the expert)
  picks every robot's action each step, compared against the expert-policy
  oracle as an upper bound, for both scenario families

Full results land in `eval_report.json`. Render a shareable summary image with
`python eval/make_results_card.py`, or record a top-down rollout video with
`python eval/record_demo.py --checkpoint ./checkpoints/swarm_laya`.

## Results

One fine-tuning run: 5,368 training states (300 episodes, 2-8 robots,
3-14 obstacles), 4 epochs, single 6 GB RTX 3060.

![Swarm-Laya results](results_card.png)

| | in-distribution | unseen swarms (9-16 robots, 16-28 obstacles) |
|---|---|---|
| decision accuracy | 83.1% (n=611) | 90.2% (n=2,000) |
| latency (p50 / p95) | 32.6 / 33.4 ms | 34.0 / 38.3 ms |
| collision rate, expert vs. model | 2.4% vs. 5.4% | 6.1% vs. 12.4% |
| task completion, expert vs. model | 19.8% vs. 18.5% | 12.4% vs. 14.3% |

`demo.mp4` is a top-down recording of the fine-tuned model driving every
robot in three rollouts (two in-distribution, one unseen-swarm).

Reading the rollout numbers: episodes are capped at 25 steps for eval speed,
which caps completion rate for both policies alike (many targets need more
steps than that to reach on foot) — the meaningful comparison is
model-vs-expert under identical conditions, not the absolute rate. The
weakest per-class accuracy is on `hold_position` (~4-5%) — it's the rarest
action in the training distribution (582 of ~24k pre-balancing rows), so the
model under-predicts it; more `hold_position` examples (or explicit
upweighting) is the obvious next lever, along with an unbalanced test-set
sanity check per `docs/finetune.md`'s advice to check per-workflow accuracy,
not just the aggregate.

## Design notes

- **Decisions are a single `choice` question** (`next_action`, five
  options), matching Laya's typed-decision primitives directly rather than
  a continuous control output — Laya is a classifier over a fixed option
  set, not a policy network, so "navigation, collision avoidance, task
  prioritization, swarm coordination and failure recovery" from the
  abstract are folded into that one categorical decision per robot per
  step; a discrete choice is then converted to a heading for the simulator.
- **State is a flat JSON object** (position, battery, distance/heading to
  target and base, nearest obstacle clearance, nearest teammate, comm and
  sensor-noise status) — this is exactly the "state" Laya reads as text, so
  it doubles as a human-readable log line.
- **Gold labels are soft distributions**, not one-hot: the expert policy's
  per-action utilities are turned into a temperature-softened distribution,
  because RLCD trains against a teacher's confidence, not just its arg-max
  — per `docs/finetune.md` in the Laya repo, "the loop is only as good as
  the targets."
- **`ood_generalization.jsonl`** is drawn from a scenario family (9-16
  robots, 16-28 obstacles) disjoint from anything in `train`/`val`/`test`
  (2-8 robots, 3-14 obstacles), so its accuracy and rollout numbers are a
  genuine generalization measurement, not a held-out slice of the same
  distribution.
