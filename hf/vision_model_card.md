---
license: apache-2.0
tags:
- robotics
- swarm-robotics
- computer-vision
- lightweight
datasets:
- Aditharavind/swarm-laya-decisions
---

# swarm-laya-vision

> **Simulation-only.** Trained and evaluated entirely on PyBullet-rendered
> camera frames — not tested on a physical camera. See
> [Aditharavind/swarm-laya](https://github.com/Aditharavind/swarm-laya) for
> the full pipeline.

A **~137K-parameter** CNN (trained from scratch, no pretrained backbone) that
reads a robot-mounted first-person camera frame (64x64 RGB, 90° FOV) and
estimates the visually-observable parts of a swarm robot's state: whether an
obstacle/teammate is visible, its distance, and its bearing relative to the
robot's own heading. This is the "lightweight vision module" half of the
project's decision+perception architecture — its output feeds into
[Aditharavind/swarm-laya](https://huggingface.co/Aditharavind/swarm-laya)
exactly like ground-truth state does, so the two together form a
camera-to-decision pipeline. It is deliberately *not* a general-purpose
vision model: battery, radio-link status and target direction are telemetry
in this design, not camera-derived, because a camera cannot literally see a
robot's own battery percentage.

## Architecture

4 conv blocks (stride-2, BatchNorm, ReLU) → global average pool → two small
linear heads (obstacle: visible/clearance/radius/bearing; teammate:
visible/distance/bearing, bearing as sin/cos to avoid angle wraparound).

## Training

- 12,112 labeled frames from the PyBullet simulator (camera render + ground-truth visibility/distance/bearing as supervision), 15 epochs, plain supervised learning (binary cross-entropy for visibility, masked MSE for distance/radius/bearing — masked so an occluded object's placeholder distance isn't a training target)
- Data generator: `vision/generate_vision_dataset.py` in the source repo

## Evaluation

| | in-distribution (test) | unseen swarms (ood) |
|---|---|---|
| obstacle-visibility accuracy | 86.7% | 72.2% |
| teammate-visibility accuracy | 78.1% | 64.9% |
| obstacle clearance MAE | 0.40 m | 0.55 m |
| teammate distance MAE | 0.71 m | 0.96 m |

**What happens when Laya decides from vision instead of ground truth**
(closed-loop rollout, same 25-step episode cap):

| | collision rate (ground truth / vision) | task completion (ground truth / vision) |
|---|---|---|
| in-distribution | 6.0% / 2.0% | 17.4% / 4.3% |
| unseen swarms | 6.9% / 4.9% | 15.0% / 3.6% |

Honest finding: vision-based perception *lowers* the collision rate (the
decision model gets more cautious when it can't clearly see what's around
it) but *cuts task completion sharply* — a real perception-limits-performance
tradeoff, not a solved problem. This is exactly the kind of result the
project's evaluation was designed to surface, not hide.

## Use

```python
import torch
from vision.model import SwarmVisionNet  # from the source repo

model = SwarmVisionNet()
ckpt = torch.load("swarm_vision.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# images: (N, 64, 64, 3) uint8
x = SwarmVisionNet.preprocess(images)
out = model(x)
```

See `vision/perceive.py` in the source repo for the full state-composition
logic (vision estimates + telemetry → the same JSON schema
`Aditharavind/swarm-laya` expects).

## License

Apache 2.0.
