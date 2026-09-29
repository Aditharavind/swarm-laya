---
license: apache-2.0
task_categories:
- text-classification
tags:
- robotics
- swarm-robotics
- synthetic
- decision-making
- laya
pretty_name: Swarm-Laya Decisions
size_categories:
- 10K<n<100K
---

# Swarm-Laya Decisions

> **Simulation-only.** Generated entirely in a PyBullet swarm simulator — no
> physical robots or real sensors were involved. See
> [Aditharavind/swarm-laya](https://github.com/Aditharavind/swarm-laya) for
> the full pipeline that produced this data.

Synthetic per-robot decision states for swarm robotics, formatted for
[Laya](https://github.com/NandhaKishorM/laya)'s typed-decision fine-tuning
recipe. Each row is one robot at one simulated timestep: a `state` (its
position, battery, distance/heading to target and base, nearest-obstacle
clearance, nearest-teammate distance, radio and sensor-noise status), a fixed
`questions` schema (one `choice` question, `next_action`, over five options),
and `gold` — a hand-coded potential-field expert policy's *soft* probability
distribution over those five actions, not a hard label, so it supervises
RLCD's proper-scoring-rule objective directly.

## How it was generated

A PyBullet simulator (headless, `DIRECT` mode) draws a randomized scenario —
robot count, obstacle count and positions, a shared target, per-robot
battery, radio dropout and sensor noise — and steps every robot in closed
loop under a potential-field expert controller that scores five actions
(`move_to_target`, `avoid_obstacle`, `return_to_base`, `request_swarm_help`,
`hold_position`) and turns the scores into a temperature-softened
distribution. See `swarm_env/expert_policy.py` and
`data_generation/generate_dataset.py` in the source repo for the exact logic.

## Splits

| split | rows | scenario family |
|---|---|---|
| `train` | 5,368 | 2-8 robots, 3-14 obstacles (class-balanced, capped per action) |
| `val` | 603 | same family, disjoint scenario seeds from `train` |
| `test` | 611 | same family, disjoint scenario seeds from `train`/`val` |
| `ood_generalization` | 24,650 | **9-16 robots, 16-28 obstacles** — a family never seen in `train`/`val`/`test`, for measuring generalization, not accuracy on a held-out slice of the same distribution |

## Schema

Three string columns, each JSON-encoded (matching the row format Laya's own
[fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
expects):

```json
{
  "state": "{\"robot_id\": 4, \"position\": [-3.43, -2.05], \"battery_percent\": 63.6, \"distance_to_target\": 8.53, \"heading_to_target_deg\": -32.06, \"distance_to_base\": 3.99, \"sensor_noise_std_m\": 0.4, \"radio_link_degraded\": true, \"teammates_in_comm_range\": 0, \"swarm_size\": 5, \"nearest_obstacle\": {\"clearance_m\": 1.99, \"radius_m\": 0.8}, \"nearest_teammate\": {\"distance_m\": 3.9, \"battery_percent\": 0.0, \"radio_link_degraded\": false}}",
  "questions": "{\"next_action\": {\"type\": \"choice\", \"instructions\": \"Given this robot's state within the swarm, what should it do next?\", \"criteria\": {\"move_to_target\": \"...\", \"avoid_obstacle\": \"...\", \"return_to_base\": \"...\", \"request_swarm_help\": \"...\", \"hold_position\": \"...\"}}}",
  "gold": "{\"next_action\": {\"probabilities\": {\"move_to_target\": 0.67, \"avoid_obstacle\": 0.03, \"return_to_base\": 0.02, \"request_swarm_help\": 0.02, \"hold_position\": 0.25}}}"
}
```

## Use

```python
from datasets import load_dataset
ds = load_dataset("Aditharavind/swarm-laya-decisions")
```

Or point Laya's fine-tuning notebook's preprocessing cell at these files
directly (see `training/preprocess.py` in the source repo, which does
exactly that).

## License

Apache 2.0, matching [Laya](https://github.com/NandhaKishorM/laya)'s own
license.
