---
license: apache-2.0
base_model: convaiinnovations/laya
tags:
- robotics
- swarm-robotics
- decision-making
- laya
- rlcd
datasets:
- Aditharavind/swarm-laya-decisions
pipeline_tag: text-classification
---

# swarm-laya

> **Simulation-only.** Fine-tuned and evaluated entirely in a PyBullet swarm
> simulator — not tested on physical robots. See
> [Aditharavind/swarm-laya](https://github.com/Aditharavind/swarm-laya) for
> the full pipeline, and the **v1 — no vision** note there: this checkpoint
> decides from ground-truth structured state, not camera images.

A [Laya](https://github.com/NandhaKishorM/laya) checkpoint (ModernBERT-large
encoder, 421M params) fine-tuned via RLCD to make one decision per robot per
timestep in a synthetic swarm-robotics setting: given a robot's state
(position, battery, distance/heading to target and base, nearest-obstacle
clearance, nearest-teammate distance, radio and sensor-noise status), pick
one of `move_to_target`, `avoid_obstacle`, `return_to_base`,
`request_swarm_help`, `hold_position`.

## Training

- Base: `convaiinnovations/laya`
- Data: [Aditharavind/swarm-laya-decisions](https://huggingface.co/datasets/Aditharavind/swarm-laya-decisions) (5,368 training states from a PyBullet simulator + potential-field expert policy, soft-label RLCD targets)
- Recipe: RLCD (GRPO-style policy gradient over proper scoring rules + soft cross-entropy), 4 epochs, single 6GB consumer GPU, 8-bit AdamW (bitsandbytes) — plain fp32 AdamW does not fit a 421M encoder in 6GB
- Calibration: one temperature per question type, fit on the held-out `val` split after training

## Evaluation

| | in-distribution (test, n=611) | unseen swarms (ood, n=2,000) |
|---|---|---|
| decision accuracy | 83.1% | 90.2% |
| latency (p50 / p95) | 31.4 / 32.1 ms | 32.4 / 35.7 ms |

Closed-loop rollouts (this model driving every robot vs. the hand-coded
expert-policy oracle, 15 episodes each; episodes capped at 25 steps for eval
speed, so completion rate is a model-vs-expert comparison under identical
conditions, not an absolute success rate):

| | collision rate (expert / model) | task completion (expert / model) |
|---|---|---|
| in-distribution | 2.1% / 6.0% | 20.3% / 17.4% |
| unseen swarms | 3.9% / 6.9% | 11.9% / 15.0% |

"Unseen swarms" = 9-16 robots / 16-28 obstacles, never seen during training
(train/val/test used 2-8 robots / 3-14 obstacles).

The weakest per-class accuracy is `hold_position` (~4-5%) — it's the rarest
action in the training distribution before balancing; see the source repo's
README for more on this and other known limits.

## Use

```python
import laya
agent = laya.Agent("Aditharavind/swarm-laya")

state = {
    "robot_id": 0, "position": [-3.4, -2.0], "battery_percent": 12.0,
    "distance_to_target": 8.5, "heading_to_target_deg": -32.0,
    "distance_to_base": 4.0, "sensor_noise_std_m": 0.0,
    "radio_link_degraded": False, "teammates_in_comm_range": 1, "swarm_size": 5,
    "nearest_obstacle": {"clearance_m": 2.0, "radius_m": 0.8},
    "nearest_teammate": {"distance_m": 3.9, "battery_percent": 40.0, "radio_link_degraded": False},
}
questions = {
    "next_action": {
        "type": "choice",
        "instructions": "Given this robot's state within the swarm, what should it do next?",
        "criteria": {
            "move_to_target": "No immediate hazard nearby and battery is sufficient; head toward the shared target.",
            "avoid_obstacle": "An obstacle or a teammate is close enough that the direct path is unsafe; steer around it.",
            "return_to_base": "Battery is low enough that continuing the mission risks stranding the robot; head back to base to recharge.",
            "request_swarm_help": "The robot is blocked, lost communication reliability, or its sensors are too noisy to act confidently, and a nearby teammate could assist or relay.",
            "hold_position": "Conditions are too uncertain or congested to move safely and no teammate is reachable to help; stay put and wait for the situation to clear.",
        },
    }
}
result = agent.predict(state, questions)
print(result["answers"]["next_action"]["choice"])
```

## License

Apache 2.0, matching [Laya](https://github.com/NandhaKishorM/laya)'s own
license.
