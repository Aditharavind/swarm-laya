Swarm robots have to decide fast: move toward the goal, dodge an obstacle, head back to recharge, ask a teammate for help, or just hold position — all with noisy sensors, a flaky radio link, and a battery that's always ticking down.

I built Swarm-Laya: a synthetic-data pipeline + fine-tuning recipe that turns Laya (an open-source non-autoregressive decision model — one forward pass, no text generation) into a per-robot decision-maker for swarm robotics.

How it works:
1. A PyBullet simulator generates randomized swarm scenarios — 2 to 16+ robots, obstacles, a shared target, battery levels, comm dropout, sensor noise.
2. A potential-field expert policy scores every action for every robot and turns the scores into a soft probability distribution — the "teacher" signal.
3. That labels the dataset automatically, no manual annotation, in Laya's native typed-decision format.
4. Laya gets fine-tuned with RLCD (policy gradient over proper scoring rules + soft cross-entropy) — entirely on a single 6GB consumer GPU, using 8-bit AdamW to fit a 421M-parameter encoder where full-precision training wouldn't.
5. Evaluated on decision accuracy, latency, and — the part I care about most — closed-loop collision rate and task completion rate when the model (not the hand-coded expert) is actually driving the robots, including on swarm sizes and obstacle densities it never saw during training.

Results from one fine-tuning run (5,368 synthetic states, 4 epochs, single 6GB GPU):
→ 83.1% decision accuracy in-distribution, 90.2% on swarm sizes/obstacle densities it never trained on
→ ~33ms per decision
→ Closed-loop collision rate and task completion rate tracked against a hand-coded expert-policy oracle, both in-distribution and on unseen scenarios

Full numbers, the results chart, and a demo video of it driving a swarm are in the repo.

Code, dataset generator, training scripts and eval harness are open-source: https://github.com/Aditharavind/swarm-laya

Built on top of @Laya (https://github.com/NandhaKishorM/laya) by Nandhakishor M — a genuinely clever piece of engineering: a typed-decision model that answers choice/score/yes-no questions in a single forward pass across 100+ languages, with a fine-tuning recipe that turns its zero-shot ~35% accuracy into 76%+ on domain data. Swarm-Laya is that same recipe pointed at robotics instead of text.

#robotics #swarmrobotics #opensource #machinelearning #edgeai #pybullet
