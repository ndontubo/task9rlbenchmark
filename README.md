# RL Testing/Benchmark Framework

`rl_benchmark.py` is a script that runs all of my from-scratch RL algorithms on the same environment, then auto-generates comparison plots and a results table.

It wraps the exact, already-validated implementations from Tasks 6–8 — it does **not** reimplement them:

| Key | Script | Algorithm |
|---|---|---|
| `ppo` | `ppo_from_scratch.py` | Task 6 PPO (clipped surrogate, GAE, multi-epoch) |
| `a2c_t7` | `a2c_from_scratch.py` | Task 7 A2C (n-step returns, single update) |
| `a2c_v2` | `a2c_v2.py` | Task 8 A2C v2 (+ frame-skip, GAE, LR decay) |

## How it works

1. **Train** — runs each script as its own subprocess, **sequentially**. This is
   faithful to the validated code (colliding class names + different env
   preprocessing make a shared import impossible) and it is the
   ReadWriteOnce-correct pattern from Task 7: a block-storage PVC is mounted by
   one pod at a time, so "benchmark several algorithms at the same time" means
   within one harness invocation, not concurrent GPU pods.
2. **Parse** — reads each run's TensorBoard event files into a unified schema
   (the scripts use different tag names, e.g. PPO `rollout/ep_reward_mean_10`
   vs A2C `rollout/ep_rew_mean`). All three count steps in env frames, so the
   x-axis is directly comparable.
3. **Evaluate** — loads each saved checkpoint and rolls out greedily, recording
   the **action distribution** (the Task 8 diagnostic: reward alone hid an
   all-GAS single-action collapse; the histogram exposes it). PPO saves no
   checkpoint, so it is omitted from the policy panel.
4. **Plot + summarise** — writes three figures and a results table.

Outputs (under `--outdir`):
```
plots/reward_curves.png        mean episodic return vs steps, all algos
plots/loss_curves.png          policy loss / value loss / entropy
plots/action_distribution.png  greedy action histogram per algo (the policy plot)
results_summary.md / .csv      best/final reward + eval action mix
runs/<algo>/                   raw TensorBoard logs + checkpoints
```

## Run locally (CartPole sanity check, no Box2D needed)

```bash
python rl_benchmark.py --env CartPole-v1 --timesteps 20000 \
    --num-envs 4 --n-steps 8 --eval-episodes 10
```

## Run on CarRacing

```bash
python rl_benchmark.py --env CarRacing-v3 --timesteps 1000000
# subset + re-plot from existing logs without retraining:
python rl_benchmark.py --algos ppo,a2c_v2 --env CarRacing-v3
python rl_benchmark.py --env CarRacing-v3 --skip-train
```

## Run on Nautilus (one Job)

```bash
# once: PVC + pre-install deps onto the PVC
kubectl apply -f ndontubo-task9-manifests.yml      # pvc + installer + shell
kubectl logs -f ndontubo-task9-installer           # wait for INSTALL_DONE
kubectl delete pod ndontubo-task9-installer

# copy the four scripts onto the PVC via the shell pod, then release the volume
kubectl cp rl_benchmark.py        ndontubo-task9-shell:/pvcvolume/
kubectl cp ppo_from_scratch.py    ndontubo-task9-shell:/pvcvolume/
kubectl cp a2c_from_scratch.py    ndontubo-task9-shell:/pvcvolume/
kubectl cp a2c_v2.py              ndontubo-task9-shell:/pvcvolume/
kubectl delete pod ndontubo-task9-shell            # frees the ReadWriteOnce PVC

# run the benchmark
kubectl apply -f ndontubo-task9-job.yml
kubectl logs -f job/ndontubo-task9-benchmark

# after it finishes, copy plots off the cluster (bring the shell pod back first)
kubectl apply -f ndontubo-task9-manifests.yml
kubectl cp ndontubo-task9-shell:/pvcvolume/benchmark_out/plots ./plots
```

## Adding an algorithm later (e.g. DQN)

Append one `AlgoSpec` to `build_registry()` with the script name, an `argv`
builder, the TensorBoard reward/loss tag names it logs, and its checkpoint path.
Nothing else changes, the parse/eval/plot stages are algorithm-agnostic.

## Fairness Condition

The wrapped scripts use different observation preprocessing: PPO trains on raw
96×96×3 RGB single frames; both A2C variants use 4-frame grayscale stacks, and
v2 adds action-repeat (frame-skip 4). The benchmark compares the algorithms **as
implemented in Tasks 6–8** which is an side-by-side of my actual agents, not a
controlled single-variable output.
