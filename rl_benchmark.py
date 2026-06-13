#!/usr/bin/env python3
"""
rl_benchmark.py  --  Task 9: one-click RL testing / benchmarking framework.

WHAT THIS IS
    A single-file harness that runs ALL of my from-scratch RL algorithms on the
    same Gymnasium environment, one after another, then automatically produces
    comparison plots and a results table. "One click": you run this script and
    it trains, evaluates, and graphs several algorithms with no babysitting.

    It wraps the *exact, already-validated* implementations from Tasks 6-8
    rather than reimplementing them:
        - ppo_from_scratch.py   (Task 6 PPO: clipped surrogate, GAE, multi-epoch)
        - a2c_from_scratch.py   (Task 7 A2C: n-step returns, single update)
        - a2c_v2.py             (Task 8 A2C v2: + frame-skip, GAE, LR decay)

WHY SUBPROCESS ORCHESTRATION (the central design decision)
    The three scripts have colliding class names (each defines its own
    CNNActorCritic / MLPActorCritic) and different env preprocessing, so they
    cannot all be imported into one namespace and driven from a shared loop
    without rewriting them. Running each as its own subprocess:
        * uses my real, validated code unchanged (no risk of altering results),
        * isolates global state / CUDA context / RNG seeding between runs,
        * survives one algorithm crashing without killing the whole benchmark,
        * runs the algorithms SEQUENTIALLY, which is the ReadWriteOnce-correct
          pattern from Task 7 -- a block-storage PVC is mounted by one pod at a
          time, so "benchmark several algorithms at the same time" means within
          one harness invocation, not literally concurrent GPU pods.

    Each script already writes TensorBoard event files. The harness parses those
    into a unified schema (different scripts use different tag names, e.g. PPO's
    rollout/ep_reward_mean_10 vs A2C's rollout/ep_rew_mean), then plots them on
    shared axes. All three count global_step in underlying ENV FRAMES, so the
    timestep x-axis is directly comparable across algorithms.

WHAT IT GRAPHS  (the task asks for reward / loss / policy)
    1. reward_curves.png       -- smoothed mean episodic return vs timesteps,
                                  all algorithms overlaid.
    2. loss_curves.png         -- policy loss, value loss, and entropy vs steps.
    3. action_distribution.png -- the POLICY plot: greedy action histogram from
                                  a post-training evaluation. This is the Task 8
                                  signature diagnostic -- a flat reward curve hid
                                  an all-GAS single-action collapse that only the
                                  action distribution revealed. The framework
                                  bakes that lesson in as a default panel.
    + results_summary.{md,csv} -- best/final reward and eval action mix per algo.

FAIRNESS CAVEAT (state this in the writeup)
    The wrapped scripts use different observation preprocessing: PPO trains on
    raw 96x96x3 RGB single frames; both A2C variants use 4-frame grayscale
    stacks, and v2 adds action-repeat (frame-skip 4). The benchmark compares the
    algorithms AS I IMPLEMENTED THEM. It is not a controlled ablation of one
    variable -- it is an honest side-by-side of my actual Task 6-8 agents.

USAGE
    # one click, all algorithms, on CarRacing (the real cluster run):
    python rl_benchmark.py --env CarRacing-v3 --timesteps 1000000

    # quick local sanity check on CartPole (no Box2D needed):
    python rl_benchmark.py --env CartPole-v1 --timesteps 20000 \
        --num-envs 4 --n-steps 8 --eval-episodes 10

    # only re-draw plots from logs of a previous run (no retraining):
    python rl_benchmark.py --env CarRacing-v3 --skip-train

    # pick a subset of algorithms:
    python rl_benchmark.py --algos ppo,a2c_v2 --env CarRacing-v3

On Nautilus, run inside the Job exactly like the individual scripts:
    PYTHONUNBUFFERED=1 PYTHONPATH=/pvcvolume/python-packages \
    python /pvcvolume/rl_benchmark.py --env CarRacing-v3 \
        --timesteps 1000000 --outdir /pvcvolume/benchmark
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# matplotlib without a display (headless pod / CI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Algorithm registry                                                          #
# --------------------------------------------------------------------------- #
# Each algorithm is described once, here. To add a new algorithm (e.g. a DQN
# script later), append one AlgoSpec -- nothing else in the file changes.
#
#   argv: builds the command-line args for the wrapped script from the shared
#         benchmark config. This is where each script's own flag spelling
#         (--timesteps vs --total-timesteps, --log_dir vs --logdir) is mapped.
#   reward_keys / *_keys: ordered candidate TensorBoard tags; the first one that
#         exists in the run is used for that unified metric.
#   checkpoint: path (relative to the run's logdir) to the saved model used for
#         the post-training action-distribution eval, or None if the script does
#         not checkpoint (PPO does not, so it is skipped in the policy plot).

@dataclass
class AlgoSpec:
    key: str                        # short id, e.g. "ppo"
    label: str                      # pretty name for plots/legends
    script: str                     # filename of the wrapped implementation
    argv: Callable                  # (cfg, abs_logdir) -> list[str]
    reward_keys: List[str]
    policy_loss_keys: List[str] = field(default_factory=lambda: ["train/policy_loss"])
    value_loss_keys: List[str] = field(default_factory=lambda: ["train/value_loss"])
    entropy_keys: List[str] = field(default_factory=lambda: ["train/entropy"])
    checkpoint: Optional[str] = None
    family: str = ""                # "ppo" or "a2c" -- selects the eval loader


def _ppo_argv(cfg, logdir):
    # ppo_from_scratch.py: --env --timesteps --log_dir --seed
    return [
        "--env", cfg.env,
        "--timesteps", str(cfg.timesteps),
        "--log_dir", logdir,
        "--seed", str(cfg.seed),
    ]


def _a2c_argv(cfg, logdir, frame_skip):
    # a2c_from_scratch.py / a2c_v2.py share these flags. frame-skip only matters
    # for CarRacing; on vector envs we force it to 1 so the step axis is honest.
    argv = [
        "--env", cfg.env,
        "--total-timesteps", str(cfg.timesteps),
        "--num-envs", str(cfg.num_envs),
        "--n-steps", str(cfg.n_steps),
        "--logdir", logdir,
        "--seed", str(cfg.seed),
        "--log-interval", str(cfg.log_interval),
    ]
    return argv


def build_registry():
    return [
        AlgoSpec(
            key="ppo",
            label="PPO (Task 6)",
            script="ppo_from_scratch.py",
            argv=_ppo_argv,
            reward_keys=[
                "rollout/ep_reward_mean_10",
                "rollout/ep_reward_mean_100",
                "rollout/ep_reward",
            ],
            checkpoint=None,            # PPO script saves no model -> no policy panel
            family="ppo",
        ),
        AlgoSpec(
            key="a2c_t7",
            label="A2C (Task 7)",
            script="a2c_from_scratch.py",
            argv=lambda cfg, ld: _a2c_argv(cfg, ld, frame_skip=1),
            reward_keys=["rollout/ep_rew_mean_10", "rollout/ep_rew_mean"],
            checkpoint="models/a2c_final.pt",
            family="a2c",
        ),
        AlgoSpec(
            key="a2c_v2",
            label="A2C v2 (Task 8)",
            script="a2c_v2.py",
            # v2 adds --frame-skip; default 4 on CarRacing, 1 elsewhere.
            argv=lambda cfg, ld: _a2c_argv(cfg, ld, frame_skip=cfg.frame_skip)
            + ["--frame-skip", str(cfg.frame_skip)],
            reward_keys=["rollout/ep_rew_mean_10", "rollout/ep_rew_mean"],
            checkpoint="models/a2c_final.pt",
            family="a2c",
        ),
    ]


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class BenchConfig:
    env: str
    timesteps: int
    num_envs: int
    n_steps: int
    frame_skip: int
    seed: int
    log_interval: int
    eval_episodes: int
    outdir: str
    script_dir: str
    python: str


# --------------------------------------------------------------------------- #
# Stage 1: training (subprocess, sequential)                                  #
# --------------------------------------------------------------------------- #
def run_training(spec: AlgoSpec, cfg: BenchConfig) -> str:
    """Launch the wrapped script as a subprocess; stream its stdout live.

    Returns the absolute logdir the run wrote its TensorBoard events to.
    """
    logdir = os.path.abspath(os.path.join(cfg.outdir, "runs", spec.key))
    # Fresh dir so EventAccumulator never mixes this run with a previous one.
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir, exist_ok=True)

    script_path = os.path.join(cfg.script_dir, spec.script)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Cannot find {spec.script} next to the harness "
            f"(looked in {cfg.script_dir}). Put the algorithm scripts there."
        )

    cmd = [cfg.python, script_path] + spec.argv(cfg, logdir)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # Task 6 lesson: keep stdout live, not buffered

    print("\n" + "=" * 78)
    print(f"[TRAIN] {spec.label}")
    print("  " + " ".join(cmd))
    print("=" * 78, flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"[WARN] {spec.label} exited with code {proc.returncode} "
              f"after {dt:.0f}s -- its partial logs will still be parsed.")
    else:
        print(f"[OK] {spec.label} finished in {dt/60:.1f} min.")
    return logdir


# --------------------------------------------------------------------------- #
# Stage 2: parse TensorBoard event files into a unified schema                #
# --------------------------------------------------------------------------- #
def _load_scalars(logdir: str):
    """Return {tag: (steps[np], values[np])} for every scalar tag in logdir."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.float64)
        vals = np.array([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


def _first_present(scalars: dict, keys: List[str]):
    for k in keys:
        if k in scalars and len(scalars[k][0]) > 0:
            return scalars[k]
    return None


def parse_run(spec: AlgoSpec, logdir: str) -> dict:
    """Pull the unified metrics (reward, losses, entropy) out of one run."""
    try:
        scalars = _load_scalars(logdir)
    except Exception as exc:  # noqa: BLE001 -- never let one bad run kill the report
        print(f"[WARN] could not read TB logs for {spec.label}: {exc}")
        scalars = {}

    result = {
        "spec": spec,
        "reward": _first_present(scalars, spec.reward_keys),
        "policy_loss": _first_present(scalars, spec.policy_loss_keys),
        "value_loss": _first_present(scalars, spec.value_loss_keys),
        "entropy": _first_present(scalars, spec.entropy_keys),
    }
    if result["reward"] is not None:
        steps, vals = result["reward"]
        finite = vals[np.isfinite(vals)]
        result["best_reward"] = float(np.nanmax(vals)) if finite.size else float("nan")
        result["final_reward"] = float(finite[-1]) if finite.size else float("nan")
    else:
        result["best_reward"] = float("nan")
        result["final_reward"] = float("nan")
    return result


# --------------------------------------------------------------------------- #
# Stage 3: post-training policy evaluation (the action-distribution panel)    #
# --------------------------------------------------------------------------- #
def _import_module(path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)        # runs top-level imports, NOT main()
    return mod


def _action_labels(env_id: str, n_actions: int):
    if "CarRacing" in env_id:
        names = ["noop", "left", "right", "gas", "brake"]
        return names[:n_actions] + [str(i) for i in range(len(names), n_actions)]
    if "CartPole" in env_id:
        return ["left", "right"][:n_actions]
    return [str(i) for i in range(n_actions)]


def evaluate_policy(spec: AlgoSpec, cfg: BenchConfig, logdir: str) -> Optional[dict]:
    """Load the trained checkpoint and roll out greedily, recording the action
    histogram and mean return. Returns None if the algorithm did not checkpoint.

    This is the Task 8 diagnostic: reward alone can hide a degenerate single
    -action policy; the action histogram makes it visible.
    """
    if spec.checkpoint is None:
        print(f"[EVAL] {spec.label}: script saves no checkpoint -> "
              f"skipping action-distribution panel.")
        return None

    ckpt = os.path.join(logdir, spec.checkpoint)
    if not os.path.isfile(ckpt):
        print(f"[EVAL] {spec.label}: no checkpoint at {ckpt} -> skipping.")
        return None

    import torch
    import gymnasium as gym

    mod = _import_module(os.path.join(cfg.script_dir, spec.script), f"_mod_{spec.key}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild a single eval env with the SAME preprocessing this script trained on.
    def build_env():
        if spec.family == "a2c":
            if "CarRacing" in cfg.env:
                env = gym.make(cfg.env, continuous=False)
                if cfg.frame_skip > 1 and hasattr(mod, "FrameSkip"):
                    env = mod.FrameSkip(env, skip=cfg.frame_skip)
                env = mod.GrayFrameStack(env, k=4)
            else:
                env = gym.make(cfg.env)
        else:  # ppo family: raw env, no wrappers (matches ppo_from_scratch.make_env)
            env = gym.make(cfg.env, continuous=False) if "CarRacing" in cfg.env \
                else gym.make(cfg.env)
        return env

    env = build_env()
    obs_space = env.observation_space
    n_actions = env.action_space.n
    is_image = len(obs_space.shape) == 3

    # Rebuild the matching network from this script's own classes, load weights.
    if spec.family == "a2c":
        if is_image:
            c, h, w = obs_space.shape
            net = mod.CNNActorCritic(c, n_actions, sample_hw=(h, w))
        else:
            net = mod.MLPActorCritic(obs_space.shape[0], n_actions)
    else:  # ppo
        if is_image:
            net = mod.CNNActorCritic(n_actions)
        else:
            net = mod.MLPActorCritic(obs_space.shape[0], n_actions)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.to(device).eval()

    def to_tensor(obs):
        t = torch.as_tensor(np.asarray(obs), device=device, dtype=torch.float32)
        if spec.family == "a2c" and is_image:
            t = t / 255.0            # a2c normalises in to_tensor; ppo does it in-net
        return t.unsqueeze(0)

    counts = np.zeros(n_actions, dtype=np.int64)
    returns = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < 2000:
            with torch.no_grad():
                logits, _ = net.forward(to_tensor(obs))
                action = int(torch.argmax(logits, dim=-1).item())   # greedy
            counts[action] += 1
            obs, reward, term, trunc, _ = env.step(action)
            ep_ret += reward
            done = bool(term or trunc)
            steps += 1
        returns.append(ep_ret)
    env.close()

    fracs = counts / max(counts.sum(), 1)
    labels = _action_labels(cfg.env, n_actions)
    dominant = int(np.argmax(fracs))
    print(f"[EVAL] {spec.label}: mean return {np.mean(returns):.1f} over "
          f"{cfg.eval_episodes} eps | action mix "
          + ", ".join(f"{labels[i]}={fracs[i]*100:.0f}%" for i in range(n_actions)))
    return {
        "labels": labels,
        "fracs": fracs,
        "mean_return": float(np.mean(returns)),
        "dominant": labels[dominant],
        "dominant_frac": float(fracs[dominant]),
    }


# --------------------------------------------------------------------------- #
# Stage 4: plotting                                                           #
# --------------------------------------------------------------------------- #
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    acc = values[0]
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = acc
            continue
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def plot_reward_curves(results: List[dict], cfg: BenchConfig, path: str):
    plt.figure(figsize=(9, 5.5))
    plotted = False
    for i, r in enumerate(results):
        if r["reward"] is None:
            continue
        steps, vals = r["reward"]
        color = _COLORS[i % len(_COLORS)]
        plt.plot(steps, vals, color=color, alpha=0.18, linewidth=1)
        plt.plot(steps, _ema(vals), color=color, linewidth=2.2,
                 label=r["spec"].label)
        plotted = True
    if not plotted:
        plt.close()
        return None
    plt.xlabel("environment timesteps")
    plt.ylabel("mean episodic return")
    plt.title(f"Reward curves on {cfg.env} (smoothed)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_loss_curves(results: List[dict], cfg: BenchConfig, path: str):
    panels = [("policy_loss", "policy loss"),
              ("value_loss", "value loss"),
              ("entropy", "policy entropy")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    any_data = False
    for ax, (key, title) in zip(axes, panels):
        for i, r in enumerate(results):
            series = r.get(key)
            if series is None:
                continue
            steps, vals = series
            ax.plot(steps, _ema(vals, 0.2), color=_COLORS[i % len(_COLORS)],
                    linewidth=1.8, label=r["spec"].label)
            any_data = True
        ax.set_title(title)
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Training diagnostics on {cfg.env}", y=1.02)
    fig.tight_layout()
    if not any_data:
        plt.close(fig)
        return None
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_action_distribution(evals: dict, cfg: BenchConfig, path: str):
    """Grouped bar chart of greedy action frequency per algorithm -- the policy
    plot. A bar that fills one action = the single-action collapse from Task 8."""
    items = [(k, v) for k, v in evals.items() if v is not None]
    if not items:
        return None
    # Union of action labels (all share the env's action set).
    labels = items[0][1]["labels"]
    n_actions = len(labels)
    x = np.arange(n_actions)
    width = 0.8 / len(items)

    plt.figure(figsize=(9, 5))
    for j, (key, ev) in enumerate(items):
        spec_label = ev.get("_label", key)
        plt.bar(x + j * width, ev["fracs"], width,
                color=_COLORS[j % len(_COLORS)], label=spec_label)
    plt.xticks(x + width * (len(items) - 1) / 2, labels)
    plt.ylabel("greedy action frequency")
    plt.ylim(0, 1)
    plt.title(f"Policy action distribution on {cfg.env} (post-training, greedy)")
    plt.legend(loc="best")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return path


# --------------------------------------------------------------------------- #
# Stage 5: summary table                                                      #
# --------------------------------------------------------------------------- #
def write_summary(results: List[dict], evals: dict, cfg: BenchConfig, outdir: str):
    rows = []
    for r in results:
        spec = r["spec"]
        ev = evals.get(spec.key)
        rows.append({
            "algorithm": spec.label,
            "best_mean_reward": r["best_reward"],
            "final_mean_reward": r["final_reward"],
            "eval_mean_return": ev["mean_return"] if ev else float("nan"),
            "dominant_action": f"{ev['dominant']} ({ev['dominant_frac']*100:.0f}%)"
            if ev else "n/a",
        })

    # CSV
    csv_path = os.path.join(outdir, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write("algorithm,best_mean_reward,final_mean_reward,"
                "eval_mean_return,dominant_action\n")
        for row in rows:
            f.write(f"{row['algorithm']},{row['best_mean_reward']:.2f},"
                    f"{row['final_mean_reward']:.2f},"
                    f"{row['eval_mean_return']:.2f},{row['dominant_action']}\n")

    # Markdown
    md_path = os.path.join(outdir, "results_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# RL benchmark on {cfg.env}\n\n")
        f.write(f"- timesteps per algorithm: {cfg.timesteps:,}\n")
        f.write(f"- eval episodes (greedy): {cfg.eval_episodes}\n\n")
        f.write("| Algorithm | Best mean reward | Final mean reward | "
                "Eval mean return | Dominant action |\n")
        f.write("|---|---|---|---|---|\n")
        for row in rows:
            f.write(f"| {row['algorithm']} | {row['best_mean_reward']:.2f} | "
                    f"{row['final_mean_reward']:.2f} | "
                    f"{row['eval_mean_return']:.2f} | {row['dominant_action']} |\n")
    return md_path, csv_path


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="One-click benchmark of my from-scratch RL algorithms.")
    ap.add_argument("--algos", type=str, default="ppo,a2c_t7,a2c_v2",
                    help="comma list of algorithm keys to run "
                         "(choices: ppo, a2c_t7, a2c_v2).")
    ap.add_argument("--env", type=str, default="CarRacing-v3")
    ap.add_argument("--timesteps", type=int, default=1_000_000,
                    help="training budget per algorithm (env frames).")
    ap.add_argument("--num-envs", type=int, default=8, help="A2C parallel envs.")
    ap.add_argument("--n-steps", type=int, default=16, help="A2C rollout length.")
    ap.add_argument("--frame-skip", type=int, default=None,
                    help="A2C v2 action-repeat. Default: 4 on CarRacing, 1 else.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="benchmark_out")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing logs in --outdir; only parse/eval/plot.")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="python executable used for the subprocess runs.")
    args = ap.parse_args()

    frame_skip = args.frame_skip
    if frame_skip is None:
        frame_skip = 4 if "CarRacing" in args.env else 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    cfg = BenchConfig(
        env=args.env, timesteps=args.timesteps, num_envs=args.num_envs,
        n_steps=args.n_steps, frame_skip=frame_skip, seed=args.seed,
        log_interval=args.log_interval, eval_episodes=args.eval_episodes,
        outdir=os.path.abspath(args.outdir), script_dir=script_dir,
        python=args.python,
    )

    registry = {a.key: a for a in build_registry()}
    selected = []
    for key in [k.strip() for k in args.algos.split(",") if k.strip()]:
        if key not in registry:
            print(f"[WARN] unknown algorithm '{key}' -- skipping. "
                  f"Known: {list(registry)}")
            continue
        selected.append(registry[key])
    if not selected:
        print("No valid algorithms selected. Exiting.")
        return

    print(f"Benchmark config: env={cfg.env}  timesteps={cfg.timesteps:,}  "
          f"frame_skip={cfg.frame_skip}  algos={[s.key for s in selected]}")

    # --- train (sequential: ReadWriteOnce-correct) ---
    logdirs = {}
    for spec in selected:
        if args.skip_train:
            logdirs[spec.key] = os.path.abspath(
                os.path.join(cfg.outdir, "runs", spec.key))
        else:
            logdirs[spec.key] = run_training(spec, cfg)

    # --- parse + eval ---
    results, evals = [], {}
    for spec in selected:
        results.append(parse_run(spec, logdirs[spec.key]))
        ev = evaluate_policy(spec, cfg, logdirs[spec.key])
        if ev is not None:
            ev["_label"] = spec.label
        evals[spec.key] = ev

    # --- plot ---
    print("\n[PLOT] writing figures ...")
    p_reward = plot_reward_curves(results, cfg, os.path.join(plots_dir, "reward_curves.png"))
    p_loss = plot_loss_curves(results, cfg, os.path.join(plots_dir, "loss_curves.png"))
    p_act = plot_action_distribution(evals, cfg, os.path.join(plots_dir, "action_distribution.png"))
    md, csv = write_summary(results, evals, cfg, cfg.outdir)

    print("\n" + "=" * 78)
    print("BENCHMARK COMPLETE")
    for p in (p_reward, p_loss, p_act):
        if p:
            print("  figure:", p)
    print("  summary:", md)
    print("  summary:", csv)
    print("=" * 78)
    # Echo the table to stdout so it shows up in `kubectl logs`.
    with open(md) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
