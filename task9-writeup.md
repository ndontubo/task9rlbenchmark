# Nautilus Onboarding — Task 9 Writeup (RL Testing / Benchmark Framework)

GitHub: https://github.com/ndontubo/task9rlbenchmark

## What this task was

Task 9 was to build a single-file, "one-click" framework that benchmarks all of my
RL algorithms on the same environment in one run, and that automatically produces
graphs of the results (reward, loss, and policy). Unlike Tasks 6–8, this task is not
another algorithm — it is the comparison *infrastructure* that runs the algorithms I
already wrote and puts their results side by side. The deliverable is this one-page
writeup, the GitHub repo, and the plots.

## Creating the framework

The framework is `rl_benchmark.py`, a single file that wraps — rather than
reimplements — the three validated agents from earlier tasks:

| Key | Wrapped script | Algorithm |
|---|---|---|
| `ppo` | `ppo_from_scratch.py` | Task 6 PPO (clipped surrogate, GAE, multi-epoch) |
| `a2c_t7` | `a2c_from_scratch.py` | Task 7 A2C (n-step returns, single update) |
| `a2c_v2` | `a2c_v2.py` | Task 8 A2C v2 (+ frame-skip, GAE, LR decay) |

The central design decision was to run each algorithm as its own **subprocess,
sequentially**, instead of importing all three into one program. Two reasons. First,
the three scripts each define their own `CNNActorCritic`/`MLPActorCritic` classes and
use different env preprocessing, so they cannot share one namespace without rewriting
code I had already validated. Subprocesses run my real code unchanged and isolate RNG
and CUDA state between runs. Second, running them sequentially is the only thing that
works on a `ReadWriteOnce` PVC, which is mounted by one pod at a time — so the task's
"benchmark several algorithms at the same time" means *within one harness
invocation*, not literally concurrent GPU pods. This is the same constraint I hit in
Task 7.

The harness then does three things automatically. It **parses** each run's
TensorBoard event files into a unified schema (the scripts log different tag names —
PPO's `rollout/ep_reward_mean_10` vs A2C's `rollout/ep_rew_mean` — so the framework
maps candidate keys to one metric; all three count steps in env frames, so the x-axis
is comparable). It **evaluates** each saved checkpoint greedily and records the
action distribution. And it **plots**: `reward_curves.png` (all algos overlaid),
`loss_curves.png` (policy loss, value loss, entropy), and `action_distribution.png`,
plus a `results_summary` table.

The action-distribution panel is the Task 8 lesson built in as a default: a flat
reward curve there hid an all-GAS single-action collapse that only the action
histogram revealed, so the framework always renders it. Adding a new algorithm later
(e.g. DQN, to make this a cross-family on-policy-vs-off-policy benchmark) is a single
`AlgoSpec` entry — the parse/eval/plot stages are algorithm-agnostic.

## What is working so far

The full pipeline is **implemented and validated end-to-end on CartPole-v1** (a fast
CPU sanity check, the same "verify on CartPole first" discipline from Task 6). A
12,000-timestep run trained all three algorithms, parsed their logs, evaluated their
policies, and produced all three figures with no manual steps:

| Algorithm | Best mean reward | Final mean reward | Eval mean return | Dominant action |
|---|---|---|---|---|
| PPO (Task 6) | 92.80 | 81.00 | n/a (no checkpoint) | n/a |
| A2C (Task 7) | 35.30 | 35.30 | 82.90 | left (53%) |
| A2C v2 (Task 8) | 33.30 | 18.10 | 9.20 | right (99%) |

Two things to read from this. PPO climbs fastest, as expected for the most
sample-efficient of the three. More importantly, the **action-distribution panel
immediately did its job**: on this short run A2C v2 collapsed onto a single action
(right, 99%) while Task 7 A2C stayed balanced (53/47) — the exact single-action
collapse signature I diagnosed by hand in Task 8, now caught automatically by the
framework. (PPO is absent from that panel because `ppo_from_scratch.py` saves no
checkpoint; one `torch.save` would add it.)

This CartPole run is a **pipeline validation, not a performance verdict** — the
numbers are from 12k CPU timesteps, and v2's hyperparameters are tuned for CarRacing,
so its weak CartPole showing is expected and not meaningful as a comparison.

## What the CarRacing benchmark is set up to show (pending run)

The full `CarRacing-v3` benchmark on Nautilus (1,000,000 timesteps × 3 algorithms,
sequential, one GPU) is **implemented but not yet run**. It is my immediate next
step but due to time constraints I want to first put this out. Based on the prior-task 
results, the benchmark should reproduce, on one set of axes: Task 5/6 PPO 
reaching positive reward with a genuine steering+gas action mix;
Task 7 A2C showing its peak-then-collapse (~+28 then back to ~−35); and Task 8 A2C v2
testing, at scale, whether frame-skip prevents the all-gas action collapse. The
framework exists precisely to make those three stories directly comparable.

## What I learned so far

The main lesson was that a benchmark's real engineering content is *fairness and
unification*, not the training itself. The three scripts log different metric names,
checkpoint differently, and preprocess observations differently (PPO uses raw RGB
single frames; both A2C variants use grayscale 4-frame stacks, and v2 adds
action-repeat). The honest framing — which I state in the repo and will state at the
meeting — is that this benchmarks the algorithms *as I implemented them in Tasks 6–8*,
an apples-to-apples comparison of my actual agents rather than a controlled
single-variable ablation. The second lesson reinforced Task 8: a benchmark that only
plotted reward would be misleading, so the action-distribution panel is a first-class
output, not an afterthought.

## Artifacts produced

- `rl_benchmark.py` — the single-file, one-click benchmark framework
- `ndontubo-task9-job.yml` — the Nautilus Job that runs the benchmark
- `ndontubo-task9-manifests.yml` — PVC + installer + shell pod (Task 7/8 pattern)
- `README.md` — local and Nautilus run instructions
- `example_cartpole_output/` — the validated CartPole figures and summary table

## Status / next step

Done: framework written, validated end-to-end on CartPole, Nautilus manifests and run
sequence prepared. Next: run the full CarRacing-v3 benchmark on Nautilus, drop the
resulting `reward_curves.png` / `loss_curves.png` / `action_distribution.png` into
this writeup, and push everything to the repo.
