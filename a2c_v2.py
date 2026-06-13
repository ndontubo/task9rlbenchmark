"""
a2c_v2.py  (Task 8 v2 — fixing the ACTION COLLAPSE seen in a2c_refined.py)

A from-scratch implementation of **A2C** (synchronous variant of A3C; Mnih et
al. 2016, https://arxiv.org/abs/1602.01783) on CarRacing-v3 (discrete mode).
Still A2C: no clipped surrogate, no ratio, one update per rollout.

WHY v2 EXISTS. The v2-refined run (a2c_refined.py: GAE + LR-decay + adv-norm +
ent-coef 0.02) did NOT collapse the Task-7 way -- the curve was stable -- but it
exposed a DIFFERENT failure: the policy collapsed onto a SINGLE action. Rendering
the 200k/300k checkpoints (eval_render.py) showed the agent picking GAS 100% of
the time, greedily AND when sampling: it floors it in a straight line and drives
off the track. Reward parks around -40 purely from the time penalty. It never
learned to steer because (a) one frame of steering barely turns the car, so the
gradient rarely sees turning pay off, and (b) ent-coef 0.02 wasn't enough to keep
the other four actions alive once gas got a head start.

THREE CHANGES vs a2c_refined.py, all aimed at keeping steering learnable:

  1. FRAME-SKIP / action-repeat (k=4)  [NEW, the primary fix].
     Each chosen action is held for 4 sim frames (rewards summed). This is the
     standard thing that makes CarRacing learnable: it makes a single decision
     actually move the car enough that the reward consequence of steering vs
     gas is visible to the gradient. --frame-skip (set 1 to disable).

  2. Higher entropy coefficient (0.02 -> 0.05).
     Directly fights the single-action collapse by paying the policy to keep
     sampling all five actions. --ent-coef.

  3. Lower learning rate (7e-4 -> 3e-4).
     The high RMSprop LR accelerated the snowball onto one action; a gentler
     step lets the value function catch up before the policy sharpens. --lr.

RETAINED from a2c_refined.py: GAE-lambda advantages, linear LR decay (now from
3e-4), per-batch advantage normalisation, global grad-norm clipping (0.5),
parallel synchronous actors with serial stepping.

NOTE on step counting: global_step counts underlying ENV FRAMES (incremented by
num_envs*frame_skip per decision), so the x-axis stays comparable to v1 and the
baseline. With frame-skip=4, --total-timesteps 1_000_000 frames == 250k agent
decisions. Raise --total-timesteps if it needs longer to develop steering.

Run locally (sanity check on CartPole, optional):
    python a2c_refined.py --env CartPole-v1 --total-timesteps 200000 \
        --num-envs 8 --n-steps 5

Run on Nautilus (CarRacing):
    PYTHONUNBUFFERED=1 PYTHONPATH=/pvcvolume/python-packages \
    python a2c_v2.py --env CarRacing-v3 \
        --total-timesteps 1000000 --num-envs 8 --n-steps 16 --frame-skip 4 \
        --logdir /pvcvolume/runs/a2c_carracing_v2

Note on throughput: the vectorised env here steps serially (robust in a
container -- no multiprocessing / /dev/shm pitfalls), so for a fixed total
timestep budget the wall-clock is roughly the same as a single env. A2C is also
less sample-efficient than the PPO you used in Task 5/6, so expect the reward
curve to develop more slowly; raise --total-timesteps if the breakthrough
hasn't happened yet.
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


# --------------------------------------------------------------------------- #
# Environment preprocessing + a tiny serial vectorised env                     #
# --------------------------------------------------------------------------- #
class FrameSkip(gym.Wrapper):
    """Action-repeat: hold each chosen action for `skip` underlying frames and
    sum the rewards. This is the key fix for CarRacing action collapse -- with
    skip=1 a single steering decision barely moves the car, so the gradient
    rarely sees turning pay off and the policy collapses onto gas. Repeating the
    action makes each decision's consequence visible. Stops early if the episode
    ends mid-skip. Exposes the actual frame count in info['skip_frames']."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = None
        info = {}
        n = 0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            n += 1
            if terminated or truncated:
                break
        info["skip_frames"] = n
        return obs, total_reward, terminated, truncated, info


class GrayFrameStack(gym.Wrapper):
    """Convert CarRacing's 96x96x3 RGB observation to a stack of k grayscale
    frames, shape (k, H, W) uint8. Frame stacking gives the agent a sense of
    velocity/direction that a single frame cant convey"""

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        h, w = env.observation_space.shape[:2]
        self.frames = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, h, w), dtype=np.uint8
        )

    @staticmethod
    def _gray(obs):
        # Luminance weights; result is (H, W) uint8.
        return np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        g = self._gray(obs)
        for _ in range(self.k):
            self.frames.append(g)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._gray(obs))
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info


class SyncVecEnv:
    """Minimal synchronous vectorised env. Steps each sub-env in a loop and
    auto-resets on done (SB3 / classic-Gym convention: the terminal observation
    is stashed in info and the returned obs is the fresh reset obs). Serial, so
    no multiprocessing"""

    def __init__(self, thunks):
        self.envs = [t() for t in thunks]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs = []
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=None if seed is None else seed + i)
            obs.append(o)
        return np.stack(obs, axis=0)

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(int(a))
            done = bool(term or trunc)
            if done:
                info["terminal_observation"] = o
                info["was_terminated"] = bool(term)  # vs truncation
                o, _ = e.reset()
            obs.append(o)
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )

    def close(self):
        for e in self.envs:
            e.close()


def make_thunk(env_id, idx, seed, frame_stack, frame_skip):
    def thunk():
        if "CarRacing" in env_id:
            env = gym.make(env_id, continuous=False)  # Discrete(5)
            if frame_skip > 1:
                env = FrameSkip(env, skip=frame_skip)
            env = GrayFrameStack(env, k=frame_stack)
        else:
            env = gym.make(env_id)
        env.action_space.seed(seed + idx)
        return env

    return thunk


# --------------------------------------------------------------------------- #
# Actor-critic networks                                                        #
# --------------------------------------------------------------------------- #
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class CNNActorCritic(nn.Module):
    """Nature-DQN-style conv stack feeding a 512-unit dense layer, with separate
    policy and value heads. Used for image observations (CarRacing)."""

    def __init__(self, in_channels, n_actions, sample_hw=(96, 96)):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(
                torch.zeros(1, in_channels, *sample_hw)
            ).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n_flat, 512)), nn.ReLU())
        self.policy_head = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        h = self.fc(self.conv(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_value(self, x):
        return self.forward(x)[1]

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class MLPActorCritic(nn.Module):
    """Two 64-unit tanh layers with separate policy/value heads. Used for
    vector observations (e.g. CartPole)."""

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.policy_head = layer_init(nn.Linear(64, n_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_value(self, x):
        return self.forward(x)[1]

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CarRacing-v3")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=16,
                   help="rollout length per env before each update")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--frame-skip", type=int, default=4,
                   help="action-repeat: hold each action this many frames "
                        "(v2 fix for action collapse; set 1 to disable)")
    # v2: LR lowered 7e-4 -> 3e-4 to slow the snowball onto a single action.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="GAE lambda. 1.0 ~= high-variance n-step/MC; "
                        "lower trades a little bias for much less variance.")
    p.add_argument("--vf-coef", type=float, default=0.5)
    # v2: ent-coef raised 0.02 -> 0.05 to keep all five actions sampled and
    # prevent the single-action (all-GAS) collapse.
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # REFINED toggles (all default ON; flip to --no-* to reproduce Task 7).
    p.add_argument("--anneal-lr", action="store_true", default=True,
                   help="linearly decay LR to 0 over training")
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    p.add_argument("--norm-adv", action="store_true", default=True,
                   help="normalise advantages per batch (zero mean, unit std)")
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--save-freq", type=int, default=100_000,
                   help="save a checkpoint every this many timesteps")
    p.add_argument("--log-interval", type=int, default=10,
                   help="print/log every this many updates")
    return p.parse_args()


def main():
    args = parse_args()

    run_name = f"a2c_{args.env}_{int(time.time())}"
    logdir = args.logdir or os.path.join("runs", run_name)
    os.makedirs(logdir, exist_ok=True)
    model_dir = os.path.join(logdir, "models")
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Build the vectorised env.
    thunks = [
        make_thunk(args.env, i, args.seed, args.frame_stack, args.frame_skip)
        for i in range(args.num_envs)
    ]
    envs = SyncVecEnv(thunks)
    obs_space = envs.single_observation_space
    n_actions = envs.single_action_space.n
    is_image = len(obs_space.shape) == 3
    print(f"obs space: {obs_space.shape}  |  n_actions: {n_actions}  |  "
          f"image_obs: {is_image}", flush=True)

    # Select the network by observation type.
    if is_image:
        c, h, w = obs_space.shape
        net = CNNActorCritic(c, n_actions, sample_hw=(h, w)).to(device)
    else:
        net = MLPActorCritic(obs_space.shape[0], n_actions).to(device)

    optimizer = torch.optim.RMSprop(
        net.parameters(), lr=args.lr, alpha=0.99, eps=1e-5
    )

    def to_tensor(np_obs):
        t = torch.from_numpy(np.asarray(np_obs)).to(device).float()
        if is_image:
            t = t / 255.0
        return t

    # Episode bookkeeping (computed on the host, not via env wrappers).
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    return_window = deque(maxlen=100)
    length_window = deque(maxlen=100)
    return_window10 = deque(maxlen=10)

    # global_step counts underlying ENV FRAMES. With frame-skip, each agent
    # decision consumes `frame_skip` frames per env, so the budget is divided
    # accordingly -> --total-timesteps stays in frames and comparable to v1.
    frames_per_step = args.num_envs * max(1, args.frame_skip)
    batch_per_update = frames_per_step * args.n_steps
    num_updates = args.total_timesteps // batch_per_update

    obs = envs.reset(seed=args.seed)
    obs_t = to_tensor(obs)
    global_step = 0
    start_time = time.time()
    next_save = args.save_freq

    for update in range(1, num_updates + 1):
        # ---- REFINEMENT 2: linear LR decay to 0 over the run ----
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            current_lr = frac * args.lr
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
        else:
            current_lr = args.lr

        obs_storage, act_storage = [], []
        rew_storage, mask_storage, val_storage = [], [], []

        # ---- collect a rollout of n_steps from all envs in lockstep ----
        for _ in range(args.n_steps):
            with torch.no_grad():
                action, _, _, value = net.get_action_and_value(obs_t)
            next_obs, reward, done, _ = envs.step(action.cpu().numpy())

            obs_storage.append(obs_t)
            act_storage.append(action)
            val_storage.append(value)
            rew_storage.append(torch.as_tensor(reward, device=device))
            mask_storage.append(torch.as_tensor(1.0 - done, device=device))

            # episode tracking (lengths counted in env FRAMES for comparability)
            ep_returns += reward
            ep_lengths += max(1, args.frame_skip)
            for i in range(args.num_envs):
                if done[i]:
                    return_window.append(ep_returns[i])
                    return_window10.append(ep_returns[i])
                    length_window.append(ep_lengths[i])
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

            global_step += frames_per_step
            obs_t = to_tensor(next_obs)

        # ---- REFINEMENT 1: GAE-lambda advantages ----
        # Was: plain n-step bootstrapped returns, then advantages = returns - V.
        # GAE computes a lower-variance advantage by exponentially averaging
        # n-step TD errors (delta_t = r_t + gamma*V(s_{t+1})*mask - V(s_t)).
        # Same done-masking convention as before: mask_storage[t] = 1 - done_t,
        # so we never bootstrap across an episode boundary. Setting
        # gae_lambda = 1.0 recovers the high-variance (n-step/MC-like) estimate.
        with torch.no_grad():
            next_value = net.get_value(obs_t)  # V(s_T), (num_envs,)
        values = torch.stack(val_storage)       # (n_steps, num_envs), detached
        advantages = torch.zeros_like(values)
        last_gae = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.n_steps)):
            next_val = next_value if t == args.n_steps - 1 else values[t + 1]
            delta = (
                rew_storage[t]
                + args.gamma * next_val * mask_storage[t]
                - values[t]
            )
            last_gae = delta + args.gamma * args.gae_lambda * mask_storage[t] * last_gae
            advantages[t] = last_gae
        # Value targets are advantages + baseline (the standard GAE return target).
        returns = advantages + values

        #single A2C update on the whole batch
        b_obs = torch.cat(obs_storage, dim=0)
        b_actions = torch.cat(act_storage, dim=0)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # ---- REFINEMENT 3a: normalise advantages (zero mean, unit std) ----
        # Keeps the policy-gradient step a consistent magnitude across rollouts.
        # Without this, large raw CarRacing advantages produced occasional huge
        # updates that saturated the logits and crushed entropy to ~0 early.
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

        _, new_logp, entropy, new_values = net.get_action_and_value(
            b_obs, b_actions
        )
        policy_loss = -(b_advantages * new_logp).mean()
        value_loss = F.mse_loss(new_values, b_returns)
        entropy_bonus = entropy.mean()
        loss = (
            policy_loss
            + args.vf_coef * value_loss
            - args.ent_coef * entropy_bonus
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        optimizer.step()

        # ---- logging ----
        if update % args.log_interval == 0:
            fps = int(global_step / (time.time() - start_time))
            mean_ret = float(np.mean(return_window)) if return_window else float("nan")
            mean_ret10 = float(np.mean(return_window10)) if return_window10 else float("nan")
            mean_len = float(np.mean(length_window)) if length_window else float("nan")

            writer.add_scalar("rollout/ep_rew_mean", mean_ret, global_step)
            writer.add_scalar("rollout/ep_rew_mean_10", mean_ret10, global_step)
            writer.add_scalar("rollout/ep_len_mean", mean_len, global_step)
            writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("train/value_loss", value_loss.item(), global_step)
            writer.add_scalar("train/entropy", entropy_bonus.item(), global_step)
            writer.add_scalar("train/learning_rate", current_lr, global_step)
            writer.add_scalar("time/fps", fps, global_step)

            print(
                f"update {update}/{num_updates}  step {global_step}  "
                f"ep_rew_mean {mean_ret:8.2f}  ep_len_mean {mean_len:6.1f}  "
                f"fps {fps}",
                flush=True,
            )

        # ---- checkpoint ----
        if global_step >= next_save:
            ckpt = os.path.join(model_dir, f"a2c_{global_step}.pt")
            torch.save(net.state_dict(), ckpt)
            print(f"saved checkpoint: {ckpt}", flush=True)
            next_save += args.save_freq

    # final save
    final = os.path.join(model_dir, "a2c_final.pt")
    torch.save(net.state_dict(), final)
    print(f"training complete. final model: {final}", flush=True)
    writer.close()
    envs.close()


if __name__ == "__main__":
    main()
