"""
a2c_from_scratch.py

A from-scratch implementation of **A2C** which is the synchronous, deterministic
variant of A3C (Mnih et al. 2016, "Asynchronous Methods for Deep RL",
https://arxiv.org/abs/1602.01783) applied to CarRacing-v3 (discrete mode).

Features:
  * No clipped surrogate objective, no probability ratio, no "old" log-probs.
    A2C uses the plain policy-gradient term  -(advantage * log_prob).
  * ONE gradient update per rollout. Collect a short rollout from all envs -> compute one loss -> one optimizer
    step. That single-update-per-batch behaviour is the defining A2C trait.
  * The paper's key idea is *parallel actors*: several environments stepped in
    lockstep so each batch is decorrelated. We run N envs synchronously and
    batch their transitions together for the update.
  * n-step bootstrapped returns (the A3C paper's estimator) rather than GAE.

Run locally (sanity check on CartPole, optional):
    python a2c_from_scratch.py --env CartPole-v1 --total-timesteps 200000 \
        --num-envs 8 --n-steps 5

Run on Nautilus (CarRacing), pointing logs/checkpoints at the PVC:
    PYTHONUNBUFFERED=1 PYTHONPATH=/pvcvolume/python-packages \
    python a2c_from_scratch.py --env CarRacing-v3 \
        --total-timesteps 1000000 --num-envs 8 --n-steps 16 \
        --logdir /pvcvolume/runs/a2c_carracing

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


def make_thunk(env_id, idx, seed, frame_stack):
    def thunk():
        if "CarRacing" in env_id:
            env = gym.make(env_id, continuous=False)  # Discrete(5)
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
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
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
        make_thunk(args.env, i, args.seed, args.frame_stack)
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

    batch_per_update = args.num_envs * args.n_steps
    num_updates = args.total_timesteps // batch_per_update

    obs = envs.reset(seed=args.seed)
    obs_t = to_tensor(obs)
    global_step = 0
    start_time = time.time()
    next_save = args.save_freq

    for update in range(1, num_updates + 1):
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

            # episode tracking
            ep_returns += reward
            ep_lengths += 1
            for i in range(args.num_envs):
                if done[i]:
                    return_window.append(ep_returns[i])
                    return_window10.append(ep_returns[i])
                    length_window.append(ep_lengths[i])
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

            global_step += args.num_envs
            obs_t = to_tensor(next_obs)

        '''n-step bootstrapped returns
        - we mask on done = terminated OR truncated, i.e. we do not bootstrap across an episode boundary.
        This is the common A2C simplification; truncated (time-limit) episodes therefore get nobootstrap,
        which costs a little but keeps the logic clean.'''
        with torch.no_grad():
            next_value = net.get_value(obs_t)  # V(s_T), (num_envs,)
        returns = [None] * args.n_steps
        R = next_value
        for t in reversed(range(args.n_steps)):
            R = rew_storage[t] + args.gamma * R * mask_storage[t]
            returns[t] = R
        returns = torch.stack(returns)              # (n_steps, num_envs)
        values = torch.stack(val_storage)           # (n_steps, num_envs)
        advantages = returns - values               # both detached here

        #single A2C update on the whole batch
        b_obs = torch.cat(obs_storage, dim=0)
        b_actions = torch.cat(act_storage, dim=0)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)

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
            writer.add_scalar("train/learning_rate", args.lr, global_step)
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
