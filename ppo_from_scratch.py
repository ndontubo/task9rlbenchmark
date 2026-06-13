"""
PPO from scratch — Task 6 implementation.

This file implements Proximal Policy Optimization (Schulman et al., 2017)
using only PyTorch for autodiff/networks and Gymnasium for envs.
All RL-specific logic (rollout collection, GAE, clipped surrogate objective,
value loss, entropy bonus, training loop) is written from scratch.

Usage:
    python ppo_from_scratch.py --env CartPole-v1 --timesteps 100000
    python ppo_from_scratch.py --env CarRacing-v3 --timesteps 500000

Run CartPole first to verify the algorithm worksThen try Car Racing.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

# Networks

class MLPActorCritic(nn.Module):
    """Actor-critic network for vector observations (e.g. CartPole)."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        # Shared trunk: extract features from the state
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Two heads: one for action logits, one for value
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        features = self.shared(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs, action=None):
        """Returns (action, log_prob, entropy, value).

        If action is None, samples a new action from the policy.
        If action is provided, returns log_prob and entropy for that action.
        """
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class CNNActorCritic(nn.Module):
    """Actor-critic network for image observations (e.g. Car Racing).

    Input expected as [B, H, W, C] uint8 (Gymnasium's default for image envs).
    Internally permutes to [B, C, H, W] and normalizes to [0, 1].
    """

    def __init__(self, n_actions, n_input_channels=3):
        super().__init__()
        # Nature DQN-style CNN: 3 conv layers + FC.
        # Input is 96x96 (Car Racing default).
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),  # 96 -> 23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),                 # 23 -> 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),                 # 10 -> 8
            nn.ReLU(),
            nn.Flatten(),
        )
        # After conv: 64 channels * 8 * 8 = 4096
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(512, n_actions)
        self.critic_head = nn.Linear(512, 1)

    def _preprocess(self, obs):
        # Accept [B, H, W, C] or [H, W, C]
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] == 3:
            obs = obs.permute(0, 3, 1, 2)
        obs = obs.float() / 255.0
        return obs

    def forward(self, obs):
        obs = self._preprocess(obs)
        features = self.cnn(obs)
        features = self.fc(features)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs, action=None):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# Rollout buffer

class RolloutBuffer:
    """Stores experience collected over n_steps and computes GAE advantages.

    The buffer holds, for each step t in the rollout:
      obs[t]      : observation at step t
      actions[t]  : action taken at step t
      log_probs[t]: log probability of that action under the policy at time of collection
      rewards[t]  : reward received after taking action at step t
      values[t]   : value estimate V(s_t) at time of collection
      dones[t]    : 1 if the episode ended after step t (s_{t+1} is terminal)

    After collection, compute_returns_and_advantages() fills in:
      advantages[t]: GAE-lambda advantage estimate
      returns[t]   : target for value function update (advantage + value)
    """

    def __init__(self, n_steps, obs_shape, obs_dtype, device):
        self.n_steps = n_steps
        self.device = device
        self.obs = torch.zeros((n_steps,) + obs_shape, dtype=obs_dtype, device=device)
        self.actions = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """Compute GAE-lambda advantages and discounted returns.

        last_value: V(s_T), the value estimate at the state AFTER the last
                    recorded step. Used to bootstrap the advantage from the
                    last step if the rollout ended mid-episode.

        For each step t (working backwards):
          delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
          A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

        The (1 - done_t) factor zeros out the bootstrap whenever the
        episode ended after step t, since there's no s_{t+1} to value.
        """
        self.advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_v = last_value
            else:
                next_v = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_v * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, minibatch_size):
        """Yield shuffled minibatches for multi-epoch PPO updates."""
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)
        for start in range(0, self.n_steps, minibatch_size):
            end = start + minibatch_size
            batch_idx = indices[start:end]
            yield (
                self.obs[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
                self.values[batch_idx],
            )

    def reset(self):
        self.ptr = 0

# PPO agent

class PPO:
    """Proximal Policy Optimization (Schulman et al., 2017).

    Implements the clipped surrogate objective variant with:
      - GAE-lambda advantage estimation
      - Multiple epochs of minibatch SGD per rollout
      - Value loss + entropy bonus
      - Advantage normalization
      - Gradient clipping
    """

    def __init__(
        self,
        env,
        is_cnn=False,
        n_steps=2048,
        n_epochs=10,
        minibatch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        log_dir="./tb_logs",
        device="auto",
    ):
        self.env = env
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Using device: {device}")

        # Build network and figure out obs storage type
        if is_cnn:
            self.net = CNNActorCritic(env.action_space.n).to(device)
            obs_shape = env.observation_space.shape  # (96, 96, 3) for Car Racing
            obs_dtype = torch.uint8                  # store images as uint8 to save memory
        else:
            obs_dim = env.observation_space.shape[0]
            self.net = MLPActorCritic(obs_dim, env.action_space.n).to(device)
            obs_shape = (obs_dim,)
            obs_dtype = torch.float32

        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer(n_steps, obs_shape, obs_dtype, device)

        # Hyperparameters
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.is_cnn = is_cnn

        # Logging
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.start_time = time.time()

    def _obs_to_tensor(self, obs):
        """Convert a numpy obs to the right tensor dtype for the buffer."""
        if self.is_cnn:
            return torch.as_tensor(obs, device=self.device, dtype=torch.uint8)
        else:
            return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

    def _obs_to_net_input(self, obs):
        """Convert a buffer obs (uint8 for CNN, float for MLP) to net input."""
        if self.is_cnn:
            return obs.float()  # CNN's _preprocess will further normalize
        else:
            return obs

    def collect_rollouts(self):
        """Run policy in the env for n_steps and fill the buffer."""
        self.buffer.reset()
        obs, _ = self.env.reset()
        obs = self._obs_to_tensor(obs)

        ep_reward = 0.0
        ep_length = 0
        done_flag = False

        for step in range(self.n_steps):
            with torch.no_grad():
                action, log_prob, _, value = self.net.get_action_and_value(
                    self._obs_to_net_input(obs).unsqueeze(0)
                )
            action_int = action.item()

            next_obs, reward, terminated, truncated, _ = self.env.step(action_int)
            done_flag = terminated or truncated

            self.buffer.add(
                obs, action_int, log_prob.squeeze(0),
                reward, value.squeeze(0), float(done_flag),
            )

            ep_reward += reward
            ep_length += 1
            self.global_step += 1

            if done_flag:
                # Episode finished: log it and reset env
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.writer.add_scalar("rollout/ep_reward", ep_reward, self.global_step)
                self.writer.add_scalar("rollout/ep_length", ep_length, self.global_step)
                if len(self.episode_rewards) >= 10:
                    self.writer.add_scalar(
                        "rollout/ep_reward_mean_10",
                        float(np.mean(self.episode_rewards[-10:])),
                        self.global_step,
                    )
                if len(self.episode_rewards) >= 100:
                    self.writer.add_scalar(
                        "rollout/ep_reward_mean_100",
                        float(np.mean(self.episode_rewards[-100:])),
                        self.global_step,
                    )
                ep_reward = 0.0
                ep_length = 0
                next_obs, _ = self.env.reset()

            obs = self._obs_to_tensor(next_obs)

        # Bootstrap the advantage of the last step with V(s_T)
        with torch.no_grad():
            _, _, _, last_value = self.net.get_action_and_value(
                self._obs_to_net_input(obs).unsqueeze(0)
            )
        last_value = last_value.squeeze(0).item()

        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

    def update(self):
        """Run n_epochs of minibatch SGD on the collected rollout."""
        # Advantage normalization (per-rollout, standard PPO trick)
        adv = self.buffer.advantages
        self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        approx_kls = []

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_minibatches(self.minibatch_size):
                obs_b, actions_b, old_log_probs_b, advantages_b, returns_b, old_values_b = batch

                # Recompute log_probs, entropy, value for current network
                _, new_log_probs, entropy, new_values = self.net.get_action_and_value(
                    self._obs_to_net_input(obs_b), actions_b
                )

                # Clipped surrogate policy loss (the heart of PPO)
                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE between predicted value and computed return)
                value_loss = ((new_values - returns_b) ** 2).mean()

                # Entropy bonus to encourage exploration
                entropy_bonus = entropy.mean()

                # Combined loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_bonus.item())
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    approx_kl = (old_log_probs_b - new_log_probs).mean().item()
                clip_fractions.append(clip_fraction)
                approx_kls.append(approx_kl)

        # Log update metrics
        self.writer.add_scalar("train/policy_loss", float(np.mean(policy_losses)), self.global_step)
        self.writer.add_scalar("train/value_loss", float(np.mean(value_losses)), self.global_step)
        self.writer.add_scalar("train/entropy", float(np.mean(entropies)), self.global_step)
        self.writer.add_scalar("train/clip_fraction", float(np.mean(clip_fractions)), self.global_step)
        self.writer.add_scalar("train/approx_kl", float(np.mean(approx_kls)), self.global_step)

    def learn(self, total_timesteps):
        n_updates = total_timesteps // self.n_steps
        print(f"Training for {total_timesteps} timesteps ({n_updates} updates)")

        for update in range(1, n_updates + 1):
            self.collect_rollouts()
            self.update()

            # Periodic logging
            elapsed = time.time() - self.start_time
            fps = int(self.global_step / max(elapsed, 1e-6))
            self.writer.add_scalar("time/fps", fps, self.global_step)
            self.writer.add_scalar("time/iterations", update, self.global_step)

            if len(self.episode_rewards) > 0:
                window = min(10, len(self.episode_rewards))
                mean_r = float(np.mean(self.episode_rewards[-window:]))
                print(
                    f"[Update {update}/{n_updates}] step={self.global_step} "
                    f"fps={fps} ep_reward_mean_{window}={mean_r:.2f} "
                    f"episodes={len(self.episode_rewards)}"
                )

        self.writer.close()
        print(f"Training complete after {self.global_step} timesteps")
        if len(self.episode_rewards) > 0:
            print(f"Total episodes: {len(self.episode_rewards)}")
            print(f"Final 10-ep mean reward: {np.mean(self.episode_rewards[-10:]):.2f}")
            print(f"Final 100-ep mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")


# Main

def make_env(env_name):
    """Construct the env and return (env, is_cnn) flag for network selection."""
    if env_name == "CartPole-v1":
        env = gym.make("CartPole-v1")
        return env, False
    elif env_name == "CarRacing-v3":
        # Discrete mode: 5 actions (do nothing, left, right, gas, brake)
        env = gym.make("CarRacing-v3", continuous=False)
        return env, True
    else:
        raise ValueError(f"Unknown env: {env_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="CartPole-v1",
        choices=["CartPole-v1", "CarRacing-v3"],
        help="Which env to train on. Test on CartPole first.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--log_dir", type=str, default="./tb_logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env, is_cnn = make_env(args.env)
    print(f"Env: {args.env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Pick reasonable hyperparams per env. CNN training uses smaller n_steps
    # to keep memory manageable; MLP uses standard PPO defaults.
    if is_cnn:
        agent = PPO(
            env, is_cnn=True,
            n_steps=512, minibatch_size=64, n_epochs=4,
            learning_rate=2.5e-4, clip_range=0.1, ent_coef=0.01,
            log_dir=args.log_dir,
        )
    else:
        agent = PPO(
            env, is_cnn=False,
            n_steps=2048, minibatch_size=64, n_epochs=10,
            learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
            log_dir=args.log_dir,
        )

    agent.learn(args.timesteps)
    env.close()


if __name__ == "__main__":
    main()
