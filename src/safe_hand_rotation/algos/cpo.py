"""
Constrained Policy Optimization (CPO).

Extends PPO with hard cost constraints using a Lagrangian approach.
Instead of fixed penalty weights, CPO automatically adjusts a Lagrange
multiplier to enforce: "average episode cost must stay below cost_limit."

When costs are high → multiplier increases → policy penalized more for unsafe actions.
When costs are low  → multiplier decreases → policy free to pursue reward.

This makes the constraint adaptive, unlike PPO's fixed reward penalties.

Reference:
    Achiam et al. "Constrained Policy Optimization." ICML 2017.
    Stooke et al. "Responsive Safety in Reinforcement Learning." 2020.
      (Lagrangian relaxation variant used here for practical simplicity)
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class CPO(PPO):
    """CPO extends PPO with a cost critic and Lagrangian constraint enforcement.

    Key additions over PPO:
      - A cost critic network that estimates future costs (like the reward critic)
      - A Lagrange multiplier that auto-tunes the cost penalty weight
      - Cost signals flowing from the environment through extras["costs"]

    The Lagrangian approach converts the constrained problem:
        maximize reward  subject to  cost <= cost_limit
    into an unconstrained problem:
        maximize reward - lambda * cost
    where lambda (the Lagrange multiplier) is adjusted to enforce the constraint.
    """

    cost_critic: MLPModel
    """Network that estimates future costs, analogous to the reward critic."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        cost_critic: MLPModel,
        storage: RolloutStorage,
        cost_limit: float = 0.1,
        lagrange_lr: float = 0.01,
        initial_lagrange: float = 0.0,
        **kwargs,
    ):
        """Initialize CPO.

        Args:
            actor: Policy network (same as PPO).
            critic: Reward value network (same as PPO).
            cost_critic: Cost value network (estimates future costs).
            storage: Rollout storage for transitions.
            cost_limit: Maximum allowed average cost per episode.
            lagrange_lr: Learning rate for the Lagrange multiplier.
            initial_lagrange: Starting value for the Lagrange multiplier.
            **kwargs: Remaining PPO arguments (learning_rate, gamma, etc.)
        """
        # Initialize PPO (actor, critic, storage, optimizer, etc.)
        super().__init__(actor, critic, storage, **kwargs)

        # --- CPO-specific setup ---
        self.cost_critic = cost_critic
        self._raw_cost_critic = cost_critic  # Keep reference for save/load
        self.cost_limit = cost_limit

        # Lagrange multiplier — starts at initial_lagrange, auto-adjusts during training.
        # Higher lambda = stronger cost penalty = safer but slower-learning policy.
        self.lagrange_multiplier = initial_lagrange
        self.lagrange_lr = lagrange_lr

        # Optimizer for the cost critic (separate from actor/reward-critic optimizer)
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(),
            lr=kwargs.get("learning_rate", 1e-3),
        )

        # Cost storage — parallel arrays to what PPO stores for rewards.
        # Shape: [num_steps_per_rollout, num_envs, 1]
        num_steps = storage.num_transitions_per_env
        num_envs = storage.num_envs
        self.cost_values = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.costs = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.cost_returns = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.cost_advantages = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.cost_dones = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self._cost_step = 0

        # Running average of episode cost for Lagrange updates
        self.mean_episode_cost = 0.0

    # ------------------------------------------------------------------
    # Rollout phase: collect experience
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data.

        Extends PPO's act() to also compute cost value estimates.
        """
        # PPO: compute actions, reward values, log probs
        actions = super().act(obs)

        # CPO: also compute cost value estimate for current observation
        cost_value = self.cost_critic(obs).detach()
        self.cost_values[self._cost_step] = cost_value

        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one environment step.

        Extends PPO's process_env_step to:
        1. Extract costs from extras["costs"]
        2. Subtract Lagrangian penalty from rewards before PPO stores them
        3. Store raw costs for cost critic training
        """
        # Get costs from the environment (default to zero if not provided)
        raw_costs = extras.get("costs", torch.zeros_like(rewards))

        # Penalize rewards with Lagrangian cost term.
        # This is the key CPO mechanism: rewards become (reward - lambda * cost).
        # PPO then optimizes this penalized reward, which implicitly enforces
        # the cost constraint because lambda adjusts to make costs stay in budget.
        penalized_rewards = rewards - self.lagrange_multiplier * raw_costs

        # Let PPO handle everything with the penalized rewards
        super().process_env_step(obs, penalized_rewards, dones, extras)

        # Store raw costs and dones for cost critic training
        self.costs[self._cost_step] = raw_costs.unsqueeze(-1) if raw_costs.dim() == 1 else raw_costs
        self.cost_dones[self._cost_step] = dones.unsqueeze(-1).float() if dones.dim() == 1 else dones.float()
        self._cost_step += 1

    # ------------------------------------------------------------------
    # Return computation: GAE for both rewards and costs
    # ------------------------------------------------------------------

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets for rewards AND costs.

        PPO only computes reward returns. CPO also computes cost returns
        using the same GAE (Generalized Advantage Estimation) algorithm,
        then updates the Lagrange multiplier based on mean episode cost.
        """
        # PPO: compute reward returns and advantages (using penalized rewards)
        super().compute_returns(obs)

        # --- CPO: compute cost returns and advantages ---
        # Same GAE algorithm as rewards, just applied to costs instead.
        last_cost_values = self.cost_critic(obs).detach()
        cost_advantage = torch.zeros_like(last_cost_values)

        for step in reversed(range(self._cost_step)):
            if step == self._cost_step - 1:
                next_cost_values = last_cost_values
            else:
                next_cost_values = self.cost_values[step + 1]

            next_is_not_terminal = 1.0 - self.cost_dones[step]

            # TD error for costs: c_t + gamma * Vc(s_{t+1}) - Vc(s_t)
            delta = (
                self.costs[step]
                + next_is_not_terminal * self.gamma * next_cost_values
                - self.cost_values[step]
            )

            # GAE: accumulate discounted advantages
            cost_advantage = (
                delta + next_is_not_terminal * self.gamma * self.lam * cost_advantage
            )

            # Cost return = cost advantage + cost value
            self.cost_returns[step] = cost_advantage + self.cost_values[step]
            self.cost_advantages[step] = cost_advantage

        # --- Update Lagrange multiplier ---
        # Compute mean cost over the rollout
        self.mean_episode_cost = self.costs[: self._cost_step].mean().item()

        # Dual gradient ascent: increase lambda if over budget, decrease if under.
        # max(0, ...) ensures lambda never goes negative (can't reward costs).
        self.lagrange_multiplier = max(
            0.0,
            self.lagrange_multiplier
            + self.lagrange_lr * (self.mean_episode_cost - self.cost_limit),
        )

        # Reset cost step counter for next rollout
        self._cost_step = 0

    # ------------------------------------------------------------------
    # Update phase: train networks
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run PPO update on penalized rewards, then log CPO metrics."""
        # PPO update: trains actor and reward critic on penalized rewards.
        losses = super().update()

        # Log CPO-specific metrics
        losses["mean_cost"] = self.mean_episode_cost
        losses["lagrange_multiplier"] = self.lagrange_multiplier
        losses["cost_limit"] = self.cost_limit

        return losses
    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        """Set training mode for all networks."""
        super().train_mode()
        self.cost_critic.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for all networks."""
        super().eval_mode()
        self.cost_critic.eval()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self) -> dict:
        """Return a dict of all model states for checkpointing."""
        saved = super().save()
        saved["cost_critic_state_dict"] = self._raw_cost_critic.state_dict()
        saved["cost_critic_optimizer_state_dict"] = (
            self.cost_critic_optimizer.state_dict()
        )
        saved["lagrange_multiplier"] = self.lagrange_multiplier
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load model states from a checkpoint."""
        result = super().load(loaded_dict, load_cfg, strict)

        if "cost_critic_state_dict" in loaded_dict:
            self._raw_cost_critic.load_state_dict(
                loaded_dict["cost_critic_state_dict"], strict=strict
            )
        if "cost_critic_optimizer_state_dict" in loaded_dict:
            self.cost_critic_optimizer.load_state_dict(
                loaded_dict["cost_critic_optimizer_state_dict"]
            )
        if "lagrange_multiplier" in loaded_dict:
            self.lagrange_multiplier = loaded_dict["lagrange_multiplier"]

        return result

    # ------------------------------------------------------------------
    # Algorithm construction (called by the runner)
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "CPO":
        """Construct CPO algorithm with actor, reward critic, and cost critic.

        Mirrors PPO.construct_algorithm but adds a cost critic network
        with the same architecture as the reward critic.
        """
        # Resolve class callables
        alg_class: type[CPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Create actor (policy network)
        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")

        # Share CNN encoders if requested
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns

        # Create reward critic (value network for rewards)
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")

        # Create cost critic (value network for costs) — same architecture as reward critic
        # We use deepcopy of the critic config so they have the same structure
        # but independent weights
        cost_critic_cfg = deepcopy(cfg["critic"])
        cost_critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cost_critic_cfg
        ).to(device)
        print(f"Cost Critic Model: {cost_critic}")

        # Create rollout storage
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )

        # Extract CPO-specific params from algorithm config
        cost_limit = cfg["algorithm"].pop("cost_limit", 0.1)
        lagrange_lr = cfg["algorithm"].pop("lagrange_lr", 0.01)
        initial_lagrange = cfg["algorithm"].pop("initial_lagrange", 0.0)

        # Create the CPO algorithm
        alg: CPO = alg_class(
            actor,
            critic,
            cost_critic,
            storage,
            cost_limit=cost_limit,
            lagrange_lr=lagrange_lr,
            initial_lagrange=initial_lagrange,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # Compile models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg
