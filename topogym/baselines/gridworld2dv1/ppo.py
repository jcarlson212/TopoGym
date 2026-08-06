"""PPO from Ray RLlib.

The algorithm is RLlib's -- TopoGym does not reimplement PPO. What
lives here is how PPO meets the protocol in
:mod:`topogym.baselines.gridworld2dv1.protocol`: hyperparameters chosen on ``tune``,
gradients taken on ``train``, stopping decided on ``val``.

Variants subclass rather than copy. An intrinsic-reward method such as
RND or ICM is PPO plus a bonus, so it overrides
:meth:`PPOBaseline.algorithm_config` (and, if it has knobs worth
searching, :attr:`PPOBaseline.tune_grid`) and inherits the training
loop, the early-stopping rule, and the evaluation protocol unchanged.

Ray and torch are imported inside methods, so importing ``topogym``
never pulls them in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from topogym.baselines.gridworld2dv1.protocol import (
    Baseline,
    Hyperparameters,
    TrainingReport,
)

logger = logging.getLogger("topogym")


def mean_return(result: dict) -> float:
    """RLlib's mean episode return, across API-stack spellings.

    ``nan`` when no episode finished inside the batch, which is a real
    state for long-horizon worlds and must not be read as a score.
    """
    for section_name in ("env_runners", "evaluation"):
        section = result.get(section_name) or {}
        for key in ("episode_return_mean", "episode_reward_mean"):
            value = section.get(key)
            if value is not None:
                return float(value)
    return float("nan")


class PPOBaseline(Baseline):
    """Vanilla PPO over the split's instance distribution."""

    name = "ppo"

    #: Searched on the tuning split. Deliberately small and legible:
    #: this is a documented reference point, not a tuned entry.
    tune_grid = (
        {"lr": 3e-4, "entropy_coeff": 0.01},
        {"lr": 1e-4, "entropy_coeff": 0.01},
        {"lr": 3e-4, "entropy_coeff": 0.001},
    )

    defaults = {"lr": 3e-4, "entropy_coeff": 0.01, "gamma": 0.99}

    def __init__(self, config=None):
        super().__init__(config)
        self._algorithm = None

    # -- the hook variants override -----------------------------------

    def algorithm_config(self, rows: list, values: dict, seed: int):
        """The RLlib config for one training run.

        Subclasses extend this: an intrinsic-reward variant adds its
        learner or connector here and inherits everything else.
        """
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env

        from topogym.baselines.gridworld2dv1.multitask import SplitEnv

        register_env("topogym_split", SplitEnv)
        params = {**self.defaults, **values}
        return (
            PPOConfig()
            .environment(
                "topogym_split",
                env_config={
                    "rows": rows, "seed": seed,
                    "env_options": self.env_options(),
                    "episodes_per_instance":
                        self.config.train_episodes_per_instance,
                },
            )
            .env_runners(
                num_env_runners=self.config.num_env_runners,
                num_envs_per_env_runner=self.config.num_envs_per_runner,
            )
            .learners(
                num_learners=self.config.num_learners,
                num_gpus_per_learner=self.config.gpus_per_learner,
            )
            .training(
                lr=params["lr"],
                gamma=params["gamma"],
                entropy_coeff=params["entropy_coeff"],
                train_batch_size_per_learner=self.config.train_batch_size,
            )
            .debugging(log_level="ERROR")
        )

    # -- the protocol -------------------------------------------------

    def select_hyperparameters(self, tune_rows: list) -> Hyperparameters:
        best, best_score, searched = None, -float("inf"), []
        for candidate in self.tune_grid:
            score = self._score_candidate(tune_rows, candidate)
            searched.append({**candidate, "score": None
                             if np.isnan(score) else score})
            logger.info("[%s] tune %s -> %.4f", self.name, candidate,
                        score)
            if np.isfinite(score) and score > best_score:
                best, best_score = dict(candidate), score
        if best is None:  # no candidate completed an episode
            logger.warning("[%s] tuning inconclusive; using defaults",
                           self.name)
            return Hyperparameters(values=dict(self.tune_grid[0]),
                                   searched=searched)
        return Hyperparameters(values=best, tuning_score=best_score,
                               searched=searched)

    def _score_candidate(self, rows: list, candidate: dict) -> float:
        algorithm = self.algorithm_config(
            rows, candidate, self.config.seed).build_algo()
        score = float("nan")
        try:
            for _ in range(self.config.tune_iterations):
                score = mean_return(algorithm.train())
        finally:
            algorithm.stop()
        return score

    def fit(self, train_rows: list, val_rows: list,
            hyperparameters: Hyperparameters) -> TrainingReport:
        from topogym.baselines.gridworld2dv1.multitask import SplitEnv

        self._algorithm = self.algorithm_config(
            train_rows, hyperparameters.values, self.config.seed
        ).build_algo()
        validator = SplitEnv({"rows": val_rows, "seed": self.config.seed,
                              "sequential": True})
        report = TrainingReport()
        best, stale, baseline_value, moved = -float("inf"), 0, None, False
        try:
            for iteration in range(1, self.config.max_iterations + 1):
                entry = {"iteration": iteration,
                         "train_return":
                             mean_return(self._algorithm.train())}
                if iteration % self.config.val_every == 0:
                    score = self._validate(validator)
                    entry["val_return"] = score
                    if baseline_value is None:
                        baseline_value = score
                    elif score != baseline_value:
                        # The objective has moved at least once, so a
                        # plateau from here is a real plateau.
                        moved = True
                    if score > best:
                        best, stale = score, 0
                        report.best_checkpoint = self._checkpoint()
                    elif moved:
                        stale += 1
                    logger.info(
                        "[%s] iter %d train %.3f val %.3f "
                        "(stale %d/%d%s)",
                        self.name, iteration, entry["train_return"],
                        score, stale, self.config.patience,
                        "" if moved else ", signal flat",
                    )
                report.history.append(entry)
                report.iterations = iteration
                if moved and stale >= self.config.patience:
                    report.stopped_early = True
                    report.stopped_because = "validation plateaued"
                    logger.info("[%s] early stop at iteration %d",
                                self.name, iteration)
                    break
        finally:
            validator.close()
        if not report.stopped_early:
            # A never-moving objective is the expected outcome for a
            # method with no exploration machinery against a sparse
            # reward. Spend the whole budget and say so, rather than
            # halting on a plateau that was never a plateau.
            report.stopped_because = ("budget exhausted" if moved
                                      else "no learning signal")
        report.best_val_return = best if np.isfinite(best) else None
        return report

    def _checkpoint(self) -> str | None:
        if self.config.run_dir is None:
            return None
        directory = self.config.run_dir / f"{self.name}-checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        return str(self._algorithm.save(str(directory)).checkpoint.path)

    def _validate(self, validator) -> float:
        """Mean greedy return over validation instances.

        ``val`` decides only *when to stop*: no gradient is taken here.
        """
        act = self.policy()
        returns = []
        for episode in range(self.config.val_episodes):
            observation, _ = validator.reset(seed=episode)
            total, done = 0.0, False
            while not done:
                observation, reward, terminated, truncated, _ = \
                    validator.step(act(observation, validator))
                total += reward
                done = terminated or truncated
            returns.append(total)
        return float(np.mean(returns)) if returns else float("nan")

    def policy(self) -> Callable:
        import torch

        if self._algorithm is None:
            raise RuntimeError("fit() must run before policy()")
        module = self._algorithm.get_module()

        def act(observation, env):
            batch = {"obs": torch.as_tensor(
                np.asarray(observation, dtype=np.float32)[None, ...])}
            with torch.no_grad():
                out = module.forward_inference(batch)
            logits = out.get("action_dist_inputs")
            if logits is None:
                return int(np.asarray(out["actions"]).reshape(-1)[0])
            return int(torch.argmax(logits, dim=-1).item())

        return act

    def close(self) -> None:
        if self._algorithm is not None:
            self._algorithm.stop()
            self._algorithm = None
