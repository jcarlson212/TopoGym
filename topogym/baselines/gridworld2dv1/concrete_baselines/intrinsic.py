"""PPO with an intrinsic reward, shared by ICM and RND.

The intrinsic module lives in the **Learner**, not in an environment
wrapper. That distinction is the whole point: both papers train one
curiosity model on the same batches as the policy, while a wrapper
under Ray would give every env runner its own copy, and sixteen
diverging curiosity models are not the algorithm either paper
describes.

RLlib's new API stack supports this directly. The policy and the
intrinsic module are two entries of a ``MultiRLModule``; a learner
connector computes the intrinsic reward and folds it into the train
batch's rewards *before* advantage estimation, so PPO itself needs no
change; and the intrinsic module's own self-supervised loss is added
by the Learner.

Subclasses supply the module and its knobs -- see
:class:`IntrinsicRewardBaseline.intrinsic_module_spec` and
:attr:`IntrinsicRewardBaseline.intrinsic_defaults`. Everything else --
the training loop, the early-stopping rule, hyperparameter selection,
evaluation -- is inherited from :class:`PPOBaseline` unchanged, which
is what makes an intrinsic-reward variant an override rather than a
fork.

References:
    D. Pathak, P. Agrawal, A. A. Efros and T. Darrell. "Curiosity-driven
    Exploration by Self-supervised Prediction." ICML 2017.
    https://arxiv.org/abs/1705.05363
    Y. Burda, H. Edwards, A. Storkey and O. Klimov. "Exploration by
    Random Network Distillation." ICLR 2019.
    https://arxiv.org/abs/1810.12894
"""

from __future__ import annotations

import logging

from topogym.baselines.gridworld2dv1.concrete_baselines.ppo import (
    PPOBaseline,
)

logger = logging.getLogger("topogym")

#: The MultiRLModule *slot* the intrinsic model occupies. The string
#: is fixed by RLlib's curiosity connector and learner, which is why it
#: says "curiosity model" -- it names the slot, not the architecture.
#: ICM and RND put entirely different models in it: ICM a feature
#: encoder with inverse and forward dynamics nets, RND a trainable
#: predictor against a frozen random target.
INTRINSIC_MODULE_ID = "_intrinsic_curiosity_model"


class IntrinsicRewardBaseline(PPOBaseline):
    """PPO whose reward is extrinsic + coefficient * intrinsic.

    Abstract: a subclass declares the intrinsic module. The extrinsic
    signal stays exactly what the benchmark defines -- sparse, +1 on
    the goal -- so what is being measured is whether the intrinsic
    signal finds that goal at all.
    """

    name = "intrinsic"

    #: Weights the intrinsic signal against the (rare) extrinsic one,
    #: and any module-specific knobs. Searched on the tuning split
    #: alongside PPO's own learning rate.
    intrinsic_defaults = {
        "intrinsic_reward_coeff": 0.05,
        "forward_loss_weight": 0.2,
    }

    #: Searched on `tune`. Learning rate matters as much as the
    #: intrinsic weight, so both vary.
    tune_grid = (
        {"lr": 3e-4, "intrinsic_reward_coeff": 0.05},
        {"lr": 3e-4, "intrinsic_reward_coeff": 0.5},
        {"lr": 1e-4, "intrinsic_reward_coeff": 0.05},
        {"lr": 1e-4, "intrinsic_reward_coeff": 0.5},
    )

    def intrinsic_module_spec(self, observation_space, action_space):
        """The intrinsic model, as an ``RLModuleSpec``."""
        raise NotImplementedError

    def learner_class(self):
        """The Learner that trains policy and intrinsic model together."""
        from ray.rllib.examples.learners.classes.\
            intrinsic_curiosity_learners import (
                PPOTorchLearnerWithCuriosity,
            )

        return PPOTorchLearnerWithCuriosity

    def algorithm_config(self, rows: list, values: dict, seed: int):
        """PPO's config, plus the intrinsic module and its Learner."""
        from ray.rllib.core import DEFAULT_MODULE_ID
        from ray.rllib.core.rl_module.multi_rl_module import (
            MultiRLModuleSpec,
        )
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec

        from topogym.baselines.gridworld2dv1.instances import make_instance

        config = super().algorithm_config(rows, values, seed)
        probe = make_instance(rows[0], **self.env_options())
        observation_space = probe.observation_space
        action_space = probe.action_space
        probe.close()

        knobs = {**self.intrinsic_defaults,
                 **{k: v for k, v in values.items()
                    if k in self.intrinsic_defaults}}
        return (
            config
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        DEFAULT_MODULE_ID: RLModuleSpec(),
                        INTRINSIC_MODULE_ID: self.intrinsic_module_spec(
                            observation_space, action_space),
                    },
                ),
                # Only the policy is trained by PPO's own loss; the
                # intrinsic module is trained by its self-supervised
                # one, which the Learner adds.
                algorithm_config_overrides_per_module={},
            )
            .learners(
                learner_class=self.learner_class(),
                learner_config_dict=knobs,
            )
            .multi_agent(
                policies={DEFAULT_MODULE_ID},
                policy_mapping_fn=lambda *_args, **_kwargs:
                    DEFAULT_MODULE_ID,
            )
        )


class ICMBaseline(IntrinsicRewardBaseline):
    """PPO + the Intrinsic Curiosity Module (Pathak et al., 2017).

    Three networks: a feature encoder, an inverse model predicting the
    action between two encoded observations, and a forward model
    predicting the next encoding. The intrinsic reward is the forward
    model's error -- surprise about what happened next.
    """

    name = "icm-ppo"

    #: ``forward_loss_weight`` is Pathak et al.'s beta: the ICM loss is
    #: ``beta * forward + (1 - beta) * inverse``, so the inverse
    #: dynamics term -- predicting which action led from phi to phi' --
    #: carries the remaining 0.8 and is what keeps the features about
    #: what the agent controls.
    intrinsic_defaults = {
        "intrinsic_reward_coeff": 0.05,
        "forward_loss_weight": 0.2,
        "feature_dim": 64,
    }

    def intrinsic_module_spec(self, observation_space, action_space):
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.examples.rl_modules.classes.\
            intrinsic_curiosity_model_rlm import (
                IntrinsicCuriosityModel,
            )

        return RLModuleSpec(
            module_class=IntrinsicCuriosityModel,
            observation_space=observation_space,
            action_space=action_space,
            learner_only=True,  # never used to act, only to learn
            model_config={
                "feature_dim": self.intrinsic_defaults["feature_dim"],
                "feature_net_hiddens": (128,),
                "feature_net_activation": "relu",
                "inverse_net_hiddens": (128,),
                "inverse_net_activation": "relu",
                "forward_net_hiddens": (128,),
                "forward_net_activation": "relu",
            },
        )


class RNDBaseline(IntrinsicRewardBaseline):
    """PPO + Random Network Distillation (Burda et al., 2019).

    Novelty is the error of predicting a fixed random embedding of the
    next observation. Compared with ICM there is no inverse model and
    nothing to learn about dynamics -- only how familiar a state is --
    which is why it needs no ``forward_loss_weight``.
    """

    name = "rnd-ppo"

    #: RND has no inverse model and no forward/inverse balance: its
    #: loss is the prediction error alone. The key is present only
    #: because the shared learner scaffolding asserts it, and
    #: RandomNetworkDistillation never reads it.
    intrinsic_defaults = {
        "intrinsic_reward_coeff": 0.05,
        "forward_loss_weight": 0.0,
        "feature_dim": 64,
    }

    def intrinsic_module_spec(self, observation_space, action_space):
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec

        from topogym.baselines.gridworld2dv1.concrete_baselines\
            .rnd_module import RandomNetworkDistillation

        return RLModuleSpec(
            module_class=RandomNetworkDistillation,
            observation_space=observation_space,
            action_space=action_space,
            learner_only=True,
            model_config={
                "feature_dim": self.intrinsic_defaults["feature_dim"],
                "net_hiddens": (128,),
            },
        )
