"""Random Network Distillation as an RLlib ``RLModule``.

    Y. Burda, H. Edwards, A. Storkey and O. Klimov. "Exploration by
    Random Network Distillation." ICLR 2019.
    https://arxiv.org/abs/1810.12894

A fixed, randomly initialised *target* network embeds an observation;
a *predictor* network is trained to reproduce that embedding. The
prediction error is the intrinsic reward: it is large where the
predictor has seen little, and falls as a state becomes familiar. The
target is never trained -- that is the whole trick, and the reason the
error is a novelty signal rather than an artefact of a moving target.

Written against the same interface as RLlib's ICM module -- a
``forward_train`` returning ``Columns.INTRINSIC_REWARDS`` and a
self-supervised loss -- so it reuses the curiosity Learner and the
connector that folds intrinsic rewards into the train batch. RND
differs from ICM in what it predicts, not in how it is wired.
"""

from __future__ import annotations

from ray.rllib.core import Columns
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.apis import SelfSupervisedLossAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def _mlp(input_dim: int, hiddens, output_dim: int) -> nn.Module:
    layers, previous = [], input_dim
    for width in hiddens:
        layers += [nn.Linear(previous, width), nn.ReLU()]
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)




class RandomNetworkDistillation(TorchRLModule, SelfSupervisedLossAPI):
    """Novelty as the error of predicting a fixed random embedding."""

    def setup(self):
        import gymnasium as gym

        from topogym.baselines.encoders import CellFeatureNet

        config = self.model_config or {}
        feature_dim = config.get("feature_dim", 64)
        hiddens = tuple(config.get("net_hiddens", (128,)))
        space = self.observation_space

        def build():
            if isinstance(space, gym.spaces.Dict):
                return CellFeatureNet(
                    space, feature_dim,
                    embed_dim=int(config.get("embed_dim", 16)),
                    out_dim=int(config.get("encoder_out_dim", 256)),
                )
            return _mlp(int(space.shape[0]), hiddens, feature_dim)

        # The target is random and frozen: never trained, never
        # updated. Training it would chase a moving goalpost and the
        # error would stop meaning novelty.
        self._target = build()
        for parameter in self._target.parameters():
            parameter.requires_grad = False
        self._predictor = build()

    def _forward(self, batch, **kwargs):
        # Never used to act; the policy module does that.
        return {}

    def _forward_train(self, batch, **kwargs):
        observations = batch[Columns.NEXT_OBS]
        if not isinstance(observations, dict):
            observations = observations.float()
        with torch.no_grad():
            target = self._target(observations)
        predicted = self._predictor(observations)
        # Per-sample squared error: the novelty of each transition.
        error = torch.square(predicted - target).mean(dim=-1)
        return {
            Columns.INTRINSIC_REWARDS: error,
            "rnd_prediction_error": error,
        }

    def compute_self_supervised_loss(self, *, learner, module_id, config,
                                     batch, fwd_out, **kwargs):
        loss = fwd_out["rnd_prediction_error"].mean()
        learner.metrics.log_dict(
            {"rnd_prediction_loss": loss.item()},
            key=module_id, window=1,
        )
        return loss


__all__ = ["RandomNetworkDistillation", "make_target_network"]
