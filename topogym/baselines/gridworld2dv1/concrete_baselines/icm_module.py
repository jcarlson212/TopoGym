"""The Intrinsic Curiosity Module over the ``dict`` observation.

    D. Pathak, P. Agrawal, A. A. Efros and T. Darrell. "Curiosity-driven
    Exploration by Self-supervised Prediction." ICML 2017.
    https://arxiv.org/abs/1705.05363

Three networks, exactly as in the paper: a feature encoder ``phi``, an
*inverse* model predicting the action that led from ``phi(s)`` to
``phi(s')``, and a *forward* model predicting ``phi(s')`` from
``phi(s)`` and the action. The intrinsic reward is the forward model's
error -- surprise about what happened next -- and the inverse model is
what keeps ``phi`` about the parts of the world the agent controls
rather than about whatever happens to vary.

What differs from RLlib's stock module is only ``phi``: it is
:class:`~topogym.baselines.encoders.CellFeatureNet` over the ``dict``
observation instead of an MLP over a flattened Box. That is a choice of
encoder, which ICM leaves open -- Pathak et al. use a convolutional
``phi`` over pixels -- not a change to the algorithm. The forward and
inverse models, the loss, and the ``beta`` balance are unchanged.

Written against the same interface as the stock module -- a
``forward_train`` returning ``Columns.INTRINSIC_REWARDS`` plus the two
feature vectors, and a self-supervised loss -- so it reuses the
curiosity Learner and the connector that folds intrinsic rewards into
the train batch.
"""

from __future__ import annotations

from ray.rllib.core import Columns
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


def _concat(first, second):
    """Stack two observations along the batch axis.

    A ``dict`` observation is concatenated channel by channel, so the
    encoder still runs once over both -- the same saved forward pass the
    stock module gets from ``tree.map_structure``.
    """
    if isinstance(first, dict):
        return {key: torch.cat([first[key], second[key]], dim=0)
                for key in first}
    return torch.cat([first, second], dim=0)


class IntrinsicCuriosityModule(TorchRLModule, SelfSupervisedLossAPI):
    """Surprise as the forward model's error in a learned feature space."""

    def setup(self):
        import gymnasium as gym

        from topogym.baselines.encoders import CellFeatureNet

        config = self.model_config or {}
        feature_dim = int(config.get("feature_dim", 64))
        n_actions = int(self.action_space.n)
        space = self.observation_space

        if isinstance(space, gym.spaces.Dict):
            self._feature_net = CellFeatureNet(
                space, feature_dim,
                embed_dim=int(config.get("embed_dim", 16)),
                out_dim=int(config.get("encoder_out_dim", 256)),
            )
        else:
            self._feature_net = _mlp(
                int(space.shape[0]),
                tuple(config.get("feature_net_hiddens", (256, 256))),
                feature_dim,
            )

        # Inverse: (phi, next_phi) -> which action was taken.
        self._inverse_net = _mlp(
            feature_dim * 2,
            tuple(config.get("inverse_net_hiddens", (256,))),
            n_actions,
        )
        # Forward: (phi, one-hot action) -> next_phi.
        self._forward_net = _mlp(
            feature_dim + n_actions,
            tuple(config.get("forward_net_hiddens", (256,))),
            feature_dim,
        )
        self._n_actions = n_actions

    def _forward(self, batch, **kwargs):
        raise NotImplementedError(
            "IntrinsicCuriosityModule is a world model, trained but never "
            "used to act; only forward_train() is supported."
        )

    def _forward_train(self, batch, **kwargs):
        # One pass over both observations, then split.
        both = _concat(batch[Columns.OBS], batch[Columns.NEXT_OBS])
        phis = self._feature_net(both)
        phi, next_phi = torch.chunk(phis, 2)

        actions = nn.functional.one_hot(
            batch[Columns.ACTIONS].long(), self._n_actions
        ).float()
        predicted_next_phi = self._forward_net(
            torch.cat([phi, actions], dim=-1)
        )

        # Per-sample forward error: the surprise of this transition, and
        # the intrinsic reward. Its mean is the forward loss.
        forward_error = 0.5 * torch.sum(
            torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
        )
        return {
            Columns.INTRINSIC_REWARDS: forward_error,
            "phi": phi,
            "next_phi": next_phi,
        }

    def compute_self_supervised_loss(self, *, learner, module_id, config,
                                     batch, fwd_out, **kwargs):
        module = learner.module[module_id].unwrapped()
        forward_loss = torch.mean(fwd_out[Columns.INTRINSIC_REWARDS])

        # Inverse: negative log-likelihood of the action actually taken.
        # Cross-entropy over the inverse net's logits is exactly that for
        # a Discrete action space, without depending on a catalog to
        # supply the distribution class.
        logits = module._inverse_net(
            torch.cat([fwd_out["phi"], fwd_out["next_phi"]], dim=-1)
        )
        inverse_loss = nn.functional.cross_entropy(
            logits, batch[Columns.ACTIONS].long()
        )

        # Pathak et al.'s beta: beta * forward + (1 - beta) * inverse.
        beta = config.learner_config_dict["forward_loss_weight"]
        total = beta * forward_loss + (1.0 - beta) * inverse_loss

        learner.metrics.log_dict(
            {
                "mean_intrinsic_rewards": forward_loss,
                "forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
            },
            key=module_id, window=1,
        )
        return total
