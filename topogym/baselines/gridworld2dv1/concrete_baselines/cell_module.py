"""A PPO policy over the ``dict`` observation, built on CellEncoder.

RLlib's default encoders take a flat Box, so a ``Dict`` observation
reaches them only after RLlib flattens and concatenates it -- which
would splice the nominal patch codes, the sparse texture block, and the
world-scaled position into one undifferentiated float vector and
reintroduce exactly the encoding this observation mode exists to avoid.

This module skips that path: it runs
:class:`~topogym.baselines.encoders.CellEncoder` over the three
channels and puts the usual policy and value heads on top. It is a
plain ``TorchRLModule`` implementing ``ValueFunctionAPI``, the same
extension point the RND module uses, rather than a Catalog override --
Catalog internals move between RLlib releases and this interface does
not.
"""

from __future__ import annotations

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class CellPPOModule(TorchRLModule, ValueFunctionAPI):
    """PPO actor-critic over the ``dict`` observation."""

    def setup(self):
        from topogym.baselines.encoders import CellEncoder

        config = self.model_config or {}
        embed_dim = int(config.get("embed_dim", 16))
        out_dim = int(config.get("encoder_out_dim", 256))
        head_hidden = int(config.get("head_hidden", 256))

        self.encoder = CellEncoder(
            self.observation_space, embed_dim=embed_dim, out_dim=out_dim
        )
        # Separate heads over a shared trunk: the value estimate and the
        # policy want different things from the same field of view, and
        # sharing the trunk is what keeps the encoder's gradient signal
        # from being dominated by whichever head learns first.
        self.pi = nn.Sequential(
            nn.Linear(out_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, int(self.action_space.n)),
        )
        self.vf = nn.Sequential(
            nn.Linear(out_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def _embeddings(self, batch):
        return self.encoder(batch[Columns.OBS])

    def _forward(self, batch, **kwargs):
        return {Columns.ACTION_DIST_INPUTS: self.pi(self._embeddings(batch))}

    def _forward_train(self, batch, **kwargs):
        embeddings = self._embeddings(batch)
        return {
            Columns.ACTION_DIST_INPUTS: self.pi(embeddings),
            Columns.EMBEDDINGS: embeddings,
        }

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._embeddings(batch)
        return self.vf(embeddings).squeeze(-1)
