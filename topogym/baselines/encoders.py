"""Encoders shared by the baselines, over the ``dict`` observation.

The ``dict`` observation has three channels with three different
natures, and the point of :class:`CellEncoder` is to stop treating them
as one undifferentiated float vector:

- ``patch`` is *nominal*: ten symbolic codes with no order. Flattening
  it and dividing by :data:`~topogym.core.constants.OBS_MAX` -- what
  the older ``FlatObservation`` path does -- tells a network that
  ``goal`` (4/9) is four times ``wall`` (1/9) and that ``hazard`` (8/9)
  and ``wormhole`` (9/9) are near neighbours. None of that is true, and
  the network spends capacity unlearning it. An embedding table imposes
  no geometry at all.
- ``textures`` is *sparse multi-hot*: around one active slot per cell in
  the Texture worlds and identically zero everywhere else. A dense
  linear layer over it would leave most weights without gradient.
- ``position`` is *continuous and world-scaled*, so it needs
  normalising against the space's own bounds rather than a constant.

Torch is imported at module scope, so import this lazily (inside a
method) exactly like the RLlib algorithms do -- importing ``topogym``
must never pull torch in.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from topogym.core import constants as C


class CellEncoder(nn.Module):
    """Embed each cell in view, then read the field of view as an image.

    Every cell becomes ``embed(code) + sum(embed(active texture slots))``
    -- one ``d``-dimensional vector carrying both what the cell is and
    what it means. The sum is what makes the two channels compose:
    a cell is *both* ``OBS_EMPTY`` and ``water``, not one or the other.

    The texture term is a matrix product against the multi-hot block,
    which is exactly ``nn.EmbeddingBag(..., mode="sum")`` over the
    active slot indices -- the same arithmetic without the ragged
    bookkeeping. Two properties follow, and both matter here:

    - An all-zero block contributes exactly zero, so on GridWorld2D and
      Top -- where no textures exist -- a cell degrades cleanly to its
      code embedding with no dead parameters and no wasted input width.
    - Multi-label cells cost nothing extra, which is what the block is:
      a cell can be ``ladder`` and ``blocked_left`` at once.

    Sizes come from the definitions rather than from literals:
    :data:`~topogym.core.constants.OBS_CODE_COUNT` for the code table
    and :attr:`TextureSlotMap.dim <topogym.core.constants.TextureSlotMap>`
    for the texture table, so appending a slot widens this encoder
    automatically.
    """

    def __init__(self, observation_space, embed_dim: int = 16,
                 out_dim: int = 256):
        super().__init__()
        patch_space = observation_space["patch"]
        texture_space = observation_space["textures"]
        n_rows, n_cols = patch_space.shape

        if texture_space.shape[-1] != C.TEXTURE_SLOTS.dim:
            raise ValueError(
                f"texture channel is {texture_space.shape[-1]} wide but the "
                f"slot map defines {C.TEXTURE_SLOTS.dim}; they must agree"
            )

        self.embed_dim = embed_dim
        #: Nominal codes -> vectors. Sized to every code that exists,
        #: never to the codes one slice happens to contain: hazard and
        #: wormhole appear only in Texture worlds, so a table fitted to
        #: a GridWorld2D run would fail or alias them at evaluation.
        self.code = nn.Embedding(C.OBS_CODE_COUNT, embed_dim)
        #: Multi-hot slots -> vectors, summed. No bias: an unannotated
        #: cell must contribute nothing, not a learned constant.
        self.texture = nn.Linear(C.TEXTURE_SLOTS.dim, embed_dim, bias=False)

        #: Position is normalised by the space's own bounds, so a
        #: 400-cell world and a 50-cell world both land in [0, 1]. The
        #: raw coordinate is unbounded across sizes, which is exactly
        #: what breaks a policy on the size-extrapolation split.
        high = np.asarray(observation_space["position"].high,
                          dtype=np.float32)
        self.register_buffer(
            "position_scale",
            torch.as_tensor(np.where(high > 0, high, 1.0)),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * n_rows * n_cols + 2, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, observation: dict) -> torch.Tensor:
        patch = observation["patch"].long()
        textures = observation["textures"].float()
        position = observation["position"].float()

        # (B, H, W, d): what the cell is, plus what it means.
        cells = self.code(patch) + self.texture(textures)
        # (B, d, H, W) for the convolution.
        features = self.conv(cells.permute(0, 3, 1, 2))
        flat = features.flatten(start_dim=1)
        scaled = position / self.position_scale
        return self.head(torch.cat([flat, scaled], dim=-1))


class CellFeatureNet(nn.Module):
    """:class:`CellEncoder` projected to a fixed feature width.

    The shape every world-model wants: an intrinsic-reward method needs
    *a* latent space to measure novelty or surprise in, and which one it
    is decides what the signal means. Over a flattened patch, RND's
    error and ICM's forward error are dominated by whichever raw code
    happens to vary most; over cell embeddings they track what the agent
    has actually seen.

    Shared by :mod:`~...concrete_baselines.rnd_module` and
    :mod:`~...concrete_baselines.icm_module` so the two measure novelty
    in comparably-built spaces rather than each rolling its own.
    """

    def __init__(self, observation_space, feature_dim: int,
                 embed_dim: int = 16, out_dim: int = 256):
        super().__init__()
        self.encoder = CellEncoder(observation_space, embed_dim=embed_dim,
                                   out_dim=out_dim)
        self.head = nn.Linear(out_dim, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, observation):
        return self.head(self.encoder(observation))
