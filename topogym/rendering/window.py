"""A pygame window for the "human" render mode.

pygame is an optional dependency (``pip install topogym[play]``); it is
imported lazily so the core library stays dependency-free.
"""

from __future__ import annotations

import numpy as np


class Window:
    """Displays RGB frames in a pygame window at a fixed frame rate."""

    def __init__(self, title: str = "TopoGym", fps: int = 8):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'the "human" render mode needs pygame: '
                "pip install topogym[play]"
            ) from exc
        self._pygame = pygame
        self.title = title
        self.fps = fps
        self._screen = None
        self._clock = None

    def show(self, frame: np.ndarray) -> None:
        """Display one (H, W, 3) uint8 frame; paces to ``fps``."""
        pygame = self._pygame
        size = (frame.shape[1], frame.shape[0])
        if self._screen is None or self._screen.get_size() != size:
            pygame.init()
            pygame.display.set_caption(self.title)
            self._screen = pygame.display.set_mode(size)
            self._clock = pygame.time.Clock()
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()
        self._clock.tick(self.fps)

    def set_caption(self, text: str) -> None:
        if self._screen is not None:
            self._pygame.display.set_caption(text)
        else:
            self.title = text

    def close(self) -> None:
        if self._screen is not None:
            self._pygame.display.quit()
            self._screen = None
