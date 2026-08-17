"""Specialized native two-card expectimax policy."""

from __future__ import annotations

from .lookahead import DepthKPolicy


class Depth2Policy(DepthKPolicy):
    """Exact depth-two expectimax with a specialized final-card expansion."""

    _specialized_depth_two = True

    def __init__(self) -> None:
        super().__init__(2)
