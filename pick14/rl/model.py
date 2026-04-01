"""
Deprecated: legacy pointer / dual-head MLP stacks were removed in favor of rl.md §2–3.

Use :class:`pick14.rl.rlmd_model.RLmdPPOAgent` and :mod:`pick14.rl.rlmd_obs`.
"""

from pick14.rl.rlmd_model import RLmdPPOAgent

__all__ = ["RLmdPPOAgent"]
