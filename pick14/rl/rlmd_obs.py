"""
Build rl.md §2 observation tensors for :class:`~pick14.rl.env.Pick14GymEnv` and legal-action helpers.

Training stacks ``seq_*`` with the env's card observation; :meth:`Pick14GymEnv.step` sets
``_post_play_obs`` after a legal PLAY so the opponent critic (§3.3) sees $S_{post\\_play}$.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import ObservationWrapper, spaces
from gymnasium.core import Env

from pick14.rl import rlmd_sequences as seq
from pick14.rl.rlmd_sequences import NUM_ROLE_TYPES
from pick14.rl.sim_core import (
    MAX_PUBLIC_SLOTS,
    RlPick14State,
    TurnPhase,
    build_match_legality_mask,
    build_play_legality_mask,
    is_finished,
    max_hand_slots,
    max_match_combo_slots,
)

# Slices on agent body (before global tokens); pool always MAX_PUBLIC_SLOTS wide.
def agent_pool_start(n_hand: int) -> int:
    return seq.agent_hand_token_slots(n_hand)


def agent_global_start(n_hand: int) -> int:
    return seq.agent_body_len(n_hand)


def critic_opp_start(n_hand: int) -> int:
    return seq.agent_hand_token_slots(n_hand) + seq.MAX_PUBLIC_SLOTS


def build_mask_match_matrix(env: Any) -> np.ndarray:
    """``[max_combo+1, MAX_PUBLIC_SLOTS]`` int8 copy of env legality (0 if not match phase)."""
    assert env.state is not None
    ctx = env._build_ctx()
    st = env.state
    if st.phase != TurnPhase.MATCH or is_finished(st):
        return np.zeros((env._max_match_combos + 1, MAX_PUBLIC_SLOTS), dtype=np.int8)
    return np.asarray(ctx.match_mask, dtype=np.int8)


def build_play_valid(env: Any) -> np.ndarray:
    assert env.state is not None
    ctx = env._build_ctx()
    slots = max_hand_slots(env.n_hand)
    v = np.zeros((slots,), dtype=np.int8)
    pm = np.asarray(ctx.play_mask, dtype=np.int8)
    v[: min(len(pm), slots)] = pm[:slots]
    return v


def encode_rlmd_tensors_for_env(env: Any) -> dict[str, np.ndarray]:
    """Full forward pass keys for :class:`~pick14.rl.rlmd_model.RLmdPPOAgent` at current env state."""
    assert env.state is not None
    st = env.state
    lp = env.learning_player
    n_hand = env.n_hand
    opp = 1 if lp == 0 else 0

    if st.phase == TurnPhase.MATCH:
        af, ar, am = seq.encode_agent_match(st, lp, n_hand)
        cf, cr, cm = seq.encode_critic_match(st, lp, opp, n_hand)
    else:
        af, ar, am = seq.encode_agent_play(st, lp, n_hand)
        cf, cr, cm = seq.encode_critic_play(st, lp, opp, n_hand)

    mm = build_mask_match_matrix(env)
    pv = build_play_valid(env)
    phase = np.array([seq.phase_obs_scalar(st)], dtype=np.float32)

    return {
        "seq_agent_feats": af,
        "seq_agent_roles": ar,
        "seq_agent_mask": am,
        "seq_critic_feats": cf,
        "seq_critic_roles": cr,
        "seq_critic_mask": cm,
        "mask_match": mm,
        "play_hand_valid": pv,
        "phase": phase,
    }


def encode_rlmd_post_play_from_state(
    state: RlPick14State,
    learning_player: int,
    n_hand: int,
) -> dict[str, np.ndarray]:
    """
    Same tensor bundle as :func:`encode_rlmd_post_play` for an arbitrary ``RlPick14State``
    (e.g. counterfactual discards). Does not require a :class:`~pick14.rl.env.Pick14GymEnv`.
    """
    lp = learning_player
    opp = 1 if lp == 0 else 0

    af, ar, am = seq.encode_agent_play(state, lp, n_hand)
    cf, cr, cm = seq.encode_critic_play(state, lp, opp, n_hand)

    rows = max_match_combo_slots(n_hand)
    if state.phase != TurnPhase.MATCH or is_finished(state):
        mm = np.zeros((rows + 1, MAX_PUBLIC_SLOTS), dtype=np.int8)
    else:
        mm = np.asarray(build_match_legality_mask(state)[0], dtype=np.int8)

    slots = max_hand_slots(n_hand)
    pv = np.zeros((slots,), dtype=np.int8)
    pm = np.asarray(build_play_legality_mask(state), dtype=np.int8)
    pv[: min(len(pm), slots)] = pm[:slots]

    phase = np.array([seq.phase_obs_scalar(state)], dtype=np.float32)

    return {
        "seq_agent_feats": af,
        "seq_agent_roles": ar,
        "seq_agent_mask": am,
        "seq_critic_feats": cf,
        "seq_critic_roles": cr,
        "seq_critic_mask": cm,
        "mask_match": mm,
        "play_hand_valid": pv,
        "phase": phase,
    }


def encode_rlmd_post_play(env: Any) -> dict[str, np.ndarray] | None:
    """
    Full rl.md observation bundle at $S_{post\\_play}$ (after learning seat's discard, before autoplay).

    Agent branch is the true post-play layout (PLAY phase) so a full :class:`~pick14.rl.rlmd_model.RLmdPPOAgent`
    forward remains well-defined; trainers typically read only ``V_{opponent}``.
    """
    if env.state is None:
        return None
    st = env.state
    lp = env.learning_player
    n_hand = env.n_hand
    opp = 1 if lp == 0 else 0

    af, ar, am = seq.encode_agent_play(st, lp, n_hand)
    cf, cr, cm = seq.encode_critic_play(st, lp, opp, n_hand)

    mm = build_mask_match_matrix(env)
    pv = build_play_valid(env)
    phase = np.array([seq.phase_obs_scalar(st)], dtype=np.float32)

    return {
        "seq_agent_feats": af,
        "seq_agent_roles": ar,
        "seq_agent_mask": am,
        "seq_critic_feats": cf,
        "seq_critic_roles": cr,
        "seq_critic_mask": cm,
        "mask_match": mm,
        "play_hand_valid": pv,
        "phase": phase,
    }


def merge_rlmd_into_obs(base_obs: dict[str, np.ndarray], env: Any) -> dict[str, np.ndarray]:
    out = {**base_obs, **encode_rlmd_tensors_for_env(env)}
    return out


def rlmd_observation_space(n_hand: int) -> spaces.Dict:
    ab = seq.agent_body_len(n_hand)
    cb = seq.critic_body_len(n_hand)
    rows = max_match_combo_slots(n_hand) + 1
    ps = max_hand_slots(n_hand)
    return spaces.Dict(
        {
            "seq_agent_feats": spaces.Box(-1e6, 1e6, shape=(ab, 9), dtype=np.float32),
            "seq_agent_roles": spaces.Box(0, NUM_ROLE_TYPES, shape=(ab,), dtype=np.int64),
            "seq_agent_mask": spaces.Box(0, 1, shape=(ab,), dtype=np.int8),
            "seq_critic_feats": spaces.Box(-1e6, 1e6, shape=(cb, 9), dtype=np.float32),
            "seq_critic_roles": spaces.Box(0, NUM_ROLE_TYPES, shape=(cb,), dtype=np.int64),
            "seq_critic_mask": spaces.Box(0, 1, shape=(cb,), dtype=np.int8),
            "mask_match": spaces.Box(0, 1, shape=(rows, MAX_PUBLIC_SLOTS), dtype=np.int8),
            "play_hand_valid": spaces.Box(0, 1, shape=(ps,), dtype=np.int8),
            "phase": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
    )


class RlmdObservationWrapper(ObservationWrapper):
    """
    Augments ``Pick14GymEnv`` observations with rl.md §2 token sequences and legality masks.

    Expects the wrapped env to expose ``state``, ``learning_player``, ``n_hand``, ``_max_match_combos``,
    ``_build_ctx``, and (after each step) ``_post_play_obs`` when applicable.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert hasattr(env, "n_hand")
        nh = int(env.n_hand)
        rlmd = rlmd_observation_space(nh)
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = spaces.Dict({**env.observation_space.spaces, **rlmd.spaces})
        else:
            self.observation_space = rlmd

    def observation(self, obs):  # type: ignore[override]
        return merge_rlmd_into_obs(dict(obs), self.env)


def masked_match_logits(
    match_logits: torch.Tensor,
    mask_match: torch.Tensor,
) -> torch.Tensor:
    """Flattened [B, R*P] with illegal cells at -1e9 (§3.1)."""
    flat = mask_match.reshape(mask_match.shape[0], -1).bool()
    return match_logits.masked_fill(~flat, -1e9)


def masked_play_logits(play_logits: torch.Tensor, play_valid: torch.Tensor) -> torch.Tensor:
    v = play_valid.bool()
    return play_logits.masked_fill(~v, -1e9)


def legal_flat_action_from_logits(
    match_logits: torch.Tensor,
    play_logits: torch.Tensor,
    mask_match: torch.Tensor,
    play_valid: torch.Tensor,
    phase_match: torch.Tensor,
    match_flat_dim: int,
    rng: torch.Generator | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Sample or argmax a **legal** env flat action. Falls back to first legal cell if the mask is all illegal
    (numerical edge case). ``phase_match`` bool [B] — True = match head.
    """
    device = match_logits.device
    bsz = match_logits.shape[0]
    out = torch.zeros((bsz,), dtype=torch.long, device=device)

    for b in range(bsz):
        if bool(phase_match[b].item()):
            m = mask_match[b].reshape(-1).bool()
            logits = match_logits[b].masked_fill(~m, -1e9)
            if deterministic:
                a = int(torch.argmax(logits).item())
            else:
                dist = torch.distributions.Categorical(logits=logits.unsqueeze(0))
                a = int(dist.sample(generator=rng).item())
            if not m[a]:
                nz = torch.nonzero(m, as_tuple=False)
                a = int(nz[0].item()) if nz.numel() else 0
            out[b] = a
        else:
            v = play_valid[b].bool()
            logits = play_logits[b].masked_fill(~v, -1e9)
            if deterministic:
                sub = int(torch.argmax(logits).item())
            else:
                dist = torch.distributions.Categorical(logits=logits.unsqueeze(0))
                sub = int(dist.sample(generator=rng).item())
            if not v[sub]:
                nz = torch.nonzero(v, as_tuple=False)
                sub = int(nz[0].item()) if nz.numel() else 0
            out[b] = sub + match_flat_dim

    return out


def numpy_legal_action(
    model: torch.nn.Module,
    device: torch.device,
    obs_dict: dict[str, np.ndarray],
    *,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> int:
    """Run ``RLmdPPOAgent`` on one observation dict and return a legal flat ``Discrete`` index."""
    from pick14.rl.rlmd_model import RLmdPPOAgent

    if not isinstance(model, RLmdPPOAgent):
        raise TypeError("model must be RLmdPPOAgent")
    obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs_dict.items() if k in _MODEL_KEYS}
    with torch.no_grad():
        ml, pl, _, _ = model(obs_t)
    phase_m = obs_t["phase"].squeeze(1) < 0.5
    mm = obs_t["mask_match"]
    pv = obs_t["play_hand_valid"]
    ml = masked_match_logits(ml, mm)
    pl = masked_play_logits(pl, pv)
    a = legal_flat_action_from_logits(
        ml,
        pl,
        mm,
        pv,
        phase_m,
        model.match_flat_dim,
        rng=generator,
        deterministic=deterministic,
    )
    return int(a.squeeze(0).item())


_MODEL_KEYS = frozenset(
    {
        "seq_agent_feats",
        "seq_agent_roles",
        "seq_agent_mask",
        "seq_critic_feats",
        "seq_critic_roles",
        "seq_critic_mask",
        "mask_match",
        "play_hand_valid",
        "phase",
    }
)
