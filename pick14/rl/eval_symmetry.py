"""
No-gradient rollouts for symmetric teacher-like play (seat 0 neural vs seat 1 baseline).

Env step rewards are sparse (0 / -1); critic bootstrap in pretrain uses those returns (often ~0).
For evaluation we also back up **terminal net score gap** (learning minus opponent) as a scalar
return so ``V_agent`` / ``V_opp`` can be compared to a non-degenerate target and **swapped-critic**
errors can be reported (sign / MSE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_obs import RlmdObservationWrapper
from pick14.rl.sim_core import total_score_points


def _to_torch_obs(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}


def total_gap_scores(state: Any) -> float:
    """Learning player (seat 0) total points minus seat 1."""
    if state is None:
        return 0.0
    return float(total_score_points(state, 0) - total_score_points(state, 1))


@dataclass(slots=True)
class StepTrace:
    is_match: bool
    rew_env: float
    gap_before: float
    gap_after: float
    g_terminal: float
    v_agent: float
    v_opp: float
    v_agent_post: float | None = None
    v_opp_post: float | None = None


@dataclass(slots=True)
class EpisodeTrace:
    final_gap: float
    seat0_win: bool
    tie: bool
    steps: list[StepTrace] = field(default_factory=list)


@dataclass(slots=True)
class SymmetryRolloutResult:
    episodes: list[EpisodeTrace]
    """Per-step env rewards (legal steps are 0.0)."""
    env_rewards: list[float]


def rollout_symmetry(
    env: RlmdObservationWrapper,
    model: RLmdPPOAgent,
    device: torch.device,
    *,
    episodes: int,
    gamma: float,
    seed_base: int,
    max_steps: int = 512,
    deterministic: bool = False,
) -> SymmetryRolloutResult:
    """
    Roll out while seat 0 to act: sample (or argmax) from the model; autoplay fills other seats.

    Each learning step records env reward, score gap before/after the step, and critic values.
    ``g_terminal`` is the Monte Carlo return **if the only non-zero reward is the terminal gap**
    paid once at the last learning step: ``G_t = gamma^{k} * final_gap`` with ``k`` steps remaining
    after the current decision (``k = 0`` on the last seat-0 decision of the episode).
    """
    model.eval()
    base = env.unwrapped
    ep_traces: list[EpisodeTrace] = []
    all_rew: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        trans: list[
            tuple[
                bool,
                float,
                float,
                float,
                float,
                float,
                dict[str, np.ndarray],
                dict[str, np.ndarray] | None,
                float | None,
                float | None,
            ]
        ] = []

        for _ in range(max_steps):
            if base.state is None or base.state.current_player != 0:
                break
            sc = base.state
            gap_before = float(total_gap_scores(sc))

            is_match = float(obs["phase"][0]) < 0.5
            with torch.no_grad():
                ml, pl, va_t, vo_t = model(_to_torch_obs(obs, device))
                logits = ml if is_match else pl
                if deterministic:
                    sub = int(torch.argmax(logits, dim=1).item())
                else:
                    sub = int(Categorical(logits=logits).sample().item())
            va0 = float(va_t.item())
            vo0 = float(vo_t.item())

            action = sub if is_match else sub + model.match_flat_dim
            next_obs, rew, term, trunc, _ = env.step(action)
            all_rew.append(float(rew))

            sc2 = base.state
            gap_after = float(total_gap_scores(sc2)) if sc2 is not None else gap_before

            obs_post = None
            vap = vop = None
            if not is_match and base._post_play_obs is not None:
                obs_post = {k: np.asarray(v).copy() for k, v in base._post_play_obs.items()}
                with torch.no_grad():
                    _, _, vapt, vopt = model(_to_torch_obs(obs_post, device))
                vap = float(vapt.item())
                vop = float(vopt.item())

            trans.append(
                (is_match, float(rew), gap_before, gap_after, va0, vo0, obs_post, vap, vop)
            )

            obs = next_obs
            if term or trunc:
                break

        final_sc = base.state
        final_gap = float(total_gap_scores(final_sc)) if final_sc is not None else 0.0
        n = len(trans)
        steps_out: list[StepTrace] = []
        for i, (is_m, r, gb, ga, va0, vo0, o_pst, vap, vop) in enumerate(trans):
            steps_after = n - 1 - i
            g_t = (gamma**steps_after) * final_gap
            steps_out.append(
                StepTrace(
                    is_match=is_m,
                    rew_env=r,
                    gap_before=gb,
                    gap_after=ga,
                    g_terminal=g_t,
                    v_agent=va0,
                    v_opp=vo0,
                    v_agent_post=vap,
                    v_opp_post=vop,
                )
            )

        ep_traces.append(
            EpisodeTrace(
                final_gap=final_gap,
                seat0_win=final_gap > 0,
                tie=final_gap == 0,
                steps=steps_out,
            )
        )

    return SymmetryRolloutResult(episodes=ep_traces, env_rewards=all_rew)


def summarize_symmetry_rollout(res: SymmetryRolloutResult) -> dict[str, float]:
    """Aggregate win rate, final-gap moments, env-reward moments, critic MSE / bias / variance."""
    eps = res.episodes
    n_ep = len(eps)
    if n_ep == 0:
        return {"n_episodes": 0.0}

    gaps = np.array([e.final_gap for e in eps], dtype=np.float64)
    wins = sum(1 for e in eps if e.seat0_win)
    ties = sum(1 for e in eps if e.tie)

    match_va: list[float] = []
    match_vo: list[float] = []
    match_g: list[float] = []
    match_dgap: list[float] = []
    play_vo: list[float] = []
    play_va_post: list[float] = []
    play_g: list[float] = []
    play_dgap: list[float] = []

    for e in eps:
        for s in e.steps:
            dg = s.gap_after - s.gap_before
            if s.is_match:
                match_va.append(s.v_agent)
                match_vo.append(s.v_opp)
                match_g.append(s.g_terminal)
                match_dgap.append(dg)
            else:
                if s.v_opp_post is not None and s.v_agent_post is not None:
                    play_vo.append(s.v_opp_post)
                    play_va_post.append(s.v_agent_post)
                    play_g.append(s.g_terminal)
                    play_dgap.append(dg)

    def _mse(pred: list[float], target: list[float]) -> float:
        if not pred:
            return float("nan")
        p = np.array(pred, dtype=np.float64)
        t = np.array(target, dtype=np.float64)
        return float(np.mean((p - t) ** 2))

    def _mean_var(x: list[float]) -> tuple[float, float]:
        if not x:
            return float("nan"), float("nan")
        a = np.array(x, dtype=np.float64)
        return float(np.mean(a)), float(np.var(a))

    er = np.array(res.env_rewards, dtype=np.float64) if res.env_rewards else np.array([])

    res_m_a = [a - b for a, b in zip(match_va, match_g, strict=True)]
    res_m_o = [a - b for a, b in zip(match_vo, match_g, strict=True)]
    res_p_o = [a - b for a, b in zip(play_vo, play_g, strict=True)]
    res_p_a = [a - b for a, b in zip(play_va_post, play_g, strict=True)]

    bm_a, vm_a = _mean_var(res_m_a)
    bm_o, _ = _mean_var(res_m_o)
    bp_o, vp_o = _mean_var(res_p_o)
    bp_a, _ = _mean_var(res_p_a)

    out: dict[str, float] = {
        "n_episodes": float(n_ep),
        "win_rate_seat0": wins / n_ep,
        "tie_rate": ties / n_ep,
        "mean_final_gap": float(np.mean(gaps)),
        "var_final_gap": float(np.var(gaps)),
        "mean_env_reward": float(np.mean(er)) if er.size else float("nan"),
        "var_env_reward": float(np.var(er)) if er.size else float("nan"),
    }

    out["mse_match_v_agent"] = _mse(match_va, match_g)
    out["mse_match_swap_v_opp"] = _mse(match_vo, match_g)
    out["bias_match_v_agent"] = bm_a
    out["bias_match_swap_v_opp"] = bm_o
    out["var_residual_match_v_agent"] = vm_a
    out["var_residual_match_swap_v_opp"] = _mean_var(res_m_o)[1]

    out["mse_play_v_opp"] = _mse(play_vo, play_g)
    out["mse_play_swap_v_agent"] = _mse(play_va_post, play_g)
    out["bias_play_v_opp"] = bp_o
    out["bias_play_swap_v_agent"] = bp_a
    out["var_residual_play_v_opp"] = vp_o
    out["var_residual_play_swap_v_agent"] = _mean_var(res_p_a)[1]

    md_m, md_v = _mean_var(match_dgap)
    pd_m, pd_v = _mean_var(play_dgap)
    out["mean_delta_gap_match"] = md_m
    out["var_delta_gap_match"] = md_v
    out["mean_delta_gap_play"] = pd_m
    out["var_delta_gap_play"] = pd_v

    return out


def critic_swap_report(res: SymmetryRolloutResult) -> dict[str, float]:
    """Subset of metrics focused on correct vs swapped critic heads (MSE, bias sign)."""
    s = summarize_symmetry_rollout(res)
    return {
        k: s[k]
        for k in (
            "mse_match_v_agent",
            "mse_match_swap_v_opp",
            "bias_match_v_agent",
            "bias_match_swap_v_opp",
            "mse_play_v_opp",
            "mse_play_swap_v_agent",
            "bias_play_v_opp",
            "bias_play_swap_v_agent",
        )
        if k in s
    }


def collect_critic_eval_rows(
    env: RlmdObservationWrapper,
    model: RLmdPPOAgent,
    device: torch.device,
    *,
    episodes: int,
    gamma: float,
    seed_base: int,
    max_steps: int = 512,
    policy: str = "teacher",
) -> tuple[
    list[tuple[dict[str, np.ndarray], float, float]],
    list[tuple[dict[str, np.ndarray], float, float]],
]:
    """
    Build (obs, g_env_mc, g_terminal_gap) rows aligned with critic training layout.

    * ``g_env_mc`` — MC backup of gym ``rew`` (mostly ``0`` on legal steps; **not** Phase-2 training).
    * ``g_terminal_gap`` — discounted terminal net score gap (seat 0 minus seat 1), same convention as
      :func:`rollout_symmetry`.

    ``policy``:

    * ``teacher`` — baseline expert ``teacher_action`` (rl.md §1.5); does not use the actor head.
    * ``sample`` / ``deterministic`` — current actor on seat-0 decisions (matches critic bootstrap).
    """
    if policy not in ("teacher", "sample", "deterministic"):
        raise ValueError(policy)
    model.eval()
    base = env.unwrapped
    match_rows: list[tuple[dict[str, np.ndarray], float, float]] = []
    play_rows: list[tuple[dict[str, np.ndarray], float, float]] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        trans: list[tuple[bool, float, dict[str, np.ndarray], dict[str, np.ndarray] | None]] = []
        for _ in range(max_steps):
            if base.state is None or base.state.current_player != 0:
                break
            is_match = float(obs["phase"][0]) < 0.5
            if policy == "teacher":
                action = int(base.teacher_action())
            else:
                with torch.no_grad():
                    ml, pl, _, _ = model(_to_torch_obs(obs, device))
                    logits = ml if is_match else pl
                    if policy == "deterministic":
                        sub = int(torch.argmax(logits, dim=1).item())
                    else:
                        sub = int(Categorical(logits=logits).sample().item())
                action = sub if is_match else sub + model.match_flat_dim

            next_obs, rew, term, trunc, _ = env.step(action)
            obs_post = None
            if not is_match and base._post_play_obs is not None:
                obs_post = {k: np.asarray(v).copy() for k, v in base._post_play_obs.items()}
            trans.append((is_match, float(rew), {k: v.copy() for k, v in obs.items()}, obs_post))
            obs = next_obs
            if term or trunc:
                break

        final_sc = base.state
        final_gap = float(total_gap_scores(final_sc)) if final_sc is not None else 0.0
        n = len(trans)
        g = 0.0
        returns_rev: list[float] = []
        for _, r, _, _ in reversed(trans):
            g = r + gamma * g
            returns_rev.append(g)
        returns_rev.reverse()

        for i, ((_is_m, _r, o_pre, o_post), g_env) in enumerate(zip(trans, returns_rev, strict=True)):
            steps_after = n - 1 - i
            g_term = (gamma**steps_after) * final_gap
            if _is_m:
                match_rows.append((o_pre, float(g_env), float(g_term)))
            else:
                if o_post is not None:
                    play_rows.append((o_post, float(g_env), float(g_term)))

    return match_rows, play_rows


def evaluate_critic_head_predictions(
    model: RLmdPPOAgent,
    device: torch.device,
    match_rows: list[tuple[dict[str, np.ndarray], float, float]],
    play_rows: list[tuple[dict[str, np.ndarray], float, float]],
    *,
    target: str,
    batch_size: int = 128,
) -> dict[str, float]:
    """
    MSE / RMSE / mean bias for correct vs swapped critics.

    * Match-phase rows: correct = ``V_agent(obs_pre)``, swap = ``V_opp(obs_pre)``.
    * Play-phase rows (post-play obs): correct = ``V_opp``, swap = ``V_agent``.

    ``target``:

    * ``env`` — index 1 in row tuples; matches Phase-2 critic training objective when returns are env MC.
    * ``terminal`` — index 2; terminal-gap return (not what bootstrap optimizes unless rew encodes it).
    """
    if target not in ("env", "terminal"):
        raise ValueError(target)
    ti = 1 if target == "env" else 2
    model.eval()

    def _run_match_batches() -> tuple[float, float, float, float, int]:
        sse_a = sse_o = s_err_a = s_err_o = 0.0
        n = 0
        for start in range(0, len(match_rows), batch_size):
            chunk = match_rows[start : start + batch_size]
            obs = {k: np.stack([row[0][k] for row in chunk], axis=0) for k in chunk[0][0].keys()}
            t = np.array([row[ti] for row in chunk], dtype=np.float64)
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
            with torch.no_grad():
                _, _, va, vo = model(obs_t)
            pa = va.detach().cpu().numpy().reshape(-1)
            po = vo.detach().cpu().numpy().reshape(-1)
            d_a = pa - t
            d_o = po - t
            sse_a += float(np.dot(d_a, d_a))
            sse_o += float(np.dot(d_o, d_o))
            s_err_a += float(d_a.sum())
            s_err_o += float(d_o.sum())
            n += len(chunk)
        return sse_a, sse_o, s_err_a, s_err_o, n

    def _run_play_batches() -> tuple[float, float, float, float, int]:
        sse_o = sse_a = s_err_o = s_err_a = 0.0
        n = 0
        for start in range(0, len(play_rows), batch_size):
            chunk = play_rows[start : start + batch_size]
            obs = {k: np.stack([row[0][k] for row in chunk], axis=0) for k in chunk[0][0].keys()}
            t = np.array([row[ti] for row in chunk], dtype=np.float64)
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
            with torch.no_grad():
                _, _, va, vo = model(obs_t)
            pa = va.detach().cpu().numpy().reshape(-1)
            po = vo.detach().cpu().numpy().reshape(-1)
            d_o = po - t
            d_a = pa - t
            sse_o += float(np.dot(d_o, d_o))
            sse_a += float(np.dot(d_a, d_a))
            s_err_o += float(d_o.sum())
            s_err_a += float(d_a.sum())
            n += len(chunk)
        return sse_o, sse_a, s_err_o, s_err_a, n

    out: dict[str, float] = {
        "n_match_rows": float(len(match_rows)),
        "n_play_rows": float(len(play_rows)),
    }

    if match_rows:
        sse_a, sse_o, s_err_a, s_err_o, n_m = _run_match_batches()
        mse_a = sse_a / max(1, n_m)
        mse_o = sse_o / max(1, n_m)
        out["mse_match_v_agent"] = mse_a
        out["rmse_match_v_agent"] = float(np.sqrt(mse_a))
        out["bias_match_v_agent"] = s_err_a / max(1, n_m)
        out["mse_match_swap_v_opp"] = mse_o
        out["rmse_match_swap_v_opp"] = float(np.sqrt(mse_o))
        out["bias_match_swap_v_opp"] = s_err_o / max(1, n_m)

    if play_rows:
        sse_o, sse_a, s_err_o, s_err_a, n_p = _run_play_batches()
        mse_o = sse_o / max(1, n_p)
        mse_a = sse_a / max(1, n_p)
        out["mse_play_v_opp"] = mse_o
        out["rmse_play_v_opp"] = float(np.sqrt(mse_o))
        out["bias_play_v_opp"] = s_err_o / max(1, n_p)
        out["mse_play_swap_v_agent"] = mse_a
        out["rmse_play_swap_v_agent"] = float(np.sqrt(mse_a))
        out["bias_play_swap_v_agent"] = s_err_a / max(1, n_p)

    return out


def full_critic_eval_suite(
    env: RlmdObservationWrapper,
    model: RLmdPPOAgent,
    device: torch.device,
    *,
    episodes: int,
    gamma: float,
    seed_teacher: int,
    seed_on_policy: int,
    max_steps: int = 512,
    batch_size: int = 128,
) -> dict[str, dict[str, float]]:
    """
    Critic metrics aligned with Phase-2 training (rl.md §4.1): ``A_t`` and play EV vs ``-O_t``.

    ``gamma`` / ``batch_size`` are accepted for CLI compatibility; this suite does not use them.
    """
    del gamma, batch_size
    from pick14.rl.pretrain_curriculum import collect_critic_bootstrap_data, eval_atomic_critic_metrics

    mr_t, pr_t = collect_critic_bootstrap_data(
        env,
        model,
        device,
        episodes,
        0.99,
        seed_teacher,
        max_steps,
        policy="teacher",
    )
    mr_p, pr_p = collect_critic_bootstrap_data(
        env,
        model,
        device,
        episodes,
        0.99,
        seed_on_policy,
        max_steps,
        policy="deterministic",
    )
    return {
        "teacher_atomic_rlmd41": eval_atomic_critic_metrics(model, device, mr_t, pr_t),
        "on_policy_det_atomic_rlmd41": eval_atomic_critic_metrics(model, device, mr_p, pr_p),
    }
