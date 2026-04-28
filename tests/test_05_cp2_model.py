"""CP2 — PickQNet forward pass: shapes, gradients, no NaNs, parameter count."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pick14.rl.q_model import (
    FFN_DIM,
    IN_CHANNELS,
    LATENT_DIM,
    N_CARDS,
    N_HEADS,
    N_LAYERS,
    PickQNet,
)


@pytest.fixture
def net() -> PickQNet:
    return PickQNet()


@pytest.fixture
def batch() -> torch.Tensor:
    return torch.randn(8, N_CARDS, IN_CHANNELS)


class TestPickQNetForward:
    def test_q_output_shape(self, net, batch):
        q, _ = net(batch)
        assert q.shape == (8, 1)

    def test_opp_hand_output_shape(self, net, batch):
        _, opp = net(batch)
        assert opp.shape == (8, N_CARDS)

    def test_opp_hand_in_unit_interval(self, net, batch):
        _, opp = net(batch)
        assert (opp >= 0).all() and (opp <= 1).all()

    def test_no_nans_in_outputs(self, net, batch):
        q, opp = net(batch)
        assert not torch.isnan(q).any()
        assert not torch.isnan(opp).any()

    def test_no_infs_in_outputs(self, net, batch):
        q, opp = net(batch)
        assert not torch.isinf(q).any()
        assert not torch.isinf(opp).any()

    def test_single_sample(self, net):
        x = torch.randn(1, N_CARDS, IN_CHANNELS)
        q, opp = net(x)
        assert q.shape == (1, 1)
        assert opp.shape == (1, N_CARDS)

    def test_predict_q_shape(self, net, batch):
        q = net.predict_q(batch)
        assert q.shape == (8, 1)


class TestPickQNetGradients:
    def test_q_loss_reaches_shared_layers(self, net, batch):
        """Q loss should flow back through proj and attention (shared layers)."""
        q, _ = net(batch)
        q.mean().backward()
        assert net.q_pool_w.grad is not None and net.q_pool_w.grad.abs().sum() > 0
        assert net.proj.weight.grad is not None and net.proj.weight.grad.abs().sum() > 0

    def test_opp_loss_reaches_shared_layers(self, net, batch):
        """Opp loss should flow back through proj and attention (shared layers)."""
        _, opp = net(batch)
        nn.BCELoss()(opp, torch.zeros_like(opp)).backward()
        assert net.opp_pool_w.grad is not None and net.opp_pool_w.grad.abs().sum() > 0
        assert net.proj.weight.grad is not None and net.proj.weight.grad.abs().sum() > 0

    def test_heads_are_independent(self, net, batch):
        """Q loss must not touch opp_pool_w; opp loss must not touch q_pool_w."""
        # Q loss only
        net.zero_grad()
        q, _ = net(batch)
        q.mean().backward()
        assert net.opp_pool_w.grad is None or net.opp_pool_w.grad.abs().sum() == 0

        # Opp loss only
        net.zero_grad()
        _, opp = net(batch)
        nn.BCELoss()(opp, torch.zeros_like(opp)).backward()
        assert net.q_pool_w.grad is None or net.q_pool_w.grad.abs().sum() == 0

    def test_combined_loss_reaches_all_params(self, net, batch):
        """Q + opp losses together should touch every parameter."""
        q, opp = net(batch)
        loss = q.mean() + nn.BCELoss()(opp, torch.zeros_like(opp))
        loss.backward()
        for name, p in net.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"No gradient for {name}"


class TestPickQNetParameterCount:
    def test_param_count_reasonable(self, net):
        n = net.param_count()
        # With 54-card seq, latent=32, 2 attention layers, MLP head — expect < 100 K
        assert n < 100_000, f"Unexpectedly large: {n} params"
        assert n > 1_000,   f"Suspiciously small: {n} params"

    def test_param_count_printed(self, net, capsys):
        n = net.param_count()
        print(f"PickQNet parameter count: {n:,}")
        captured = capsys.readouterr()
        assert "PickQNet" in captured.out


class TestPickQNetVariantConfig:
    def test_one_layer(self):
        net = PickQNet(n_layers=1)
        q, opp = net(torch.randn(2, N_CARDS, IN_CHANNELS))
        assert q.shape == (2, 1)

    def test_three_layers(self):
        net = PickQNet(n_layers=3)
        q, opp = net(torch.randn(2, N_CARDS, IN_CHANNELS))
        assert q.shape == (2, 1)

    def test_with_dropout(self):
        net = PickQNet(dropout=0.1)
        net.train()
        q, _ = net(torch.randn(4, N_CARDS, IN_CHANNELS))
        assert q.shape == (4, 1)
