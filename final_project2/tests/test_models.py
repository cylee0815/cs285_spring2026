"""Tests for neural network models (Milestone 4)."""

import pytest
import torch
import torch.nn as nn

from models.mlp import build_mlp
from models.q_network import QNetwork
from models.value_network import ValueNetwork
from models.policy_network import PolicyNetwork

B = 4   # batch size
F = 10  # state features
A = 8   # actions (assets)


class TestMLP:
    def test_forward_shape(self):
        mlp = build_mlp(input_dim=F, output_dim=1, hidden_sizes=(256, 256))
        x = torch.randn(B, F)
        out = mlp(x)
        assert out.shape == (B, 1)

    def test_custom_hidden_sizes(self):
        mlp = build_mlp(input_dim=5, output_dim=3, hidden_sizes=(64, 32))
        x = torch.randn(B, 5)
        out = mlp(x)
        assert out.shape == (B, 3)

    def test_gradient_flow(self):
        mlp = build_mlp(input_dim=F, output_dim=1)
        x = torch.randn(B, F)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        for p in mlp.parameters():
            assert p.grad is not None
            assert p.grad.shape == p.shape


class TestQNetwork:
    def test_output_shape(self):
        q = QNetwork(state_dim=F, action_dim=A)
        s = torch.randn(B, F)
        a = torch.randn(B, A)
        out = q(s, a)
        assert out.shape == (B, 1)

    def test_gradient_flow(self):
        q = QNetwork(state_dim=F, action_dim=A)
        s = torch.randn(B, F)
        a = torch.randn(B, A)
        out = q(s, a)
        loss = out.sum()
        loss.backward()
        for p in q.parameters():
            assert p.grad is not None


class TestValueNetwork:
    def test_output_shape(self):
        v = ValueNetwork(state_dim=F)
        s = torch.randn(B, F)
        out = v(s)
        assert out.shape == (B, 1)

    def test_gradient_flow(self):
        v = ValueNetwork(state_dim=F)
        s = torch.randn(B, F)
        out = v(s)
        loss = out.sum()
        loss.backward()
        for p in v.parameters():
            assert p.grad is not None


class TestPolicyNetwork:
    def test_output_shape(self):
        pi = PolicyNetwork(state_dim=F, action_dim=A)
        s = torch.randn(B, F)
        out = pi(s)
        assert out.shape == (B, A)

    def test_weights_sum_to_one(self):
        pi = PolicyNetwork(state_dim=F, action_dim=A)
        s = torch.randn(B, F)
        out = pi(s)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-6)

    def test_weights_non_negative(self):
        pi = PolicyNetwork(state_dim=F, action_dim=A)
        s = torch.randn(B, F)
        out = pi(s)
        assert (out >= 0).all()

    def test_gradient_flow(self):
        pi = PolicyNetwork(state_dim=F, action_dim=A)
        s = torch.randn(B, F)
        out = pi(s)
        loss = out.sum()
        loss.backward()
        for p in pi.parameters():
            assert p.grad is not None


class TestBatching:
    def test_different_batch_sizes(self):
        q = QNetwork(state_dim=F, action_dim=A)
        v = ValueNetwork(state_dim=F)
        pi = PolicyNetwork(state_dim=F, action_dim=A)
        for b in [1, 8, 32]:
            s = torch.randn(b, F)
            a = torch.randn(b, A)
            assert q(s, a).shape == (b, 1)
            assert v(s).shape == (b, 1)
            assert pi(s).shape == (b, A)

    def test_batch_gradient_backprop(self):
        q = QNetwork(state_dim=F, action_dim=A)
        v = ValueNetwork(state_dim=F)
        pi = PolicyNetwork(state_dim=F, action_dim=A)
        s = torch.randn(B, F)
        a = torch.randn(B, A)
        loss = q(s, a).sum() + v(s).sum() + pi(s).sum()
        loss.backward()
        for model in [q, v, pi]:
            for p in model.parameters():
                assert p.grad is not None


class TestDeviceTransfer:
    def test_cpu(self):
        q = QNetwork(state_dim=F, action_dim=A).to("cpu")
        s = torch.randn(B, F)
        a = torch.randn(B, A)
        out = q(s, a)
        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_gpu(self):
        q = QNetwork(state_dim=F, action_dim=A).to("cuda")
        s = torch.randn(B, F, device="cuda")
        a = torch.randn(B, A, device="cuda")
        out = q(s, a)
        assert out.device.type == "cuda"
