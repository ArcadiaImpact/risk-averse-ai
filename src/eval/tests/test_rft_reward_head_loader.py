"""Tests for the rft_pipeline.py reward_head.pt loader path.

These tests avoid downloading any base model: they construct a tiny fake
backbone in pure torch and verify that the wrapper reproduces the pipeline's
forward math (right-padded last-token pooling + fp32 linear head).
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - matches evaluate_reward_model.py guard
    torch = None

import evaluate_reward_model as erm


def _save_reward_head_ckpt(tmpdir: str, hidden_size: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(1, hidden_size, generator=gen, dtype=torch.float32)
    bias = torch.randn(1, generator=gen, dtype=torch.float32)
    path = os.path.join(tmpdir, "reward_head.pt")
    torch.save(
        {
            "reward_head_state_dict": {"weight": weight, "bias": bias},
            "hidden_size": hidden_size,
        },
        path,
    )
    return path, weight, bias


@unittest.skipIf(torch is None, "torch not available")
class BuildRewardHeadTests(unittest.TestCase):
    def test_loaded_weights_match_saved_state(self):
        hidden = 32
        with tempfile.TemporaryDirectory() as tmp:
            path, weight, bias = _save_reward_head_ckpt(tmp, hidden)
            head, ckpt_hidden = erm.build_reward_head_from_checkpoint(path)
            self.assertEqual(ckpt_hidden, hidden)
            self.assertTrue(torch.equal(head.weight.detach().cpu(), weight))
            self.assertTrue(torch.equal(head.bias.detach().cpu(), bias))
            self.assertEqual(head.weight.dtype, torch.float32)

    def test_rejects_checkpoint_missing_expected_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reward_head.pt")
            torch.save({"something_else": {}}, path)
            with self.assertRaisesRegex(ValueError, "reward_head_state_dict"):
                erm.build_reward_head_from_checkpoint(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            erm.build_reward_head_from_checkpoint("/no/such/reward_head.pt")


@unittest.skipIf(torch is None, "torch not available")
class ResolveRewardHeadPathTests(unittest.TestCase):
    def _args(self, **overrides):
        ns = types.SimpleNamespace(
            reward_head_path=None,
            model_path=None,
            base_model=None,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_autodetects_inside_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reward_head.pt")
            torch.save({"reward_head_state_dict": {"weight": torch.zeros(1, 4)}}, path)
            resolved = erm.resolve_reward_head_path(self._args(model_path=tmp))
            self.assertEqual(resolved, path)

    def test_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(erm.resolve_reward_head_path(self._args(model_path=tmp)))

    def test_explicit_empty_string_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reward_head.pt")
            torch.save({"reward_head_state_dict": {"weight": torch.zeros(1, 4)}}, path)
            self.assertIsNone(
                erm.resolve_reward_head_path(self._args(model_path=tmp, reward_head_path=""))
            )

    def test_explicit_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = os.path.join(tmp, "custom_head.pt")
            torch.save({"reward_head_state_dict": {"weight": torch.zeros(1, 4)}}, explicit)
            resolved = erm.resolve_reward_head_path(self._args(reward_head_path=explicit))
            self.assertEqual(resolved, explicit)


class _FakeBackboneOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeBackbone(torch.nn.Module if torch is not None else object):
    """Returns a fixed (B, T, H) tensor so the wrapper's pooling can be checked."""

    def __init__(self, hidden):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden.shape[-1])
        self.register_buffer("hidden", hidden)
        # A parameter so next(self.parameters()).device resolves predictably.
        self._probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return _FakeBackboneOutput(self.hidden)


@unittest.skipIf(torch is None, "torch not available")
class WrapperForwardTests(unittest.TestCase):
    def test_forward_picks_last_non_pad_token_and_applies_head(self):
        torch.manual_seed(0)
        batch, seq, hidden = 2, 5, 8
        hidden_states = torch.randn(batch, seq, hidden, dtype=torch.float16)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
        backbone = _FakeBackbone(hidden_states)

        weight = torch.randn(1, hidden, dtype=torch.float32)
        bias = torch.randn(1, dtype=torch.float32)
        head = torch.nn.Linear(hidden, 1, bias=True)
        with torch.no_grad():
            head.weight.copy_(weight)
            head.bias.copy_(bias)

        model = erm.RftRewardModelWrapper(backbone, head, backbone.config)
        model.eval()

        with torch.inference_mode():
            out = model(input_ids=torch.zeros_like(attention_mask), attention_mask=attention_mask)

        expected_idx = attention_mask.sum(dim=-1) - 1  # [2, 4]
        picked = hidden_states[torch.arange(batch), expected_idx].to(torch.float32)
        expected = (picked @ weight.T + bias).to(torch.float32)

        self.assertIn("logits", out)
        self.assertTrue(torch.allclose(out["logits"].to(torch.float32), expected, atol=1e-6))

    def test_roundtrip_through_build_helper(self):
        """End-to-end: checkpoint -> build_reward_head_from_checkpoint -> wrapper -> scores
        match a manual reference computed from the same saved weights."""
        torch.manual_seed(1)
        batch, seq, hidden = 2, 4, 8
        hidden_states = torch.randn(batch, seq, hidden, dtype=torch.float16)
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmp:
            path, weight, bias = _save_reward_head_ckpt(tmp, hidden, seed=7)
            head, _ = erm.build_reward_head_from_checkpoint(path)
            model = erm.RftRewardModelWrapper(
                _FakeBackbone(hidden_states), head, types.SimpleNamespace(hidden_size=hidden)
            )
            model.eval()
            with torch.inference_mode():
                got = model(input_ids=torch.zeros_like(attention_mask), attention_mask=attention_mask)

        last_idx = attention_mask.sum(dim=-1) - 1
        picked = hidden_states[torch.arange(batch), last_idx].to(torch.float32)
        expected = picked @ weight.T + bias
        self.assertTrue(torch.allclose(got["logits"].to(torch.float32), expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
