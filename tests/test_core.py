import math
import numpy as np
import pandas as pd
import torch
from torch import nn

from speech_negotiation_kv.calibration import fit_ridge_opponent_code
from speech_negotiation_kv.crad import normalized_creditor_utility, parse_offer_days, parse_agreement_days, split_crad
from speech_negotiation_kv.glm_voice import audio_ids_to_prompt, partition_generated_token_ids, normalize_transcript
from speech_negotiation_kv.kv_hooks import split_fused_qkv, apply_kv_delta, FusedQKVRecorder, FusedQKVSteerer
from speech_negotiation_kv.subspace import advantage_memory_directions, fit_low_rank_subspace, subspace_overlap
from speech_negotiation_kv.sweep import MockSpeechBackend, run_one_turn_matched_sweep


def test_crad_score_and_parsing():
    assert normalized_creditor_utility(30, 120, 30) == 1.0
    assert normalized_creditor_utility(30, 120, 120) == 0.0
    assert math.isclose(normalized_creditor_utility(30, 120, 75), 0.5)
    assert normalized_creditor_utility(30, 120, None) == 0.0
    assert parse_offer_days("We can repay the balance in 60 days.") == 60
    assert parse_agreement_days("Agreed. We accept repayment within 55 days.") == 55
    assert parse_agreement_days("We propose 55 days instead.") is None


def test_fixed_crad_split():
    df = pd.DataFrame({"x": list(range(100))})
    train, test = split_crad(df)
    assert train["x"].tolist() == list(range(80))
    assert test["x"].tolist() == list(range(80, 100))


def test_glm_audio_token_helpers():
    assert audio_ids_to_prompt([3, 9]) == "<|begin_of_audio|><|audio_3|><|audio_9|><|end_of_audio|>"
    text, audio = partition_generated_token_ids([10, 1000, 1001, 11, 9999], audio_offset=1000, audio_vocab_size=4, stop_token_ids={9999})
    assert text == [10, 11] and audio == [0, 1]
    assert normalize_transcript("We need 30 days!") == normalize_transcript("we need 30 days")


def test_advantage_direction_cancels_state_content():
    records = [
        {"state_id": "s1", "utility": 0.0, "memory": np.array([9.0, 100.0])},
        {"state_id": "s1", "utility": 0.5, "memory": np.array([10.0, 100.0])},
        {"state_id": "s1", "utility": 1.0, "memory": np.array([11.0, 100.0])},
        {"state_id": "s2", "utility": 0.0, "memory": np.array([19.0, -70.0])},
        {"state_id": "s2", "utility": 0.5, "memory": np.array([20.0, -70.0])},
        {"state_id": "s2", "utility": 1.0, "memory": np.array([21.0, -70.0])},
    ]
    states, G = advantage_memory_directions(records)
    assert states == ["s1", "s2"]
    assert np.all(np.abs(G[:, 0]) > 0.99)
    assert np.all(np.abs(G[:, 1]) < 1e-8)


def test_low_rank_fit_and_overlap():
    G = np.array([[1.0, 0.0], [2.0, 0.05], [3.0, -0.05], [4.0, 0.0]])
    result = fit_low_rank_subspace(G, rank=1)
    assert abs(result.basis[:, 0][0]) > 0.99
    assert result.explained_variance > 0.99
    a = np.array([[1.0], [0.0]])
    assert np.isclose(subspace_overlap(a, a), 1.0)
    assert np.isclose(subspace_overlap(a, np.array([[0.0], [1.0]])), 0.0)


def test_ridge_code_recovery():
    C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    true_w = np.array([2.0, -1.0])
    assert np.allclose(fit_ridge_opponent_code(C, C @ true_w, ridge=1e-8), true_w, atol=1e-6)


def test_fused_qkv_split_and_delta():
    x = torch.arange(8.0).reshape(1, 1, 8)
    q, k, v = split_fused_qkv(x, num_attention_heads=2, kv_channels=2, multi_query_group_num=1)
    assert q.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert k.flatten().tolist() == [4.0, 5.0]
    assert v.flatten().tolist() == [6.0, 7.0]
    out = apply_kv_delta(torch.zeros(1, 1, 8), key_delta=torch.tensor([1.0, 2.0]), value_delta=torch.tensor([3.0, 4.0]), num_attention_heads=2, kv_channels=2, multi_query_group_num=1, scale=0.5)
    assert out[..., :4].abs().sum().item() == 0.0
    assert out[..., 4:6].flatten().tolist() == [0.5, 1.0]


def test_kv_recorder_and_steerer():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_key_value = nn.Linear(2, 8, bias=False)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block(), Block()])
        def forward(self, x):
            return [layer.query_key_value(x) for layer in self.layers]
    model = Toy()
    rec = FusedQKVRecorder(model, layer_indices=[0, 2], num_attention_heads=2, kv_channels=2, multi_query_group_num=1)
    with rec:
        model(torch.ones(1, 1, 2))
    assert rec.vector().numel() == 8
    for p in model.parameters():
        nn.init.zeros_(p)
    direction = torch.tensor([1.0, 0.0, 0.0, 2.0])
    with FusedQKVSteerer(model, layer_indices=[1], direction=direction, num_attention_heads=2, kv_channels=2, multi_query_group_num=1):
        outputs = model(torch.ones(1, 1, 2))
    assert outputs[0].abs().sum().item() == 0.0
    assert outputs[1][..., 4:6].flatten().tolist() == [1.0, 0.0]
    assert outputs[1][..., 6:8].flatten().tolist() == [0.0, 2.0]


def test_mock_sweep_keeps_semantics_fixed():
    df = pd.DataFrame([{"Creditor Name": "A", "Debtor Name": "B", "Creditor Target Days": 30, "Debtor Target Days": 120}])
    backend = MockSpeechBackend(style_offer_shift={"neutral": 0, "assertive": -10})
    rows = run_one_turn_matched_sweep(df, backend=backend, styles=["neutral", "assertive"], seeds=[0])
    assert len(rows) == 2
    assert len({r.semantic_id for r in rows}) == 1
    assert len({r.transcript for r in rows}) == 1
    by_style = {r.style: r for r in rows}
    assert by_style["assertive"].utility > by_style["neutral"].utility
