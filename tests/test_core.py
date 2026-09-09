import math
import numpy as np
import pandas as pd
import torch
from torch import nn

from speech_negotiation_kv.calibration import fit_ridge_opponent_code
from speech_negotiation_kv.crad import normalized_creditor_utility, parse_offer_days, parse_agreement_days, split_crad
from speech_negotiation_kv.glm_voice import VoiceGeneration, audio_ids_to_prompt, partition_generated_token_ids, normalize_transcript
from speech_negotiation_kv.kv_hooks import (
    split_fused_qkv,
    apply_kv_delta,
    FusedQKVRecorder,
    FusedQKVSteerer,
    pool_kv_tokens,
)
from speech_negotiation_kv.long_horizon import (
    is_strategically_valid_move,
    negotiation_turn_prompt,
    paired_turn_seed,
    parse_negotiation_move,
    run_long_horizon_branch,
    score_terminal_outcome,
)
from speech_negotiation_kv.long_horizon_analysis import style_ranking_correlations
from speech_negotiation_kv.subspace import (
    advantage_memory_directions,
    balanced_scenario_splits,
    fit_low_rank_subspace,
    subspace_overlap,
)
from speech_negotiation_kv.observations import observation_audio_ids
from speech_negotiation_kv.strategy_geometry import (
    benjamini_hochberg,
    center_within_states,
    cosine_centroid_accuracy,
    pairwise_style_directions,
    shuffle_labels_within_states,
)
from speech_negotiation_kv.gate_f import evaluate_state_predictions, leave_one_scenario_out
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


def test_negotiation_move_parser_is_conservative_about_terminal_outcomes():
    agreement = parse_negotiation_move("AGREED: 45 days.", latest_offer=45)
    assert agreement.kind == "agreement" and agreement.days == 45
    inferred = parse_negotiation_move("I accept your latest proposal.", latest_offer=50)
    assert inferred.kind == "agreement" and inferred.days == 50
    assert parse_negotiation_move("NO DEAL. We are ending talks.", latest_offer=45).kind == "no_deal"
    proposal = parse_negotiation_move("PROPOSE: 60 days.", latest_offer=45)
    assert proposal.kind == "proposal" and proposal.days == 60
    negated = parse_negotiation_move(
        "I can't agree to 30 days. Let's aim for 21 days.", latest_offer=30
    )
    assert negated.kind == "proposal" and negated.days == 21
    assert parse_negotiation_move("Let us keep discussing.", latest_offer=45).kind == "invalid"


def test_terminal_scoring_and_paired_turn_seed():
    assert math.isclose(score_terminal_outcome("agreement", 45, 30, 120), 5 / 6)
    assert score_terminal_outcome("no_deal", None, 30, 120) == 0.0
    assert score_terminal_outcome("censored", None, 30, 120) is None
    assert paired_turn_seed(4242424242, branch_seed=3, transition=2, actor="creditor") == \
        paired_turn_seed(4242424242, branch_seed=3, transition=2, actor="creditor")
    assert paired_turn_seed(4242424242, branch_seed=3, transition=2, actor="creditor") != \
        paired_turn_seed(4242424242, branch_seed=3, transition=2, actor="debtor")


def test_strategic_move_validation_enforces_counteroffer_direction():
    assert is_strategically_valid_move(
        parse_negotiation_move("PROPOSE: 45 days", latest_offer=60),
        role="creditor", latest_offer=60, creditor_target=30, debtor_target=120, transition=2,
    )
    assert not is_strategically_valid_move(
        parse_negotiation_move("PROPOSE: 60 days", latest_offer=60),
        role="creditor", latest_offer=60, creditor_target=30, debtor_target=120, transition=2,
    )
    assert is_strategically_valid_move(
        parse_negotiation_move("PROPOSE: 70 days", latest_offer=60),
        role="debtor", latest_offer=60, creditor_target=30, debtor_target=120, transition=2,
    )
    assert is_strategically_valid_move(
        parse_negotiation_move("AGREED: 60 days", latest_offer=60),
        role="debtor", latest_offer=60, creditor_target=30, debtor_target=120, transition=2,
    )


def test_long_horizon_branch_restores_neutral_policy_and_terminates():
    class Backend:
        def __init__(self):
            self.calls = []

        def render_exact(self, text, style, *, seed):
            self.calls.append(("renderer", style, seed, 0))
            return VoiceGeneration(text, [101], [101])

        def respond_audio(self, audio_ids, scenario, *, seed):
            self.calls.append(("debtor_opening", "base", seed, 1))
            return VoiceGeneration("PROPOSE: 60 days.", [60], [60])

        def respond_negotiation_turn(self, *, role, opponent_audio_ids, scenario, history,
                                     transition, seed, policy_style, force_terminal=False):
            self.calls.append((role, policy_style, seed, transition))
            if role == "creditor" and transition == 2:
                return VoiceGeneration("PROPOSE: 45 days.", [45], [45])
            if role == "debtor":
                return VoiceGeneration("PROPOSE: 55 days.", [55], [55])
            return VoiceGeneration("AGREED: 55 days.", [54], [54])

    scenario = {
        "Creditor Name": "A", "Debtor Name": "B",
        "Creditor Target Days": 30, "Debtor Target Days": 120,
    }
    backend = Backend()
    result = run_long_horizon_branch(
        scenario, scenario_id=0, style="firm and assertive", branch_seed=2,
        backend=backend, horizon=4, max_horizon=4, base_seed=100,
    )
    assert result["outcome"] == "agreement"
    assert result["agreement_days"] == 55
    assert result["rounds"] == 2
    assert result["policy_valid"] is True
    assert result["turns"][0]["policy_style"] == "firm and assertive"
    assert backend.calls[1][0] == "debtor_opening"
    assert all(call[1] == "neutral" for call in backend.calls[2:])


def test_long_horizon_branch_keeps_unresolved_outcome_censored():
    class Backend:
        def render_exact(self, text, style, *, seed):
            return VoiceGeneration(text, [101], [101])

        def respond_audio(self, audio_ids, scenario, *, seed):
            return VoiceGeneration("PROPOSE: 70 days.", [70], [70])

        def respond_negotiation_turn(self, *, role, opponent_audio_ids, scenario, history,
                                     transition, seed, policy_style, force_terminal=False):
            days = 70 if role == "debtor" else 40
            return VoiceGeneration(f"PROPOSE: {days} days.", [days], [days])

    scenario = {
        "Creditor Name": "A", "Debtor Name": "B",
        "Creditor Target Days": 30, "Debtor Target Days": 120,
    }
    result = run_long_horizon_branch(
        scenario, scenario_id=0, style="neutral", branch_seed=0,
        backend=Backend(), horizon=4, max_horizon=4, base_seed=100,
    )
    assert result["outcome"] == "censored"
    assert result["utility"] is None
    assert result["rounds"] == 4


def test_negotiation_turn_prompt_fixes_role_policy_and_output_contract():
    scenario = {
        "Creditor Name": "A", "Debtor Name": "B",
        "Creditor Target Days": 30, "Debtor Target Days": 120,
    }
    prompt = negotiation_turn_prompt(
        role="creditor", opponent_audio_ids=[3, 9], scenario=scenario,
        history=[
            {"actor": "creditor", "move_kind": "proposal", "move_days": 30},
            {"actor": "debtor", "move_kind": "proposal", "move_days": 90},
        ],
        transition=2, policy_style="neutral",
    )
    assert "You are A, the creditor" in prompt
    assert "prefer repayment within 30 days" in prompt
    assert "neutral" in prompt
    assert "AGREED: N days" in prompt and "NO DEAL" in prompt and "PROPOSE: N days" in prompt
    assert "must counteroffer" in prompt
    assert "may accept the exact latest proposal" in prompt
    assert "do not use NO DEAL" in prompt
    assert "strictly fewer days" in prompt
    assert "do not repeat the opponent's number" in prompt
    assert "between 30 and 89 days" in prompt
    assert "Latest opponent proposal is 90 days" in prompt
    assert "Never output PROPOSE: 90 days" in prompt
    assert "audio tokens are required" in prompt
    assert "Creditor proposed 30 days" in prompt and "Debtor proposed 90 days" in prompt
    assert "Choose the next move from the structured history" in prompt
    assert "<|audio_3|><|audio_9|>" in prompt


def test_negotiation_turn_prompt_includes_terminal_contract_for_final_subset_turn():
    scenario = {
        "Creditor Name": "A", "Debtor Name": "B",
        "Creditor Target Days": 30, "Debtor Target Days": 120,
    }
    prompt = negotiation_turn_prompt(
        role="debtor", opponent_audio_ids=[7], scenario=scenario,
        history=[
            {"actor": "creditor", "move_kind": "proposal", "move_days": 45},
        ],
        transition=8, policy_style="neutral", force_terminal=True,
    )
    assert "<|audio_7|>" in prompt
    assert "final decision" in prompt
    assert "must output either AGREED:" in prompt and "or NO DEAL" in prompt
    assert "must not use PROPOSE" in prompt


def test_long_horizon_style_ranking_detects_reversal():
    rows = []
    for state in ("s0", "s1"):
        for index, style in enumerate(("a", "b", "c")):
            rows.append({
                "state_id": state,
                "style": style,
                "immediate_offer_utility": float(index),
                "utility": float(2 - index),
            })
    result = style_ranking_correlations(rows, expected_styles={"a", "b", "c"})
    assert result["n_complete_states"] == 2
    assert result["spearman"]["median"] == -1.0
    assert result["best_style_agreement_rate"] == 0.0


def test_exploratory_gate_f_uses_scenario_held_out_state_metrics():
    rows = [
        {"state_id": "s0", "scenario_id": 0, "style": "a", "utility": 0.1, "prediction": 0.2},
        {"state_id": "s0", "scenario_id": 0, "style": "b", "utility": 0.9, "prediction": 0.8},
        {"state_id": "s1", "scenario_id": 1, "style": "a", "utility": 0.8, "prediction": 0.7},
        {"state_id": "s1", "scenario_id": 1, "style": "b", "utility": 0.2, "prediction": 0.3},
    ]
    result = evaluate_state_predictions(rows, expected_styles={"a", "b"})
    assert result["n_complete_states"] == 2
    assert math.isclose(result["spearman"]["median"], 1.0)
    assert result["top1_accuracy"] == 1.0
    assert leave_one_scenario_out([0, 1]) == [({1}, {0}), ({0}, {1})]


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


def test_kv_pooling_can_isolate_audio_positions():
    k = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    v = torch.tensor([[[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]])
    mask = torch.tensor([[False, True, True]])
    k_audio, v_audio = pool_kv_tokens(k, v, mode="audio_only", token_mask=mask)
    assert k_audio.tolist() == [2.5, 25.0]
    assert v_audio.tolist() == [5.5, 55.0]
    k_last, v_last = pool_kv_tokens(k, v, mode="last_audio", token_mask=mask)
    assert k_last.tolist() == [3.0, 30.0]
    assert v_last.tolist() == [6.0, 60.0]


def test_observation_audio_ids_selects_action_or_response():
    rec = {"audio_token_ids": [1, 2], "opponent_audio_token_ids": [3, 4]}
    assert observation_audio_ids(rec, "action") == [1, 2]
    assert observation_audio_ids(rec, "response") == [3, 4]


def test_balanced_scenario_splits_enumerates_unique_complements():
    splits = balanced_scenario_splits(list(range(10)), half_size=5)
    assert len(splits) == 126
    assert all(len(a) == len(b) == 5 and a.isdisjoint(b) for a, b in splits)
    assert len({frozenset(a) for a, _ in splits}) == 126


def test_state_centered_style_geometry_transfers_across_scenarios():
    features = np.array([
        [11.0, 0.0], [9.0, 0.0],
        [21.0, 2.0], [19.0, 2.0],
        [-4.0, 7.0], [-6.0, 7.0],
        [4.0, -3.0], [2.0, -3.0],
    ])
    states = np.array(["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3"])
    styles = np.array(["a", "b"] * 4)
    scenarios = np.repeat(np.arange(4), 2)
    centered = center_within_states(features, states)
    assert np.allclose(centered.mean(axis=0), 0.0)
    accuracy, _ = cosine_centroid_accuracy(centered, styles, scenarios, {0, 1})
    assert accuracy == 1.0


def test_pairwise_style_direction_is_stable_across_scenarios():
    features = np.array([[1.0, 0.0], [-1.0, 0.0]] * 3)
    states = np.array(["s0", "s0", "s1", "s1", "s2", "s2"])
    styles = np.array(["a", "b"] * 3)
    scenarios = np.repeat(np.arange(3), 2)
    result = pairwise_style_directions(features, styles, scenarios, states)
    assert result[("a", "b")]["cross_scenario_mean_cosine"] > 0.99


def test_style_shuffle_is_within_state_and_bh_is_monotone():
    labels = np.array(["a", "b", "c", "a", "b", "c"])
    states = np.array(["s0"] * 3 + ["s1"] * 3)
    shuffled = shuffle_labels_within_states(labels, states, np.random.default_rng(7))
    for state in ("s0", "s1"):
        mask = states == state
        assert sorted(shuffled[mask]) == sorted(labels[mask])
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_mock_sweep_keeps_semantics_fixed():
    df = pd.DataFrame([{"Creditor Name": "A", "Debtor Name": "B", "Creditor Target Days": 30, "Debtor Target Days": 120}])
    backend = MockSpeechBackend(style_offer_shift={"neutral": 0, "assertive": -10})
    rows = run_one_turn_matched_sweep(df, backend=backend, styles=["neutral", "assertive"], seeds=[0])
    assert len(rows) == 2
    assert len({r.semantic_id for r in rows}) == 1
    assert len({r.transcript for r in rows}) == 1
    by_style = {r.style: r for r in rows}
    assert by_style["assertive"].utility > by_style["neutral"].utility
