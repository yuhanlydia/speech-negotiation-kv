import math
import numpy as np

from speech_negotiation_kv.short_horizon_selector import (
    teacher_distribution,
    state_spread_weights,
    opening_state_features,
    prototype_style_coordinates,
    fit_geometry_selector,
    fit_onehot_selector,
    score_geometry_selector,
    score_onehot_selector,
    evaluate_selector_states,
    paired_bootstrap_mean_delta,
    pairwise_unseen_style_accuracy,
    relabel_teacher_rows,
    teacher_targets_and_weights,
)


def test_teacher_distribution_is_tie_aware_and_spread_weight_zero_for_flat_state():
    p = teacher_distribution(np.array([0.2, 0.8, 0.8]), temperature=0.1)
    assert p[1] == p[2]
    assert p[1] > p[0]
    flat = state_spread_weights(np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]]))
    assert np.allclose(flat, [0.0, 0.8])


def test_teacher_targets_use_soft_distribution_and_raw_utility_spread_weights():
    utilities = np.array([0.0, 1.0, 0.5, 0.5])
    targets, weights = teacher_targets_and_weights(
        utilities, ["s0", "s0", "s1", "s1"], temperature=0.1
    )
    assert targets[1] > targets[0]
    assert np.allclose(targets[2:], [0.5, 0.5])
    assert np.allclose(weights, [1.0, 1.0, 0.0, 0.0])


def test_opening_state_features_are_finite_and_scale_invariant_in_ratio_feature():
    a = opening_state_features(30, 120)
    b = opening_state_features(60, 240)
    assert a.shape == (5,)
    assert np.isfinite(a).all()
    assert math.isclose(a[-1], b[-1])


def test_style_prototypes_recover_low_rank_coordinates():
    features = np.array([
        [1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [-1.0, 0.0, 10.0],
        [1.1, 0.0, -4.0], [0.0, 1.1, -4.0], [-1.1, 0.0, -4.0],
    ])
    styles = np.array(["a", "b", "c", "a", "b", "c"])
    coords, metadata = prototype_style_coordinates(features, styles, ["a", "b", "c"], rank=2)
    assert set(coords) == {"a", "b", "c"}
    assert metadata["rank"] == 2
    assert np.linalg.norm(coords["a"] - coords["c"]) > 1.0


def _synthetic_rows():
    styles = ["a", "b", "c"]
    coords = {"a": np.array([-1.0]), "b": np.array([0.0]), "c": np.array([1.0])}
    state_ids = []
    state_features = []
    style_coords = []
    style_labels = []
    utilities = []
    for s, x in enumerate([-2.0, -1.0, 1.0, 2.0]):
        for style in styles:
            c = coords[style][0]
            utilities.append(0.5 + 0.2 * x * c)
            state_ids.append(f"s{s}")
            state_features.append([x])
            style_coords.append([c])
            style_labels.append(style)
    return (
        np.asarray(state_features), np.asarray(style_coords), np.asarray(style_labels),
        np.asarray(utilities), np.asarray(state_ids), styles, coords,
    )


def test_geometry_and_onehot_selectors_fit_state_conditioned_rankings():
    phi, c, labels, y, state_ids, styles, _ = _synthetic_rows()
    g = fit_geometry_selector(phi, c, y, state_ids, ridge=1e-6)
    gp = score_geometry_selector(g, phi, c)
    o = fit_onehot_selector(phi, labels, y, state_ids, styles=styles, ridge=1e-6)
    op = score_onehot_selector(o, phi, labels)
    assert np.corrcoef(gp, y)[0, 1] > 0.99
    assert np.corrcoef(op, y)[0, 1] > 0.99


def test_geometry_selector_accepts_explicit_teacher_row_weights():
    phi = np.zeros((4, 1))
    coords = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    targets = np.array([0.0, 1.0, 1.0, 0.0])
    states = np.array(["signal", "signal", "ignored", "ignored"])
    model = fit_geometry_selector(
        phi, coords, targets, states, ridge=1e-6,
        row_weights=np.array([1.0, 1.0, 0.0, 0.0]),
    )
    prediction = score_geometry_selector(model, phi[:2], coords[:2])
    assert prediction[1] > prediction[0]


def test_state_metrics_are_tie_aware_and_report_regret():
    rows = [
        {"state_id": "s", "style": "a", "utility": 0.9, "prediction": 1.0},
        {"state_id": "s", "style": "b", "utility": 0.7, "prediction": 1.0},
        {"state_id": "s", "style": "c", "utility": 0.1, "prediction": 0.0},
    ]
    result = evaluate_selector_states(rows, expected_styles={"a", "b", "c"})
    assert result["n_complete_states"] == 1
    assert result["top1_agreement_rate"] == 1.0
    assert math.isclose(result["regret"]["mean"], 0.1)
    assert math.isclose(result["selected_utility"]["mean"], 0.8)


def test_bootstrap_delta_and_unseen_pairwise_accuracy():
    boot = paired_bootstrap_mean_delta([0.8, 0.9, 1.0], [0.4, 0.5, 0.6], repeats=500, seed=1)
    assert math.isclose(boot["mean_delta"], 0.4)
    assert boot["lower_95"] > 0
    rows = [
        {"state_id": "s1", "style": "u", "utility": 0.9, "prediction": 0.8},
        {"state_id": "s1", "style": "a", "utility": 0.2, "prediction": 0.1},
        {"state_id": "s1", "style": "b", "utility": 0.5, "prediction": 0.4},
        {"state_id": "s2", "style": "u", "utility": 0.1, "prediction": 0.2},
        {"state_id": "s2", "style": "a", "utility": 0.8, "prediction": 0.7},
        {"state_id": "s2", "style": "b", "utility": 0.6, "prediction": 0.5},
    ]
    assert pairwise_unseen_style_accuracy(rows, held_out_style="u", seen_styles={"a", "b"}) == 1.0


def test_geometry_selector_can_value_unseen_style_coordinate():
    coords = {"a": -1.0, "b": -0.3, "c": 0.4, "d": 1.0}
    held_out = "b"
    train_phi, train_c, train_y, train_states = [], [], [], []
    for state_index, x in enumerate([-2.0, -1.0, 1.0, 2.0]):
        for style, c in coords.items():
            if style == held_out:
                continue
            train_phi.append([x])
            train_c.append([c])
            train_y.append(0.5 + 0.1 * x * c)
            train_states.append(f"s{state_index}")
    model = fit_geometry_selector(
        np.asarray(train_phi), np.asarray(train_c), np.asarray(train_y), np.asarray(train_states), ridge=1e-6
    )
    test_phi = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    test_c = np.full((4, 1), coords[held_out])
    prediction = score_geometry_selector(model, test_phi, test_c)
    truth = np.asarray([0.5 + 0.1 * x * coords[held_out] for x in [-2.0, -1.0, 1.0, 2.0]])
    assert np.corrcoef(prediction, truth)[0, 1] > 0.99


def test_relabel_teacher_rows_reparses_proposal_and_marks_truncation_missing():
    rows = [
        {
            "scenario_id": 30,
            "opponent_transcript": "20 days is too tight. How about 120 days?",
            "opponent_offer_days": 20,
            "utility": 1.0,
        },
        {
            "scenario_id": 30,
            "opponent_transcript": "Our target is 158 days. What about",
            "opponent_offer_days": 158,
            "utility": 0.0,
        },
    ]
    corrected = relabel_teacher_rows(rows, scenario_targets={30: (20, 158)})
    assert corrected[0]["opponent_offer_days"] == 120
    assert math.isclose(corrected[0]["utility"], (158 - 120) / (158 - 20))
    assert corrected[0]["original_opponent_offer_days"] == 20
    assert corrected[1]["opponent_offer_days"] is None
    assert corrected[1]["utility"] is None
