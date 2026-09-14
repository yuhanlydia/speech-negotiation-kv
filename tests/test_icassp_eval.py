import json

from speech_negotiation_kv.icassp_eval import (
    bootstrap_preference, choose_alpha, latex_escape, pairwise_preference,
    paired_wer_degradation, preference_by_dimension, render_ablation_table,
    render_geometry_table, render_main_results_table, render_result_macros,
)


def test_pairwise_preference_supports_dynamic_and_multidimension_formats():
    rows = [
        {"status": "Success", "candidate_position": 1, "winner_position": 1, "dimensions": ["Pitch"]},
        {"status": "Success", "candidate_position": 2, "winner_position": 0, "dimensions": ["Pitch"]},
        {"status": "Success", "candidate_position": 2, "winner_position_1": 2, "winner_position_2": 1, "dimensions": ["Pitch", "Pace"]},
    ]
    result = pairwise_preference(rows)
    assert result["n_samples"] == 3
    assert result["wins"] == 2 and result["ties"] == 1 and result["losses"] == 1
    assert result["preference_score"] == 100 * ((1 + .5 + .5) / 3)
    by_dim = preference_by_dimension(rows)
    assert by_dim["Pitch"]["n"] == 3 and by_dim["Pace"]["preference_score"] == 0.0


def test_bootstrap_is_reproducible_and_centered_on_preference():
    rows = [{"candidate_position": 1, "winner_position": 1, "dimensions": ["Pitch"]}] * 5
    a = bootstrap_preference(rows, repeats=100, seed=3); b = bootstrap_preference(rows, repeats=100, seed=3)
    assert a == b and a["preference_score"] == 100.0 and a["gain_vs_tie"] == 50.0


def test_paired_wer_and_alpha_selection(tmp_path):
    baseline, candidate = tmp_path / "baseline.jsonl", tmp_path / "candidate.jsonl"
    baseline.write_text('\n'.join([json.dumps({"item_id": "a", "text_channel_wer": .1}), json.dumps({"item_id": "b", "text_channel_wer": .2})]) + '\n')
    candidate.write_text('\n'.join([json.dumps({"item_id": "a", "text_channel_wer": .11}), json.dumps({"item_id": "b", "text_channel_wer": .21})]) + '\n')
    assert abs(paired_wer_degradation(candidate, baseline)["mean_degradation"] - .01) < 1e-12
    selected = choose_alpha([{"alpha": .25, "preference_score": 60, "wer_degradation": .01},
                             {"alpha": .5, "preference_score": 70, "wer_degradation": .03},
                             {"alpha": 1.0, "preference_score": 60, "wer_degradation": .01}])
    assert selected["alpha"] == .25


def test_latex_renderers_emit_expected_tables_and_macros():
    assert latex_escape("Pitch_rate&x") == r"Pitch\_rate\&x"
    summary = {"decision": "continue_to_paper",
        "geometry": {"attributes": 80, "leave_one_content_out_centroid_accuracy": .75, "chance": .0125,
                     "explained_variance_at_main_rank": .92, "same_attribute_cross_content_cosine": {"mean": .64}},
        "tasks": {
            "static": {"main": {"preference_score": 56, "gain_vs_tie": 6, "lower_95": 52, "upper_95": 60}, "random": {"preference_score": 49}, "fidelity": {"mean_degradation": .01}},
            "composed": {"main": {"preference_score": 60, "gain_vs_tie": 10, "lower_95": 55, "upper_95": 65}, "random": {"preference_score": 50}, "fidelity": {"mean_degradation": 0}},
            "dynamic": {"main": {"preference_score": 59, "gain_vs_tie": 9, "lower_95": 53, "upper_95": 64}, "random": {"preference_score": 48}, "fidelity": {"mean_degradation": .015}}},
        "ablations": {"composed": {"comp_mean": {"preference_score": 54}}}}
    assert "Composition" in render_main_results_table(summary)
    assert "LOCO acc." in render_geometry_table(summary)
    assert "composition: mean" in render_ablation_table(summary)
    macros = render_result_macros(summary)
    assert r"\newcommand{\CompGain}{10.0}" in macros
    assert r"\newcommand{\GeometryCosine}{0.640}" in macros
