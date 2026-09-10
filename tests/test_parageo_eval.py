from speech_negotiation_kv.parageo_eval import select_scale, pilot_decision


def test_select_scale_uses_dev_score_and_smallest_tie():
    assert select_scale({0.25: 61.0, 0.5: 64.0, 1.0: 64.0, 1.5: 60.0}) == 0.5


def test_pilot_decision_requires_gain_fidelity_and_nonrandom_control():
    result = pilot_decision(
        baseline={"static": 60.0, "dynamic": 40.0},
        ours={"static": 66.0, "dynamic": 45.0},
        random={"static": 61.0, "dynamic": 42.0},
        wer_degradation={"static": 0.01, "dynamic": 0.00},
        static_min_gain=5.0,
        dynamic_min_gain=8.0,
        max_wer_degradation=0.02,
    )
    assert result["parageo_pilot_passes"] is True
    blocked = pilot_decision(
        baseline={"static": 60.0}, ours={"static": 66.0}, random={"static": 66.0},
        wer_degradation={"static": 0.0}, static_min_gain=5.0,
        dynamic_min_gain=8.0, max_wer_degradation=0.02,
    )
    assert blocked["parageo_pilot_passes"] is False
