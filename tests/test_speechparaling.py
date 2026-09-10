from speech_negotiation_kv.speechparaling import (
    extract_target_text,
    parse_static_control,
    parse_dynamic_control,
)


def test_static_control_parser_matches_benchmark_prompt_shape():
    prompt = "Please read this sentence with a very high pitch: 'Ah! There is a mouse!'"
    assert parse_static_control(prompt) == "very high pitch"
    assert extract_target_text(prompt) == "Ah! There is a mouse!"


def test_dynamic_parser_supports_gradual_and_step_transitions():
    gradual = "Please read this sentence starting with a very low pitch and gradually transitioning to a medium pitch: 'Late at night.'"
    assert parse_dynamic_control(gradual) == ("very low pitch", "medium pitch", "linear")
    sudden = "Please read this sentence starting with a high pitch and suddenly dropping to a very low pitch: 'Wait!'"
    assert parse_dynamic_control(sudden) == ("high pitch", "very low pitch", "step")


def test_dynamic_parser_handles_volume_wording():
    prompt = "Please read this sentence starting with a whisper and gradually increasing the volume to a normal volume: 'Did you hear?'"
    assert parse_dynamic_control(prompt) == ("whisper", "normal volume", "linear")


def test_catalog_matching_supports_compositional_multi_control():
    from speech_negotiation_kv.speechparaling import match_catalog_controls
    catalog = {
        "Pitch::very high pitch": {"control": "very high pitch"},
        "Pace::fast pace": {"control": "fast pace"},
        "Pitch::high pitch": {"control": "high pitch"},
    }
    prompt = "Please read this sentence with a very high pitch and a fast pace: 'Look!'"
    result = match_catalog_controls(prompt, catalog)
    assert "Pitch::very high pitch" in result
    assert "Pace::fast pace" in result
    assert "Pitch::high pitch" not in result


def test_word_error_rate_is_zero_for_identical_content():
    from speech_negotiation_kv.speechparaling import word_error_rate
    assert word_error_rate("Hello, world!", "hello world") == 0.0
    assert word_error_rate("one two", "one three") == 0.5
