from scripts.compare_shortest_path import summarize


def test_summarize_single_value() -> None:
    mean, std, ci95 = summarize([2.5])
    assert mean == 2.5
    assert std == 0.0
    assert ci95 == 0.0


def test_summarize_multiple_values() -> None:
    mean, std, ci95 = summarize([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert std > 0.0
    assert ci95 > 0.0


def test_summarize_rejects_empty_input() -> None:
    try:
        summarize([])
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
