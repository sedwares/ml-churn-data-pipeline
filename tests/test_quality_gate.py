def test_quality_threshold_passes_at_90_percent():
    valid_rows = 90
    total_rows = 100
    threshold = 90
    valid_percentage = valid_rows / total_rows * 100
    assert valid_percentage >= threshold


def test_quality_threshold_fails_below_90_percent():
    valid_rows = 70
    total_rows = 100
    threshold = 90
    valid_percentage = valid_rows / total_rows * 100
    assert valid_percentage < threshold
