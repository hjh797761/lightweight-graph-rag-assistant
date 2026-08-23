from graphrag.normalization import normalize_evidence


def test_normalization_is_generic():
    assert normalize_evidence("收入 1,234.50 万元，同比增长 8 %") == "收入1234.50万元同比增长8%"
