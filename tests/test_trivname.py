import pytest

def test_trivname():
    """
    Test prop_trivname
    """

    import wilson_suite as ws

    assert ws.utils.prop_trivname.prop_trivname(1, 0, 0) == 'grad'
    assert ws.utils.prop_trivname.prop_trivname(1, 1, 0) == 'dipgrad'
    assert ws.utils.prop_trivname.prop_trivname(0, 0, 1) == 'B'
    assert ws.utils.prop_trivname.prop_trivname(3, 5, 0) == 'thypcff'

    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(0, 3, 1)
    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(3, 5, 1)
    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(0, 7, 0)
    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(0, 7, 4)

    return