import pytest

def test_trivname():
    """
    Test prop_trivname
    """

    import wilson_suite as ws

    assert ws.utils.prop_trivname.prop_trivname(ord_geo=1, ord_el=0, ord_rot=0) == 'grad'
    assert ws.utils.prop_trivname.prop_trivname(ord_geo=1, ord_el=1, ord_rot=0) == 'dipgrad'
    assert ws.utils.prop_trivname.prop_trivname(ord_geo=0, ord_el=0, ord_rot=1) == 'B'
    assert ws.utils.prop_trivname.prop_trivname(ord_geo=3, ord_el=5, ord_rot=0) == 'thypcff'

    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(ord_geo=0, ord_el=3, ord_rot=1)
    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(ord_geo=3, ord_el=5, ord_rot=1)
    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(ord_geo=0, ord_el=7, ord_rot=0)
    with pytest.raises(AssertionError):
        a = ws.utils.prop_trivname.prop_trivname(ord_geo=0, ord_el=7, ord_rot=4)

    return