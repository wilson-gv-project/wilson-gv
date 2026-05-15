import pytest

from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos
from wilson_suite.wilson_derive.abstractions import QOperator, VibStateSymbolic
from wilson_suite.wilson_utils import common_labels as wu_common

def test_get_vib_sos():

    order = 6

    R_sos_new = get_vib_sos(order)
