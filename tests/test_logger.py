from wilson_utils.printing import printtest, separatorprint

import logging
from wilson_utils.serialization import check_if_jsonsafe
from dataclasses import dataclass

@dataclass
class Mock:
    a: int
    b: float


def test_logger_with_caplog_serialization(caplog):
    separatorprint()
    # Set the logging level to capture all messages
    with caplog.at_level(logging.INFO, logger="wilson.wilson_utils.serialization"):
        check_if_jsonsafe(Mock(1, {(0,1): 'str'}))
    # Check the captured log messages
    printtest(caplog.records)
    printtest(len(caplog.records))
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "🔍 Offending object: {'a': 1, 'b': {(0, 1): 'str'}}"
    assert caplog.records[1].levelname == "ERROR"
    assert caplog.records[1].message == "❌ Not JSON-safe: keys must be str, int, float, bool or None, not tuple"
    log_output = caplog.text.replace("  ", " ")  # Replace multiple spaces with a single space
    assert "WARNING wilson.wilson_utils.serialization:serialization.py:60 🔍 Offending object: {'a': 1, 'b': {(0, 1): 'str'}}" in log_output

