from ...wilson_utils.printing import printtest, separatorprint

import logging
from ...wilson_utils.serialization import check_if_jsonsafe
from dataclasses import dataclass

@dataclass
class Mock:
    a: int
    b: float

'''
def test_logger_with_caplog_serialization(caplog):
    separatorprint()
    # Set the logging level to capture all messages
    with caplog.at_level(logging.INFO, logger="wilson.wilson_utils.serialization"):
        check_if_jsonsafe(Mock(1, {(0,1): 'str'}))
    # Check the captured log messages
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "🔍 Offending object: {'a': 1, 'b': {(0, 1): 'str'}}"
    assert caplog.records[1].levelname == "ERROR"
    assert caplog.records[1].message == "❌ Not JSON-safe: keys must be str, int, float, bool or None, not tuple"
    log_output = caplog.text.replace("  ", " ")  # Replace multiple spaces with a single space
    assert "WARNING wilson.wilson_utils.serialization:serialization.py:60 🔍 Offending object: {'a': 1, 'b': {(0, 1): 'str'}}" in log_output
'''


def logfunc_mock():
    """
    A mock function with an intentionally incorrect usage of logger.info: 
        argument should be one string or a formatted string:
            logger.info(''+''+'fg') - is OK
            logger.debug(f"number is {number}")
    """
    import logging
    logger = logging.getLogger('testlogger')
    logger.info('', '', 'fg')


def test_log_messages():
    """
    Trying to show that argument to logger message functions should be one string or a formatted string.
    """
    import pytest
    from ...wilson_utils.logger import setup_logger
    setup_logger('testlogger', level=logging.INFO)
    # test logfunc_mock and ensure it raises a specific error if the log message is incorrect
    try:
        logfunc_mock()
        # fail test if no error raised
        pytest.fail("logfunc_mock() did not raise an error when it should have")
    except TypeError as e:
        assert "not all arguments converted during string formatting" in str(e), \
            f"Unexpected error message: {e}"