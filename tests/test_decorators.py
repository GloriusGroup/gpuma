import logging
import time

from gpuma.decorators import time_it, timed_block


def test_time_it(caplog):
    @time_it
    def sample_func(x):
        return x + 1

    with caplog.at_level(logging.INFO):
        res = sample_func(1)
        assert res == 2

    assert "Function: 'sample_func' took:" in caplog.text

def test_time_it_wraps():
    @time_it
    def sample_func(x):
        """Docstring."""
        return x

    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Docstring."


def test_timed_block_logs_and_stores_elapsed(caplog):
    with caplog.at_level(logging.INFO):
        with timed_block("test block") as tb:
            time.sleep(0.01)

    assert tb.elapsed >= 0.01
    assert "test block took" in caplog.text


def test_timed_block_custom_level(caplog):
    with caplog.at_level(logging.DEBUG):
        with timed_block("debug block", level=logging.DEBUG) as tb:
            pass

    assert tb.elapsed >= 0.0
    assert "debug block took" in caplog.text
