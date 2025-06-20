import sys
import os
from nProphet import NProphetForecaster


def _minimal_forecaster(monkeypatch):
    monkeypatch.setattr(NProphetForecaster, "_connect_db", lambda self: None)
    return NProphetForecaster({"SEED": 1})


def test_suppress_stdout_hides_stderr(monkeypatch, capsys):
    with NProphetForecaster.suppress_stdout():
        fc = _minimal_forecaster(monkeypatch)

    def emit():
        print("visible")
        print("error", file=sys.stderr)
        # simulate C-level writes directly to file descriptors
        os.write(1, b"native")
        os.write(2, b"native")

    with fc.suppress_stdout():
        emit()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


