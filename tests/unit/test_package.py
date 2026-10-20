"""Enforce the educational engine's independence from production frameworks."""

import subprocess
import sys


def test_numpy_core_imports_without_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import neuropulse.nn_numpy; "
            "import neuropulse.contracts; "
            "assert 'torch' not in sys.modules; assert 'fastapi' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_rejects_missing_configuration_file():
    result = subprocess.run(
        [sys.executable, "-m", "neuropulse", "--config", "does-not-exist.yaml"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Configuration error" in result.stderr
