"""Verify the installed package boundary without relying on repository imports."""

import subprocess
import sys
from pathlib import Path


def test_package_imports_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import decisionguard; import decisionguard.config",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
