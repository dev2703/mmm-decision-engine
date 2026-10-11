"""Test-session environment shared by every test module."""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _darwin_linker_rejects_pytensor_flags() -> bool:
    """PyTensor <=2.35 passes `-ld64` on macOS>=15; newer Apple clang rejects it."""
    compiler = shutil.which("clang++")
    if sys.platform != "darwin" or compiler is None:
        return False
    if int(platform.mac_ver()[0].split(".")[0]) < 15:
        return False
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "probe.cpp"
        source.write_text("int probe() { return 0; }\n")
        result = subprocess.run(
            [
                compiler,
                "-dynamiclib",
                "-undefined",
                "dynamic_lookup",
                "-ld64",
                str(source),
                "-o",
                str(Path(directory) / "probe.so"),
            ],
            capture_output=True,
            check=False,
        )
    return result.returncode != 0


# Must run before any test imports PyTensor; an explicit user setting always wins.
if "PYTENSOR_FLAGS" not in os.environ and _darwin_linker_rejects_pytensor_flags():
    os.environ["PYTENSOR_FLAGS"] = "cxx="
