#!/usr/bin/env python3
"""Build the single-file Kaggle entrypoint for the LOSO rerun."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "train.py"
RUNNER = ROOT / "train_loso.py"
PROTOCOL = ROOT / "loso_protocol.py"
DEFAULT_OUTPUT = ROOT / "train_loso_bundle.py"
MAIN_GUARD = '\nif __name__ == "__main__":\n    main()\n'
RUNNER_START = "SEED = 23\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    base_source = BASE.read_text()
    if MAIN_GUARD not in base_source:
        raise ValueError("Could not locate the v5 train.py main guard")
    base_module = base_source.split(MAIN_GUARD, 1)[0].rstrip()

    runner_source = RUNNER.read_text()
    if RUNNER_START not in runner_source:
        raise ValueError("Could not locate the LOSO runner body")
    runner_body = runner_source.split(RUNNER_START, 1)[1]

    protocol_source = PROTOCOL.read_text()
    protocol_body = protocol_source.split("def leave_one_animal_out", 1)[1]
    protocol_body = "def leave_one_animal_out" + protocol_body

    base_hash = hashlib.sha256(BASE.read_bytes()).hexdigest()
    bundled = "\n\n".join([
        base_module,
        "from collections.abc import Sequence\n\n" + protocol_body.rstrip(),
        "import sys\ntrain = sys.modules[__name__]\nBASE_V5_SOURCE_SHA256 = " + repr(base_hash),
        RUNNER_START + runner_body.rstrip(),
    ]) + "\n"
    output.write_text(bundled)
    return output


if __name__ == "__main__":
    import sys
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT))
