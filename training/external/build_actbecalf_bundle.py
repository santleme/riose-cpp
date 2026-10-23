#!/usr/bin/env python3
"""Bundle the v5 CNR parser, ActBeCalf loader, and transfer runner for Kaggle."""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "train.py"
LOADER = HERE / "actbecalf.py"
RUNNER = HERE / "train_actbecalf_transfer.py"
OUTPUT = HERE / "train_actbecalf_transfer_bundle.py"
GUARD = '\nif __name__ == "__main__":\n    main()\n'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_header(source: str) -> str:
    lines = source.splitlines()
    if lines and lines[0].startswith("#!"):
        lines.pop(0)
    while lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
        quote = '"""' if lines[0].startswith('"""') else "'''"
        if lines[0].count(quote) >= 2:
            lines.pop(0)
        else:
            lines.pop(0)
            while lines and quote not in lines[0]:
                lines.pop(0)
            if lines:
                lines.pop(0)
    return "\n".join(line for line in lines if line != "from __future__ import annotations").strip()


def build() -> Path:
    base = BASE.read_text()
    if GUARD not in base:
        raise ValueError("Could not find the v5 training module main guard")
    base_module = base.split(GUARD, 1)[0].rstrip()
    loader_source = LOADER.read_text()
    if GUARD not in loader_source:
        raise ValueError("Could not find the ActBeCalf loader main guard")
    loader_module = loader_source.split(GUARD, 1)[0]
    loader = strip_header(loader_module)
    runner = RUNNER.read_text()
    if GUARD not in runner:
        raise ValueError("Could not find the transfer runner main guard")
    runner_module = runner.split(GUARD, 1)[0]
    runner_body = strip_header(runner_module)
    bundle = "\n\n".join((
        base_module,
        "import hashlib\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n" + loader,
        "import argparse\nimport json\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom sklearn.ensemble import ExtraTreesClassifier\nfrom sklearn.metrics import classification_report, confusion_matrix, f1_score\n" + runner_body,
        f"BASE_TRAIN_SOURCE_SHA256 = {sha256(BASE)!r}\nACTBECALF_LOADER_SOURCE_SHA256 = {sha256(LOADER)!r}\nTRANSFER_RUNNER_SOURCE_SHA256 = {sha256(RUNNER)!r}",
        'if __name__ == "__main__":\n    main()',
    )) + "\n"
    OUTPUT.write_text(bundle)
    return OUTPUT


if __name__ == "__main__":
    print(build())
