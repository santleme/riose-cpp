#!/usr/bin/env python3
"""Bundle shared window preprocessing with the standalone Kaggle experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "kernels" / "cow-activity-temporal-v1"


def build() -> Path:
    base = (ROOT / "train.py").read_text()
    guard = '\nif __name__ == "__main__":\n    main()\n'
    if guard not in base:
        raise ValueError("Could not find shared trainer entrypoint guard")
    base_module = base.split(guard, 1)[0].rstrip()
    experiment = (ROOT / "activity_tcn_loso.py").read_text()
    experiment = experiment.replace("import train\n", "")
    experiment = experiment.replace("from __future__ import annotations\n", "")
    experiment = experiment.replace("if __name__ == \"__main__\":\n    main()", "")
    entry = "\n".join([
        "import sys",
        "train = sys.modules[__name__]",
        f"BASE_TRAIN_SHA256 = {hashlib.sha256((ROOT / 'train.py').read_bytes()).hexdigest()!r}",
        experiment.strip(),
        "",
        "if __name__ == \"__main__\":",
        "    main()",
    ])
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = KERNEL_DIR / "activity_tcn_bundle.py"
    bundle.write_text(base_module + "\n\n" + entry + "\n")
    metadata = {
        "id": "pascalcoimbra/campo-vivo-cow-activity-temporal-cnn-loso-v1",
        "title": "Campo Vivo Cow Activity Temporal CNN LOSO v1",
        "code_file": "activity_tcn_bundle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "dataset_sources": ["fandaoerji/cow-nose-ring-data-set"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return bundle


if __name__ == "__main__":
    print(build())
