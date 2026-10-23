from pathlib import Path

from training.external import build_actbecalf_bundle as builder


def test_generated_bundle_embeds_current_source_hashes():
    bundle_path = builder.build()
    source = bundle_path.read_text()
    assert f"BASE_TRAIN_SOURCE_SHA256 = '{builder.sha256(builder.BASE)}'" in source
    assert f"ACTBECALF_LOADER_SOURCE_SHA256 = '{builder.sha256(builder.LOADER)}'" in source
    assert f"TRANSFER_RUNNER_SOURCE_SHA256 = '{builder.sha256(builder.RUNNER)}'" in source
    compile(source, str(bundle_path), "exec")


def test_bundle_is_self_contained_for_kaggle_working_directory():
    source = builder.OUTPUT.read_text()
    assert "from training.train import" in source  # guarded fallback is retained
    assert "BASE_TRAIN_SOURCE_SHA256 =" in source
    assert "ACTBECALF_LOADER_SOURCE_SHA256 =" in source
    assert "TRANSFER_RUNNER_SOURCE_SHA256 =" in source
    assert "def extract_actbecalf(" in source
    assert "def _extract_windows(" in source
