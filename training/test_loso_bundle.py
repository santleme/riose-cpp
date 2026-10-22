import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARITY_FUNCTIONS = {
    "_window_features",
    "_parse_sample_timestamps",
    "_extract_windows",
    "verify_dataset_hash",
    "_cuda_available",
    "_gpu_name",
    "_train_torch_mlp",
}
PARITY_ASSIGNMENTS = {
    "DATASET",
    "DATASET_LICENSE",
    "DATASET_CSV_SHA256",
    "LABELS",
    "AXES",
    "REQUIRED",
    "WINDOW_SAMPLES",
    "SAMPLING_HZ",
    "FEATURE_NAMES",
}


def selected_nodes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    selected = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PARITY_FUNCTIONS:
            selected[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            for name in names & PARITY_ASSIGNMENTS:
                selected[name] = ast.dump(node, include_attributes=False)
    return selected


class LosoBundleParityTest(unittest.TestCase):
    def test_bundle_contains_byte_for_byte_ast_equivalent_v5_core(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle_path = Path(temp) / "train_loso_bundle.py"
            subprocess.run(
                [sys.executable, str(ROOT / "build_loso_bundle.py"), str(bundle_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            source_nodes = selected_nodes(ROOT / "train.py")
            bundle_nodes = selected_nodes(bundle_path)
            self.assertEqual(set(source_nodes), PARITY_FUNCTIONS | PARITY_ASSIGNMENTS)
            self.assertEqual(bundle_nodes, source_nodes)


if __name__ == "__main__":
    unittest.main()
