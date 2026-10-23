import ast
import json
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ActivityTcnBundleTests(unittest.TestCase):
    def test_generated_kaggle_bundle_compiles_imports_and_exposes_help(self):
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(
                [sys.executable, str(ROOT / "build_activity_tcn_bundle.py")],
                check=True, capture_output=True, text=True,
            )
            bundle = ROOT / "kernels" / "cow-activity-temporal-v1" / "activity_tcn_bundle.py"
            py_compile.compile(str(bundle), doraise=True)
            ast.parse(bundle.read_text())
            result = subprocess.run(
                [sys.executable, str(bundle), "--help"],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("--require-cuda", result.stdout)
            metadata = json.loads((bundle.parent / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["code_file"], bundle.name)
            self.assertEqual(metadata["dataset_sources"], ["fandaoerji/cow-nose-ring-data-set"])
            self.assertEqual(metadata["enable_gpu"], "true")


if __name__ == "__main__":
    unittest.main()
