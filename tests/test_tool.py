import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import audar_mlx


class ToolTests(unittest.TestCase):
    def test_output_layer_survives_both_loading_formats(self):
        class Loader:
            @staticmethod
            def sanitize(weights):
                weights.pop("thinker.lm_head.weight", None)
                weights.pop("lm_head.weight", None)
                return weights

        audar_mlx.patch_loader(Loader)
        patched = Loader.sanitize
        audar_mlx.patch_loader(Loader)
        self.assertIs(Loader.sanitize, patched)
        for name in ("thinker.lm_head.weight", "lm_head.weight"):
            tensor = object()
            source = {name: tensor, "lm_head.scales": object()}
            result = Loader.sanitize(source)
            self.assertIs(result["lm_head.weight"], tensor)
            self.assertIn("lm_head.scales", result)
            self.assertIn(name, source)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                audar_mlx.convert_model(folder)

    def test_download_requires_explicit_license_acceptance(self):
        result = subprocess.run(
            [sys.executable, "audar_mlx.py", "convert", "--output", "models/test"],
            capture_output=True,
            text=True,
            cwd=Path(audar_mlx.__file__).parent,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--accept-audar-license", result.stderr)


if __name__ == "__main__":
    unittest.main()
