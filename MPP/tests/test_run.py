import unittest
import subprocess
import tempfile
import os
from pathlib import Path
import numpy as np
import yaml

# DATASETS = ["HP35", "PDZ3", "aSyn"]
DATASETS = ["HP35", "PDZ3"]
MAPPING_FILE = Path(__file__).parent / "data" / "lumpings.yaml"


class TestRunScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MAPPING_FILE, "r") as f:
            cls.param_map = yaml.safe_load(f)

    def setUp(self):
        self.base_data_dir = Path(__file__).parent / "data"

    def _run_command(self, config_path, d, g, output_file):
        """Helper to invoke the run script."""
        cmd = [
            "python",
            "-m",
            "MPT.run",
            str(config_path),
            d,
            g,
            "-Z",
            str(output_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def _get_key(self, d, g):
        """Returns the key like 'kl', 't_js', etc. from the mapping."""
        for key, val in self.param_map.items():
            if val["kernel similarity"] == d and val["feature kernel"] == g:
                return key
        raise ValueError(f"No key found for d={d}, g={g}")

    def test_run_and_validate_output(self):
        for dataset in DATASETS:
            config_file = self.base_data_dir / dataset / "config.yaml"
            for key, val in self.param_map.items():
                d = val["kernel similarity"]
                g = val["feature kernel"]
                with self.subTest(dataset=dataset, d=d, g=g):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        z_output = Path(tmpdir) / "Z.npy"

                        # Run first time: should compute and save
                        result = self._run_command(config_file, d, g, z_output)
                        self.assertEqual(
                            result.returncode, 0, f"Failed for {dataset} {d}-{g}"
                        )
                        self.assertTrue(
                            z_output.exists(),
                            f"Z.npy not created for {dataset} {d}-{g}",
                        )

                        # Compare with expected
                        expected_path = (
                            self.base_data_dir
                            / dataset
                            / "expected_output"
                            / key
                            / "Z.npy"
                        )
                        self.assertTrue(
                            expected_path.exists(),
                            f"Expected file missing: {expected_path}",
                        )

                        output_data = np.load(z_output)
                        expected_data = np.load(expected_path)

                        np.testing.assert_allclose(
                            output_data,
                            expected_data,
                            rtol=1e-5,
                            err_msg=f"Mismatch in Z for {dataset} {d}-{g}",
                        )

                        # Second run: should load existing file (tests the from_Z logic indirectly)
                        result2 = self._run_command(config_file, d, g, z_output)
                        self.assertEqual(
                            result2.returncode,
                            0,
                            f"Reload failed for {dataset} {d}-{g}",
                        )

                        # Verify "Loading existing Z" is printed (indicating from_Z was called)
                        self.assertIn(
                            "Loading existing Z",
                            result2.stdout,
                            f"Z not loaded from file for {dataset} {d}-{g}",
                        )
