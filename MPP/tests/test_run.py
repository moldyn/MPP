import unittest


import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import tempfile
from pathlib import Path
import numpy as np
import yaml
import MPT.run as run_module

# DATASETS = ["HP35", "PDZ3", "aSyn"]
# DATASETS = ["HP35", "PDZ3"]
DATASETS = ["HP35"]
MAPPING_FILE = Path(__file__).parent / "data" / "lumpings.yaml"


def _run_main_with_args(args_list):
    """Helper to run run.main() with patched sys.argv and capture output."""
    saved_argv = sys.argv
    sys.argv = ["run.py"] + args_list
    stdout, stderr = StringIO(), StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            run_module.main()
        return 0, stdout.getvalue(), stderr.getvalue()
    except SystemExit as e:
        return e.code, stdout.getvalue(), stderr.getvalue()
    finally:
        sys.argv = saved_argv


class TestRunScript(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     with open(MAPPING_FILE, "r") as f:
    #         cls.param_map = yaml.safe_load(f)

    def setUp(self):
        self.base_data_dir = Path(__file__).parent / "data"
        with open(MAPPING_FILE, "r") as f:
            self.param_map = yaml.safe_load(f)

    def _run_command(self, config_path, d, g, output_file):
        """Helper to invoke the run script."""
        args = [
            str(config_path),
            d,
            g,
            "-Z",
            str(output_file),
        ]
        return _run_main_with_args(args)

    def _get_key(self, d, g):
        """Returns the key like 'kl', 't_js', etc. from the mapping."""
        for key, val in self.param_map.items():
            if val["kernel similarity"] == d and val["feature kernel"] == g:
                return key
        raise ValueError(f"No key found for d={d}, g={g}")

    def run_and_validate_output(self, dataset, d, g, stochastic=False):
        config_file = (
            self.base_data_dir
            / dataset
            / f"config{'_stochastic' if stochastic else ''}.yaml"
        )
        key = self._get_key(d, g)
        with self.subTest(dataset=dataset, d=d, g=g):
            with tempfile.TemporaryDirectory() as tmpdir:
                z_output = Path(tmpdir) / "Z.npy"

                # Run first time: should compute and save
                exit_code, stdout, stderr = self._run_command(
                    config_file, d, g, z_output
                )
                self.assertEqual(exit_code, 0, f"Failed for {stderr} {d}-{g}")
                self.assertTrue(
                    z_output.exists(),
                    f"Z.npy not created for {dataset} {d}-{g}",
                )

                # Compare with expected
                expected_path = (
                    self.base_data_dir / dataset / "expected_output" / key / "Z.npy"
                )
                self.assertTrue(
                    expected_path.exists(),
                    f"Expected file missing: {expected_path}",
                )

                output_data = np.load(z_output)
                expected_data = np.load(expected_path)

                if not stochastic:
                    np.testing.assert_allclose(
                        output_data,
                        expected_data,
                        rtol=1e-5,
                        err_msg=f"Mismatch in Z for {dataset} {d}-{g}",
                    )

                # Second run: should load existing file (tests the from_Z logic indirectly)
                exit_code2, stdout2, stderr2 = self._run_command(
                    config_file, d, g, z_output
                )
                self.assertEqual(
                    exit_code2,
                    0,
                    f"Reload failed for {stderr2} {d}-{g}",
                )

                # Verify "Loading existing Z" is printed (indicating from_Z was called)
                self.assertIn(
                    "Loading existing Z",
                    stdout2,
                    f"Z not loaded from file for {stderr2} {d}-{g}",
                )

    def test_HP35_t(self):
        self.run_and_validate_output("HP35", "T", "none")

    def test_HP35_t_stoch(self):
        self.run_and_validate_output("HP35", "T", "none", stochastic=True)

    def test_HP35_kl(self):
        self.run_and_validate_output("HP35", "KL", "none")

    def test_HP35_t_js(self):
        self.run_and_validate_output("HP35", "T", "JS")

    def test_HP35_gpcca(self):
        self.run_and_validate_output("HP35", "gpcca", "ref")

    def test_PDZ3_kl(self):
        self.run_and_validate_output("PDZ3", "KL", "none")

    def test_aSyn_t(self):
        self.run_and_validate_output("aSyn", "T", "none")
