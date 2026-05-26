import unittest


import sys
import warnings
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import tempfile
from pathlib import Path
import numpy as np
import yaml
import MPP.run as run_module


# TODO:
# - MultiFeatureKernel.full_feature_from_Z


DATASETS = ["HP35"]
MAPPING_FILE = Path(__file__).parent / "data" / "lumpings.yml"


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
    def setUp(self):
        self.base_data_dir = Path(__file__).parent / "data"
        with open(MAPPING_FILE, "r") as f:
            self.param_map = yaml.safe_load(f)

    def _run_command(self, config_path, d, g, output_file, r=None, o=None):
        """Helper to invoke the run script."""
        args = [
            str(config_path),
            d,
            g,
            "-Z",
            str(output_file),
        ]
        if r is not None:
            args.append("-r")
            args.append(str(r))
        if o is not None:
            args.append("-o")
            args.append(str(o))
        return _run_main_with_args(args)

    def _get_key(self, d, g):
        """Returns the key like 'kl', 't_js', etc. from the mapping."""
        for key, val in self.param_map.items():
            if val["kernel_similarity"] == d and val["feature_kernel"] == g:
                return key
        raise ValueError(f"No key found for d={d}, g={g}")

    def run_and_validate_output(self, dataset, d, g, stochastic=False):
        config_file = (
            self.base_data_dir
            / dataset
            / "input"
            / f"config{'_stochastic' if stochastic else ''}.yml"
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
                    self.base_data_dir
                    / dataset
                    / "expected_output"
                    / key
                    / f"Z{'_stochastic' if stochastic else ''}.npy"
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
                    "Loading existing Z matrix",
                    stdout2,
                    f"Z not loaded from file for {stderr2} {d}-{g}",
                )

    def test_HP35_t_ref(self):
        self.run_and_validate_output("HP35", "T", "none")

    def test_HP35_t_stoch(self):
        self.run_and_validate_output("HP35", "T", "none", stochastic=True)

    def test_HP35_kl(self):
        self.run_and_validate_output("HP35", "KL", "none")

    def test_HP35_t_js(self):
        self.run_and_validate_output("HP35", "T", "JS")

    def test_HP35_js(self):
        self.run_and_validate_output("HP35", "none", "JS")

    def test_HP35_gpcca(self):
        self.run_and_validate_output("HP35", "gpcca", "reference_count")

    def test_PDZ3_kl(self):
        self.run_and_validate_output("PDZ3", "KL", "none")

    def test_aSyn_t(self):
        self.run_and_validate_output("aSyn", "T", "none")

    def test_aSyn_kl_js(self):
        self.run_and_validate_output("aSyn", "KL", "JS")

    def test_aSyn_t_stoch(self):
        self.run_and_validate_output("aSyn", "T", "none", stochastic=True)

    def test_macrostate_map_saved_alongside_z(self):
        """macrostate_map.npy must be written to the same directory as Z.npy."""
        config_file = self.base_data_dir / "HP35" / "input" / "config.yml"
        with tempfile.TemporaryDirectory() as tmpdir:
            z_output = Path(tmpdir) / "Z.npy"
            exit_code, stdout, stderr = self._run_command(
                config_file, "T", "none", z_output
            )
            self.assertEqual(exit_code, 0, f"Run failed: {stderr}")

            map_output = Path(tmpdir) / "macrostate_map.npy"
            self.assertTrue(
                map_output.exists(),
                "macrostate_map.npy not created alongside Z.npy",
            )
            macrostate_map = np.load(map_output)
            self.assertEqual(macrostate_map.ndim, 1, "macrostate_map should be 1D")
            self.assertTrue(
                np.all(macrostate_map >= 0),
                "macrostate_map values must be non-negative",
            )

    def test_macrostate_map_reloaded_on_load(self):
        """macrostate_map.npy must be (re-)written when loading an existing Z."""
        config_file = self.base_data_dir / "HP35" / "input" / "config.yml"
        key = "t"
        z_file = self.base_data_dir / "HP35" / "expected_output" / key / "Z.npy"
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy Z.npy into tmpdir so perform_mpp loads it rather than computing
            import shutil

            z_copy = Path(tmpdir) / "Z.npy"
            shutil.copy(z_file, z_copy)

            exit_code, stdout, stderr = self._run_command(
                config_file, "T", "none", z_copy
            )
            self.assertEqual(exit_code, 0, f"Reload failed: {stderr}")
            self.assertIn("Loading existing Z matrix", stdout)

            map_output = Path(tmpdir) / "macrostate_map.npy"
            self.assertTrue(
                map_output.exists(),
                "macrostate_map.npy not written when loading existing Z",
            )

    def assert_same_file_count(self, expected_dir, actual_dir, pattern="*"):
        expected_files = list(Path(expected_dir).glob(pattern))
        actual_files = list(Path(actual_dir).glob(pattern))

        self.assertEqual(
            len(actual_files),
            len(expected_files),
            f"Mismatch in file count: expected {len(expected_files)} but got {len(actual_files)}",
        )

    def _run_random_frames_indices(self, dataset, d, g, r=20):
        key = self._get_key(d, g)
        z_file = self.base_data_dir / dataset / "expected_output" / key / "Z.npy"
        config_file = self.base_data_dir / dataset / "input" / "config.yml"
        with self.subTest(dataset=dataset, d=d, g=g, r=r):
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)

                exit_code, stdout, stderr = self._run_command(
                    config_file, d, g, z_file, r=r, o=output_dir
                )
                self.assertEqual(exit_code, 0, f"Failed for {stderr} {d}-{g}")

                # Compare with expected
                expected_path = (
                    self.base_data_dir
                    / dataset
                    / "expected_output"
                    / key
                    / "random_frames"
                )
                self.assertTrue(
                    expected_path.exists(),
                    f"Expected directory missing: {expected_path}",
                )

                self.assert_same_file_count(expected_path, output_dir, pattern="*.ndx")

    def test_random_frames_indices_aSyn_t_ref(self):
        self._run_random_frames_indices("aSyn", "T", "none")


class TestConfigNormalization(unittest.TestCase):
    """Tests for backward-compatible YAML config key normalization."""

    def _write_config(self, path, extra_keys):
        """Write a minimal valid config to *path* with the given extra keys."""
        config = {
            "source": "tests/data/HP35/input/",
            "lagtime": 1,
            "pop_thr": 0.005,
            "q_min": 0.5,
            "frame_length": 10,
        }
        config.update(extra_keys)
        import yaml as _yaml
        with open(path, "w") as f:
            _yaml.dump(config, f)

    def test_canonical_keys_load_without_warning(self):
        """New snake_case keys must load without any DeprecationWarning."""
        with tempfile.NamedTemporaryFile(
            suffix=".yml", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name
        self._write_config(
            tmp_path,
            {
                "microstate_trajectory": "microstate_trajectory",
                "multi_feature_trajectory": "contact_distances_trajectory",
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_module.Data(tmp_path)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(deprecations),
            0,
            f"Unexpected DeprecationWarning(s): {[str(w.message) for w in deprecations]}",
        )

    def test_legacy_keys_emit_deprecation_warning(self):
        """Old space-separated keys must emit DeprecationWarning and still load."""
        with tempfile.NamedTemporaryFile(
            suffix=".yml", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name
        self._write_config(
            tmp_path,
            {
                "microstate trajectory": "microstate_trajectory",
                "multi feature trajectory": "contact_distances_trajectory",
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = run_module.Data(tmp_path)
        deprecation_messages = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        self.assertTrue(
            any("microstate trajectory" in m for m in deprecation_messages),
            f"Expected DeprecationWarning for 'microstate trajectory', got: {deprecation_messages}",
        )
        self.assertTrue(
            any("multi feature trajectory" in m for m in deprecation_messages),
            f"Expected DeprecationWarning for 'multi feature trajectory', got: {deprecation_messages}",
        )
        # Value must still be accessible under the canonical key
        self.assertEqual(data.d["microstate_trajectory"], "microstate_trajectory")

    def test_duplicate_legacy_and_canonical_keys_raises(self):
        """Specifying both a legacy key and its canonical equivalent must raise ValueError."""
        with tempfile.NamedTemporaryFile(
            suffix=".yml", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name
        # Write raw YAML to avoid dict deduplication
        with open(tmp_path, "w") as f:
            f.write(
                "source: tests/data/HP35/input/\n"
                "lagtime: 1\npop_thr: 0.005\nq_min: 0.5\n"
                "microstate trajectory: old_traj\n"
                "microstate_trajectory: new_traj\n"
                "multi_feature_trajectory: contact_distances_trajectory\n"
            )
        with self.assertRaises(ValueError) as ctx:
            run_module.Data(tmp_path)
        self.assertIn("microstate trajectory", str(ctx.exception))
        self.assertIn("microstate_trajectory", str(ctx.exception))

    def test_normalize_config_canonical_keys_unchanged(self):
        """_normalize_config must return canonical keys unchanged."""
        config = {"microstate_trajectory": "foo", "lagtime": 1}
        result = run_module._normalize_config(config)
        self.assertEqual(result["microstate_trajectory"], "foo")
        self.assertNotIn("microstate trajectory", result)

    def test_normalize_config_all_aliases(self):
        """Every legacy alias must be renamed to its canonical form."""
        legacy = {
            "microstate trajectory": "a",
            "multi feature trajectory": "b",
            "contact threshold": 0.45,
            "cluster file": "c",
            "contact index file": "d",
            "topology file": "e",
            "xtc file": "f",
            "frame length": 10,
            "xtc stride": 1000,
            "n timescales": 3,
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = run_module._normalize_config(legacy)
        for old_key in legacy:
            self.assertNotIn(old_key, result, f"Legacy key '{old_key}' should be removed")
        expected_canonical = [
            "microstate_trajectory",
            "multi_feature_trajectory",
            "contact_threshold",
            "cluster_file",
            "contact_index_file",
            "topology_file",
            "xtc_file",
            "frame_length",
            "xtc_stride",
            "n_timescales",
        ]
        for key in expected_canonical:
            self.assertIn(key, result, f"Canonical key '{key}' missing from result")


class TestCLIValidation(unittest.TestCase):
    """Tests for CLI argument validation and user-facing error messages."""

    def setUp(self):
        self.base_data_dir = Path(__file__).parent / "data"
        self.valid_config = str(
            self.base_data_dir / "HP35" / "input" / "config.yml"
        )
        self.valid_z = str(
            self.base_data_dir / "HP35" / "expected_output" / "t" / "Z.npy"
        )

    def _run(self, args_list):
        return _run_main_with_args(args_list)

    def test_invalid_d_exits_with_error(self):
        """An unrecognised dynamic similarity selector must produce a clear error."""
        code, stdout, stderr = self._run(
            [self.valid_config, "INVALID", "none", "-Z", self.valid_z]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("INVALID", stderr)
        self.assertIn("dynamic similarity selector", stderr)

    def test_invalid_g_exits_with_error(self):
        """An unrecognised feature similarity selector must produce a clear error."""
        code, stdout, stderr = self._run(
            [self.valid_config, "T", "INVALID", "-Z", self.valid_z]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("INVALID", stderr)
        self.assertIn("feature similarity selector", stderr)

    def test_missing_z_for_mpp_exits_with_error(self):
        """Omitting -Z for a non-gpcca run must produce a clear error."""
        code, stdout, stderr = self._run([self.valid_config, "T", "none"])
        self.assertNotEqual(code, 0)
        self.assertIn("-Z", stderr)

    def test_plot_without_out_exits_with_error(self):
        """Specifying -p without -o must produce a clear error."""
        code, stdout, stderr = self._run(
            [self.valid_config, "T", "none", "-Z", self.valid_z, "-p", "dendrogram"]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("-o", stderr)

    def test_missing_required_config_key_raises(self):
        """A config file missing required keys must raise a ValueError with the key name."""
        import tempfile
        import yaml as _yaml

        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as tmp:
            tmp_path = tmp.name
        # Write a config missing 'lagtime', 'pop_thr', 'q_min', 'frame_length'
        with open(tmp_path, "w") as f:
            _yaml.dump(
                {
                    "source": str(self.base_data_dir / "HP35" / "input"),
                    "microstate_trajectory": "microstate_trajectory",
                    "multi_feature_trajectory": "contact_distances_trajectory",
                },
                f,
            )
        with self.assertRaises(ValueError) as ctx:
            run_module.Data(tmp_path)
        self.assertIn("lagtime", str(ctx.exception))

    def test_nonexistent_config_file_gives_argparse_error(self):
        """A non-existent config file must produce a non-zero exit code."""
        code, stdout, stderr = self._run(
            ["/nonexistent/path/config.yml", "T", "none", "-Z", self.valid_z]
        )
        self.assertNotEqual(code, 0)

    def test_metrics_flag_prints_all_keys(self):
        """--metrics must print all expected metric keys to stdout."""
        code, stdout, stderr = self._run(
            [self.valid_config, "T", "none", "-Z", self.valid_z, "--metrics"]
        )
        self.assertEqual(code, 0)
        expected_keys = [
            "shannon_entropy",
            "davies_bouldin",
            "gmrq",
            "gmrq2",
            "silhouette",
            "calinski_harabasz",
        ]
        for key in expected_keys:
            self.assertIn(key + "=", stdout, msg=f"Missing metric key: {key}")
