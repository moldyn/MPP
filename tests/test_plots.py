import unittest
import unittest.mock

import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# import subprocess
import tempfile
from pathlib import Path
import yaml
import hashlib
import matplotlib.figure
from matplotlib import pyplot as plt
import MPP.run as run_module
from MPP._style import FONT_FAMILY


DATASETS = ["HP35", "PDZ3", "aSyn"]
PLOT_KINDS = [
    "dendrogram",
    "timescales",
    "sankey",
    "contacts",
    "macrotraj",
    "ck_test",
    "state_network",
]
MAPPING_FILE = Path(__file__).parent / "data" / "lumpings.yml"


def _run_main_with_args(args_list):
    """Helper to run run.main() with patched sys.argv and capture output."""
    saved_argv = sys.argv
    sys.argv = ["run.py"] + args_list
    stdout, stderr = StringIO(), StringIO()
    plt.close("all")
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            run_module.main()
        return 0, stdout.getvalue(), stderr.getvalue()
    except SystemExit as e:
        return e.code, stdout.getvalue(), stderr.getvalue()
    finally:
        plt.close("all")
        sys.argv = saved_argv


class TestPlotting(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     with open(MAPPING_FILE, "r") as f:
    #         cls.param_map = yaml.safe_load(f)

    def setUp(self):
        self.data_root = Path(__file__).parent / "data"
        with open(MAPPING_FILE, "r") as f:
            self.param_map = yaml.safe_load(f)

    def _run_plot(self, dataset, config, d, g, kind, output_file, stochastic=False):
        key = self._get_key(d, g)
        args = [
            str(config),
            d,
            g,
            "-p",
            kind,
            "-o",
            str(output_file),
            "-Z",
            str(
                Path(__file__).parent
                / "data"
                / dataset
                / "expected_output"
                / key
                / f"Z{'_stochastic' if stochastic else ''}.npy"
            ),
        ]
        return _run_main_with_args(args)

    def _get_key(self, d, g):
        for key, val in self.param_map.items():
            if val["kernel_similarity"] == d and val["feature_kernel"] == g:
                return key
        raise ValueError(f"No mapping found for d={d}, g={g}")

    def run_single_plot_test(
        self, dataset, kind, d, g, manual_inspection=False, stochastic=False
    ):
        config = (
            self.data_root
            / dataset
            / "input"
            / f"config{'_stochastic' if stochastic else ''}.yml"
        )
        key = self._get_key(d, g)
        expected_file = (
            self.data_root / dataset / "expected_output" / key / f"{kind}.pdf"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if manual_inspection:
                plot_path = (
                    self.data_root
                    / dataset
                    / f"output{'_stochastic' if stochastic else ''}"
                    / key
                    / expected_file.name
                )
                plot_path.unlink(missing_ok=True)
                plot_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                plot_path = tmpdir / expected_file.name

            exit_code, stdout, stderr = self._run_plot(
                dataset, config, d, g, kind, plot_path, stochastic=stochastic
            )
            self.assertEqual(exit_code, 0, f"Plot command failed: {stderr}")
            if manual_inspection:
                self.assertTrue(
                    plot_path.exists(), f"Plot file not created: {plot_path}"
                )
            else:
                # Hashed comparison
                expected_hash = file_hash(expected_file)
                generated_hash = file_hash(plot_path)
                self.assertEqual(
                    expected_hash,
                    generated_hash,
                    f"Hash mismatch for {dataset} {d}-{g} {kind}",
                )

    def test_manual_dendrogram(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "dendrogram"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_timescales(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "timescales"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_sankey(self):
        dataset = "HP35"
        d, g = "KL", "none"
        kind = "sankey"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_contacts(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "contacts"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_macrotraj_ref(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "macrotraj"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_macrotraj_PDZ3(self):
        dataset = "PDZ3"
        d, g = "KL", "none"
        kind = "macrotraj"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_ck_test(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "ck_test"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_state_network(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "state_network"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_macro_feature(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "macro_feature"
        self.run_single_plot_test(
            dataset, kind, d, g, manual_inspection=True, stochastic=True
        )

    def test_manual_rmsd(self):
        dataset = "PDZ3"
        d, g = "KL", "none"
        kind = "rmsd"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_delta_rmsd(self):
        dataset = "PDZ3"
        d, g = "KL", "none"
        kind = "delta_rmsd"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_stochastic_state_similarity(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "stochastic_state_similarity"
        self.run_single_plot_test(
            dataset, kind, d, g, manual_inspection=True, stochastic=True
        )

    def test_manual_relative_implied_timescales(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "relative_implied_timescales"
        self.run_single_plot_test(
            dataset, kind, d, g, manual_inspection=True, stochastic=True
        )

    def test_manual_transition_matrix(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "transition_matrix"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_transition_time(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "transition_time"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)


class TestFontFamily(unittest.TestCase):
    """Verify that all plot functions use FONT_FAMILY at save time."""

    def setUp(self):
        self.data_root = Path(__file__).parent / "data"
        with open(MAPPING_FILE, "r") as f:
            self.param_map = yaml.safe_load(f)

    def _get_key(self, d, g):
        for key, val in self.param_map.items():
            if val["kernel_similarity"] == d and val["feature_kernel"] == g:
                return key
        raise ValueError(f"No mapping found for d={d}, g={g}")

    def _assert_font_family(self, dataset, kind, d, g, stochastic=False):
        key = self._get_key(d, g)
        config = (
            self.data_root
            / dataset
            / "input"
            / f"config{'_stochastic' if stochastic else ''}.yml"
        )
        z_file = (
            self.data_root
            / dataset
            / "expected_output"
            / key
            / f"Z{'_stochastic' if stochastic else ''}.npy"
        )

        observed = []

        original_savefig = matplotlib.figure.Figure.savefig

        def capturing_savefig(fig_self, fname, *args, **kwargs):
            observed.append({
                "usetex": plt.rcParams.get("text.usetex"),
                "font.family": list(plt.rcParams.get("font.family", [])),
            })
            return original_savefig(fig_self, fname, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / f"{kind}.pdf")
            args = [str(config), d, g, "-p", kind, "-o", out, "-Z", str(z_file)]
            with unittest.mock.patch.object(
                matplotlib.figure.Figure, "savefig", capturing_savefig
            ):
                _run_main_with_args(args)

        self.assertTrue(
            len(observed) > 0,
            f"savefig was never called for plot kind '{kind}'",
        )
        for snapshot in observed:
            self.assertTrue(
                snapshot["usetex"],
                f"text.usetex is not True for plot kind '{kind}': {snapshot}",
            )
            self.assertIn(
                FONT_FAMILY,
                snapshot["font.family"],
                f"Font family '{FONT_FAMILY}' not set for plot kind '{kind}': {snapshot}",
            )

    def test_font_dendrogram(self):
        self._assert_font_family("HP35", "dendrogram", "T", "none")

    def test_font_timescales(self):
        self._assert_font_family("HP35", "timescales", "T", "none")

    def test_font_sankey(self):
        self._assert_font_family("HP35", "sankey", "KL", "none")

    def test_font_contacts(self):
        self._assert_font_family("HP35", "contacts", "T", "none")

    def test_font_macrotraj(self):
        self._assert_font_family("HP35", "macrotraj", "T", "none")

    def test_font_ck_test(self):
        self._assert_font_family("HP35", "ck_test", "T", "none")

    def test_font_state_network(self):
        self._assert_font_family("HP35", "state_network", "T", "none")

    def test_font_macro_feature(self):
        self._assert_font_family("HP35", "macro_feature", "T", "none", stochastic=True)

    def test_font_stochastic_state_similarity(self):
        self._assert_font_family(
            "HP35", "stochastic_state_similarity", "T", "none", stochastic=True
        )

    def test_font_relative_implied_timescales(self):
        self._assert_font_family(
            "HP35", "relative_implied_timescales", "T", "none", stochastic=True
        )

    def test_font_transition_matrix(self):
        self._assert_font_family("HP35", "transition_matrix", "T", "none")

    def test_font_transition_time(self):
        self._assert_font_family("HP35", "transition_time", "T", "none")


def file_hash(path, algo="sha256"):
    """Returns the hash digest of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    unittest.main()
