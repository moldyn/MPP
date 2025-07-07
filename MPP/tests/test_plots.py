import unittest

import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# import subprocess
import tempfile
from pathlib import Path
import yaml
import hashlib
import MPT.run as run_module

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
MAPPING_FILE = Path(__file__).parent / "data" / "lumpings.yaml"


# TODO:
# Add correlation plots


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


class TestPlotting(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     with open(MAPPING_FILE, "r") as f:
    #         cls.param_map = yaml.safe_load(f)

    def setUp(self):
        self.data_root = Path(__file__).parent / "data"
        with open(MAPPING_FILE, "r") as f:
            self.param_map = yaml.safe_load(f)

    def _run_plot(self, config, d, g, kind, output_file):
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
                / config.parent.name
                / "expected_output"
                / key
                / "Z.npy"
            ),
        ]
        return _run_main_with_args(args)

    def _get_key(self, d, g):
        for key, val in self.param_map.items():
            if val["kernel similarity"] == d and val["feature kernel"] == g:
                return key
        raise ValueError(f"No mapping found for d={d}, g={g}")

    def run_single_plot_test(
        self, dataset, kind, d, g, manual_inspection=False, stochastic=False
    ):
        config = (
            self.data_root
            / dataset
            / f"config{'_stochastic' if stochastic else ''}.yaml"
        )
        key = self._get_key(d, g)
        expected_file = (
            self.data_root / dataset / "expected_output" / key / f"{kind}.pdf"
        )
        # if not expected_file.exists():
        #     expected_file = expected_file.with_suffix(".pdf")

        # self.assertTrue(
        #     expected_file.exists(), f"Expected plot not found: {expected_file}"
        # )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if manual_inspection:
                plot_path = self.data_root / dataset / "output" / expected_file.name
                plot_path.unlink(missing_ok=True)
            else:
                plot_path = tmpdir / expected_file.name

            exit_code, stdout, stderr = self._run_plot(config, d, g, kind, plot_path)
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

    def _test_partial_plotting(self):
        # ⚙️ Define filters here — adjust for partial testing
        # selected_datasets = ["HP35", "PDZ3", "aSyn"]  # change to [] for all
        selected_datasets = ["HP35"]  # change to [] for all
        # selected_kinds = ["sankey", "dendrogram", "ck_test"]  # change to [] for all
        selected_kinds = [
            "timescales",
            # "contacts",
            # "ck_test",
            # "dendrogram",
            # "sankey",
            # "macrotraj",
            # "state_network",
        ]  # change to [] for all
        selected_combos = [
            ("T", "none"),
            # ("KL", "none"),
            # ("T", "JS"),
            # ("KL", "JS"),
            # ("gpcca", "ref"),
        ]  # change to [] for all

        datasets = selected_datasets or DATASETS
        kinds = selected_kinds or PLOT_KINDS
        combos = selected_combos or [
            (v["kernel similarity"], v["feature kernel"])
            for v in self.param_map.values()
        ]

        for dataset in datasets:
            for kind in kinds:
                for d, g in combos:
                    with self.subTest(dataset=dataset, kind=kind, d=d, g=g):
                        self.run_single_plot_test(dataset, kind, d, g)

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

    def test_manual_macrotraj(self):
        dataset = "HP35"
        d, g = "T", "none"
        kind = "macrotraj"
        self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)

    def test_manual_macrotraj_PDZ3(self):
        dataset = "PDZ3"
        d, g = "T", "none"
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


def file_hash(path, algo="sha256"):
    """Returns the hash digest of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    unittest.main()
