import unittest
import subprocess
import tempfile
from pathlib import Path
import yaml
import hashlib

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


class TestPlotting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MAPPING_FILE, "r") as f:
            cls.param_map = yaml.safe_load(f)

    def setUp(self):
        self.data_root = Path(__file__).parent / "data"

    def _run_plot(self, config, d, g, kind, output_file):
        cmd = [
            "python",
            "-m",
            "MPT.run",
            str(config),
            d,
            g,
            "-p",
            kind,
            "-o",
            str(output_file),
            "-Z",
            str(output_file.parent / "Z.npy"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def _get_key(self, d, g):
        for key, val in self.param_map.items():
            if val["kernel similarity"] == d and val["feature kernel"] == g:
                return key
        raise ValueError(f"No mapping found for d={d}, g={g}")

    def run_single_plot_test(self, dataset, kind, d, g, manual_inspection=False):
        config = self.data_root / dataset / "config.yaml"
        key = self._get_key(d, g)
        expected_file = (
            self.data_root / dataset / "expected_output" / key / f"{kind}.pdf"
        )
        # if not expected_file.exists():
        #     expected_file = expected_file.with_suffix(".pdf")

        self.assertTrue(
            expected_file.exists(), f"Expected plot not found: {expected_file}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if manual_inspection:
                plot_path = self.data_root / dataset / "output" / expected_file.name
                plot_path.unlink(missing_ok=True)
            else:
                plot_path = tmpdir / expected_file.name

            result = self._run_plot(config, d, g, kind, plot_path)
            self.assertEqual(
                result.returncode, 0, f"Plot command failed: {result.stderr}"
            )
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

    def test_manual_inspection_of_plots(self):
        dataset = "HP35"
        d, g = "T", "none"
        # for kind in PLOT_KINDS:
        # for kind in ["state_network"]:
        for kind in ["dendrogram"]:
            with self.subTest(
                dataset=dataset, kind=kind, d=d, g=g, manual_inspection=True
            ):
                self.run_single_plot_test(dataset, kind, d, g, manual_inspection=True)


def file_hash(path, algo="sha256"):
    """Returns the hash digest of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    unittest.main()
