import unittest

import os
import yaml
import numpy as np
import tempfile
from pathlib import Path
import MPP.run as run_module


config_dir = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/sm/config/"
root = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/sm/results/"

SYSTEMS = [
    "HP35",
    "PDZ3_7",
    "aSyn_rdc_200ns",
    "HP35_stoch",
]
SETUPS = [
    "t",
    "kl",
    "t_js",
    "kl_js",
    "gpcca",
]

with open(f"{config_dir}lumpings.yaml") as f:
    lumpings = yaml.safe_load(f)


def get_d(system, setup, rmsd=False):
    d = run_module.Data(
        f"/data/evaluation/MPP/stochastic_MPP_Felix/data_production/sm/config/{system}.yaml"
    )
    d.setup_mpp(
        lumpings[setup]["kernel similarity"],
        lumpings[setup]["feature kernel"],
    )
    if setup == "gpcca":
        d.perform_gpcca("ref", f"{root}{system}/{setup}/Z.npy")
    else:
        d.perform_mpp(f"{root}{system}/{setup}/Z.npy")
    if rmsd:
        d.mpp.load_rmsd(f"{root}{system}/{setup}/rmsd.npy")
    return d


class TestRMSD_HP35(unittest.TestCase):
    def setUp(self):
        self.d = get_d("HP35", "kl")
        self.mpp = self.d.mpp

    def compare_text_files(self, file1, file2):
        """Compares two text files, ignoring the first line (timestamp)."""
        with open(file1, "r") as f1, open(file2, "r") as f2:
            lines1 = f1.readlines()[1:]  # skip first line
            lines2 = f2.readlines()[1:]
            self.assertEqual(
                lines1,
                lines2,
                f"Mismatch in file content (excluding header): {file1} vs {file2}",
            )

    def test_draw_random_indices(self):
        np.random.seed(123456)
        drawn_frames = self.mpp.draw_random_frames_indices()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir) / "random_indices"
            np.random.seed(123456)
            self.mpp.draw_random_frames_indices(tmpdir)

            drawn_frames_from_file = np.zeros(drawn_frames.shape, dtype=int)
            for s, i in enumerate(drawn_frames):
                drawn_frames_from_file[s] = np.loadtxt(tmpdir / f"{s + 1:02d}.ndx")

        np.testing.assert_allclose(drawn_frames, drawn_frames_from_file)

    def test_draw_random_frames(self):
        expected_path = (
            Path(__file__).parent / "data" / "HP35" / "expected_output" / "t" / "pdbs"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            np.random.seed(123456)
            self.mpp.draw_random_frames(tmpdir)

            for expected_file in expected_path.glob("*.pdb"):
                actual_file = tmpdir / expected_file.name
                self.assertTrue(actual_file.exists(), f"Missing file: {actual_file}")
                # self.compare_text_files(expected_file, actual_file)


class TestRMSD_aSyn(unittest.TestCase):
    def setUp(self):
        self.d = get_d("aSyn_rdc_200ns", "t")
        self.mpp = self.d.mpp

    def compare_text_files(self, file1, file2):
        """Compares two text files, ignoring the first line (timestamp)."""
        with open(file1, "r") as f1, open(file2, "r") as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            self.assertEqual(
                lines1,
                lines2,
                f"Mismatch in file content (excluding header): {file1} vs {file2}",
            )

    def test_write_least_moving_residues(self):
        expected_output = (
            Path(__file__).parent
            / "data"
            / "aSyn"
            / "expected_output"
            / "t"
            / "least_moving_residues.ndx"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "least_moving_residues.ndx"
            self.mpp.write_least_moving_residues(
                self.d.d["source"] + self.d.d["contact index file"],
                tmpfile,
            )
            self.compare_text_files(tmpfile, expected_output)


class TestRMSD_PDZ3(unittest.TestCase):
    def setUp(self):
        self.d1 = get_d("PDZ3_7", "kl")
        self.d2 = get_d("PDZ3_7", "kl")

    def test_rmsd_property(self):
        expected_output = (
            Path(__file__).parent
            / "data"
            / "PDZ3"
            / "expected_output"
            / "kl"
            / "rmsd.npy"
        )
        self.d2.mpp.load_rmsd(expected_output)
        with tempfile.TemporaryDirectory() as tmpdir:
            rmsd_file = Path(tmpdir) / "rmsd.npy"
            self.d1.mpp.save_rmsd(rmsd_file)

        np.testing.assert_allclose(self.d2.mpp.rmsd, self.d1.mpp.rmsd, rtol=1e-5)

    def test_rmsd_sharpness(self):
        expected_output = (
            Path(__file__).parent
            / "data"
            / "PDZ3"
            / "expected_output"
            / "kl"
            / "rmsd.npy"
        )
        self.d1.mpp.load_rmsd(expected_output)
        np.testing.assert_allclose(self.d1.mpp.rmsd_sharpness(), 2.1352340796266334)
