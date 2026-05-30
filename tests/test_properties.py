import unittest

import yaml
import numpy as np
import MPP.run as run_module


config_dir = "tests/data/"
root = "tests/data/HP35/input/"

SYSTEMS = [
    "HP35",
    "PDZ3",
    "aSyn",
    "HP35_stoch",
]
SETUPS = [
    "t",
    "kl",
    "t_js",
    "kl_js",
    "gpcca",
]

with open(f"{config_dir}lumpings.yml") as f:
    lumpings = yaml.safe_load(f)


def get_d(system, setup, rmsd=False):
    d = run_module.Data(
        f"tests/data/{system}/input/config.yml"
    )
    d.setup_mpp(
        lumpings[setup]["kernel_similarity"],
        lumpings[setup]["feature_kernel"],
    )
    z_path = f"tests/data/{system}/expected_output/{setup}/Z.npy"
    if setup == "gpcca":
        d.perform_gpcca("ref", z_path)
    else:
        d.perform_mpp(z_path)
    if rmsd:
        d.mpp.load_rmsd(f"{root}{system}/{setup}/rmsd.npy")
    return d


class TestProperties(unittest.TestCase):
    def setUp(self):
        self.d = get_d("HP35", "t")
        self.mpp = self.d.mpp

    def test_shannon_entropy(self):
        np.testing.assert_allclose(self.mpp.shannon_entropy[0], 0.7440447)

    def test_gmrq(self):
        np.testing.assert_allclose(self.mpp.gmrq[0], 2.65830228)

    def test_davies_bouldin_index(self):
        np.testing.assert_allclose(self.mpp.davies_bouldin_index[0], 2.18738, atol=1e-6)

    def test_silhouette(self):
        np.testing.assert_allclose(self.mpp.silhouette[0], 0.20912119, atol=1e-6)

    def test_calinski_harabasz(self):
        np.testing.assert_allclose(self.mpp.calinski_harabasz[0], 6498.04436, atol=1e-3)

    def test_silhouette_single_macrostate(self):
        """Silhouette should raise ValueError when only 1 macrostate exists."""
        # Force a single-macrostate scenario by patching macrostate_trajectory
        import numpy as np
        self.mpp._silhouette = None  # reset cache
        original = self.mpp.macrostate_trajectory
        self.mpp.macrostate_trajectory = np.zeros_like(original)
        try:
            with self.assertRaises(ValueError):
                _ = self.mpp.silhouette
        finally:
            self.mpp.macrostate_trajectory = original
            self.mpp._silhouette = None

    def test_calinski_harabasz_single_macrostate(self):
        """Calinski-Harabasz should raise ValueError when only 1 macrostate exists."""
        import numpy as np
        self.mpp._calinski_harabasz = None  # reset cache
        original = self.mpp.macrostate_trajectory
        self.mpp.macrostate_trajectory = np.zeros_like(original)
        try:
            with self.assertRaises(ValueError):
                _ = self.mpp.calinski_harabasz
        finally:
            self.mpp.macrostate_trajectory = original
            self.mpp._calinski_harabasz = None
