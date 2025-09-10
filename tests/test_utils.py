import unittest


import yaml
import numpy as np
from pathlib import Path
import MPP
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


class TestProperties(unittest.TestCase):
    def setUp(self):
        self.d = get_d("HP35", "t")
        self.mpp = self.d.mpp

    def test_Z_to_linkage(self):
        linkage = MPP.utils.Z_to_linkage(self.mpp.Z[self.mpp.n_i])
        expected_linkage = np.load(
            Path(__file__).parent
            / "data"
            / "HP35"
            / "expected_output"
            / "t"
            / "linkage.npy"
        )
        np.testing.assert_allclose(linkage, expected_linkage)

    def test_linkage_to_Z(self):
        expected_linkage = np.load(
            Path(__file__).parent
            / "data"
            / "HP35"
            / "expected_output"
            / "t"
            / "linkage.npy"
        )
        z_i, full_pop = MPP.utils.linkage_to_Z(expected_linkage, self.mpp.pop)
        expected_z = np.load(
            Path(__file__).parent / "data" / "HP35" / "expected_output" / "t" / "Z.npy"
        )
        np.testing.assert_allclose(z_i, expected_z[0])

    def test_calc_full_tmat(self):
        expected_tmat = np.load(
            Path(__file__).parent
            / "data"
            / "HP35"
            / "expected_output"
            / "t"
            / "full_tmat.npy"
        )
        expected_pop = np.load(
            Path(__file__).parent
            / "data"
            / "HP35"
            / "expected_output"
            / "t"
            / "full_pop.npy"
        )
        full_tmat, full_pop = MPP.utils.calc_full_tmat(
            self.mpp.tmat, self.mpp.pop, self.mpp.Z
        )
        np.testing.assert_allclose(full_tmat, expected_tmat)
        np.testing.assert_allclose(full_pop, expected_pop)

    def test_Z_to_mask(self):
        expected_mask = np.load(
            Path(__file__).parent
            / "data"
            / "HP35"
            / "expected_output"
            / "t"
            / "full_mask.npy"
        )
        full_mask = MPP.utils.Z_to_mask(self.mpp.Z[0])
        np.testing.assert_allclose(full_mask, expected_mask)


class TestFullFeature(unittest.TestCase):
    def setUp(self):
        self.d = get_d("HP35", "t_js")

    def test_full_feature_from_Z(self):
        expected_full_feature = np.load(
            Path(__file__).parent
            / "data"
            / "HP35"
            / "expected_output"
            / "t_js"
            / "full_feature.npy"
        )
        full_feature = self.d.feature_kernel.full_feature_from_Z(self.d.mpp.Z)
        np.testing.assert_allclose(full_feature, expected_full_feature)
