import unittest

import yaml
import numpy as np
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

    def test_shannon_entropy(self):
        np.testing.assert_allclose(self.mpp.shannon_entropy[0], 0.73862646)

    def test_gmrq(self):
        np.testing.assert_allclose(self.mpp.gmrq[0], 2.5071318)

    def test_davies_bouldin_index(self):
        np.testing.assert_allclose(self.mpp.davies_bouldin_index[0], 2.20325216)
