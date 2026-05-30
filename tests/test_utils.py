import unittest


import yaml
import numpy as np
from pathlib import Path
import MPP
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

    def test_Z_to_linkage(self):
        linkage = MPP.utils.Z_to_linkage(self.mpp.Z[self.mpp.run_index])
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


class TestPureUtils(unittest.TestCase):
    """Unit tests for pure utility functions that require no external data."""

    def test_argmedian_odd(self):
        """argmedian returns the index of the median element."""
        arr = np.array([3, 1, 4, 1, 5])
        idx = MPP.utils.argmedian(arr)
        # median of sorted [1,1,3,4,5] is 3; index of 3 in original is 0
        self.assertEqual(arr[idx], sorted(arr)[len(arr) // 2])

    def test_argmedian_even(self):
        arr = np.array([2, 0, 4, 6])
        idx = MPP.utils.argmedian(arr)
        self.assertTrue(0 <= idx < len(arr))

    def test_weighting_function_single(self):
        """Single-element input: returns exp(-dq)."""
        dq = np.array([2.0])
        result = MPP.utils.weighting_function(dq)
        np.testing.assert_allclose(result, np.exp(-2.0))

    def test_weighting_function_multi(self):
        """Multi-element input: Gaussian kernel, no negative outputs."""
        dq = np.array([0.1, 0.5, 1.0, 2.0])
        result = MPP.utils.weighting_function(dq)
        self.assertEqual(result.shape, dq.shape)
        self.assertTrue(np.all(result >= 0))
        # Smallest divergence → highest weight
        self.assertEqual(np.argmax(result), np.argmin(dq))

    def test_find_state_lengths_simple(self):
        """Run-length encoding of a simple sequence."""
        arr = [0, 0, 1, 1, 1, 2]
        states, lengths = MPP.utils.find_state_lengths(arr)
        np.testing.assert_array_equal(states, [0, 1, 2])
        np.testing.assert_array_equal(lengths, [2, 3, 1])

    def test_find_state_lengths_single(self):
        arr = [5]
        states, lengths = MPP.utils.find_state_lengths(arr)
        np.testing.assert_array_equal(states, [5])
        np.testing.assert_array_equal(lengths, [1])

    def test_get_multi_state_trajectory_none_limits(self):
        """None limits returns the original array."""
        traj = np.array([0, 1, 2, 3])
        result = MPP.utils.get_multi_state_trajectory(traj, None)
        np.testing.assert_array_equal(result, traj)

    def test_get_multi_state_trajectory_splits(self):
        """Non-None limits splits into sub-trajectories of correct lengths."""
        traj = np.array([0, 1, 2, 3, 4])
        limits = np.array([2, 3])
        result = MPP.utils.get_multi_state_trajectory(traj, limits)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], [0, 1])
        np.testing.assert_array_equal(result[1], [2, 3, 4])
