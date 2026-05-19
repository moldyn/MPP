"""Regression tests against scientific baseline artifacts.

These tests regenerate deterministic outputs with the current package version
and compare them to committed baseline artifacts.
"""

from pathlib import Path

import numpy as np

import MPP.run as run_module


BASE_DIR = Path(__file__).parent / "data"

# dataset, baseline key, kernel similarity (d), feature kernel (g)
CASE_MAP = [
    ("HP35", "t", "T", "none"),
    ("HP35", "kl", "KL", "none"),
    ("HP35", "js", "none", "JS"),
    ("HP35", "t_js", "T", "JS"),
    ("PDZ3", "kl", "KL", "none"),
    ("aSyn", "t", "T", "none"),
    ("aSyn", "kl_js", "KL", "JS"),
]


def _generate_outputs(dataset: str, d: str, g: str, tmp_path: Path):
    config_path = BASE_DIR / dataset / "input" / "config.yml"
    data = run_module.Data(config_path)
    data.setup_mpp(d, g)

    z_path = tmp_path / dataset / d / g / "Z.npy"
    data.perform_mpp(str(z_path), overwrite=True)

    generated_z = np.load(z_path)
    generated_assignment = data.mpp.macrostate_assignment[0]
    return generated_z, generated_assignment


def test_generated_z_matches_baseline(tmp_path):
    """Regenerated ``Z.npy`` values must match committed baseline files."""
    for dataset, baseline_key, d, g in CASE_MAP:
        generated_z, _ = _generate_outputs(dataset, d, g, tmp_path)
        baseline_z = np.load(BASE_DIR / dataset / "baseline" / baseline_key / "Z.npy")
        np.testing.assert_allclose(
            generated_z,
            baseline_z,
            rtol=1e-7,
            atol=0.0,
            err_msg=f"Generated Z mismatch for {dataset}/{baseline_key}",
        )


def test_generated_macrostate_assignment_matches_baseline(tmp_path):
    """Regenerated macrostate assignments must match committed baselines."""
    for dataset, baseline_key, d, g in CASE_MAP:
        _, generated_assignment = _generate_outputs(dataset, d, g, tmp_path)
        baseline_assignment = np.load(
            BASE_DIR / dataset / "baseline" / baseline_key / "macrostate_assignment.npy"
        )
        np.testing.assert_array_equal(
            generated_assignment,
            baseline_assignment,
            err_msg=(
                f"Generated macrostate assignment mismatch for "
                f"{dataset}/{baseline_key}"
            ),
        )
