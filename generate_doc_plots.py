#!/usr/bin/env python3
"""Regenerate all docs/assets/plots/ figures."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "docs/assets/plots"
TEST_HP35 = ROOT / "tests/data/HP35"
TEST_PDZ3 = ROOT / "tests/data/PDZ3"
REAL_HP35_CFG = Path("/tmp/hp35_real_config.yml")   # patched copy with absolute source path
REAL_HP35_Z   = Path("/home/felixg/Documents/uni/MPP/data/HP35/results/t/Z.npy")
REAL_PDZ3_CFG = Path("/tmp/pdz3_real_config.yml")   # patched copy with absolute source path

def run(args, cwd=None):
    cmd = [sys.executable, "-m", "MPP.run"] + [str(a) for a in args]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        print("STDOUT:", r.stdout[-500:] if r.stdout else "")
        print("STDERR:", r.stderr[-500:] if r.stderr else "")
        raise RuntimeError(f"Command failed (exit {r.returncode})")
    print("  OK")

HP35_CFG   = TEST_HP35 / "input/config.yml"
HP35_Z_T   = TEST_HP35 / "expected_output/t/Z.npy"
HP35_Z_KL  = TEST_HP35 / "expected_output/kl/Z.npy"
HP35_STOCH_CFG = TEST_HP35 / "input/config_stochastic.yml"
HP35_Z_STOCH   = TEST_HP35 / "expected_output/t_stochastic/Z.npy"
PDZ3_CFG   = TEST_PDZ3 / "input/config.yml"
PDZ3_Z_KL  = TEST_PDZ3 / "expected_output/kl/Z.npy"
# rmsd/delta_rmsd need real topology+trajectory; use real PDZ3 data, Z computed to /tmp
REAL_PDZ3_Z_KL = Path("/tmp/pdz3_real_kl_Z.npy")

plots = [
    # (config, d, g, Z, kind, outname, extra_args)
    (HP35_CFG,        "T",  "none", HP35_Z_T,    "dendrogram",   "dendrogram.png",   []),
    (HP35_CFG,        "T",  "none", HP35_Z_T,    "timescales",   "timescales_t.png", []),
    (HP35_CFG,        "KL", "none", HP35_Z_KL,   "timescales",   "timescales_kl.png",[]),
    (HP35_CFG,        "KL", "none", HP35_Z_KL,   "sankey",       "sankey.png",       []),
    (HP35_CFG,        "T",  "none", HP35_Z_T,    "contacts",     "contacts.png",     []),
    (HP35_CFG,        "T",  "none", HP35_Z_T,    "macrotraj",    "macrotraj.png",    []),
    # ck_test uses the real HP35 system; config uses source: relative to project root
    # so run from the real HP35 project root with an absolute output path
    ("REAL_HP35",     "T",  "none", REAL_HP35_Z, "ck_test",      "ck_test.png",      []),
    # rmsd/delta_rmsd need real topology+trajectory; use real PDZ3 data entirely
    (REAL_PDZ3_CFG,   "KL", "none", REAL_PDZ3_Z_KL, "rmsd",      "rmsd.png",         []),
    (REAL_PDZ3_CFG,   "KL", "none", REAL_PDZ3_Z_KL, "delta_rmsd","delta_rmsd.png",   []),
    (HP35_CFG,        "T",  "none", HP35_Z_T,    "state_network","state_network.png",[]),
    (HP35_CFG,        "T",  "none", HP35_Z_T,    "transition_matrix","transition_matrix.png",[]),
    (HP35_CFG,        "T",  "none", HP35_Z_T,    "transition_time",  "transition_time.png",  []),
    (HP35_STOCH_CFG,  "T",  "none", HP35_Z_STOCH,"stochastic_state_similarity","stochastic_state_similarity.png",[]),
    (HP35_STOCH_CFG,  "T",  "none", HP35_Z_STOCH,"relative_implied_timescales","relative_implied_timescales.png",[]),
    (HP35_STOCH_CFG,  "T",  "none", HP35_Z_STOCH,"macro_feature","macro_feature.png",[]),
]

REAL_HP35_ROOT = Path("/home/felixg/Documents/uni/MPP")

for cfg, d, g, z, kind, outname, extra in plots:
    out = OUT / outname
    if cfg == "REAL_HP35":
        args = [REAL_HP35_CFG, d, g, "-Z", z, "-p", kind, "-o", out] + extra
        run(args)
    else:
        args = [cfg, d, g, "-Z", z, "-p", kind, "-o", out] + extra
        run(args)

print("\nAll plots generated successfully.")
