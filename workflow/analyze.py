import sys
import MPP
import MPP.run as r
import numpy as np
import yaml
import itertools


root = "/data/evaluation/MPP/stochastic_MPP_Felix/data/"
workflow = "/data/evaluation/MPP/stochastic_MPP_Felix/MPP/workflow/"
systems = [
    "HP35",
    "PDZ3",
    "aSyn",
]
setups = [
    "t",
    "kl",
    "t_js",
    "kl_js",
]


interesting_lumpings = {
    "HP35": {
        "name": "HP35",
        "setups": ["t", "kl", "t_js", "kl_js", "gpcca"],
    },
    "aSyn": {
        "name": "aSyn_rdc_200ns",
        "setups": ["t", "kl", "t_js", "kl_js", "gpcca"],
    },
}

il2 = {
    "HP35": {
        "name": "HP35",
        "setups": ["t"],
    }
}


with open(f"{workflow}lumpings.yml") as f:
    lumpings = yaml.safe_load(f)


def get_d(system, setup):
    d = r.Data(f"{root}{system}/input/config.yml")
    d.setup_mpp(
        lumpings[setup]["kernel similarity"],
        lumpings[setup]["feature kernel"],
    )
    if setup == "gpcca":
        d.perform_gpcca(
            "ref",
            f"{root}{system}/results/{setup}/Z.npy",
        )
    else:
        d.perform_mpp(f"{root}{system}/results/{setup}/Z.npy")
    return d


def return_rmsd_str(d):
    s = ""
    for i in d.mpp.rmsd.sum(axis=1):
        s += f"{i:.2f} & "
    return s[:-3]


def return_rmsd_str2(d):
    s = ""
    for i in d.mpp[:, 2:-2].rmsd.sum(axis=1):
        s += f"{i:.2f} & "
    return s[:-3]


def nested_dict_to_latex_table(data):
    # Collect first-level keys
    top_keys = list(data.keys())
    # Collect second-level keys (column subheaders) for each top key
    subkeys_per_top = [list(data[top].keys()) for top in top_keys]

    # Determine the maximum number of rows needed (max array length across all subkeys)
    max_len = max(len(arr) for top in data.values() for arr in top.values())

    # Build LaTeX table
    latex = []
    latex.append(
        "\\begin{tabular}{c" + "".join("c" * sum(len(s) for s in subkeys_per_top)) + "}"
    )
    latex.append("\\toprule")

    # First header row (merged cells for top keys)
    header1 = ["Macrostate"]
    for top, subs in zip(top_keys, subkeys_per_top):
        header1.append(f"\\multicolumn{{{len(subs)}}}{{c}}{{{top}}}")
    latex.append(" & ".join(header1) + " \\\\")

    # Second header row (subkeys)
    header2 = [""]
    for subs in subkeys_per_top:
        header2.extend(subs)
    latex.append(" & ".join(header2) + " \\\\")
    latex.append("\\midrule")

    # Data rows
    for i in range(max_len):
        row = [str(i + 1)]
        for top, subs in zip(top_keys, subkeys_per_top):
            for sub in subs:
                arr = data[top][sub]
                if i < len(arr):
                    row.append(f"{arr[i]:.3g}")
                else:
                    row.append("")  # blank if index out of range
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    print("\n".join(latex))


def create_dict(d_lum, func):
    """Apply a function to all lumpings in d_lum and return the
    corresponding dict.
    """
    new_dict = {}
    for system in interesting_lumpings:
        new_dict[system] = {}
        for setup in interesting_lumpings[system]["setups"]:
            new_dict[system][setup] = func(d_lum[system][setup])
    return new_dict


# Functions utilized for create_dict. Must return a list or array.
def rmsd_sharpness(d):
    return [d.mpp.rmsd_sharpness()]


def rmsd_sharpness_detailed(d):
    return (
        d.mpp.rmsd.mean(axis=1)
        * d.mpp.macrostate_population[0]
        / d.mpp.macrostate_population[0].sum()
        * 1000
    )


def rmsd(d):
    d.mpp.rmsd_feature = rmsd_feature
    return (d.mpp.rmsd**2).sum(axis=1)


def rmsd_sum(d):
    """Intra"""
    return [(rmsd(d)).sum() / d.mpp.n_macrostates[0]]


def rmsd_inv(d):
    return [1 / rmsd_sum(d)[0]]


def shannon_entropy(d):
    return [d.mpp.shannon_entropy[0]]


def dbi(d):
    return [d.mpp.davies_bouldin_index[0]]


def gmrq(d):
    return [d.mpp.gmrq[0]]


def gmrq2(d):
    return [d.mpp.gmrq2[0]]


def implied_timescales(d):
    return d.mpp.timescales[0] * d.mpp.frame_length


def population(d):
    return d.mpp.macrostate_population[0] / d.mpp.macrostate_population[0].sum()


def macrostate_feature(d):
    return d.mpp.macrostate_feature[0]


def interstate_rmsd(d):
    features = np.array(d.mpp.macrostate_multi_feature[0])  # (e.g. 12x42)
    mask1 = np.array(
        [[i] * (features.shape[0] - 1) for i in range(features.shape[0])]
    )  # e.g. 12 x 11
    considered_features = features[mask1]  # e.g. 12 x 11
    mask2 = [
        np.arange(features.shape[0], dtype=int) != i for i in range(features.shape[0])
    ]  # 12 x 11
    other_features = np.array([features[i] for i in mask2])  # 12 x 11 x 42

    # Calculate RMSD
    return np.sqrt(((other_features - considered_features) ** 2).mean(axis=2)).sum(
        axis=1
    )  # 12 x 11 x 42 -> 12 x 11 -> 12


def interstate_rmsd_sum(d):
    return [interstate_rmsd(d).sum()]


def inter_rmsd_mean(d):
    p = population(d)
    rmsd = interstate_rmsd(d)
    return [(rmsd * p).sum()]


def mean_square_distance(A: np.ndarray, B: np.ndarray) -> float:
    # A: (n_a, d), B: (n_b, d)
    mean_norm_A = np.mean(np.sum(A**2, axis=1))  # average squared norm of A
    mean_norm_B = np.mean(np.sum(B**2, axis=1))  # average squared norm of B
    mean_A = np.mean(A, axis=0)
    mean_B = np.mean(B, axis=0)

    return mean_norm_A + mean_norm_B - 2 * np.dot(mean_A, mean_B)


def msd(state_traj, feature_traj):
    states, counts = np.unique(state_traj, return_counts=True)
    n_states = len(states)
    masks = [np.where(state_traj == i)[0] for i in states]
    msd_abs = []
    for i, j in itertools.combinations(states, 2):
        a = feature_traj[masks[i]]
        b = feature_traj[masks[j]]
        msd_abs.append(mean_square_distance(a, b))
    return 2 / (n_states * (n_states - 1)) * sum(msd_abs)


def inter(d):
    """Compare all frames for state distance."""
    return [msd(d.mpp.macrostate_trajectory[0], d.mpp.multi_feature_trajectory)]


def Q(d):
    return [inter(d)[0] / rmsd_sum(d)[0]]


def Q_H(d):
    return [inter(d)[0] / rmsd_sum(d)[0] * shannon_entropy(d)[0]]


rmsd_feature = "feature"
d_lum = {}
for system in interesting_lumpings:
    d_lum[system] = {}
    for setup in interesting_lumpings[system]["setups"]:
        d = get_d(system, setup)
        d.mpp.load_rmsd(f"{root}{system}/results/{setup}/rmsd_{rmsd_feature}.npy")
        d_lum[system][setup] = d


dhp = d_lum["HP35"]["t"]

d_gmrq = create_dict(d_lum, gmrq)
d_gmrq2 = create_dict(d_lum, gmrq2)

d_its = create_dict(d_lum, implied_timescales)