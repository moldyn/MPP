import numpy as np

def Z_to_linkage(Z):
    l = Z[:, :, :3].copy()
    for r, z in enumerate(l):
        for i, row in enumerate(z):
            mask = np.where(l[r, :, :2]==i+Z.shape[1]+1)
            l[r, :, :2][mask] = row[1]
    l[:, :, :2] += 1
    return l
