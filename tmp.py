#!/usr/bin/env python

import MPT.run as mpp_run

cfg = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/sm/config/aSyn_kmeans.yaml"

d = mpp_run.Data(cfg)
# Try the following combinations:
# "T", "none" - this is the referenece
# "T", "JS"
# "KL", "none"
# "KL", "JS"
d = mpp_run.setup_mpp("T", "JS", d)
d.mpp.mpt(d.kernel, feature_kernel=d.feature_kernel)

