# Workflow Organization

The MPP package provides an integration into a Snakemake workflow. To use it, you need to copy this workflow directory into your working directory where you keep your data. A sample structure might look like this:

```bash
data/
├── System1
│   ├── input
│   │   ├── clusters
│   │   ├── config.yml
│   │   ├── contact_distances_trajectory
│   │   ├── contacts.ndx
│   │   ├── microstate_trajectory
│   │   ├── README.md
│   │   ├── topology.pdb
│   │   ├── trajectory.xtc
│   │   └── view
│   └── results
│       ├── t
└── workflow
    ├── Snakefile
```


