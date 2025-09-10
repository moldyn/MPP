from MPP import __version__

from setuptools import setup, find_packages

setup(
    name="MPP",  # Module name
    version=__version__,
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[  # List of dependencies for your module
        "anytree>=2.12.1",
        "matplotlib>=3.10.0",
        "mdtraj>=1.9.9",
        "msmhelper>=1.1.1",
        "numba>=0.59.1",
        "numpy>=1.23",
        "pandas>=2.2.3",
        "prettypyplot>=0.11.0",
        "scikit_learn>=1.6.1",
        "scipy>=1.15.1",
        "seaborn>=0.13.2",
        "tqdm>=4.66.2",
    ],
)
