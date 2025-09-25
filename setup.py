from MPP import __version__

from setuptools import setup, find_packages

setup(
    name="MPP",  # Module name
    version=__version__,
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[  # List of dependencies for your module
        "tqdm>=4.67.1",
        "numpy>=2.2.5",
        "seaborn>=0.13.2",
        "networkx>=3.5",
        "mdtraj>=1.11.0",
        "scikit_learn>=1.7.2",
        "pygpcca>=1.0.4",
        "anytree>=2.12.1",
        "fa2_modified>=0.3.10",
        "msmhelper>=1.1.1",
        "bezier>=2024.6.20",
    ],
)
