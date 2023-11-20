#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="MRI Reconstruction using Lighting and Hydra",
    author="Yuyang Xue",
    author_email="yuyang.xue@ed.ac.uk",
    url="https://github.com/YuyangXueEd/ReconHydra",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
