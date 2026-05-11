from setuptools import setup, find_packages

setup(
    name="pipeline-sim",
    version="0.3.0",
    description="Single-phase liquid pipeline transient simulator (MOC + FD)",
    author="Orbit / chendw79",
    author_email="chendw79@gmail.com",
    url="https://github.com/chendw79/pipeline-sim",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "plotting": ["matplotlib>=3.4"],
        "hdf5": ["h5py>=3.0"],
        "all": ["matplotlib>=3.4", "h5py>=3.0"],
    },
    entry_points={
        "console_scripts": [
            "pipeline-sim=examples.pipeline_sim_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="pipeline transient simulation MOC waterhammer fluid-dynamics",
)
