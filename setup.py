from setuptools import setup, find_packages

setup(
    name="recovar_torch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0",
        "pandas",
        "scipy",
        "scikit-learn",
        "h5py",
        "obspy",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.10",
)
