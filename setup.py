from setuptools import setup, find_packages

setup(
    name="hrp-trading",
    version="0.1.0",
    packages=find_packages(),
    description="Hierarchical Risk Parity and other trading strategies",
    author="Alex",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
    ],
)
