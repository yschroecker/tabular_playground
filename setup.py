import setuptools

setuptools.setup(
    name="tabular",
    version="3.0",
    description="Tools for RL & IL experiments on tabular domains",
    packages=setuptools.find_packages("tabular"),
    python_requires=">3.6.0",
    install_requires=["numpy>=1.12", "matplotlib>=3.0.0", "scipy>=1.1"]
)
