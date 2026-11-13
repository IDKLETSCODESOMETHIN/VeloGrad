from setuptools import setup, find_packages

setup(
    name="velograd",
    version="1.1.0b0",
    author="Ayushman Singh Rathore",
    description="VeloGrad: Pure Compiled, Bare-Metal High-Performance Machine Learning Core Engine",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "velograd=velograd.cli:main",
            "velograd-convert=velograd.converter:main",
            "velograd-studio=velograd.studio:run_studio_server",
        ],
    },
)
