from setuptools import setup, find_packages

setup(
    name="unixcoder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.20.0",
        "fuzzywuzzy>=0.18.0",
        "transformers>=4.6.0"
    ]
)
