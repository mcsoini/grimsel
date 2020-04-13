import fnmatch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

with open("description_pypi.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="Grimsel",
    version="0.8.0",
    author="Grimsel contributors listed in AUTHORS",
    author_email="mcsoini_dev@posteo.org",
    description=("GeneRal Integrated Modeling environment for the Supply of"
                 "Electricity and Low-temperature heat"),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/mcsoini/grimsel",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent"],
)

