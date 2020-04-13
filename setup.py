import fnmatch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="Grimsel",
    version="0.0.12",
    author="Grimsel contributors listed in AUTHORS",
    author_email="mcsoini_dev@posteo.org",
    description=("GeneRal Integrated Modeling environment for the Supply of"
                 " Electricity and Low-temperature heat"),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/mcsoini/grimsel",
    packages=find_packages(),
    install_requires=['pyutilib>=5.8.0',
                      'pandas>=1.0.0', 
                      'pyomo==5.6.9',
                      'wrapt>=1.12.1', 
                      'psycopg2>=2.8.5', 
                      'numpy>=1.18.2', 
                      'sqlalchemy>=1.3.16',
                      'statsmodels>=0.11.1',
                      'tables>=3.6.1',
                      'fastparquet>=0.3.3',
                      'matplotlib>=3.2.1',
                      'tabulate>=0.8.7'
                      ],
     classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent"],
)

