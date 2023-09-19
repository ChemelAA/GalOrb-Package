import os
from setuptools import setup, find_packages

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()

def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

setup(
    name = "citros_data_analysis",
    version=get_version("citros_data_analysis/__init__.py"),
    author ="Chemel", 
    author_email="",
    license = "LICENSE.txt",
    description = "Orbit calculations",
    long_description=open('README.md').read(),
    packages=find_packages(),
    python_requires='<3.11',
    install_requires = [
    "numpy<=1.21.2",
    "scipy",
    "pandas",
    "matplotlib",
    "astropy",
    ],
    url = "", 
)