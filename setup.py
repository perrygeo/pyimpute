import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name="pyimpute",
    version="0.2",
    author="Matthew Perry",
    author_email="perrygeo@gmail.com",
    description=("Utilities for applying scikit-learn to spatial datasets"),
    license="BSD",
    keywords="gis geospatial geographic raster vector zonal statistics machinelearning",
    url="https://github.com/perrygeo/pyimpute",
    package_dir={"": "src"},
    packages=["pyimpute"],
    long_description="See documentation at https://github.com/perrygeo/pyimpute",
    install_requires=["scikit-learn", "scipy", "numpy", "rasterstats", "rasterio"],
    tests_require=["pytest", "pyshp>=1.1.4", "coverage"],
    cmdclass={"test": PyTest},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
