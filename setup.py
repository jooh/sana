#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["numpy"]

setup_requirements = ["pytest-runner"]

# NB tensorflow is an optional dependency but if you're going to run the test suite you
# will need it
test_requirements = ["pytest", "tensorflow"]

setup(
    author="Johan Carlin",
    author_email="johan.carlin@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
    ],
    description="Sana  is for distance matrix similarity analysis in python / numpy / tensorflow",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="sana",
    name="sana",
    packages=find_packages(include=["sana"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jooh/sana",
    version="0.1.0",
    zip_safe=False,
)
