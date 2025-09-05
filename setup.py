"""
Setup.py for PyMapAccuracy package.

This file provides backward compatibility for environments that require setup.py.
The canonical package configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="pymapaccuracy",
    version="0.1.0",
    author="Andrew Copenhaver", 
    author_email="andrew.copenhaver@example.edu",  # Replace with actual email
    description="A package for calculating thematic map accuracy and area under stratified random sampling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/acopenhaver/PyMapAccuracy",  # Replace with actual repo
    project_urls={
        "Bug Tracker": "https://github.com/acopenhaver/PyMapAccuracy/issues",
        "Documentation": "https://github.com/acopenhaver/PyMapAccuracy",
        "Source Code": "https://github.com/acopenhaver/PyMapAccuracy",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords=["thematic map", "accuracy", "stratified sampling", "remote sensing", "GIS"],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
