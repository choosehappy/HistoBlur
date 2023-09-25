from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as rq:
    requirements = rq.readlines()

setup(
    
    #Library name

    name="HistoBlur",

    version="1.0.0",

    author= "Petros Liakopoulos, Rahul Nair and Andrew Janowczyk",

    author_email= "liakopoulos.petros@gmail.com",

    description="HistoBlur is a Deep Learning based tool for fast and accurate blur detection on Whole Slide Images",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/choosehappy/HistoBlur.git",

    install_requires = requirements,

    packages= find_packages(),

    include_package_data=True,

    python_requires=">=3.7",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux "
    ],
    entry_points={"console_scripts" : ["HistoBlur=histoblur.main:main"],
    }

)