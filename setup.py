from distutils.core import setup

setup(
    name="micropolarray",
    description="Micro-Polarizer array and PolarCam images processing libraries",
    url="https://github.com/Hevil33/micropolarray_master",
    author="Hervé Haudemand",
    author_email="herve.haudemand@inaf.it",
    packages=[
        "micropolarray",
        "micropolarray.processing",
    ],  # name of the uppermost package directory
    # package_dir={"micropolarray": "./micropolarray"},
    project_urls={
        "Source Code": "https://github.com/Hevil33/micropolarray_master",
        "Documentation": "https://micropolarray.readthedocs.io/en/latest/",
    },
)
