from distutils.core import setup

setup(
    name="micropolarray",
    version="1.0 beta",
    description="Micro-Polarizer array and PolarCam images processing libraries",
    url="https://github.com/Hevil33/micropolarray_master",
    author="Hervé Haudemand",
    author_email="herveh96@hotmail.it",
    install_requires=[
        "numpy<1.24.0",  # compatibility with pip installation
        "pandas",
        "numba",
        "dataclasses",
        "astropy",
        "matplotlib",
        "scipy",
        "wheel",
        "tqdm",
    ],
    packages=[
        "micropolarray",
        "micropolarray.processing",
    ],  # name of the uppermost package directory
    package_dir={"micropolarray": "build/lib/micropolarray"},
    # py_modules=[
    #    "micropolarray",
    #    # "micropolarray.processing",
    #    "micropolarray.image",
    #    "micropolarray.utils",
    #    "micropolarray.micropol_image",
    #    "micropolarray.parallelize",
    #    "micropolarray.polarization_functions",
    #    "micropolarray.cameras",
    # ],
)
