# micropolarray

Python open-source module for loading and using micropolarizer array and PolarCam images.


## Installation 

Run one of the following commands in your terminal:

```
pip install micropolarray
```

OR

```
pip install git+https://github.com/Hevil33/micropolarray_master
```

If an error message about wheels appear, launch it again.
You can run the `test.py` script to verify the correct installation of the micopolarray package.

## Features

- Automatic polarization calculation
- Fast and optimized operations on the micropolarizer array
- Basic image cleaning (dark/flat subtraction)


## Documentation

Documentation is hosted at ReadTheDocs and can be found [HERE](https://micropolarray.readthedocs.io/en/latest/) (html format).


## Usage

Get the simple [jupyter tutorial](https://github.com/Hevil33/micropolarray_master/blob/main/TUTORIAL.ipynb) for a brief introduction.

After installation, you can import the library in your python application

```
import micropolarray as ml
```

The main class is `MicropolImage()`, which can be initialized from

1. A `numpy` 2D array 
2. A list of .fits filenames
3. Another `MicropolImage()`


Some useful member functions are :

MicropolImage()
- .show()
- .show_with_pol_params()
- .rebin()
- .demosaic()

Information on polarization is automatically calculated and stored in the class members as `numpy` arrays

MicropolImage()
- .data
- single_pol_subimages
- .Stokes_vec
- .I.data
- .Q.data
- .U.data
- pB.data (polarized brightness)
- AoLP.data (angle of linear polarization)
- DoLP.data (degree of linear polarization)


## Additional modules

micropolarray:

- .processing
  - .congrid (_experimental_) : fast congrid operations
  - .convert : raw (binary) files to fits conversion
  - .new_demodulation : image demodulation and demodulation tensor calculation
  - .demosaic : fast image demosaicing
  - .nrgf : occulter finding and roi selection, nrgf filter
  - .rebin : fast image binning
  - .shift : image shifting
- .cameras (_experimental_) : classes for sensor informations
- .image : general image handling
- .utils 
