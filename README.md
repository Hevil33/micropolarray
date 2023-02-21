# micropolarray

Python module for loading and using micropolarizer array and polarcam images.


## Installation 

Run the following command inside a terminal.
If an error message about wheels appear, launch it again.

```
pip install git+https://github.com/Hevil33/micropolarray_master
```

You can run the `test.py` script to verify the correct installation of the micopolarray package.


## Usage

```
import micropolarray
import numpy as np

# Can initialize images from np arrays or fitsfilenames.
# If multiple filenames are used then the average is taken 

micropolimage_from_file = micropolarray.MicroPolarizerArrayImage("test_data/image.fits.fits")
micropolimage_from_nparray = micropolarray.MicroPolarizerArrayImage(np.ones(shape=(30,30)))
image = micropolarray.MicroPolarizerArrayImage(image)
```

Multiple useful members can be used to get polarization parameters

```
angle_of_linear_polarization = image.AoLP.data  # Get the angle of linear polarization
Stokes_I, Stokes_Q, Stokes_U = self.Stokes_vec  # Get the stokes vector components as np.ndarray
pol_0_image = image.single_pol_subimages[image.angle_dic[0]]

demosaiced_image = image.demosaic() 
binned_image = image.rebin(binning=4)  # binned 4x4 image

```
