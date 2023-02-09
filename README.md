# micropolarray_lib

Python module for loading and using micropolarizer array and polarcam images.


## Installation 

Open a terminal inside the master directory and then launch TWICE the following command

```
pip install .
```


## Usage

```
import micropolarray
import numpy as np

# Can initialize images from np arrays or filenames. If multiple filenames are used
# the average is taken
micropolimage_from_file = micropolarray.MicroPolarizerArrayImage("image.fits")
micropolimage_from_nparray = micropolarray.MicroPolarizerArrayImage(np.ones(shape=(30,30)))
micropolimage_from_image = micropolarray.MicroPolarizerArrayImage(image)


```
