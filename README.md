# PIXER
PIXER: An Automated Particle-selection Method Based on Segmentation Using a Deep Neural Network
# Installation 
## Requirements
1.[Deeplab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2)  
2. pycuda  
3. mrcfile  
4. mpi4py    
## Possible problems
1. `fatal error: matio.h: No such file or directory `
Solution: 
> sudo apt-get install libmatio-dev 
2.  `./include/caffe/common.cuh(9): error: function "atomicAdd(double *, double)" has already been defined`
Solution : remove this defination.
3. `Aborted at 1534155481 (unix time) try "date -d @1534155481" if you are using GNU date ***`  
Solution:
This may be caused by the wrong data input directory.
4. ```Traceback (most recent call last):
  File "/home/jingrong/software/EM/Pixer/PIXER/gpuNonMax.py", line 7, in <module>
    import Image, ImageDraw
ImportError: No module named Image
Command exited with non-zero status 1
```
Solution:
Modify `#import Image, ImageDraw` to: 
> from PIL import Image
from PIL import ImageDraw  

5. `ImportError: No module named google.protobuf.internal`
Solution:
Try to use `pip install` can not solve this problem
> sudo apt-get install python-protobuf
 
# Usage 
# Introduction of each source file 
