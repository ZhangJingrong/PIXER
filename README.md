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
4. `ImportError: No module named Image`
Solution:
Modify `#import Image, ImageDraw` to: `from PIL import Image` `from PIL import ImageDraw `   
5. `ImportError: No module named google.protobuf.internal` . 
Solution:
Try to use `pip install` can not solve this problem
> sudo apt-get install python-protobuf

# Code Structure  
1. PIXER.sh: One example of how to run the code.    
2. run.sh: The basic workflow of PIXER.     
3. mpiPre.py: Pre-processing the micrograph. The input can be cropped with overlap according to the size of global memory of GPU card.    
4. runNetwork.sh: Run the segmentation network and generate the probability density map.    
5. mpiPost.py: Post-processing, convert data format merge cropped data.    
6. gpuNonMax.py: Run grid-based local-maximum particle locating method.    
7. getStop.py (optional): Eliminate the particles of according to the score generated from probability density map.     
8. genPar.py: Get the priliminary results.    
9. testCls.py: Put the results to the classification network.    
10. erasePar.py: Eliminate the particles according to the output of classification network.    
11. drawRec.py: Draw the results to show the results directly.    
12. color.py: Draw the particles according to the score of the particles.   

# Usage 
To run our program, you can modify the PIXER.sh file.
```
mrcArr=(testDataSet pdb1f07) # This is the name of output directory. 
                             # Here can write multiple directories to process data in batch
dinArr=('./testDataSet/mrc2' './pdb1f07/mrc' ) # The directory of micrograph input 
sizeArr=(128 100) # The particle size (in pixel )
numArr=(250 110)  # The maximum number of particles in each micrograph
gpuID=0 # The id number of GPU card
mpiNum=6 # The number of mpi processes

pre=/home/ict/git/python/PIXER #The pre directories of the program PIXER
```

# Input
This code takes micrographs (`dinArr`)and particle size (`sizeArr`) as input data.

# Output
The particle coordinates in RELION's format (`.star`).
Run `color.py` or `drawRec.py` can generate images in `.png` format. 

