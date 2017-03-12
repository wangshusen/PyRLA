# PyRLA: Randomized Linear Algebra in Python

This package implements the most common randomized matrix computation algorithms. 


## Demos

### 1. Prepare: download and process data

Download the "Year Prediction Million Song Dataset"

>- go to the directory "data/"
>- Linux: bash LinuxDownloadData.sh
>- Mac: bash MacDownloadData.sh

Convert the data to NumPy data file

>- python processLibSVMData.py

Wait a while.
The output file is "YearPredictionMSD.npy"

### 2. Run demos. 

Here are some examples.

* Matrix sketching
  * "sketch/demo/demo\_rft.py": matrix coherence after the randomized Fourier transform (RFT) gets much smaller.
  * "sketch/demo/demo\_sketch.py": apply SRFT, count sketch, and leverage score sampling to matrix multiplication and compare their errors.

* Optimization
  * "optimization/demo/demo\_precondition\_cg": the converge of CG with/without preconditioning.