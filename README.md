# p8-pipeline

contains the proof-of-concept reconstruction pipeline used in this project-8. It uses Coordinate Descent to reconstruct an image of a Radio Interferometer. It reads CASA MS files and outputs a numpy array as image.

The actual code is located in ./pipeline

## dependencies
The pipeline needs two python dependencies, which in turn require linux-only binaries.

pydata
 * Casacore <https://kernsuite.info/>

pynfft
  * NFFT 3.2.4 <https://www-user.tu-chemnitz.de/~potts/nfft/download.php>
  
## pipeline folder structure
 **./pipeline/single_run.py** is the main project file, it contains the routines for loading simulated MS files and reconstructing an image with Coordinate Descent

 * **./pipeline/algorithms** --> contains the coordinate descent implementation
 * **./pipeline/benchmark** --> contains the simulated measurement sets
 * **./pipeline/debug_data** --> contains matlab files from a different observation
 * **./pipeline/msinput.py** --> contains modified pydata code for loading simulated MeerKAT MS
 * **./pipeline/nufftwrapper.py** --> contains the wrapper code of the pynfft code
 
