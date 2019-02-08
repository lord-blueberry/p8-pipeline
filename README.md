# p8-pipeline

contains the proof-of-concept reconstruction pipeline used in this project-8. It uses Coordinate Descent to reconstruct an image of a Radio Interferometer. It reads CASA MS files and outputs a numpy array as image.

## dependencies
The pipeline needs two python dependencies, which in turn require linux-only binaries.

pydata
 * Casacore <https://kernsuite.info/>

pynufft
  * NFFT 3.2.4 <https://www-user.tu-chemnitz.de/~potts/nfft/download.php>
  
