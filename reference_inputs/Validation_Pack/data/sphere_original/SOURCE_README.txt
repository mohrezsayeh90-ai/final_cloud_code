DATA FROM DNS OF A SINGLE SPHERE SETTLING IN AMBIENT FLUID, 
BENCHMARK USING SPECTRAL/SPECTRAL-ELEMENT METHOD AND 
MOVING GRID

Flow configuration:

The motion of a single spherical particle settling in ambient flow is
computed with a spectral/spectral-element method. Data is provided for
the purpose of benchmarking. A single density ratio of 1.5 is
considered. The Galileo number is varied in the range of 144-250,
covering steady vertical, steady oblique, oscillating oblique and
chaotic motion (corresponding respectively to axi-symmetric,
double-threaded oblique, periodic hairpin-shedding and chaotic
wakes). 

How to obtain the data:

The entire data-set (including utilities and documentation) is
contained in the file "all_data_etc.tgz". Download and unpack it. Then
edit the provided Matlab script "matlab/test_read1.m" (adapt the
top-level directory path such that it points to the unpack directory)
and execute it; it will read the data and plot it.

Documentation:

Please check the document "doc/data-sheet.pdf". It contains basic
information and further pointers.

References:

M. Uhlmann & J. Dusek, "The motion of a single heavy sphere in ambient
fluid: a benchmark for interface-resolved particulate flow simulations
with significant relative velocities", Int. J. Multiphase Flow, 59:221-243, 
2014. 
http://dx.doi.org/10.1016/j.ijmultiphaseflow.2013.10.010
http://arxiv.org/abs/1310.5358

History:

Apr-02-2013 announcement
Jul-29-2013 manuscript submitted
Oct-09-2013 repository first filled with data
Oct-19-2013 manuscript accepted in IJMF
Oct-22-2013 manuscript available on arxiv

Contact:

markus.uhlmann@kit.edu