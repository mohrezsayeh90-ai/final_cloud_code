DATA FROM DNS OF A SINGLE OBLATE SPHEROID SETTLING IN AMBIENT FLUID,
BENCHMARK USING SPECTRAL/SPECTRAL-ELEMENT METHOD AND MOVING GRID

# Flow configuration

The results contained in the data files correspond to a single 
oblate spheroid with geometrical aspect ratio 1.1 or 1.5. Different
Galileo numbers ranging from 100 to 220 as well as different density
ratios can be found (see table below).

  |    caseid  |  chi |  Ga | mstar | rhop/rhof | regime            | 
  | `A11-M100` |  1.1 | 100 | 1.00  |  2.10     | Steady vertical   |
  | `B11-M100` |  1.1 | 115 | 1.00  |  2.10     | Steady oblique    |
  | `B15-M075` |  1.5 | 110 | 0.75  |  2.14     | Steady oblique    |
  | `C15-M075` |  1.5 | 150 | 0.75  |  2.14     | Vertical periodic |
  | `D11-M100` |  1.1 | 200 | 1.00  |  2.10     | Chaotic           |
  | `D15-M500` |  1.5 | 220 | 5.00  | 14.32     | Chaotic           |

The data presented here has been obtained with a spectral/spectral-element
method and is intended to be used for benchmarking purposes.

All quantities are made non-dimensional with the gravitational velocity Ug
and the equatorial diameter of the particle d. Note that the definition of 
the gravitational velocity is `Ug=sqrt((rhop/rhof-1)*abs(g)*V/d^2)` and
the Galileo number `Ga=Ug*d/nu`.

# How to obtain the data

The entire data-set (including utilities and documentation) is
contained in the file "all_data_etc.tgz". Download and unpack it.
The provided Matlab script `matlab/test.m` can be used to generate
figures in a similar fashion as they are presented in the manuscript.

If the structure of the directories is the same as in the orginal 
`tgz` file and the script `test.m` is called from the directory 
`mfiles`, set 'caseid' to generate figures of any case presented 
in the manuscript, e.g.:

>> caseid='A11-M100';
>> test;

If the script is called from a different path, set also  `basedir`
to load the data from the path where the data is saved.

>> basedir='/path/to/the/data';
>> caseid='A11-M100';
>> test;


# Documentation:

TO-BE-ADDED!
[Please check the document "doc/data-sheet.pdf". It contains basic
information and further pointers.]

# References:

M. Moriche & M. Uhlmann, "A single oblate spheroid settling in 
unbounded ambient fluid: a benchmark for simulations in steady 
and unsteady wake regimes", to be submitted to Int. J. Multiphase Flow

# History:

Aug-07-2020 creation & data added

# Contact

manuel.guerrero@kit.edu
markus.uhlmann@kit.edu
