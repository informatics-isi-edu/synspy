# Miscellaneous Matlab functions

## Contents

### Scripts

graph1DSynapses.m is a Matlab function that can be used to show a 1D simplification of the synapse identification algorithm in [synspy](https://github.com/informatics-isi-edu/synspy/). This file contains a function that was created and tested in Matlab 2019a. Raw data line profiles from the [synapse data management system](synapse.isrd.isi.edu) can be analyzed with this software to create a 1-D identification of synapses in a manner that is similar (but not identical) to the 3-D case that is used in the manuscript. 

The two .csv files in this folder represent individual line profile intensity data that can be used as a test case. These files were the data used in the manuscript itself. RawData_Profile.csv contains a line profile of a 2-dimensional raw intensity image (1-dimensional data), while GaussianBlurDataPixelVersion_Profile.csv contains the same line profile but after a 2.16 pixel Gaussian blur was applied to the raw image. These files can be used as input filenames into the graph1DSynapses.m function to demonstrate how the function works.