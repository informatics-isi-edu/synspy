
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

"""Synapse detection in 3D microscopy images using size-specific blob detection.

The method uses several convolution kernels which have been
experimentally derived:

   Low: a gaussian distribution approximating a synapse's signal
   distribution

   Red: a gaussian distribution to blur an optional autofluorescence
   channel

   Core: a kernel containing mostly central/high-intensity voxels
   within synapse signal blobs
 
   Span: a larger kernel containing the entire local region of
   synapse centroids

   Hollow: a difference of (Span - Core) containing mostly
   peripheral/low-intensity voxels around synapse signal blobs

The 3D image convolution I*Low is trivially separated into 1D gaussian
convolutions on each axis for efficiency.

Candidate synapses are detected by finding local maxima in the I*Low
convoluton result, i.e. voxels where the measured synapse core signal
is equal to the maximum within a local box centered on the same voxel.

Additional measures are computed sparsely at each candidate centroid
location:

   A. I*Core

   B. I*Hollow

   C. I*Red

These measures use direct (non-separated) 3D convolution over the
small image region surrounding the centroid.  Because the set of
candidate centroids is so small relative to the total volume size,
this is faster than convolving the entire image even with separated
kernels.

Centroid classification is based on the computed measures for each
centroid.

Currently, the Core and Span kernels are simple box kernels, but we
may vary these empirically to improve our synapse blob classification.
The sparse measurement setup allows arbitrary 3D kernels since they
do not need to be separable.

"""
