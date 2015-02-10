
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

"""Synapse detection in 3D microscopy images using size-specific blob detection.

This solution uses simple Gaussian distributions to model the signal
of a fluorescent synapse and its background, using experimentally
derived feature characteristics.

The method uses several convolution kernels:

   Core: a guassian distribution approximating a synapse's signal
   distribution

   Vicinity: a larger gaussian distribution approximating a vicinity
   surrounding (and including a synapse)

   Hollow: a renormalized difference of gaussians (Hollow - Core)
   approximating the surrounding background of a synapse.

The 3D image convolution I*Core is trivially separated into 1D
convolutions on each axis.

The 3D image convolution I*Hollow is algebraically decomposed into a
combination of convolutions which can each be separated into 1D
convolutions:

   I*Hollow = I*(Vicinity - Core + k)/s)
            = (I*(Vicinity - Core + k))/s
            = (I*Vicinity - I*Core + I*k)/s

where Vicinity and Core are gaussian kernels as described above, k is
a constant box filter kernel, and s is a scalar.

Additionally, the method uses a local maxima filter using a box filter
size related to the Hollow filter size.

Candidate synapses are detected by finding local maxima in the I*Core
convoluton result, i.e. voxels where the measured synapse core signal
is equal to the local maximum of that signal. 

Candidate synapses are characterized by the measured synapse core
signal and the measured surrounding background signal at the same
location.  These measurements are compared to manually determined
thresholds to accept synapses that are:

   A. bright enough core signal to be considered significant

   B. dark enough background signal to be considered distinctly
      visible in intercellular space

   C. (optional) with dark enough auto-fluorescence channel to not be
      considered junk

"""

import np as numpylib

import datetime
import sys
import random
import numpy as np
from numpy import array, float32, empty, newaxis, dot, cross, zeros, ones
from numpy.linalg import norm
import scipy
from scipy import ndimage
import json
import math
import re
import csv

def Gsigma(sigma):
    """Pickle a gaussian function G(x) for given sigma"""
    def G(x):
        return (math.e ** (-(x**2)/(2*sigma**2)))/(2 * math.pi* sigma**2)**0.5
    return G

def sigma_micron_ndim(diam_iter, meta):
    for d, u in zip(diam_iter, (meta.z_microns, meta.y_microns, meta.x_microns)):
        #print "sigma_micron_ndim: diam=%s unit=%s" % (d, u)
        yield (d / u) / 6.

def gaussian_kernel(s):
    G = Gsigma(s) # G(x) gaussian function
    kernel_width = 2 * (int(6.0 * s - 1) / 2) + 1 # force always odd
    kernel_radius = (kernel_width - 1) / 2 # doesn't include central cell
    kernel = map(G, range(-kernel_radius, kernel_radius+1))
    mag = sum(kernel)
    kernel = map(lambda x: x / mag, kernel)
    return kernel

def pad_centered(orig, newshape, pad=0):
    assert len(newshape) == orig.ndim
    
    for d in range(len(newshape)):
        assert newshape[d] >= orig.shape[d]
        
    na = zeros(newshape, dtype=orig.dtype)
    
    def helper1d(dstlen, srclen):
        if dstlen > srclen:
            return slice((dstlen-srclen)/2, -(dstlen-srclen)/2)
        else:
            return None

    na[ tuple( map(helper1d, na.shape, orig.shape) ) ] = orig

    return na

def compose_3d_kernel(klist):
    zk, yk, xk = klist
    def mult(a, b):
        return array(
            [ a[i] * array(b) for i in range(len(a)) ],
            dtype=float32
        )
    result = mult(zk, mult(yk, xk))
    return result
            

def numstr(x):
    s = "%f" % x
    m = re.match("""^(?P<whole>[0-9]*)[.](?P<frac>(?:[0-9]*[1-9])?)(?P<trail>0*)$""", s)
    g = m.groupdict()
    if g['frac']:
        return "%(whole)s.%(frac)s" % g
    else:
        return "%(whole)s" % g

def kernel_diameters(s):
    """Return a 3-tuple from JSON string X or array [Z, Y, X].

       If a single value X is provided, 
    """
    v = json.loads(s)
    if type(v) is array:
        assert len(v) == 3
        return tuple(map(float, v))
    else:
        v = float(v)
        return tuple(2*v, v, v)


def prepare_kernels(meta, synapse_diam_microns, vicinity_diam_microns, redblur_microns):
    """Prepare synapse-detection convolution kernels.

       Parameters:

         meta: image metadata from load_image() with voxel size info
         synapse_diam_microns: core synapse feature span
         vicinity_diam_microns: synapse local background span
         redblur_microns: auto-fluourescence blurring span

       All span arguments are 3-tuples of micron lengths of the
       standard-deviation of the related Gaussian distribution in each
       image dimension (Z, Y, X).

       Result is a 4-tuple:

         ( kernels_separated, kernels_3d, hollow_offset, hollow_scale ).

       The kernels_separated result is a 4-tuple:

         ( syn_kernels, vcn_kernels, red_kernels, k_kernels )

       where each field is a 3-tuple of numpy arrays, each array being
       weights of a 1D kernel for each image dimension (Z, Y, X).

       The kernels_3d result is a pair:

         ( syn_kernel_3d, hollow_kernel_3d )

       where each field is a numpy array, each array being weights of
       a 3D kernel.

       The hollow_offset and hollow_scale fields are floating point
       values.

    """

    # these are separated 1d gaussian kernels
    syn_kernels = map(
        gaussian_kernel,
        sigma_micron_ndim(synapse_diam_microns, meta)
        )

    vcn_kernels = map(
        gaussian_kernel,
        sigma_micron_ndim(vicinity_diam_microns, meta)
        )

    red_kernels = map(
        gaussian_kernel,
        sigma_micron_ndim(redblur_microns, meta)
        )

    #print "kernel shapes:", map(lambda kl: map(len, kl), (syn_kernels, vcn_kernels, red_kernels))

    # we need the combined 3d kernels to compute some constants
    syn_kernel_3d = compose_3d_kernel(syn_kernels)
    vcn_kernel_3d = compose_3d_kernel(vcn_kernels)
    syn_kernel_3d_pad = pad_centered(syn_kernel_3d, vcn_kernel_3d.shape)

    hollow_kernel_3d = vcn_kernel_3d - syn_kernel_3d_pad
    #print "3d_kernel sums:", syn_kernel_3d.sum(), vcn_kernel_3d.sum(), hollow_kernel_3d.sum()

    hollow_offset = - hollow_kernel_3d.min()

    hollow_kernel_3d = hollow_kernel_3d + hollow_offset
    #print "3d_kernel sums:", syn_kernel_3d.sum(), vcn_kernel_3d.sum(), hollow_kernel_3d.sum()

    hollow_scale = hollow_kernel_3d.sum()
    
    #print "3D hollow constants:", hollow_offset, hollow_scale
    hollow_kernel_3d = hollow_kernel_3d / hollow_scale
    #print "3d_kernel sums:", syn_kernel_3d.sum(), vcn_kernel_3d.sum(), hollow_kernel_3d.sum()

    k_kernels = [
        [ (hollow_offset)**(1./3) for i in range(len(k)) ]
        for k in vcn_kernels
    ]

    return (
        (syn_kernels, vcn_kernels, red_kernels, k_kernels),
        (syn_kernel_3d, hollow_kernel_3d),
        hollow_offset, hollow_scale
        )

class BlockedAnalyzer (object):
    """Analyze image using block decomposition for scalability.

       Conceptually we perform the sequence:

          prepare_kernels  (cheap, one-shot)
          process_volume   (expensive, scales with image size)
          analyze          (expensive, scales with image size)
          
       This can be decomposed into blocks and operate on 
       spatial sub-problems:

          process_volume_block  (totally independent)
          analyze_block (might be indepndent in practice?)

       The output of process_volume undergoes a global feature
       labeling step and measurement in the analyze phase.  If
       features include multiple voxels and might span a block
       boundary, then analysis cannot be decomposed.

    """

    def convNx1d(self, *args):
        return numpylib.convNx1d(*args)

    def maxNx1d(self, *args):
        return numpylib.maxNx1d(*args)

    def array_mult(self, a1, a2):
        return a1 * a2

    def sum_labeled(self, src, labels, n):
        return ndimage.sum(src, labels, range(n))

    def __init__(self, raw_channel, mask_channel, image_meta, synapse_diam_micron, vicinity_diam_micron, maskblur_micron, desired_block_size=(1024,1024,1024)):

        assert raw_channel.shape == mask_channel.shape

        self.raw_channel = raw_channel
        self.mask_channel = mask_channel
        self.raw_shape = raw_channel.shape

        # may modify previous fields as side-effect...
        self.block_size, self.num_blocks = self.find_blocking(desired_block_size)
        
        self.image_meta = image_meta
        self.separated_kernels, self.kernels_3d, self.hollow_offset, self.hollow_scale = prepare_kernels(image_meta, synapse_diam_micron, vicinity_diam_micron, maskblur_micron)

        def kernel_radii(k):
            result = []
            for k1d in k:
                result.append( len(k1d)/2 )
            return result

        self.separated_kernel_radii = map(kernel_radii, self.separated_kernels)

        # self.separated_kernels contains
        # 0: synapse_kernels
        # 1: vicinity_kernels
        # 2: mask_kernels
        # 3: offset_kernels

        self.src_border_widths = list(self.separated_kernel_radii)

        # synapse_kernels convolution result feed into a regional box
        # filter with same kernel width so twice as much border is required
        self.src_border_widths[0] = tuple(map(lambda w: 2*w, self.src_border_widths[0]))

        # this is how much border we trim at image boundaries to find
        # valid results
        self.max_border_widths = map(max, *self.src_border_widths)

        print "Kernel radii %s implies max border width %s" % (
            self.separated_kernel_radii,
            self.max_border_widths
            )

        for d in range(self.raw_channel.ndim):
            if self.num_blocks[d] > 1:
                # block has border trimmed from one edge
                trim_factor = 1
            else:
                # single block has border trimmed from two edges
                trim_factor = 2
            if self.block_size[d] <= self.max_border_widths[d] * trim_factor:
                raise ValueError("Block size %s too small for filter borders %s" % (self.block_size, self.max_border_widths))

        self.dst_shape = tuple(
            [ 
                self.raw_shape[d] - 2 * self.max_border_widths[d]
                for d in range(self.raw_channel.ndim)
            ]
        )

        print "Using %s blocks of size %s to process %s into %s" % (
            self.num_blocks,
            self.block_size,
            self.raw_shape,
            self.dst_shape
            )

    def block_slice_src(self, kn, blockpos):
        """Return slice for source block and separated_kernel[kn].

           This slice is used to extract a source sub-array from the
           original input image channels.

        """
        def slice1d(d):
            if blockpos[d] == 0:
                lower = 0 + self.max_border_widths[d] - self.src_border_widths[kn][d]
            else:
                lower = self.block_size[d] * blockpos[d] - self.src_border_widths[kn][d]
        
            if blockpos[d] == self.num_blocks[d] - 1:
                upper = self.raw_shape[d] - self.max_border_widths[d] + self.src_border_widths[kn][d]
            else:
                upper = self.block_size[d] * (1 + blockpos[d]) + self.src_border_widths[kn][d]

            return slice(lower,upper)
        
        slc = tuple([ slice1d(d) for d in range(self.raw_channel.ndim)])
        #print "block_slice_src(%s,%s): %s" % (kn, blockpos, slc)
        return slc

    def block_slice_dst(self, blockpos):
        """Return slice for dest. block.

           This slice is used to store a destination sub-array into a
           global result image channel, if reassembling a full image.

        """
        def slice1d(d):
            # invalid border gets trimmed from first and last blocks
            if blockpos[d] == 0:
                lower = 0
            else:
                lower = self.block_size[d] * blockpos[d] - self.max_border_widths[d]
        
            if blockpos[d] == (self.num_blocks[d] - 1):
                upper = self.block_size[d] * (1 + blockpos[d]) - self.max_border_widths[d] * 2
            else:
                upper = self.block_size[d] * (1 + blockpos[d]) - self.max_border_widths[d]

            return slice(lower, upper)
        
        slc = tuple([ slice1d(d) for d in range(self.raw_channel.ndim)])
        #print "block_slice_dst(%s): %s" % (blockpos, slc)
        return tuple([ slice1d(d) for d in range(self.raw_channel.ndim)])

    def block_iter(self):
        def helper(counts):
            if counts[1:]:
                for block in range(counts[0]):
                    for position in helper(counts[1:]):
                        yield (block,) + position
            else:
                for block in range(counts[0]):
                    yield (block,)
        for position in helper(self.num_blocks):
            yield position

    def find_blocking(self, desired_block_size):
        """Return N-dimensional block_size, num_blocks plan for image.

           The N-dimensional desired_block_size is the prefered size,
           and a size smaller (but at least half desired size) or
           larger (but at most twice desired size) are considered
           until an evenly divisibe size is found.

           As a last-ditch effort, the source data may be trimmed by
           one pixel in each dimension to try to find a divisible
           size, in case there is no match.  In this case, the image
           channels and shape of the object are modified as
           side-effects.

           This may all fail, raising a ValueError if no match is
           possible.

        """
        def find_blocking_1d(d):
            if self.raw_shape[d] < desired_block_size[d]:
                return self.raw_shape[d], 1

            # prefer desired_block_size or something a bit smaller
            for w in xrange(desired_block_size[d], desired_block_size[d]/2, -1):
                if (self.raw_shape[d] % w) == 0:
                    return w, self.raw_shape[d] / w
            # also consider something larger
            for w in xrange(desired_block_size[d], desired_block_size[d]*2):
                if (self.raw_shape[d] % w) == 0:
                    return w, self.raw_shape[d] / w
            raise ValueError("No blocking found for image dimension %d, length %d, desired block size %d"
                             % (d, self.raw_shape[d], desired_block_size[d]))

        block_size = []
        num_blocks = []

        for d in range(self.raw_channel.ndim):
            try:
                w, n = find_blocking_1d(d)
            except ValueError:
                # try trimming one voxel and repeating
                print "WARNING: trimming image dimension %d to try to find divisible block size" % d
                trim_slice = tuple(
                    [ slice(None,None) for i in range(d) ]
                    + [ slice(0, self.raw_channel.shape[d] - 1) ]
                    + [ slice(None,None) for i in range(d+1, self.raw_channel.ndim) ]
                )
                self.raw_channel = self.raw_channel[trim_slice]
                self.mask_channel = self.mask_channel[trim_slice]
                self.raw_shape = list(self.raw_shape)
                self.raw_shape[d] -= 1
                self.raw_shape = tuple(self.raw_shape)
                
                w, n = find_blocking_1d(d)

            block_size.append(w)
            num_blocks.append(n)

        return tuple(block_size), tuple(num_blocks)
                        
    def volume_process(self):

        syn_channel = empty(self.dst_shape, dtype=self.raw_channel.dtype)
        pks_channel = empty(self.dst_shape, dtype=self.raw_channel.dtype)
        vcn_channel = empty(self.dst_shape, dtype=self.raw_channel.dtype)
        msk_channel = empty(self.dst_shape, dtype=self.mask_channel.dtype)

        noise = None
        
        for blockpos in self.block_iter():
            dslc = self.block_slice_dst(blockpos)
            syn_channel[dslc], pks_channel[dslc], vcn_channel[dslc], msk_channel[dslc], blocknoise \
                = self.block_process(blockpos)

            if noise is None:
                noise = blocknoise
            else:
                noise = min(noise, blocknoise)

        raw_channel = self.raw_channel[
            tuple([ 
                slice( self.max_border_widths[d], - self.max_border_widths[d]  )
                for d in range(self.raw_channel.ndim)
            ]
              )
        ]

        self.noise = noise
        print "estimating noise as %f" % self.noise

        return raw_channel, syn_channel, pks_channel, vcn_channel, msk_channel

    def block_process(self, blockpos):
        """Process block data to return convolved results.

           Parameters:

              blockpos: N-dimensional block numbers

           Result is a K-tuple:

              (synapse, peaks, hollow, mask, noise)

           where all fields are Numpy arrays with the same shape and
           different result fields:

              synapse: convolution measuring synapse core intensity
              peaks: synapse core values only at local maxima
              hollow: convolution measuring synapse background intensity
              mask: convolution giving low-pass filtered red channel

           For edge blocks, the shape is smaller due to invalid border
           regions being trimmed.

        """
        syn_channel = self.convNx1d(
            self.raw_channel[self.block_slice_src(0, blockpos)], 
            self.separated_kernels[0]
        )
        
        max_channel = self.maxNx1d(
            syn_channel, 
            tuple([ len(k) for k in self.separated_kernels[0] ])
            )

        # need to trim border meant for max_channel computation
        # so we can use this below
        syn_channel = syn_channel[
            tuple([
                slice( self.separated_kernel_radii[0][d], - self.separated_kernel_radii[0][d] )
                for d in range(self.raw_channel.ndim)
            ]
              )
        ]
    
        vcn_channel = self.convNx1d(
            self.raw_channel[self.block_slice_src(1, blockpos)], 
            self.separated_kernels[1]
        )
        msk_channel = self.convNx1d(
            self.mask_channel[self.block_slice_src(2, blockpos)], 
            self.separated_kernels[2]
        )
        k_channel = self.convNx1d(
            self.raw_channel[self.block_slice_src(3, blockpos)], 
            self.separated_kernels[3]
        )

        noise = vcn_channel.min()

        # compute hollow vicinity convolution using separated convolutions
        #  ((B - A + k)/s)*I 
        #  = ((B - A + k)*I)/s
        #  = (B*I - A*I + k*I)/s
        vcn_hollow = (
            vcn_channel
            - syn_channel
            + k_channel
        ) / self.hollow_scale
        vcn_channel = None
        k_channel = None

        # find syn cores via local maxima test
        pks_channel = syn_channel * (syn_channel == max_channel)
        max_channel = None

        result = syn_channel, pks_channel, vcn_hollow, msk_channel, noise
    
        print "Calculated block shapes: %s" % map(lambda a: a.shape, result)
        return result

    def get_peaks(self, synapse, hollow, syn_lvl, vcn_lvl):
        peaks = (synapse > syn_lvl)# & (synapse > hollow)
        if vcn_lvl is not None:
            return peaks & (hollow < vcn_lvl)
        else:
            return peaks

    def analyze(self, synapse, peaks, hollow, syn_lvl=None, vcn_lvl=None):
        """Analyze synapse features and return measurements.

           Parameters:

              synapse: from block_process results
              peaks: from block_process results
              hollow: from block_process results
              syn_lvl: minimum synapse core measurement accepted
              vcn_lvl: maximum synapse background measurement accepted

           Returns 4-tuple:

              (syn_values, vcn_values, centroids, widths)

           all of which have same length N for N synapses
           accepted. The first two contain the measured core and
           background levels while centroids contains the (Z, Y, X)
           voxel coordinates of the detected synapse center, where (0,
           0, 0) is the least corner voxel of syn_channel.  Widths are
           (d, h, w) full-width-half-maximum spans of the feature on
           the corresponding (Z, Y, X) axes.

        """
        if syn_lvl is None:
            syn_lvl = 0

        t0 = datetime.datetime.now()
        peaksf = self.get_peaks(peaks, hollow, syn_lvl, vcn_lvl)

        t1 = datetime.datetime.now()
        label_im, nb_labels = ndimage.label(peaksf)
        peaksf = None

        print "found %d centroids" % nb_labels

        t2 = datetime.datetime.now()
        sizes = self.sum_labeled(
            label_im > 0,
            label_im,
            nb_labels + 1
        )[1:]

        try:
            print "centroid sizes: %s, %s, %s" % (
                sizes.min(),
                sizes.mean(),
                sizes.max()
            )
        except:
            pass

        t3 = datetime.datetime.now()
        sums = self.sum_labeled(
            peaks,
            label_im,
            nb_labels + 1
        )[1:]

        syn_vals = sums / sizes

        t4 = datetime.datetime.now()
        sums = self.sum_labeled(
            hollow,
            label_im,
            nb_labels + 1
        )[1:]

        vcn_vals = sums / sizes

        try:
            print "centroid values: %s, %s, %s" % (
                syn_vals.min(),
                syn_vals.mean(),
                syn_vals.max()
            )

            print "centroid background: %s, %s, %s" % (
                vcn_vals.min(),
                vcn_vals.mean(),
                vcn_vals.max()
            )
        except:
            pass    

        centroid_components = [ ]

        t5 = datetime.datetime.now()
        for d in range(3):
            coords = self.array_mult(
                array(
                    range(0, peaks.shape[d]) 
                ).astype('float32')[
                    [ None for i in range(d) ]  # add dims before axis
                    + [ slice(None) ]              # get axis
                    + [ None for i in range(peaks.ndim - 1 - d) ] # add dims after axis
                ],
                ones(peaks.shape, 'float32') # broadcast to full volume
            )

            centroid_components.append(
                (self.sum_labeled(
                    coords,
                    label_im,
                    nb_labels + 1
                    )[1:] / sizes).astype(np.int)
                )

        centroids = zip(*centroid_components)
        t6 = datetime.datetime.now()

        centroid_widths = []

        for i in range(len(syn_vals)):
            centroid = centroids[i]
            # use synapse core value as proxy for maximum
            # since we did peak detection
            hm = (syn_vals[i] - self.noise) / 2 + self.noise
            widths = []

            assert hm >= 0

            def slice_d(d, pos):
                return tuple(
                    [ centroid[a] for a in range(d) ]
                    + [ pos ]
                    + [ centroid[a] for a in range(d+1, 3) ]
                )

            def interp_d(d, p0, p1, v):
                v0 = synapse[slice_d(d, p0)]
                if p1 >= 0 and p1 < synapse.shape[d]:
                    v1 = synapse[slice_d(d, p1)]
                else:
                    v1 = v0

                if v0 < v and v < v1 \
                   or v0 > v and v > v1:
                    return float(p0) + (v - v0) / (v1 - v0)
                else:
                    return p0

            for d in range(3):
                # scan from center along axes in negative and positive 
                # directions until half-maximum is found
                for pos in range(centroid[d], -1, -1):
                    lower = pos
                    if synapse[ slice_d(d, pos) ] < hm:
                        break

                # interpolate to find hm sub-pixel position
                lower = interp_d(d, lower, lower+1, hm)

                for pos in range(centroid[d], synapse.shape[d]):
                    upper = pos
                    if synapse[slice_d(d, pos)] < hm:
                        break

                # interpolate to find hm sub-pixel position
                upper = interp_d(d, upper, upper-1, hm)

                # accumulate N-d measurement for centroid
                widths.append( (upper - lower) * [
                    self.image_meta.z_microns,
                    self.image_meta.y_microns,
                    self.image_meta.x_microns
                ][d]
                )

            # accumulate measurements for all centroids
            centroid_widths.append( tuple(widths) )

        t7 = datetime.datetime.now()

        try:
            print "centroids:", centroids[0:10], "...", centroids[-1]
        except:
            pass

        print "\nanalyze splits: %s" % map(lambda p: (p[1]-p[0]).total_seconds(), [ (t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t5, t6), (t6, t7) ])

        return syn_vals, vcn_vals, centroids, centroid_widths

BlockedAnalyzerOpt = BlockedAnalyzer
assign_voxels_opt = numpylib.assign_voxels

try:       
    import nexpr as numerexprlib
    class BlockedAnalyzerNumerexpr (BlockedAnalyzer):

        def convNx1d(self, *args):
            return numerexprlib.convNx1d(*args)

        def array_mult(self, a1, a2):
            return numerexprlib.array_mult(a1, a2)

        def get_peaks(self, synapse, hollow, syn_lvl, vcn_lvl):
            expr = "(synapse > syn_lvl)"
            #expr += " & (synapse > hollow)"
            if vcn_lvl is not None:
                expr += " & (hollow < vcn_lvl)"

            return numerexprlib.neval(expr)

    BlockedAnalyzerOpt = BlockedAnalyzerNumerexpr
except:
    pass

try:
    import ocl as opencllib
    class BlockedAnalyzerOpenCL (BlockedAnalyzerOpt):

        def convNx1d(self, *args):
            return opencllib.convNx1d(*args)

        def maxNx1d(self, *args):
            return opencllib.maxNx1d(*args)

        def sum_labeled(self, src, labels, n):
            return opencllib.sum_labeled(src, labels, n)

        def block_process(self, blockpos):
            """Process block data to return convolved results.

               Overrides parent implementation to keep more work on
               OpenCL device...

            """
            syn_channel = opencllib.convNx1d(
                self.raw_channel[self.block_slice_src(0, blockpos)], 
                self.separated_kernels[0]
            )
        
            clq = opencllib.cl.CommandQueue( opencllib.ctx )

            max_channel_dev = opencllib.maxNx1d(
                syn_channel, 
                tuple([ len(k) for k in self.separated_kernels[0] ]),
                clq=clq
            )

            # need to trim border meant for max_channel computation
            # so we can use this below
            syn_channel = syn_channel[
                tuple([
                    slice( self.separated_kernel_radii[0][d], - self.separated_kernel_radii[0][d] )
                    for d in range(self.raw_channel.ndim)
                ]
                  )
            ]
            syn_channel = syn_channel.astype(opencllib.float32, copy=True)
            syn_channel_dev = opencllib.cl_array.to_device(clq, syn_channel)
    
            vcn_channel_dev = opencllib.convNx1d(
                self.raw_channel[self.block_slice_src(1, blockpos)], 
                self.separated_kernels[1],
                clq=clq
            )
            msk_channel = opencllib.convNx1d(
                self.mask_channel[self.block_slice_src(2, blockpos)], 
                self.separated_kernels[2]
            )
            k_channel_dev = opencllib.convNx1d(
                self.raw_channel[self.block_slice_src(3, blockpos)], 
                self.separated_kernels[3],
                clq=clq
            )

            # compute hollow vicinity convolution using separated convolutions
            #  ((B - A + k)/s)*I 
            #  = ((B - A + k)*I)/s
            #  = (B*I - A*I + k*I)/s
            vcn_hollow_dev = (
                vcn_channel_dev
                - syn_channel_dev
                + k_channel_dev
            ) / self.hollow_scale

            noise = vcn_channel_dev.map_to_host(clq).min()
            vcn_channel_dev = None
            k_channel_dev = None

            # find syn cores via local maxima test
            pks_channel_dev = syn_channel_dev * (syn_channel_dev == max_channel_dev)
            max_channel_dev = None

            syn_channel = syn_channel_dev.map_to_host(clq)
            pks_channel = pks_channel_dev.map_to_host(clq)
            vcn_hollow = vcn_hollow_dev.map_to_host(clq)
            clq.finish()

            result = syn_channel, pks_channel, vcn_hollow, msk_channel, noise
            return result

    BlockedAnalyzerOpt = BlockedAnalyzerOpenCL
    #assign_voxels_opt = opencllib.assign_voxels
except:
    pass


## TEST HARNESS
## read parameter CSV format from standard-input and process each record as separate file+params

def batch_stdin(engine=None):

    param_reader = csv.DictReader(sys.stdin)

    for params in param_reader:
        params['engine'] = engine

        pass


