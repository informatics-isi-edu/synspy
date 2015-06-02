
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

import np as numpylib

import datetime
import sys
import random
import numpy as np
from numpy import array, float32, int32, empty, newaxis, dot, cross, zeros, ones
from numpy.linalg import norm
import scipy
from scipy import ndimage
import json
import math
import re
import csv
from volspy.util import bin_reduce

def Gsigma(sigma):
    """Pickle a gaussian function G(x) for given sigma"""
    def G(x):
        return (math.e ** (-(x**2)/(2*sigma**2)))/(2 * math.pi* sigma**2)**0.5
    return G

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


def prepare_kernels(gridsize, synapse_diam_microns, vicinity_diam_microns, redblur_microns):
    """Prepare synapse-detection convolution kernels.

       Parameters:

         gridsize: the micron step size of the image in (Z, Y, X) axes
         synapse_diam_microns: core synapse feature span
         vicinity_diam_microns: synapse local background span
         redblur_microns: auto-fluourescence blurring span

       All span arguments are 3-tuples of micron lengths of the
       standard-deviation of the related Gaussian distribution in each
       image dimension (Z, Y, X).

       Result is a 2-tuple:

         ( kernels_3x1d, kernels_3d ).

       The kernels_3x1d result is a 2-tuple:

         ( low_3x1d, span_3x1d )

       where each kernel is a 3-tuple of numpy arrays, each array
       being weights of a 1D kernel for each image dimension (Z, Y,
       X).  The low_3x1d kernel is float weights summing to 1.0 while
       the span_3x1d kernel is a binary mask.

       The kernels_3d result is a 3-tuple

         ( core_3d, hollow_3d, red_3d )

       where each field is a numpy array, each array being weights of
       a 3D kernel. The kernels are float weights summing to 1.0.

    """
    # these are separated 1d gaussian kernels
    low_kernels = map(lambda d, s: gaussian_kernel(0.5*d/s/6.), vicinity_diam_microns, gridsize)
    span_kernels = map(lambda d, s: (1,) * (2*(int(d/s)/2)+1), vicinity_diam_microns, gridsize)

    # TODO: investigate variants?
    #  adjust diameter by a fudge factor?
    #  splat an elliptic core instead of a box?
    core_kernel = ones(tuple(map(lambda d, s: 2*(int(0.8*d/s)/2)+1, synapse_diam_microns, gridsize)), dtype=float32)
    span_kernel = ones(tuple(map(lambda d, s: 2*(int(1.0*d/s)/2)+1, vicinity_diam_microns, gridsize)), dtype=float32)
    hollow_kernel = span_kernel - pad_centered(core_kernel, span_kernel.shape)

    core_kernel /= core_kernel.sum()
    hollow_kernel /= hollow_kernel.sum()
    
    red_kernel = compose_3d_kernel(
        map(lambda d, s: gaussian_kernel(d/s/6.), redblur_microns, gridsize)
    )

    return (
        (low_kernels, span_kernels),
        (core_kernel, hollow_kernel, red_kernel)
        )

class BlockedAnalyzer (object):
    """Analyze image using block decomposition for scalability.

       Conceptually we perform the sequence:

          prepare_kernels  (cheap, one-shot)
          process_volume   (expensive, scales with image size)
          analyze          (expensive, scales with image size)
          
       This can be decomposed into blocks to operate on 
       spatial sub-problems:

          process_volume_block  (totally independent)
          analyze_block (might be independent in practice?)

    """

    def convNx1d(self, *args):
        return numpylib.convNx1d(*args)

    def convNd_sparse(self, *args):
        return numpylib.convNd_sparse(*args)

    def maxNx1d(self, *args):
        return numpylib.maxNx1d(*args)

    def array_mult(self, a1, a2):
        return a1 * a2

    def sum_labeled(self, src, labels, n):
        return ndimage.sum(src, labels, range(n))

    def __init__(self, image, synapse_diam_micron, vicinity_diam_micron, maskblur_micron, desired_block_size=(384,384,450)):

        self.image = image

        self.view_reduction = tuple(map(lambda vs, ps: max(int(ps/vs), 1), self.image.micron_spacing, (0.16, 0.15, 0.15)))
        self.kernels_3x1d, self.kernels_3d = prepare_kernels(image.micron_spacing, synapse_diam_micron, vicinity_diam_micron, maskblur_micron)
        
        self.kernel_radii = [
            tuple(len(k1d)/2 for k1d in k3x1d)
            for k3x1d in self.kernels_3x1d
        ] + [
            tuple(map(lambda d: d/2, k3d.shape))
            for k3d in self.kernels_3d
        ]
        
        self.src_border_widths = list(self.kernel_radii)

        # low_kernel convolution result feeds into a regional box
        # filter with same kernel width so twice as much border is required
        self.src_border_widths[0] = tuple(map(lambda w: 2*w, self.src_border_widths[0]))

        # this is how much border we trim at image boundaries to find
        # valid results, rounding up to multiple of reduction step size
        # and adding a bit more padding to locate centroids near block boundaries
        self.max_border_widths = tuple(map(
            lambda w, r: ((w%r) and (w+r-w%r) or w) + (r>=3 and r or 2*r),
            map(max, *self.src_border_widths),
            self.view_reduction
        ))

        print "Kernel radii %s implies max border width %s" % (
            self.kernel_radii,
            self.max_border_widths
            )

        self.block_size, self.num_blocks = self.find_blocking(desired_block_size)
        
        for d in range(3):
            if self.num_blocks[d] > 1:
                # block has border trimmed from one edge
                trim_factor = 1
            else:
                # single block has border trimmed from two edges
                trim_factor = 2
            if self.block_size[d] <= self.max_border_widths[d] * trim_factor:
                raise ValueError("Block size %s too small for filter borders %s" % (self.block_size, self.max_border_widths))

        self.dst_shape = tuple(
            self.image.shape[d] - 2 * self.max_border_widths[d]
            for d in range(3)
        )

        print "Using %s blocks of size %s to process %s into %s" % (
            self.num_blocks,
            self.block_size,
            self.image.shape,
            self.dst_shape
            )

    def block_slice_src(self, blockpos):
        """Return slice for source block.

           This slice is used to extract a source sub-array from the
           original input image channels.

        """
        def slice1d(d):
            if blockpos[d] == 0:
                lower = 0
            else:
                lower = self.block_size[d] * blockpos[d] - self.max_border_widths[d]
        
            if blockpos[d] == self.num_blocks[d] - 1:
                upper = self.image.shape[d]
            else:
                upper = self.block_size[d] * (1 + blockpos[d]) + self.max_border_widths[d]

            return slice(lower,upper)
        
        slc = tuple([ slice1d(d) for d in range(3)] + [slice(None)])
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
        
        slc = tuple([ slice1d(d) for d in range(3)] + [slice(None)])
        return slc

    def block_slice_viewdst(self, blockpos):
        """Return slice for view_image dest. block.

           This slice is used to store a destination sub-array into a
           global result image channel, if reassembling a full image.

        """
        def slice1d(d):
            # invalid border gets trimmed from first and last blocks
            if blockpos[d] == 0:
                lower = self.max_border_widths[d]
            else:
                lower = self.block_size[d] * blockpos[d]
        
            if blockpos[d] == (self.num_blocks[d] - 1):
                upper = self.block_size[d] * (1 + blockpos[d]) - self.max_border_widths[d]
            else:
                upper = self.block_size[d] * (1 + blockpos[d])

            assert lower % self.view_reduction[d] == 0
            assert upper % self.view_reduction[d] == 0
            return slice(lower/self.view_reduction[d], upper/self.view_reduction[d])
       
        slc = tuple([ slice1d(d) for d in range(3)] + [slice(None)])
        return slc

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
            if self.image.shape[d] < desired_block_size[d]:
                if self.image.shape[d] % self.view_reduction[d] == 0:
                    return self.image.shape[d], 1
                else:
                    raise ValueError("Dimension %d, length %d, smaller than desired block size %d but not divisible by reduction %d" % (d, self.image.shape[d], desired_block_size[d], self.view_reduction[d]))

            # prefer desired_block_size or something a bit smaller
            for w in xrange(desired_block_size[d], max(desired_block_size[d]/2, 2*self.max_border_widths[d]), -1):
                if (self.image.shape[d] % w) == 0 and (w % self.view_reduction[d]) == 0:
                    return w, self.image.shape[d] / w
            # also consider something larger
            for w in xrange(max(desired_block_size[d], 2*self.max_border_widths[d]), desired_block_size[d]*2):
                if (self.image.shape[d] % w) == 0 and (w % self.view_reduction[d]) == 0:
                    return w, self.image.shape[d] / w
            raise ValueError("No blocking found for image dimension %d, length %d, desired block size %d, reduction %d"
                             % (d, self.image.shape[d], desired_block_size[d], self.view_reduction[d]))

        block_size = []
        num_blocks = []

        for d in range(3):
            try:
                w, n = find_blocking_1d(d)
            except ValueError:
                # try trimming one voxel and repeating
                print "WARNING: trimming image dimension %d to try to find divisible block size" % d
                axis_size = self.view_reduction[d]*(desired_block_size[d]/self.view_reduction[d])
                trimmed_shape = axis_size*(self.image.shape[d]/axis_size)
                trim_slice = tuple(
                    [ slice(None) for i in range(d) ]
                    + [ slice(0, trimmed_shape) ]
                    + [ slice(None) for i in range(d+1, self.image.ndim) ]
                )
                self.image = self.image.lazyget(trim_slice)
                w, n = find_blocking_1d(d)

            block_size.append(w)
            num_blocks.append(n)

        return tuple(block_size), tuple(num_blocks)
                        
    def volume_process(self):
        view_image = zeros(tuple(
            map(lambda w, r: w/r, self.image.shape[0:3], self.view_reduction)
            + [self.image.shape[-1]]
        ))

        print "Allocated %s view_image with %s voxel size for %s reduction of %s source image with %s voxel size." % (view_image.shape, map(lambda a, b: a*b, self.image.micron_spacing, self.view_reduction), self.view_reduction, self.image.shape, self.image.micron_spacing)

        centroids = []
        centroid_measures = []
        perf_vector = None

        total_blocks = reduce(lambda a, b: a*b, self.num_blocks, 1)
        done_blocks = 0
        last_progress = 0

        sys.stderr.write("Progress processing %d blocks:\n" % total_blocks)
        
        for blockpos in self.block_iter():
            view, cent, meas, perf = self.block_process(blockpos)
            view_image[self.block_slice_viewdst(blockpos)] = view
            centroids.extend(cent)
            centroid_measures.extend(meas)
            if perf_vector is None:
                perf_vector = perf
            else:
                perf_vector = map(lambda a, b: (a[0]+b[0], a[1]), perf_vector, perf)
            done_blocks += 1
            progress = int(100 * done_blocks / total_blocks)
            for i in range(last_progress, progress, 2):
                sys.stderr.write('%x' % (i/10))
            last_progress = progress
        sys.stderr.write(' DONE.\n')
                
        total = 0.
        for elapsed, desc in perf_vector:
            total += elapsed
            print "%8.2fs %s task time" % (elapsed, desc)
        print "%8.2fs TOTAL processing time" % total

        print "Found %d centroids" % len(centroids)
            
        return view_image, centroids, centroid_measures

    def block_process(self, blockpos):
        """Process block data to return convolved results.

           Parameters:

              blockpos: N-dimensional block numbers

           Result is a 3-tuple:

              (view_image, centroids, centroid_measures, perf_vector)

        """
        splits = [(datetime.datetime.now(), None)]
        
        image = self.image[self.block_slice_src(blockpos)]
        splits.append((datetime.datetime.now(), 'image load'))

        low_channel = self.convNx1d(image[:,:,:,0], self.kernels_3x1d[0])
        splits.append((datetime.datetime.now(), 'image*low'))

        max_channel = self.maxNx1d(
            low_channel, 
            tuple([ len(k) for k in self.kernels_3x1d[0] ])
        )
        splits.append((datetime.datetime.now(), 'local maxima'))

        # need to trim borders discarded by max_channel computation
        low_channel = low_channel[
            tuple(slice(k/2, -k/2) for k in map(lambda a, b: b-a, max_channel.shape, low_channel.shape))
        ]

        # find syn cores via local maxima test
        assert low_channel.shape == max_channel.shape
        peaks = low_channel > (max_channel * 0.9999)

        clipbox = tuple(
            slice(peaks_border, peaks_width-peaks_border)
            for peaks_width, peaks_border in map(
                    lambda iw, bw, pw: (pw, bw - (iw-pw)/2),
                    image.shape[0:3],
                    self.max_border_widths,
                    peaks.shape
                    )
        )
        splits.append((datetime.datetime.now(), 'mask peaks'))

        label_im, nb_labels = ndimage.label(peaks)
        splits.append((datetime.datetime.now(), 'label peaks'))
        
        sizes = self.sum_labeled(
            label_im > 0,
            label_im,
            nb_labels + 1
        )[1:]
        splits.append((datetime.datetime.now(), 'centroid sizes'))

        centroid_components = [ ]

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
                    )[1:] / sizes)#.astype(np.int32)
                )

        # centroids are in block peaks grid
        centroids = zip(*centroid_components)

        if centroids:
            # discard centroids outside clipbox (we searched slightly
            # larger to handle peaks at edges
            filtered_centroids = []
            for i in range(len(centroids)):
                clip = False
                for d in range(3):
                    if int(centroids[i][d]) < clipbox[d].start or int(centroids[i][d]) >= clipbox[d].stop:
                        clip = True
                if not clip:
                    filtered_centroids.append(centroids[i])

            # centroids are in block core grid
            centroids = array(filtered_centroids, int32) - array([slc.start for slc in clipbox], int32)
            # image_centroids are in block image grid
            image_centroids = centroids + array(self.max_border_widths, int32)
            # global_centroids are in self.image grid
            global_centroids = (
                array([slc.start or 0 for slc in self.block_slice_src(blockpos)[0:3]], int32)
                + image_centroids
            )

        else:
            image_centroids = []
            global_centroids = []
            
        splits.append((datetime.datetime.now(), 'centroid coords'))

        centroid_measures = [self.convNd_sparse(image[:,:,:,0], self.kernels_3d[0], image_centroids)]
        splits.append((datetime.datetime.now(), 'centroid corevals'))

        centroid_measures.append(self.convNd_sparse(image[:,:,:,0], self.kernels_3d[1], image_centroids))
        splits.append((datetime.datetime.now(), 'centroid hollowvals'))

        if image.shape[3] > 1:
            centroid_measures.append(self.convNd_sparse(image[:,:,:,1], self.kernels_3d[2], image_centroids))
            splits.append((datetime.datetime.now(), 'centroid redvals'))

        centroid_measures = zip(*centroid_measures)
        splits.append((datetime.datetime.now(), 'zip centroid measures'))

        perf_vector = map(lambda t0, t1: ((t1[0]-t0[0]).total_seconds(), t1[1]), splits[0:-1], splits[1:])

        view_image = image[tuple(
            slice(self.max_border_widths[d], -self.max_border_widths[d])
            for d in range(3)
        ) + (slice(None),)]
        view_image = bin_reduce(view_image, self.view_reduction + (1,))
        splits.append((datetime.datetime.now(), 'image reduce'))

        return view_image, global_centroids, centroid_measures, perf_vector

    def fwhm_estimate(self, synapse, centroids, syn_vals, vcn_vals, noise):
        """Estimate FWHM measures for synapse candidates."""
        centroid_widths = []
        for i in range(len(syn_vals)):
            centroid = centroids[i]
            # use synapse core value as proxy for maximum
            # since we did peak detection

            # treat vicinity measure as another local background estimate
            # and give it a fudge-factor
            floor_value = max(vcn_vals[i] * 1.5, noise)
            fm = max(syn_vals[i] - floor_value, 0)
            hm = fm / 2 + floor_value

            widths = []

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
                    if synapse[slice_d(d, pos)] <= hm:
                        break

                # interpolate to find hm sub-pixel position
                lower = interp_d(d, lower, lower+1, hm)

                for pos in range(centroid[d], synapse.shape[d]):
                    upper = pos
                    if synapse[slice_d(d, pos)] <= hm:
                        break

                # interpolate to find hm sub-pixel position
                upper = interp_d(d, upper, upper-1, hm)

                # accumulate N-d measurement for centroid
                widths.append( 
                    (upper - lower) * [
                        self.image_meta.z_microns,
                        self.image_meta.y_microns,
                        self.image_meta.x_microns
                    ][d]
                )

            # accumulate measurements for all centroids
            centroid_widths.append( tuple(widths) )

        return centroid_widths

BlockedAnalyzerOpt = BlockedAnalyzer
assign_voxels_opt = numpylib.assign_voxels

try:
    import nexpr as numerexprlib
    class BlockedAnalyzerNumerexpr (BlockedAnalyzer):

        def convNx1d(self, *args):
            return numerexprlib.convNx1d(*args)

        def array_mult(self, a1, a2):
            return numerexprlib.array_mult(a1, a2)

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

        def fwhm_estimate(self, synapse, centroids, syn_vals, vcn_vals, noise):
            return opencllib.fwhm_estimate(
                synapse, centroids, syn_vals, vcn_vals, noise,
                (self.image_meta.z_microns, self.image_meta.y_microns, self.image_meta.x_microns)
            )

        # This is actually slower due to data input bottleneck!
        #def convNd_sparse(self, data, kernel, centroids):
        #    return opencllib.weighted_measure(data, centroids, kernel)
        
    BlockedAnalyzerOpt = BlockedAnalyzerOpenCL
    assign_voxels_opt = opencllib.assign_voxels
except:
    pass


## TEST HARNESS
## read parameter CSV format from standard-input and process each record as separate file+params

def batch_stdin(engine=None):

    param_reader = csv.DictReader(sys.stdin)

    for params in param_reader:
        params['engine'] = engine

        pass


