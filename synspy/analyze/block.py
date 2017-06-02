
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from . import np as numpylib
import re
import datetime
import sys
import os
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
from volspy.util import bin_reduce, load_and_mangle_image

from .util import *
from functools import reduce

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
        return ndimage.sum(src, labels, list(range(n)))

    def __init__(self, image, synapse_diam_micron, vicinity_diam_micron, maskblur_micron, view_reduction, desired_block_size=None):
        if desired_block_size is None:
            try:
                desired_block_size = tuple(map(int, os.getenv('ZYX_BLOCK_SIZE').split(",")))
                assert len(desired_block_size) == 3
            except:
                desired_block_size = (384,384,450)
            print("Using %s voxel preferred sub-block size. Override with ZYX_BLOCK_SIZE='int,int,int'" % (desired_block_size,))

        view_mode = os.getenv('VIEW_MODE', 'raw')
        if view_mode.lower() not in ['raw', 'dog']:
            raise ValueError('Unknown VIEW_MODE "%s"' % view_mode)
        self.view_raw = view_mode.lower() == 'raw'
        print("Using %s viewing mode. Override with VIEW_MODE=raw or VIEW_MODE=dog." % (self.view_raw and 'raw' or 'dog'))
        
        self.image = image
        self.view_reduction = view_reduction
        
        self.kernels_3x1d, self.kernels_3d = prepare_kernels(image.micron_spacing, synapse_diam_micron, vicinity_diam_micron, maskblur_micron)

        # maximum dependency chain of filters trims this much invalid border data
        self.max_border_widths = (
            # DoG is largest separated filter
            radii_3x1d(self.kernels_3x1d[3])
            # sparse measures consume DoG output
            + reduce(
                lambda a, b: np.maximum(a, b),
                [radii_3d(k) for k in self.kernels_3d]
            )
            # add some padding for peak detection at block borders
            + radii_3x1d(self.kernels_3x1d[3])
        )

        # round up to multiple of reduction size
        self.max_border_widths += np.where(
            (self.max_border_widths % np.array(self.view_reduction, dtype=np.int32)),
            np.array(self.view_reduction, dtype=np.int32) - self.max_border_widths % np.array(self.view_reduction, dtype=np.int32),
            np.zeros((3,), dtype=np.int32)
        )

        print("Kernel radii %s, %s implies max-border %s" % (
            [tuple(radii_3x1d(k)) for k in self.kernels_3x1d],
            [tuple(radii_3d(k)) for k in self.kernels_3d],
            self.max_border_widths
        ))
        
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

        print("Using %s blocks of size %s to process %s into %s" % (
            self.num_blocks,
            self.block_size,
            self.image.shape,
            self.dst_shape
            ))

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
            for w in range(desired_block_size[d], max(desired_block_size[d]/2, 2*self.max_border_widths[d]), -1):
                if (self.image.shape[d] % w) == 0 and (w % self.view_reduction[d]) == 0:
                    return w, self.image.shape[d] / w
            # also consider something larger
            for w in range(max(desired_block_size[d], 2*self.max_border_widths[d]), desired_block_size[d]*2):
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
                print("WARNING: trimming image dimension %d to try to find divisible block size" % d)
                axis_size = self.view_reduction[d]*(min(desired_block_size[d], self.image.shape[d])/self.view_reduction[d])
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
            list(map(lambda w, r: w/r, self.image.shape[0:3], self.view_reduction))
            + [self.image.shape[-1]]
        ), dtype=np.float32)

        print("Allocated %s %s view_image with %s voxel size for %s reduction of %s source image with %s voxel size." % (view_image.shape, view_image.dtype, list(map(lambda a, b: a*b, self.image.micron_spacing, self.view_reduction)), self.view_reduction, self.image.shape, self.image.micron_spacing))

        centroids = None
        centroid_measures = None
        perf_vector = None

        total_blocks = reduce(lambda a, b: a*b, self.num_blocks, 1)
        done_blocks = 0
        last_progress = 0

        sys.stderr.write("Progress processing %d blocks:\n" % total_blocks)
        
        for blockpos in self.block_iter():
            view, cent, meas, perf = self.block_process(blockpos)
            view_image[self.block_slice_viewdst(blockpos)] = view
            if centroids is None:
                centroids = cent
                centroid_measures = meas
                perf_vector = perf
            else:
                centroids = np.concatenate((centroids, cent))
                centroid_measures = np.concatenate((centroid_measures, meas))
                perf_vector = list(map(lambda a, b: (a[0]+b[0], a[1]), perf_vector, perf))
            done_blocks += 1
            progress = int(100 * done_blocks / total_blocks)
            for i in range(last_progress, progress, 2):
                sys.stderr.write('%x' % (i/10))
            last_progress = progress
        sys.stderr.write(' DONE.\n')

        #view_image -= view_image.min()
        #view_image *= self.image.max() / view_image.max()
        
        total = 0.
        for elapsed, desc in perf_vector:
            total += elapsed
            print("%8.2fs %s task time" % (elapsed, desc))
        print("%8.2fs TOTAL processing time" % total)

        print("Found %d centroids" % len(centroids))
            
        return view_image, centroids, centroid_measures

    def block_process(self, blockpos):
        """Process block data to return convolved results.

           Parameters:

              blockpos: N-dimensional block numbers

           Result is a 3-tuple:

              (view_image, centroids, centroid_measures, perf_vector)

        """
        splits = [(datetime.datetime.now(), None)]
        
        image = self.image[self.block_slice_src(blockpos)].astype(np.float32, copy=False)
        splits.append((datetime.datetime.now(), 'image load'))

        low_channel = self.convNx1d(image[:,:,:,0], self.kernels_3x1d[0])
        splits.append((datetime.datetime.now(), 'image*low'))

        scale1_channel = self.convNx1d(image[:,:,:,0], self.kernels_3x1d[2])
        splits.append((datetime.datetime.now(), 'image*syn'))
        
        scale2_channel = self.convNx1d(image[:,:,:,0], self.kernels_3x1d[3])
        dog = crop_centered(scale1_channel, scale2_channel.shape) - scale2_channel
        splits.append((datetime.datetime.now(), 'image*vlow'))

        # allow tinkering w/ multiple peak detection fields
        max_inputs = [
            low_channel,
            # dog,
        ]

        if len(max_inputs) > 1:
            crop_shape = list(map(min, *[img.shape for img in max_inputs]))
        else:
            crop_shape = max_inputs[0].shape

        max_inputs = [crop_centered(img, crop_shape) for img in max_inputs]

        if self.view_raw:
            view_image = crop_centered(
                image,
                list(map(lambda w, b: w-2*b, image.shape[0:3], self.max_border_widths)) + [image.shape[3]]
            )
        else:
            # caller expects view_image to have same number of channels as raw image
            view_image = zeros(
                tuple(list(map(lambda w, b: w-2*b, image.shape[0:3], self.max_border_widths)) + [image.shape[3]]),
                dtype=dog.dtype
            )
            view_image[:,:,:,0] = crop_centered(
                dog,
                list(map(lambda w, b: w-2*b, image.shape[0:3], self.max_border_widths))
            )
            splits.append((datetime.datetime.now(), 'view image DoG'))

        view_image = bin_reduce(view_image, self.view_reduction + (1,))
        splits.append((datetime.datetime.now(), 'view image reduce'))

        max_kernel = self.kernels_3d[3].shape
        max_channels = [self.maxNx1d(img, max_kernel) for img in max_inputs]
        splits.append((datetime.datetime.now(), 'local maxima'))
            
        # need to trim borders discarded by max_channel computation
        max_inputs = [crop_centered(img, max_channels[0].shape) for img in max_inputs]

        # find syn cores via local maxima test
        peaks = np.zeros(max_channels[0].shape, dtype=np.bool)
        for i in range(len(max_inputs)):
            assert max_inputs[i].shape == max_channels[i].shape
            peaks += max_inputs[i] >= (max_channels[i])

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
                    list(range(0, peaks.shape[d])) 
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
        centroids = list(zip(*centroid_components))

        filtered_centroids = []
        if centroids:
            # discard centroids outside clipbox (we searched slightly
            # larger to handle peaks at edges
            for i in range(len(centroids)):
                clip = False
                for d in range(3):
                    if int(centroids[i][d]) < clipbox[d].start or int(centroids[i][d]) >= clipbox[d].stop:
                        clip = True
                if not clip:
                    filtered_centroids.append(centroids[i])

        if filtered_centroids:
            # centroids are in block core grid
            centroids = array(filtered_centroids, int32) - array([slc.start for slc in clipbox], int32)
            # image_centroids are in block image grid
            image_centroids = centroids + array(self.max_border_widths, int32)
            # dog_centroids are in difference-of-gaussians grid
            dog_centroids = centroids + array(list(map(lambda iw, dw: (iw-dw)/2, image.shape[0:3], dog.shape)))
            # global_centroids are in self.image grid
            global_centroids = (
                array([slc.start or 0 for slc in self.block_slice_src(blockpos)[0:3]], int32)
                + image_centroids
            )

            splits.append((datetime.datetime.now(), 'centroid coords'))

            centroid_measures = [self.convNd_sparse(image[:,:,:,0], self.kernels_3d[0], image_centroids)]
            splits.append((datetime.datetime.now(), 'raw corevals'))

            centroid_measures.append(self.convNd_sparse(image[:,:,:,0], self.kernels_3d[1], image_centroids))
            splits.append((datetime.datetime.now(), 'raw hollowvals'))

            centroid_measures.append(self.convNd_sparse(dog, self.kernels_3d[0], dog_centroids))
            splits.append((datetime.datetime.now(), 'DoG corevals'))

            centroid_measures.append(self.convNd_sparse(dog, self.kernels_3d[1], dog_centroids))
            splits.append((datetime.datetime.now(), 'DoG hollowvals'))

            if image.shape[3] > 1:
                centroid_measures.append(self.convNd_sparse(image[:,:,:,1], self.kernels_3d[2], image_centroids))
                splits.append((datetime.datetime.now(), 'centroid redvals'))
        else:
            # defaults if we have no centroids in block...
                image_centroids = zeros((0,3), int32)
                global_centroids = zeros((0,3), int32)
                centroid_measures = [
                    zeros((0,), float32), # raw coreval
                    zeros((0,), float32), # raw hollowval
                    zeros((0,), float32), # dog coreval
                    zeros((0,), float32), # dog hollowval
                ]
                if image.shape[3] > 1:
                    centroid_measures.append(
                        zeros((0,), float32), # redvals
                    )

                # need to keep same shape for splits list
                splits.append((datetime.datetime.now(), 'centroid coords'))
                splits.append((datetime.datetime.now(), 'raw corevals'))
                splits.append((datetime.datetime.now(), 'raw hollowvals'))
                splits.append((datetime.datetime.now(), 'DoG corevals'))
                splits.append((datetime.datetime.now(), 'DoG hollowvals'))

        centroid_measures = np.column_stack(tuple(centroid_measures))
        splits.append((datetime.datetime.now(), 'stack centroid measures'))

        perf_vector = list(map(lambda t0, t1: ((t1[0]-t0[0]).total_seconds(), t1[1]), splits[0:-1], splits[1:]))
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
    from . import nexpr as numerexprlib
    class BlockedAnalyzerNumerexpr (BlockedAnalyzer):

        def convNx1d(self, *args):
            return numerexprlib.convNx1d(*args)

        def array_mult(self, a1, a2):
            return numerexprlib.array_mult(a1, a2)

    BlockedAnalyzerOpt = BlockedAnalyzerNumerexpr
except:
    pass

try:
    from . import ocl as opencllib
    class BlockedAnalyzerOpenCL (BlockedAnalyzerOpt):

        def convNx1d(self, *args):
            return opencllib.convNx1d(*args)

        def maxNx1d(self, *args):
            return opencllib.maxNx1d(*args)

        def sum_labeled(self, src, labels, n, clq=None):
            return opencllib.sum_labeled(src, labels, n, clq=clq)

        def fwhm_estimate(self, synapse, centroids, syn_vals, vcn_vals, noise):
            return opencllib.fwhm_estimate(
                synapse, centroids, syn_vals, vcn_vals, noise,
                (self.image_meta.z_microns, self.image_meta.y_microns, self.image_meta.x_microns)
            )

        def convNd_sparse(self, data, kernel, centroids, clq=None):
            if clq is None:
                # CL would actually slower due to data input bottleneck!
                return BlockedAnalyzer.convNd_sparse(self, data, kernel, centroids)
            else:
                return opencllib.weighted_measure(data, centroids, kernel, clq=clq)
        
        def block_process(self, blockpos):
            """Process block data to return convolved results.
            """
            clq = opencllib.cl.CommandQueue(opencllib.ctx)
            
            splits = [(datetime.datetime.now(), None)]

            image = self.image[self.block_slice_src(blockpos)].astype(np.float32, copy=False)
            splits.append((datetime.datetime.now(), 'image load'))

            # PyOpenCL complains about discontiguous arrays when we project C dimension
            if image.strides[3] == 0:
                # but, a volspy.util TiffLazyNDArray slice repacks implicitly
                image0_dev = opencllib.cl_array.to_device(clq, image[:,:,:,0])
            else:
                # while a regular ndarray needs repacking here
                # this happens with the VOLSPY_ZNOISE_PERCENTILE pre-filtering hack
                image0_dev = opencllib.cl_array.empty(clq, image.shape[0:3], image.dtype)
                image0_tmp = image0_dev.map_to_host()
                image0_tmp[...] = image[:,:,:,0]
                del image0_tmp
                
            clq.finish()
            splits.append((datetime.datetime.now(), 'image to dev'))
            
            low_channel = self.convNx1d(image0_dev, self.kernels_3x1d[0], clq).map_to_host()
            splits.append((datetime.datetime.now(), 'image*low'))

            scale1_channel = self.convNx1d(image0_dev, self.kernels_3x1d[2], clq).map_to_host()
            splits.append((datetime.datetime.now(), 'image*syn'))

            scale2_channel = self.convNx1d(image0_dev, self.kernels_3x1d[3], clq).map_to_host()
            clq.finish()
            dog = crop_centered(scale1_channel, scale2_channel.shape) - scale2_channel
            splits.append((datetime.datetime.now(), 'image*vlow'))

            # allow tinkering w/ multiple peak detection fields
            max_inputs = [
                low_channel,
                # dog,
            ]

            if len(max_inputs) > 1:
                crop_shape = list(map(min, *[img.shape for img in max_inputs]))
            else:
                crop_shape = max_inputs[0].shape

            max_inputs = [crop_centered(img, crop_shape) for img in max_inputs]

            if self.view_raw:
                view_image = crop_centered(
                    image,
                    list(map(lambda w, b: w-2*b, image.shape[0:3], self.max_border_widths)) + [image.shape[3]]
                )
            else:
                view_image = crop_centered(
                    dog,
                    list(map(lambda w, b: w-2*b, image.shape[0:3], self.max_border_widths))
                )
                view_image = view_image[:,:,:,None]
                splits.append((datetime.datetime.now(), 'view image DoG'))

            view_image = bin_reduce(view_image, self.view_reduction + (1,))
            splits.append((datetime.datetime.now(), 'view image reduce'))

            max_kernel = self.kernels_3d[3].shape
            max_channels = [self.maxNx1d(img, max_kernel) for img in max_inputs]
            splits.append((datetime.datetime.now(), 'local maxima'))

            # need to trim borders discarded by max_channel computation
            max_inputs = [crop_centered(img, max_channels[0].shape) for img in max_inputs]

            # find syn cores via local maxima test
            peaks = np.zeros(max_channels[0].shape, dtype=np.bool)
            for i in range(len(max_inputs)):
                assert max_inputs[i].shape == max_channels[i].shape
                peaks += max_inputs[i] >= (max_channels[i])

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
            label_im_dev = opencllib.cl_array.to_device(clq, label_im)
            splits.append((datetime.datetime.now(), 'label peaks'))

            sizes = self.sum_labeled(
                label_im_dev > 0,
                label_im_dev,
                nb_labels + 1,
                clq=clq
            )[1:].map_to_host()
            splits.append((datetime.datetime.now(), 'centroid sizes'))

            centroid_components = [ ]

            for d in range(3):
                coords_dev = opencllib.nd_arange(peaks.shape, d, 0, 1, clq)
                centroid_components.append(
                    (self.sum_labeled(
                        coords_dev,
                        label_im_dev,
                        nb_labels + 1,
                        clq=clq
                        )[1:].map_to_host()/sizes)#.astype(np.int32)
                    )

            # centroids are in block peaks grid
            centroids = list(zip(*centroid_components))

            filtered_centroids = []
            if centroids:
                # discard centroids outside clipbox (we searched slightly
                # larger to handle peaks at edges
                for i in range(len(centroids)):
                    clip = False
                    for d in range(3):
                        if int(centroids[i][d]) < clipbox[d].start or int(centroids[i][d]) >= clipbox[d].stop:
                            clip = True
                    if not clip:
                        filtered_centroids.append(centroids[i])

            if filtered_centroids:
                centroids = array(filtered_centroids, int32) - array([slc.start for slc in clipbox], int32)
                # image_centroids are in block image grid
                image_centroids = centroids + array(self.max_border_widths, int32)
                # dog_centroids are in difference-of-gaussians grid
                dog_centroids = centroids + array(list(map(lambda iw, dw: (iw-dw)/2, image.shape[0:3], dog.shape)))
                # global_centroids are in self.image grid
                global_centroids = (
                    array([slc.start or 0 for slc in self.block_slice_src(blockpos)[0:3]], int32)
                    + image_centroids
                )

                splits.append((datetime.datetime.now(), 'centroid coords'))

                image_centroids_dev = opencllib.cl_array.to_device(clq, image_centroids)
                centroid_measures = [
                    self.convNd_sparse(
                        image0_dev,
                        opencllib.cl_array.to_device(clq, self.kernels_3d[0]),
                        image_centroids_dev,
                        clq=clq
                    ).map_to_host()
                ]
                splits.append((datetime.datetime.now(), 'raw corevals'))

                centroid_measures.append(
                    self.convNd_sparse(
                        image0_dev,
                        opencllib.cl_array.to_device(clq, self.kernels_3d[1]),
                        image_centroids_dev,
                        clq=clq
                    ).map_to_host()
                )
                del image0_dev
                del image_centroids_dev
                splits.append((datetime.datetime.now(), 'raw hollowvals'))

                dog_dev = opencllib.cl_array.to_device(clq, dog)
                dog_centroids_dev = opencllib.cl_array.to_device(clq, dog_centroids)
                centroid_measures.append(
                    self.convNd_sparse(
                        dog_dev,
                        opencllib.cl_array.to_device(clq, self.kernels_3d[0]),
                        dog_centroids_dev,
                        clq=clq
                    ).map_to_host()
                )
                splits.append((datetime.datetime.now(), 'DoG corevals'))

                centroid_measures.append(
                    self.convNd_sparse(
                        dog_dev,
                        opencllib.cl_array.to_device(clq, self.kernels_3d[1]),
                        dog_centroids_dev,
                        clq=clq
                    ).map_to_host()
                )
                del dog_dev
                del dog_centroids_dev
                splits.append((datetime.datetime.now(), 'DoG hollowvals'))

                if image.shape[3] > 1:
                    centroid_measures.append(self.convNd_sparse(image[:,:,:,1], self.kernels_3d[2], image_centroids))
                    splits.append((datetime.datetime.now(), 'centroid redvals'))
            else:
                # defaults if we have no centroids in block...
                image_centroids = zeros((0,3), int32)
                global_centroids = zeros((0,3), int32)
                centroid_measures = [
                    zeros((0,), float32), # raw coreval
                    zeros((0,), float32), # raw hollowval
                    zeros((0,), float32), # dog coreval
                    zeros((0,), float32), # dog hollowval
                ]
                if image.shape[3] > 1:
                    centroid_measures.append(
                        zeros((0,), float32), # redvals
                    )

                # need to keep same shape for splits list
                splits.append((datetime.datetime.now(), 'centroid coords'))
                splits.append((datetime.datetime.now(), 'raw corevals'))
                splits.append((datetime.datetime.now(), 'raw hollowvals'))
                splits.append((datetime.datetime.now(), 'DoG corevals'))
                splits.append((datetime.datetime.now(), 'DoG hollowvals'))

            centroid_measures = np.column_stack(tuple(centroid_measures))
            splits.append((datetime.datetime.now(), 'stack centroid measures'))

            perf_vector = list(map(lambda t0, t1: ((t1[0]-t0[0]).total_seconds(), t1[1]), splits[0:-1], splits[1:]))
            return view_image, global_centroids, centroid_measures, perf_vector

    BlockedAnalyzerOpt = BlockedAnalyzerOpenCL
    assign_voxels_opt = opencllib.assign_voxels
except:
    pass

def batch_analyze(image, cdiam_microns, vdiam_microns, rdiam_microns, view_reduction=(1,1,1)):
    analyzer = BlockedAnalyzerOpt(image, cdiam_microns, vdiam_microns, rdiam_microns, view_reduction)
    view_image, centroids, centroid_measures = analyzer.volume_process()
    return analyzer, view_image, centroids, centroid_measures

synaptic_footprints = (
    (2.75, 1.5, 1.5),
    (4.0, 2.75, 2.75),
    (3.0, 3.0, 3.0),
)

nucleic_footprints = (
    (8., 8., 8.),
    (16., 16., 16.),
    (3.0, 3.0, 3.0),
)

def get_mode_and_footprints():
    do_nuclei = {'true': True}.get(os.getenv('SYNSPY_DETECT_NUCLEI'), False)
    footprints = nucleic_footprints if do_nuclei else synaptic_footprints
    return do_nuclei, footprints

def batch_analyze_cli(fname):
    """Analyze file given as argument and write NPZ output file.

       Arguments:
         fname: OME-TIFF input file name

       Environment parameters:
         DUMP_PREFIX: defaults to './basename' where '.ome.tiff' suffix has been stripped
         ZYX_SLICE: selects ROI within full image
         ZYX_IMAGE_GRID: overrides image grid step metadata
         SYNSPY_NUCLEI_DETECT: 'true' for nuclei mode, else synapse mode

       Output NPZ array keys:
         'properties.json': various metadata as 1D uint8 array of UTF-8 JSON data
         'voxels': 4D voxel data with axes (channel, z, y, x)
         'centroids': 2D centroid list with axes (N, c) for coords [z y x]
         'measures':  2D measure list with axes (N, m) for measures []

       Output is written to file names by DUMP_PREFIX + '.roi.npz'
    """
    dump_prefix = os.path.basename(fname)
    try:
        m = re.match('^(?P<accession>.*)(?P<ome>[.]ome)[.]tif+$', dump_prefix)
        dump_prefix = m.groupdict()['accession']
    except:
        pass
    dump_prefix = os.getenv('DUMP_PREFIX', dump_prefix)

    image, meta, slice_origin = load_and_mangle_image(fname)
    do_nuclei, footprints = get_mode_and_footprints()
    cdiam, vdiam, rdiam = footprints
    analyzer, view_image, centroids, measures = batch_analyze(image, cdiam, vdiam, rdiam)

    props = {
        "image_grid": list(image.micron_spacing),
        "shape": list(image.shape),
        "slice_origin": list(slice_origin),
        "core_diam_microns": list(footprints[0]),
        "vicinity_diam_microns": list(footprints[1]),
        "synspy_nuclei_mode": do_nuclei,
    }
    if image.shape[0] > 1:
        props['redblur_diam_mirons'] = list(footprints[2])

    if view_image.dtype == np.float32 and measures.dtype == np.float32:
        maxval = max(view_image.max(), measures.max())
        view_image = view_image * 1.0/maxval
        view_image = view_image.astype(np.float16)
        measures = measures * 1.0/maxval
        measures = measures.astype(np.float16)
        props['voxel_divisor'] = float(maxval)
        props['measures_divisor'] = float(maxval)

    if centroids.dtype == np.int32 and centroids.max() < 2**16-1 and centroids.min() >= 0:
        centroids = centroids.astype(np.uint16)

    dump_fname = '%s.npz' % dump_prefix
    outf = open(dump_fname, 'wb')

    np.savez(
        outf,
        properties=np.fromstring(json.dumps(props), np.uint8),
        voxels=view_image,
        centroids=centroids,
        measures=measures
    )
    outf.close()
    print('Dumped ROI analysis data to %s' % dump_fname)

    return 0
