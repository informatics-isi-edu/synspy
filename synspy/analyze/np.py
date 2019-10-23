
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import os
from numpy import array, concatenate
import scipy
from scipy import ndimage

import numpy as np

def convNd_sparse(src, kernel, centroids):
    """ND convolution at sparse sampling centroids.

       For input of K N-dimensional centroids which are locations
       within N-dimensional src image, K scalars are produced as if
       the convolution src*kernel was sampled at those centroids.

       The coordinates of each centroid MUST lie within the valid
       sub-region of the src image grid, i.e. at least kernel-radius
       distance from image edges in each dimension.

    """
    results = []
    kernel_radii = [w//2 for w in kernel.shape]
    for centroid in centroids:
        slc = tuple(
            slice(int(centroid[d] - kernel_radii[d]), int(centroid[d] + kernel_radii[d] + 1))
            for d in range(len(src.shape))
        )
        box = src[slc]
        results.append((box * kernel).sum())

    return array(results, dtype=src.dtype)

def convNx1d(src, kernels):

    """ND convolution using 1D kernels

       Trims borders by filter kernel width in each dimension.

       This version uses ndimage.convolve1d() and produces float32
       intermediate and final results regardless of input type.
    """
    for d in range(len(kernels)):
        L = src.shape[d]
        kernel = kernels[d]
        kernel_width = len(kernel)
        if (kernel_width % 2) != 1:
            raise NotImplementedError('convNx1d on even-length kernel')
        kernel_radius = kernel_width//2

        if kernel_radius < 1:
            print("warning: dimension %d kernel %d is too small, has no effect" % (d, kernel_width))
            continue
        elif kernel_radius > L:
            raise ValueError("dimension %d length %d too small for kernel %d" % (d, L, kernel_width))

        src = ndimage.convolve1d(
            src.astype('float32', copy=False), 
            array(kernel, dtype='float32'),
            mode='constant',
            axis=d
        )

        # trim off invalid borders
        src = src[ tuple([slice(None) for j in range(d)] + [slice(kernel_radius,kernel_radius and -kernel_radius or None)] + [ Ellipsis ]) ]

    return src
        

def maxNx1d(src, lengths):
    """ND maximum filter using 1D kernels

    """
    for d in range(len(lengths)):
        L = src.shape[d]
        kernel_width = lengths[d]
        if (kernel_width % 2) != 1:
            raise NotImplementedError('maxNx1d on even-length %d kernel' % lengths[d])
        kernel_radius = kernel_width//2

        if kernel_radius < 1:
            print("warning: dimension %d kernel %d is too small, has no effect" % (d, kernel_width))
            continue
        elif kernel_width > L:
            raise ValueError("dimension %d length %d too small for kernel %d" % (d, L, kernel_width))

        src = ndimage.maximum_filter1d(
            src,
            lengths[d],
            mode='constant',
            axis=d
        )

        # trim off invalid borders
        src = src[ tuple([slice(None) for j in range(d)] + [slice(kernel_radius,kernel_radius and -kernel_radius or None)] + [ Ellipsis ]) ]

    return src

def equitrim(arrays):
    minshape = None
    for a in arrays:
        if a is None:
            continue
        if minshape is None:
            minshape = a.shape
        else:
            assert len(minshape) == len(a.shape)
            minshape = list(map(min, minshape, a.shape))

    for a in arrays:
        if a is None:
            yield a
        else:
            yield a[
                tuple(
                    [
                        a.shape[d] > minshape[d] 
                        and slice(
                            (a.shape[d]-minshape[d])//2, 
                            -(a.shape[d]-minshape[d])//2
                        )
                        or None
                        for d in range(len(minshape))
                    ]
                )
            ]
            

def assign_voxels(syn_values, centroids, valid_shape, syn_kernel_3d, gridsize=None):
    """Assign voxels to features and fill with segment ID.

       Parameters:

          syn_values: measured synapse core intensity as from analyze
          centroids: synapse locations as from analyze
          valid_shape: the size of the processed volume
          syn_kernel_3d: the 3D kernel representing synapse cores

       Results:

          a Numpy array with voxels filled with segment IDs

       The length N of syn_values and centroids is mapped to segment
       ID range (1...N) while non-synapse voxels are labeled zero.

       The array has shape matching valid_shape and integer type large
       enough to hold all segment IDs.

    """

    # compute mutually-exclusive gaussian segments
    # in overlaps, voxel assigned by max kernel-weighted segment core value
    assert len(centroids) == len(syn_values)

    N = len(syn_values)
    if N < 2**8:
        dtype = np.uint8
    elif N < 2**16:
        dtype = np.uint16
    elif N < 2**32:
        dtype = np.uint32
    elif N < 2**64:
        dtype = np.uint64
    else:
        raise NotImplementedError("Absurdly large segment count %s" % N)

    gaussian_map = np.zeros( valid_shape, dtype=np.float32 ) # voxel -> weighted core values
    segment_map = np.zeros( valid_shape, dtype=dtype )  # voxel -> label

    # use a slight subset as the splatting body
    body_shape = syn_kernel_3d.shape
    D, H, W = body_shape
    radial_fraction = np.clip(float(os.getenv('SYNSPY_SPLAT_SIZE', '1.0')), 0, 2)
    limits = (
        syn_kernel_3d[D//2-1,0,W//2-1],
        syn_kernel_3d[D//2,H//2,W//2]
    )
    limit = limits[1] - (limits[1] - limits[0]) * radial_fraction
    mask_3d = syn_kernel_3d >= limit
    mask_3d[tuple([w//2 for w in mask_3d.shape])] = 1 # fill at least central voxel
    print("SPLAT BOX SHAPE %s   USER COEFFICIENT %f   MASK VOXELS %d" % (
        body_shape,
        radial_fraction,
        mask_3d.sum()
    ))
    weights = syn_kernel_3d * mask_3d

    def splat_segment(label):
        weighted = weights * syn_values[label]
        centroid = centroids[label]

        def map_slice(centroid):
            # splats are confined to boundaries of valid_shape map
            def helper(d):
                lower = centroid[d] - body_shape[d]//2
                upper = centroid[d] + body_shape[d]//2 + body_shape[d]%2
                if lower < 0:
                    lower = 0
                if upper > valid_shape[d]:
                    upper = valid_shape[d]
                return slice(lower,upper)

            return tuple(map(helper, list(range(3))))

        def body_slice(centroid):
            # splats are cropped by boundaries of map
            def helper(d):
                lower = 0
                upper = body_shape[d]
                if centroid[d] < body_shape[d]//2:
                    lower = body_shape[d]//2 - centroid[d]
                if (centroid[d] + body_shape[d]//2 + body_shape[d]%2) > valid_shape[d]:
                    upper -= (centroid[d] + body_shape[d]//2 + body_shape[d]%2) - valid_shape[d]
                return slice(lower,upper)

            return tuple(map(helper, list(range(3))))

        mslc = map_slice(centroid)
        bslc = body_slice(centroid)

        # update maps for assigned voxels
        try:
            segvoxels = gaussian_map[mslc] < weighted[bslc]
            segment_map[mslc] = segment_map[mslc] * (~segvoxels) + (label+1) * segvoxels
            gaussian_map[mslc] = gaussian_map[mslc] * (~segvoxels) + weighted[bslc] * segvoxels
        except:
            print(label, centroid, mslc, bslc, valid_shape, body_shape)

    for label in range(len(syn_values)):
        splat_segment(label)

    gaussian_map = None
    return segment_map

