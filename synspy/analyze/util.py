
#
# Copyright 2014-2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import re
import sys
import os
import csv
import random
import numpy as np
from numpy import array, float32, int32, empty, newaxis, dot, cross, zeros, ones
from numpy.linalg import norm
import json
import math

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

def crop_centered(orig, newshape):
    return orig[tuple(
        diff and slice(diff/2, -(diff/2)) or slice(None)
        for diff in map(lambda l, s: l-s, orig.shape, newshape)
    )]

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

def clamp_center_edge(orig, axis=0):
    return orig * (orig >= orig[
        tuple(
            orig.shape[d]/2 for d in range(axis)
        ) + (0,) + tuple(
            orig.shape[d]/2 for d in range(axis+1, 3)
        )
    ])

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
    try:
        peak_factor = float(os.getenv('PEAKS_DIAM_FACTOR', 0.75))
    except ValueError, te:
        print 'ERROR: invalid PEAKS_DIAM_FACTOR environment value'
        raise
    
    # these are separated 1d gaussian kernels
    syn_kernels = map(lambda d, s: gaussian_kernel(d/s/6.), synapse_diam_microns, gridsize)
    low_kernels = map(lambda d, s: gaussian_kernel(peak_factor*d/s/6.), synapse_diam_microns, gridsize)
    vlow_kernels = map(lambda d, s: gaussian_kernel(d/s/6.), vicinity_diam_microns, gridsize)
    span_kernels = map(lambda d, s: (1,) * (2*(int(d/s)/2)+1), vicinity_diam_microns, gridsize)

    # TODO: investigate variants?
    #  adjust diameter by a fudge factor?
    core_kernel = compose_3d_kernel(syn_kernels)
    span_kernel = compose_3d_kernel(vlow_kernels)

    if True:
        # truncate to ellipsoid region
        core_kernel = clamp_center_edge(core_kernel)
        span_kernel = clamp_center_edge(span_kernel)

    hollow_kernel = span_kernel * (pad_centered(core_kernel, span_kernel.shape) <= 0)
        
    max_kernel = ones(map(lambda d, s: 2*(int(0.7*d/s)/2)+1, synapse_diam_microns, gridsize), dtype=float32)

    # sanity check kernel shapes
    for d in range(3):
        if len(syn_kernels[d]) <= 1:
            raise ValueError(
                'Synapse diameter %f and gridsize %f result in undersized synapse kernel!'
                % (synapse_diam_microns[d], gridsize[d])
            )
        if len(low_kernels[d]) <= 1:
            raise ValueError(
                'Synapse diameter %f, peak_diam_factor %f, and gridsize %f result in undersized low-pass kernel!'
                % (synapse_diam_microns[d], peak_factor, gridsize[d])
            )
        if hollow_kernel.shape[d] - core_kernel.shape[d] <= 1:
            raise ValueError(
                'Synapse diameter %f, vicinity diameter %f, and gridsize %f result in undersized hollow span!'
                % (synapse_diam_microns[d], synapse_diam_microns[d], gridsize[d])
            )
    
    core_kernel /= core_kernel.sum()
    hollow_kernel /= hollow_kernel.sum()

    red_kernel = compose_3d_kernel(
        map(lambda d, s: gaussian_kernel(d/s/6.), redblur_microns, gridsize)
    )

    return (
        (low_kernels, span_kernels, syn_kernels, vlow_kernels),
        (core_kernel, hollow_kernel, red_kernel, max_kernel)
        )

def radii_3x1d(k3x1d):
    return np.array([len(k1d)/2 for k1d in k3x1d], dtype=np.int32)

def radii_3d(k3d):
    return np.array([d/2 for d in k3d.shape], dtype=np.int32)

def load_segment_status_from_csv(centroids, offset_origin, infilename):
    """Load a segment list with manual override status values validating against expected centroid list.

       Arguments:
         centroids: Nx3 array of Z,Y,X segment coordinates
         offset_origin: CSV coordinates = offset_origin + centroid coordinates
         infilename: file to open to read CSV content

       Returns tuple with:
         status array (1D),
         saved params dict or None
    """
    # assume that dump is ordered subset of current analysis
    csvfile = open(infilename, 'r')
    reader = csv.DictReader(csvfile)
    i = 0
    status = np.zeros((centroids.shape[0],), dtype=np.uint8)
    saved_params = None
    for row in reader:
        # newer dump files have an extra saved-parameters row first...
        if row['Z'] == 'saved' and row['Y'] == 'parameters':
            saved_params = row
            continue

        # convert global unsliced coordinates back into sliced image coordinates
        Z = int(row['Z']) - offset_origin[0]
        Y = int(row['Y']) - offset_origin[1]
        X = int(row['X']) - offset_origin[2]

        # scan forward until we find same centroid, since CSV is a subset
        while i < centroids.shape[0] and (Z, Y, X) != tuple(centroids[i]):
            i += 1
            
        assert i < centroids.shape[0], ("CSV dump does not match image analysis!", csv_name)

        if row['override']:
            status[i] = int(row['override'])

    csvfile.close()
    return status, saved_params

def dump_segment_info_to_csv(centroids, measures, status, offset_origin, outfilename, saved_params=None, all_segments=True):
    """Load a segment list with manual override status values validating against expected centroid list.

       Arguments:
         centroids: Nx3 array of Z,Y,X segment coordinates
         measures: NxK array of segment measures
         status: N array of segment status
         offset_origin: CSV coordinates = offset_origin + centroid coordinates
         outfilename: file to open to write CSV content
         saved_params: dict or None if saving threshold params row
         all_segments: True: dump all, False: dump only when status > 0
    """
    # correct dumped centroids to global coordinate space of unsliced source image
    centroids = centroids + np.array(offset_origin, np.int32)
    csvfile = open(outfilename, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(
        ('Z', 'Y', 'X', 'raw core', 'raw hollow', 'DoG core', 'DoG hollow')
        + ((measures.shape[1] == 5) and ('red',) or ())
        + ('override',)
    )
    if saved_params:
        writer.writerow(
            (
                'saved',
                'parameters',
                saved_params.get('X', ''),
                saved_params.get('raw core', ''),
                saved_params.get('raw hollow', ''),
                saved_params.get('DoG core', ''),
                saved_params.get('DoG hollow', ''),
            )
            + ((saved_params.get('red', ''),) if 'red' in saved_params else ())
            + (saved_params.get('override', ''),)
        )
    if all_segments:
        indices = range(measures.shape[0])
    else:
        indices = (status > 0).nonzero()[0]
    for i in indices:
        Z, Y, X = centroids[i]
        writer.writerow( 
            (Z, Y, X) + tuple(measures[i,m] for m in range(measures.shape[1])) + (status[i] or '',)
        )
    del writer
    csvfile.close()


