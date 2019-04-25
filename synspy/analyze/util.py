
#
# Copyright 2014-2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import re
import sys
import os
import csv
import random
import tempfile
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
    kernel_width = 2 * (int(6.0 * s - 1) // 2) + 1 # force always odd
    kernel_radius = (kernel_width - 1) // 2 # doesn't include central cell
    kernel = list(map(G, list(range(-kernel_radius, kernel_radius+1))))
    mag = sum(kernel)
    kernel = [x / mag for x in kernel]
    return kernel

def crop_centered(orig, newshape):
    return orig[tuple(
        diff and slice(diff//2, -(diff//2)) or slice(None)
        for diff in map(lambda l, s: l-s, orig.shape, newshape)
    )]

def pad_centered(orig, newshape, pad=0):
    assert len(newshape) == orig.ndim
    
    for d in range(len(newshape)):
        assert newshape[d] >= orig.shape[d]
        
    na = zeros(newshape, dtype=orig.dtype)
    
    def helper1d(dstlen, srclen):
        if dstlen > srclen:
            return slice((dstlen-srclen)//2, -(dstlen-srclen)//2)
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
            orig.shape[d]//2 for d in range(axis)
        ) + (0,) + tuple(
            orig.shape[d]//2 for d in range(axis+1, 3)
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
    except ValueError as te:
        print('ERROR: invalid PEAKS_DIAM_FACTOR environment value')
        raise
    
    # these are separated 1d gaussian kernels
    syn_kernels = list(map(lambda d, s: gaussian_kernel(d/s/6.), synapse_diam_microns, gridsize))
    low_kernels = list(map(lambda d, s: gaussian_kernel(peak_factor*d/s/6.), synapse_diam_microns, gridsize))
    vlow_kernels = list(map(lambda d, s: gaussian_kernel(d/s/6.), vicinity_diam_microns, gridsize))
    span_kernels = list(map(lambda d, s: (1,) * (2*(int(d/s)//2)+1), vicinity_diam_microns, gridsize))

    # TODO: investigate variants?
    #  adjust diameter by a fudge factor?
    core_kernel = compose_3d_kernel(syn_kernels)
    span_kernel = compose_3d_kernel(vlow_kernels)

    if True:
        # truncate to ellipsoid region
        core_kernel = clamp_center_edge(core_kernel)
        span_kernel = clamp_center_edge(span_kernel)

    hollow_kernel = span_kernel * (pad_centered(core_kernel, span_kernel.shape) <= 0)
        
    max_kernel = ones(list(map(lambda d, s: 2*(int(0.7*d/s)//2)+1, synapse_diam_microns, gridsize)), dtype=float32)

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
        list(map(lambda d, s: gaussian_kernel(d/s/6.), redblur_microns, gridsize))
    )

    return (
        (low_kernels, span_kernels, syn_kernels, vlow_kernels),
        (core_kernel, hollow_kernel, red_kernel, max_kernel)
        )

def radii_3x1d(k3x1d):
    return np.array([len(k1d)//2 for k1d in k3x1d], dtype=np.int32)

def radii_3d(k3d):
    return np.array([d//2 for d in k3d.shape], dtype=np.int32)

def centroids_zx_swap(centroids):
    """Return a copy of centroids array with Z and X swapped, e.g. ZYX<->XYZ."""
    copy = np.zeros(centroids.shape, dtype=centroids.dtype)
    copy[:,0] = centroids[:,2]
    copy[:,1] = centroids[:,1]
    copy[:,2] = centroids[:,0]
    return copy

def transform_points(M, v, dtype=np.float32):
    """Apply transform matrix to k XYZ points in v[k,3] array.

       This does a standard np.matmul(v, M) after extending all points
       in v with the W coordinate 1.0, and renormalizing the results
       back to XYZ w/o W coordinate.

    """
    assert v.shape[1] == 3
    a1 = np.ones((v.shape[0], 4), dtype=np.float64)
    a1[:,0:3] = v
    a2 = np.matmul(a1,M)
    return (a2[:,0:3] / a2[:,3,None]).astype(dtype)

def transform_centroids(M, centroids):
    """Transform each ZYX point in centroids[k,3] using np.dot(M, xyzw) intermediate representation.

       Note, the 4x4 transform M may need to be transposed for
       np.dot(M, xyzw) to give you the transform you expect!

    """
    assert centroids.shape[1] == 3
    a1 = np.zeros((centroids.shape[0], 4), dtype=np.float64)
    a2 = np.zeros(centroids.shape, dtype=np.float64)

    a1[:,0:3] = centroids_zx_swap(centroids)
    a1[:,3] = 1.0

    for i in range(a1.shape[0]):
        p = np.dot(M, a1[i])
        a2[i,:] = p[0:3] / p[3]

    return centroids_zx_swap(a2).astype(np.float32)

def load_segment_info_from_csv(infilename, zyx_grid_scale=None, zx_swap=False, filter_status=None):
    """Load a segment list and return content as arrays.

    """
    csvfile = open(infilename, 'r')
    reader = csv.DictReader(csvfile)
    centroids = []
    measures = []
    status = []
    saved_params = None
    for row in reader:
        # newer dump files have an extra saved-parameters row first...
        if row['Z'] == 'saved' and row['Y'] == 'parameters':
            saved_params = row
            continue

        centroids.append(
            (int(row['Z']), int(row['Y']), int(row['X']))
        )
        measures.append(
            (float(row['raw core']), float(row['raw hollow']), float(row['DoG core']), float(row['DoG hollow']))
            + (float(row['red']) if 'red' in row else ())
        )
        status.append(
            int(row['override']) if row['override'] else 0
        )
    centroids = np.array(centroids, dtype=np.int32)
    measures = np.array(measures, dtype=np.float32)
    status = np.array(status, dtype=np.uint8)
    if zyx_grid_scale is not None:
        zyx_grid_scale = np.array(zyx_grid_scale, dtype=np.float32)
        assert zyx_grid_scale.shape == (3,)
        centroids = (centroids * zyx_grid_scale).astype(np.float32)
    if filter_status is not None:
        filter_idx = np.zeros(status.shape, dtype=np.bool)
        for value in filter_status:
            filter_idx += (status == value)
        centroids = centroids[filter_idx]
        measures = measures[filter_idx]
        status = status[filter_idx]
    return (
        centroids_zx_swap(centroids) if zx_swap else centroids,
        measures,
        status,
        saved_params
    )

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
    csv_centroids, csv_measures, csv_status, saved_params = load_segment_info_from_csv(infilename)
    csv_centroids -= np.array(offset_origin, dtype=np.int32)
    return dense_segment_status(centroids, csv_centroids, csv_status), saved_params
    
def dense_segment_status(centroids, sparse_centroids, sparse_status):
    """Construct dense segment status from sparse info, e.g. previously loaded from CSV."""
    # assume that dump is ordered subset of current analysis
    status = np.zeros((centroids.shape[0],), dtype=np.uint8)

    i = 0
    for row in range(sparse_centroids.shape[0]):
        # scan forward until we find same centroid in sparse subset
        while i < centroids.shape[0] and tuple(sparse_centroids[row]) != tuple(centroids[i]):
            i += 1
            
        assert i < centroids.shape[0], ("Sparse dump does not match image analysis!", sparse_centroids[row])

        if sparse_status[row]:
            status[i] = sparse_status[row]

    return status

def dump_segment_info_to_csv(centroids, measures, status, offset_origin, outfilename, saved_params=None, all_segments=True, zx_swap=False, zyx_grid_scale=None, filter_status=None):
    """Load a segment list with manual override status values validating against expected centroid list.

       Arguments:
         centroids: Nx3 array of Z,Y,X segment coordinates
         measures: NxK array of segment measures
         status: N array of segment status
         offset_origin: CSV coordinates = offset_origin + centroid coordinates
         outfilename: file to open to write CSV content
         saved_params: dict or None if saving threshold params row
         all_segments: True: dump all, False: dump only when matching filter_status values
         zx_swap: True: input centroids are in X,Y,Z order
         zyx_grid_scale: input centroids have been scaled by these coefficients in Z,Y,X order
         filter_status: set of values to include in outputs or None implies all non-zero values
    """
    if zx_swap:
        centroids = centroids_zx_swap(centroids)
    if zyx_grid_scale is not None:
        zyx_grid_scale = np.array(zyx_grid_scale, dtype=np.float32)
        assert zyx_grid_scale.shape == (3,)
        centroids = centroids * zyx_grid_scale
    # correct dumped centroids to global coordinate space of unsliced source image
    centroids = centroids + np.array(offset_origin, np.int32)
    csvfile = open(outfilename, 'w', newline='')
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

    filter_idx = np.zeros(status.shape, dtype=np.bool)
    if all_segments:
        filter_idx += np.bool(1)
    elif filter_status is not None:
        for value in filter_status:
            filter_idx += (status == value)
    else:
        filter_idx += (status > 0)

    indices = (status > 0).nonzero()[0]

    for i in indices:
        Z, Y, X = centroids[i]
        writer.writerow( 
            (Z, Y, X) + tuple(measures[i,m] for m in range(measures.shape[1])) + (status[i] or '',)
        )
    del writer
    csvfile.close()

def load_registered_csv(hatrac_store, object_path):
    """Load a registered segment list from the object store.

       Arguments:
         hatrac_store: an instance of deriva.core.HatracStore
         object_path: the path of the registered segment list CSV object in the store

       Returns:
         a: an array of shape (N, k) of type float32

       The array has N centroids and k values packed as:
          Z, Y, X, raw core, raw hollow, DoG core, DoG hollow (, red)?, override

    """
    r = hatrac_store.get(object_path, stream=True)
    r.raise_for_status()
    reader = csv.DictReader(r.iter_lines(decode_unicode=True, chunk_size=1024**2))
    rows = []
    for row in reader:
        if row['Z'] == 'saved':
            continue
        rows.append(
            tuple([
                row[k]
                for k in ['Z', 'Y', 'X', 'raw core', 'raw hollow', 'DoG core', 'DoG hollow', 'red', 'override']
                if k in row
            ])
        )
    rows = np.array(rows, dtype=np.float32)
    return rows

def get_hatrac_object_cached(hatrac_store, object_path, cache_dir, suffix=''):
    resp = hatrac_store.head(object_path)
    md5 = resp.headers['content-md5']
    if cache_dir is not None:
        fname = '%s/%s%s' % (cache_dir, md5, suffix)
        if os.path.isfile(fname):
            return fname
    else:
        fd, fname = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
    hatrac_store.get_obj(object_path, destfilename=fname)
    return fname

def load_registered_npz(hatrac_store, object_path, alignment=None, csv_object_path=None, cache_dir=None):
    """Load registered segment list from the object store.

       Arguments:
         hatrac_store: from where to fetch object content
         object_path: specific NPZ object within hatrac_store
         alignment: 4x4 matrix to transform micron-spaced coordinates (or None)
         csv_object_path: specific CSV object within hatrac_store (or None)
         cache_dir: name of local directory to use as download cache for object reuse

       If csv_object_path is specified, interpret it as a sparse
       subject of the same pointcloud and merge its status field into
       result.

       Returns:
         a: an array of shape (N, k) of type float32
    
       The array has N centroids and k values packed as:
          Z, Y, X, raw core, raw hollow, DoG core, DoG hollow (, red)?, override

    """
    if alignment is not None:
        alignment = np.array(alignment, np.float64)
    else:
        alignment = np.eye(4, dtype=np.float64)
    try:
        fname = get_hatrac_object_cached(hatrac_store, object_path, cache_dir, '.npz')
        if csv_object_path:
            fnamec = get_hatrac_object_cached(hatrac_store, csv_object_path, cache_dir, '.csv')
            csv_centroids, csv_measures, csv_status, csv_saved_params = load_segment_info_from_csv(fnamec)
        with np.load(fname) as parts:
            properties = json.loads(parts['properties'].tostring().decode('utf8'))
            measures = parts['measures'].astype(np.float32) * np.float32(properties['measures_divisor'])
            slice_origin = np.array(properties['slice_origin'], dtype=np.int32)
            centroids = parts['centroids'].astype(np.int32) + slice_origin
            image_grid = np.array(properties['image_grid'], dtype=np.float32)
            result = np.zeros((centroids.shape[0], 3 + measures.shape[1] + 1), np.float32)
            result[:,0:3] = transform_centroids(alignment, centroids * image_grid)
            result[:,3:-1] = measures[:,:]
            if csv_object_path:
                result[:,-1] = dense_segment_status(centroids, csv_centroids, csv_status)
            return result
    finally:
        if cache_dir is None:
            if csv_object_path and fnamec:
                os.unlink(fnamec)
            if fname:
                os.unlink(fname)

def matrix_ident():
    """Produce indentity transform."""
    return np.identity(4).astype(np.float64)

def matrix_translate(displacement_xyz):
    """Produce translation transform matrix."""
    M = matrix_ident()
    M[3,0:3] = displacement_xyz
    return M

def matrix_scale(s):
    """Produce scaling transform matrix with uniform scale s in all 3 dimensions."""
    M = matrix_ident()
    M[0:3,0:3] = np.diag([ s, s, s ]).astype(np.float64)
    return M

def matrix_rotate(axis, radians):
    """Produce rotation transform matrix about axis through origin."""
    s = math.sin(radians)
    c = math.cos(radians)
    C = 1 - c
    x, y, z = axis / np.linalg.norm(axis)
    R = np.array(
        [
            [ x*x*C + c,   x*y*C - z*s, x*z*C + y*s ],
            [ y*x*C + z*s, y*y*C + c,   y*z*C - x*s ],
            [ z*x*C - y*s, z*y*C + x*s, z*z*C + c   ],
        ],
        dtype=np.float64
    )
    M = matrix_ident()
    M[0:3,0:3] = R
    return M

x_axis = np.array([1,0,0], np.float64)
y_axis = np.array([0,1,0], np.float64)
z_axis = np.array([0,0,1], np.float64)

