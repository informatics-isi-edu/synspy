#!/usr/bin/python
#
# Copyright 2015-2018 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import os
import sys
import pcl
import pcl.registration
import numpy as np
import csv
import math
from transformations import decompose_matrix

from .analyze.util import load_segment_info_from_csv, dump_segment_info_to_csv, centroids_zx_swap, transform_centroids

def get_env_grid_scale():
    grid = list(os.getenv('SEGMENTS_ZYX_GRID', '0.4,0.26,0.26').split(','))
    grid = np.array(tuple(map(float, grid)), dtype=np.float32)
    assert grid.shape == (3,), grid.shape
    return grid

def centroids2pointcloud(centroids):
    pc = pcl.PointCloud()
    pc.from_array(centroids)
    return pc

def csv2pointcloud_weights(filename, zyx_grid_scale=None):
    centroids, measures, status, saved_params = load_segment_info_from_csv(filename, zyx_grid_scale, True, (3,7))
    pc = pcl.PointCloud()
    pc.from_array(centroids)
    return pc, measures[0]

def align_centroids(centroids1, centroids2):
    """Compute alignment matrix for centroids2 into centroids1 coordinates.

       Arguments:
         centroids1: Nx3 array in Z,Y,X order
         centroids2: Mx3 array in Z,Y,X order

       Returns:
         M: 4x4 transformation matrix
         angles: decomposed rotation vector from M (in radians)
    """
    pc1 = centroids2pointcloud(centroids_zx_swap(centroids1))
    pc2 = centroids2pointcloud(centroids_zx_swap(centroids2))
    results = pcl.registration.icp_nl(pc2, pc1)
    if not results[0]:
        raise ValueError("point-cloud registration did not converge")
    M = results[1]
    parts = decompose_matrix(M.T)
    angles = parts[2]
    # sanity check that our transform is using same math as pcl did
    nuc1 = centroids_zx_swap(centroids1)
    nuc2 = centroids_zx_swap(transform_centroids(M, centroids2))
    diff = nuc2 - results[2]
    assert diff.min() <= 0.00001, 'Our transform differs from PCL by %s minimum.' % diff.min()
    assert diff.max() <= 0.001, 'Our transform differs from PCL by %s maximum.' % diff.max()
    return M, angles

def dump_registered_file_pair(dstfilenames, parts):
    """Dump registered data.

       Arguments:
         dstfilenames: sequence of two filenames to use for outputs
         parts: sequence of two CMSP 4-tuples of registered data

       The parts should be processed as in results from the register()
       function in this module.
    """
    for filename, parts in zip(dstfilenames, parts):
        c, m, s, p = parts
        dump_segment_info_to_csv(c, m, s, (0,0,0), filename)

def register(nuc_filenames, zyx_grid_scale, syn_filenames=None):
    """Find alignment based on nuclei and return registered data in micron coordinates.

       Arguments:
         nuc_filenames: sequence of two filenames for tpt1, tpt2 nucleic clouds
         zyx_grid_scale: voxel grid spacing as microns per pixel
         syn_filenames: sequence of two filenames for tpt1, tpt2 synaptic clouds or None
       Returns:
         M: alignment matrix
         angles: rotation decomposed from M
         nuc_parts: sequence of two CMSP 4-tuples (centroids, measures, status, saved params)
         syn_parts: sequence of two CMSP 4-tuples if syn_filenames was provided or None
    """
    # load and immediately re-scale into micron coordinates, filtering for only manually classified segments
    nuc1file, nuc2file = nuc_filenames
    nuc1cmsp = load_segment_info_from_csv(nuc1file, zyx_grid_scale, filter_status=(3,7))
    nuc2cmsp = load_segment_info_from_csv(nuc2file, zyx_grid_scale, filter_status=(3,7))

    M, angles = align_centroids(nuc1cmsp[0], nuc2cmsp[0])
    nuc_parts = (nuc1cmsp, (transform_centroids(M, nuc2cmsp[0]),) + nuc2cmsp[1:])

    if syn_filenames is not None:
        syn1file, syn2file = syn_filenames
        syn1cmsp = load_segment_info_from_csv(syn1file, zyx_grid_scale, filter_status=(3,7))
        syn2cmsp = load_segment_info_from_csv(syn2file, zyx_grid_scale, filter_status=(3,7))
        syn_parts = (syn1cmsp, (transform_centroids(M, syn2cmsp[0]),) + syn2cmsp[1:])
    else:
        syn_parts = None

    return (M, angles, nuc_parts, syn_parts)
