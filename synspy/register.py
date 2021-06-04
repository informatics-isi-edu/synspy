#!/usr/bin/python
#
# Copyright 2015-2020 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import os
import sys
import numpy as np
import vtk
import csv
import math
from transformations import decompose_matrix

from .analyze.util import load_segment_info_from_csv, dump_segment_info_to_csv, centroids_zx_swap, transform_centroids

def align_centroids(centroids1, centroids2, maxiters=50):
    """Compute alignment matrix for centroids2 into centroids1 coordinates.

       Arguments:
         centroids1: Nx3 array in Z,Y,X order
         centroids2: Mx3 array in Z,Y,X order

       Returns:
         M: 4x4 transformation matrix
         angles: decomposed rotation vector from M (in radians)
    """
    def make_pc_poly(a):
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()

        for i in range(a.shape[0]):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(points.InsertNextPoint(a[i,:]))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)
        return poly

    def do_icp(src, tgt):
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(src)
        icp.SetTarget(tgt)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(maxiters)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()
        M = icp.GetMatrix()
        return np.array(
            [ [ M.GetElement(i, j) for j in range(4) ] for i in range(4) ],
            dtype=np.float64
        )

    pc1 = make_pc_poly(centroids_zx_swap(centroids1).astype(np.float32))
    pc2 = make_pc_poly(centroids_zx_swap(centroids2).astype(np.float32))
    M = do_icp(pc2, pc1)
    parts = decompose_matrix(M)
    angles = parts[2]
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
