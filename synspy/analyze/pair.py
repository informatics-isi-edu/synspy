
#
# Copyright 2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import math
import numpy as np
from scipy.spatial import cKDTree
from .util import load_registered_csv, x_axis, y_axis, z_axis, matrix_ident, matrix_translate, matrix_scale, matrix_rotate, transform_points, centroids_zx_swap

from deriva.core import urlquote

def nearest_pairs(v1, kdt1, v2, radius, out1, out2):
    """Find nearest k-dimensional point pairs between v1 and v2 and return via output arrays.

       Inputs:
         v1: array with first pointcloud with shape (m, k)
         kdt1: must be cKDTree(v1) for correct function
         v2: array with second pointcloud with shape (m, k)
         radius: maximum euclidean distance between points in a pair
         out1: output adjacency matrix of shape (n,)
         out2: output adjacency matrix of shape (m,)

       Use greedy algorithm to assign nearest neighbors without
       duplication of any point in more than one pair.

       Outputs:
         out1: for each point in kdt1, gives index of paired point from v2 or -1
         out2: for each point in v2, gives index of paired point from v1 or -1

    """
    depth = max(out1.shape[0], out2.shape[0])
    out1[:] = -1
    out2[:] = -1
    dx, pairs = kdt1.query(v2, depth, distance_upper_bound=radius)
    for d in range(depth):
        for idx2 in np.argsort(dx[:,d]):
            if dx[idx2,d] < radius:
                if out2[idx2] == -1 and out1[pairs[idx2,d]] == -1:
                    out2[idx2] = pairs[idx2,d]
                    out1[pairs[idx2,d]] = idx2

def intersection_sweep(v1, v2, w1, w2, radius_seq, dx_weight_ratio=None, weight_ratio_threshold=None):
    """Find intersection and return adjacency matrices.

       Inputs:
         v1: array with shape (n, k) of k-dimensional vertices
         v2: array with shape (m, k) of k-dimensional vertices
         w1: array with shape (n,) weights
         w2: array with shape (m,) weights
         radius: maximum euclidean distance for pairs
         dx_weight_ratio: if not None, weight * dx_weight_ratio forms a pseudo dimension
         weight_ratio_threshold: maximum weight ratio for paired points

       Point sets v1 and v2 are intersected to find pair-mappings
       based on nearest neighbor from each set without duplicates.
       This uses euclidean distance in the k dimensions or in k+1
       dimensions if dx_weight_ratio is not None.

       If dx_weight_ratio is not None, the vectors are extended with
       weights multiplied by this scaling coefficient which converts a
       weight into a pseudo spatial position. This step is performed
       prior to nearest-neighbor search.

       If weight_ratio_threshold is not None, pair candidates are
       disregarded if their intensity ratio is higher than the given
       ratio or lower than the inverse of the given ratio. This step
       is performed after nearest-neighor search.

       Results:
         v1_to_v2: adjacency matrix (n,) containing indices -1 < idx < m
         v2_to_v1: adjacency matrix (m,) containing indices -1 < idx < n

       Each entry v1_to_v2[i] is -1 if the i-th point in v1 is
       unpaired or an index into v2 identifying the paired point.

       Each entry v2_to_v1[j] is -1 if the j-th point in v2 is
       unpaired or an index into v2 identifying the paired point.

    """
    if dx_weight_ratio is not None:
        # convert into k+1 dimensions
        ve1 = np.zeros((v1.shape[0], v1.shape[1]+1), dtype=v1.dtype)
        ve1[:,0:v1.shape[1]] = v1[:,:]
        ve1[:,v1.shape[1]] = w1[:] * dx_weight_ratio
        ve2 = np.zeros((v2.shape[0], v2.shape[1]+1), dtype=v2.dtype)
        ve2[:,0:v2.shape[1]] = v2[:,:]
        ve2[:,v2.shape[1]] = w2[:] * dx_weight_ratio
    else:
        ve1 = v1
        ve2 = v2

    radius_seq = list(radius_seq)

    kdt1 = cKDTree(ve1)
    v1_to_v2 = np.zeros((len(radius_seq), ve1.shape[0]), dtype=np.int32)
    v2_to_v1 = np.zeros((len(radius_seq), ve2.shape[0]), dtype=np.int32)

    for r_idx in range(len(radius_seq)):
        nearest_pairs(ve1, kdt1, ve2, radius_seq[r_idx], v1_to_v2[r_idx,:], v2_to_v1[r_idx,:])
    
    if weight_ratio_threshold is not None:
        # we want to disregard some pairings with extreme intensity ratios
        pair_weights = np.zeros((v1_to_v2.shape[0],2), dtype=np.float32)
        pair_weights[:,0] = (v1_to_v2 >= 0) * w1[:]
        pair_weights[:,1] = (v1_to_v2 >= 0) * w2[(v1_to_v2,) * (v1_to_v2 >= 0)]
        pair_weights[:,:] += 0.0001
        pair_ratios = np.abs(pair_weights[:,0] / pair_weights[:,1])
        v1_to_v2[ np.where((pair_ratios > weight_ratio_threshold) + (pair_ratios < 1/weight_ratio_threshold)) ] = -1
        v2_to_v1[ (v1_to_v2[np.where((pair_ratios > weight_ratio_threshold) + (pair_ratios < 1/weight_ratio_threshold))],) ] = -1

    return v1_to_v2, v2_to_v1

class NucleicPairStudy (object):
    """Local representation of one remote Nucliec Pair Study record.

       WORK IN PROGRESS...

       Basic usage:

           study = NucleicPairStudy.from_study_id(ermrest_catalog, study_id)
           study.retrieve_data(hatrac_store)

       The above will populate instance fields:

           id: the same study_id passed to from_study_id()
           spacing: the ZYX grid spacing
           alignment: the 4x4 transform matrix to align second image to first
           n1, n2: numpy arrays of shape [n_i, k] where k is 8 or 9

       The pointclouds n1, n2 have n records of k float32
       scalars packed in the standard synspy CSV column order:

           Z, Y, X, raw core, raw hollow, DoG core, DoG hollow, (red,)? override
    """
    @classmethod
    def from_study_id(cls, ermrest_catalog, study_id):
        """Instantiate class by finding metadata for a given study_id in ermrest_catalog."""
        r = ermrest_catalog.get(cls.metadata_query_url(study_id))
        r.raise_for_status()
        result = r.json()
        if len(result) != 1:
            raise ValueError('Expected exactly 1 catalog result for %s but found %d.' % (study_id, len(result)))
        return cls(result[0])

    @staticmethod
    def metadata_query_url(study_id):
        """Build ERMrest query URL returning metadata record needed by class."""
        return (
            '/attributegroup/'
            'NPS:=%(nps)s/ID=%(sid)s/'
            'IPS:=(NPS:Study)/'
            'N1:=(NPS:%(n1)s)/'
            'N2:=(NPS:%(n2)s)/'
            'I1:=(N1:%(si)s)/'
            '$NPS/'
            '*;'
            'I1:%(zs)s,'
            'IPS:Alignment,'
            'n1:=N1:%(sfu)s,'
            'n2:=N2:%(sfu)s,'
        ) % {
            'sid': urlquote(study_id),
            'nps': urlquote('Nucleic Pair Study'),
            'n1': urlquote('Nucleic Region 1'),
            'n2': urlquote('Nucleic Region 2'),
            'si': urlquote('Source Image'),
            'zs': urlquote('ZYX Spacing'),
            'sfu': urlquote('Segments Filtered URL'),
        }

    def __init__(self, metadata):
        """Instantiate with record metadata retrieved from study and related entities.

           The exact format of metadata is an implementation detail
           and will be produced by the metadata_query_url(study_id)
           class-method.

        """
        self._metadata = metadata
        self.id = self._metadata['ID']
        self.spacing = self._metadata['ZYX Spacing']
        self.alignment = self._metadata['Alignment']

    def retrieve_data(self, hatrac_store):
        """Download raw CSV pointcloud data from Hatrac object store and register it.

           Registered pointclouds are saved to self.n1, self.n2
        """
        raise NotImplementedError()

    def nuc_pairing_maps(self, max_dx_seq=(4.0,), dx_w_ratio=None, max_w_ratio=None):
        """Return (n1_to_n2, n2_to_n1) adjacency matrices after pairing search.

           Arguments:
               max_dx_seq: sequence of x maximum distance thresholds
               dx_w_ratio: scaling coefficient to convert weights into 4th dimension (or None to use 3D points)
               max_w_ratio: weight ratio threshold to discard pairs with widely different weights

           Result:
               n1_to_n2: shape of (x, self.n1.shape[0],) adjacency matrix containing indices of self.n2 or -1
               n2_to_n1: shape of (x, self.n2.shape[0],) adjacency matrix containing indices of self.n1 or -1

           The output slices a[i,:] is an adjacency matrix for the ith element of max_dx_seq.
        """
        return intersection_sweep(
            self.n1[:,0:3],
            self.n2[:,0:3],
            self.n1[:,3],
            self.n2[:,3],
            max_dx_seq,
            dx_w_ratio,
            max_w_ratio
        )

    def get_unpaired(self, adjacency, points):
        """Get subset of points which are unpaired in adjacency matrix.

           Arguments:
              adjacency: adjacency matrix with shape (N,)
              points: pointcloud array with shape (N, k)

           Results:
              array with shape (N-P, k) if there are P pairs in adjacency matrix
        """
        return points[np.where(adjacency == -1) + (slice(None),)]

    def get_pairs(self, adjacency, points1, points2):
        """Get subset of paired points which are paired in adjacency matrix.

           Arguments:
              adjacency: adjacency matrix with shape (N,)
              points1: pointcloud array with shape (N, k)
              points2: pointclouds array with shape (M, k)

           The range of possible indices in adjacency is -1 < i < M.

           Results: (a1, a2)
              a1: shape (P, k) drawn from points1
              a2: shape (P, k) drawn from points2
        """
        paired_idxs = np.where(adjacency >= 0)[0]
        return (
            points1[(paired_idxs, slice(None))],
            points2[(adjacency[(paired_idxs,)], slice(None))]
        )

    @classmethod
    def get_alignment(cls, a0, a1):
        """Compute alignment matrix to fit a1 into a0 coordinates.

           Input point coordinates are for corresponding anatomical
           points i=0..2 and dimensions d=0..2 in ZYX order.

           Arguments:
             a0: coordinates array shaped (i,d)
             a1: coordinates array shaped (i,d)

           Alignment is determined in this order:
             1. Scale
                a1[1,:]-a1[0,:] length same as a0[1,:]-a0[0,:]
             2. Translate
                a1[0,:] colocated with a0[0,:]
             3. Rotate about point a1[0,:]
                a1[1,:] colocated with a0[1,:]
             4. Rotate about line a1[0,:]..a1[1,:]
                a1[2,:] on line a0[2,:]..a0[0,:]

           Results:
              m: 4x4 matrix
        """
        pass

class SynapticPairStudy (NucleicPairStudy):
    """Local representation of one remote Synaptic Pair Study record.

       Basic usage:

           study = SynapticPairStudy.from_study_id(ermrest_catalog, study_id)
           study.retrieve_data(hatrac_store)

       The above will populate instance fields:

           id: the same study_id passed to from_study_id()
           spacing: the ZYX grid spacing
           alignment: the 4x4 transform matrix to align second image to first
           n1, n2, s1, s2: numpy arrays of shape [n_i, k] where k is 8 or 9

       The pointclouds n1, n2, s1, s2 have n records of k float32
       scalars packed in the standard synspy CSV column order:

           Z, Y, X, raw core, raw hollow, DoG core, DoG hollow, (red,)? override

    """
    @staticmethod
    def metadata_query_url(study_id):
        """Build ERMrest query URL returning metadata record needed by class."""
        return (
            '/attributegroup/'
            'SPS:=%(sps)s/ID=%(sid)s/'
            'IPS:=(SPS:Study)/'
            'S1:=(SPS:%(s1)s)/'
            'S2:=(SPS:%(s2)s)/'
            'N1:=(IPS:%(n1)s)/'
            'N2:=(IPS:%(n2)s)/'
            'I1:=(N1:%(si)s)/'
            '$SPS/'
            '*;'
            'I1:%(zs)s,'
            'IPS:Alignment,'
            'n1:=IPS:%(r1u)s,'
            'n2:=IPS:%(r2u)s,'
            's1:=SPS:%(r1u)s,'
            's2:=SPS:%(r2u)s'
        ) % {
            'sid': urlquote(study_id),
            'sps': urlquote('Synaptic Pair Study'),
            's1': urlquote('Synaptic Region 1'),
            's2': urlquote('Synaptic Region 2'),
            'n1': urlquote('Nucleic Region 1'),
            'n2': urlquote('Nucleic Region 2'),
            'si': urlquote('Source Image'),
            'zs': urlquote('ZYX Spacing'),
            'r1u': urlquote('Region 1 URL'),
            'r2u': urlquote('Region 2 URL'),
        }

    def retrieve_data(self, hatrac_store):
        """Download registered CSV pointcloud data from Hatrac object store.

           Pointclouds are saved to self.n1, self.n2, self.s1, self.s2
        """
        self.n1 = load_registered_csv(hatrac_store, self._metadata['n1'])
        self.n2 = load_registered_csv(hatrac_store, self._metadata['n2'])
        self.s1 = load_registered_csv(hatrac_store, self._metadata['s1'])
        self.s2 = load_registered_csv(hatrac_store, self._metadata['s2'])

    def syn_pairing_maps(self, max_dx_seq=(4.0,), dx_w_ratio=None, max_w_ratio=None):
        """Return (s1_to_s2, s2_to_s1) adjacency matrices after pairing search.

           Arguments:
               max_dx_seq: sequence of x maximum distance thresholds
               dx_w_ratio: scaling coefficient to convert weights into 4th dimension (or None to use 3D points)
               max_w_ratio: weight ratio threshold to discard pairs with widely different weights

           Result:
               s1_to_s2: shape of (x, self.s1.shape[0],) adjacency matrix containing indices of self.s2 or -1
               s2_to_s1: shape of (x, self.s2.shape[0],) adjacency matrix containing indices of self.s1 or -1

           The output slices a[i,:] is an adjacency matrix for the ith element of max_dx_seq.
        """
        return intersection_sweep(
            self.s1[:,0:3],
            self.s2[:,0:3],
            self.s1[:,3],
            self.s2[:,3],
            max_dx_seq,
            dx_w_ratio,
            max_w_ratio
        )

def gross_unit_alignment(xyz_triple):
    """Return transformation to convert XYZ points into unit space.

       This is a gross anatomical alignment based on three reference
       points chosen consistently in two images.

       Arguments:
          xyz_triple[0,:]: P0
          xyz_triple[1,:]: P1
          xyz_triple[2,:]: P2

       Alignment:
          1. P0 will be translated to origin (0,0,0)
          2. P2 will be aligned to Y-axis unit vector (0,1,0) via scale+rotate
          3. P1 will be aligned into Z=0 plane via roll around Y-axis

       Results:
          M: 4x4 transform matrix
          length: micron per unit distance for this image

    """
    v = xyz_triple

    # scale to have unit length P2-P0
    length = np.linalg.norm(v[2,:]-v[0,:])
    Ms = matrix_scale(1./length)
    v = transform_points(Ms, v, np.float64)

    # translate to have P0 as origin
    Mt = matrix_translate(0 - v[0,:])
    v = transform_points(Mt, v, np.float64)

    # rotate to align P2 to (0,1,0) on Y axis
    Mr1 = matrix_rotate(
        # axis perpindicular to plane
        np.cross(y_axis, v[2,:]),
        # rotate by angle within plane
        math.acos(
            np.inner(y_axis, v[2,:])
            / (np.linalg.norm(y_axis) * np.linalg.norm(v[2,:]))
        )
    )
    v = transform_points(Mr1, v, np.float64)

    # roll on Y axis to place P1 into Z=0 plane
    u1 = np.cross(y_axis, x_axis) # use X-axis as reference for XY plane
    u2 = np.cross(y_axis, v[1,:]) # find comparable vector
    roll = math.acos(
        np.inner(u1, u2)
        / (np.linalg.norm(u1) * np.linalg.norm(u2))
    )
    if v[1,0] > 0:
        if v[1,2] >= 0:
            roll = 0 - roll
    else:
        raise NotImplementedError('P1 outside of expected right-handed semi-space during final roll calculation!')
    Mr2 = matrix_rotate(y_axis, roll)
    #Mr2 = matrix_ident()
    v = transform_points(Mr2, v, np.float64)

    # compose stacked transforms as single matrix
    M = np.matmul(np.matmul(np.matmul(Ms, Mt), Mr1), Mr2)
    return M, length

class ImageGrossAlignment (object):
    """Local representation of one image and its alignment data.

       WORK IN PROGRESS...

    """
    @classmethod
    def from_image_id(cls, ermrest_catalog, image_id):
        """Instantiate class by finding metadata for a given image_id in ermrest_catalog."""
        r = ermrest_catalog.get(cls.metadata_query_url(image_id))
        r.raise_for_status()
        result = r.json()
        if len(result) != 1:
            raise ValueError('Expected exactly 1 catalog result for %s but found %d.' % (image_id, len(result)))
        return cls(result[0])

    @staticmethod
    def metadata_query_url(image_id):
        """Build ERMrest query URL returning metadata record needed by class."""
        return (
            '/attributegroup/'
            'I:=Zebrafish:Image/ID=%(id)s/'
            '*'
        ) % {
            'id': urlquote(image_id),
        }

    def __init__(self, metadata):
        """Instantiate with record metadata retrieved from image and related entities.

           The exact format of metadata is an implementation detail
           and will be produced by the metadata_query_url(id)
           class-method.

        """
        self._metadata = metadata
        self.id = self._metadata['ID']

        grid_zyx = np.array([0.4, 0.26, 0.26], dtype=np.float32)
        def get_align_coord(colname, axis):
            p = self._metadata[colname]
            if isinstance(p, dict):
                if axis in p:
                    return p[axis]
                else:
                    raise ValueError('"%s" lacks field "%s"' % (colname, axis))
            else:
                raise ValueError('"%s" should be an object, not %s' % (colname, type(p)))

        self.alignment_points_xyz = centroids_zx_swap(
            np.array(
                [
                    [
                        get_align_coord(colname, axis)
                        for axis in ['z', 'y', 'x']
                    ]
                    for colname in ['Align P0 ZYX', 'Align P1 ZYX', 'Align P2 ZYX']
                ],
                dtype=np.float32
            ) * grid_zyx
        )

        self.M, self.length = gross_unit_alignment(self.alignment_points_xyz)
