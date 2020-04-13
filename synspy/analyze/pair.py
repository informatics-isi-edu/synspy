
#
# Copyright 2017-2018 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import sys
import math
import numpy as np
from scipy.spatial import cKDTree
from .util import load_registered_csv, load_registered_npz, load_segment_status_from_csv, x_axis, y_axis, z_axis, matrix_ident, matrix_translate, matrix_scale, matrix_rotate, transform_points, transform_centroids, centroids_zx_swap

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
    depth = min(max(out1.shape[0], out2.shape[0]), 100)
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
            'NPS:=%(nps)s/ID=%(sid)s;RID=%(sid)s/'
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
            'SPS:=%(sps)s/ID=%(sid)s;RID=%(sid)s/'
            'IPS:=%(ips)s/'
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
            's2:=SPS:%(r2u)s,'
            's1raw:=S1:%(sfu)s,'
            's2raw:=S2:%(sfu)s,'
            's1box:=S1:%(slice)s,'
            's2box:=S2:%(slice)s,'
            's1n:=S1:%(nu)s,'
            's2n:=S2:%(nu)s'
        ) % {
            'sid': urlquote(study_id),
            'sps': urlquote('Synaptic Pair Study'),
            'ips': urlquote('Image Pair Study'),
            'sfu': urlquote('Segments Filtered URL'),
            's1': urlquote('Synaptic Region 1'),
            's2': urlquote('Synaptic Region 2'),
            'n1': urlquote('Nucleic Region 1'),
            'n2': urlquote('Nucleic Region 2'),
            'si': urlquote('Source Image'),
            'zs': urlquote('ZYX Spacing'),
            'r1u': urlquote('Region 1 URL'),
            'r2u': urlquote('Region 2 URL'),
            'slice': urlquote('ZYX Slice'),
            'nu': urlquote('Npz URL'),
        }

    def retrieve_data(self, hatrac_store, classifier_override=None, use_intersect=False, cache_dir=None):
        """Download registered CSV pointcloud data from Hatrac object store.

           Arguments:
             hatrac_store: Instance of HatracStore from which to retrieve files
             classifier_override: Dictionary of override parameters

           Pointclouds are saved to self.n1, self.n2, self.s1, self.s2
        """
        self.n1 = load_registered_csv(hatrac_store, self._metadata['n1'])
        self.n2 = load_registered_csv(hatrac_store, self._metadata['n2'])

        s1raw, s2raw = self._metadata['s1raw'], self._metadata['s2raw']
        self.s1 = load_registered_npz(hatrac_store, self._metadata['s1n'], None, s1raw, cache_dir)
        self.s2 = load_registered_npz(hatrac_store, self._metadata['s2n'], self._metadata['Alignment'], s2raw, cache_dir)

        if classifier_override is None:
            def prune(centroids):
                cond = centroids[:,-1] == 7
                return centroids[np.nonzero(cond)[0],:]
            self.s1 = prune(self.s1)
            self.s2 = prune(self.s2)

        if use_intersect:
            # conservative border padding (rounded up slightly)
            zyx_pad = np.array((16, 16, 16), dtype=np.float32)
            zyx_scale = np.array((0.4, 0.26, 0.26), dtype=np.float32)

            def bbox(zyxslice):
                res = np.zeros((2,3), dtype=np.float32)
                aslices = zyxslice.split(',')
                for a in range(3):
                    l, u = aslices[a].split(':')
                    res[0,a] = float(l) if l != '' else 0.
                    res[1,a] = float(u) if u != '' else 2048.
                res[0,:] = res[0,:] + zyx_pad
                res[1,:] = res[1,:] - zyx_pad
                res = res * zyx_scale
                return res

            # use ZYX Slice metadata to determine ROI
            bbox1 = bbox(self._metadata['s1box'])
            bbox2 = bbox(self._metadata['s2box'])

            # we need tpts in opposing coordinate space to complete intersection test
            Minv = np.linalg.inv(np.array(self._metadata['Alignment']))
            s1inv = transform_centroids(Minv, self.s1[:,0:3])
            s2inv = transform_centroids(Minv, self.s2[:,0:3])

            def clip(a1, a2, bbox1, bbox2):
                assert a1.shape[0] == a2.shape[0]
                # re-clip in both spaces for better consistency
                cond = np.all(a1[:,0:3] >= bbox1[0,:], axis=1) \
                    & np.all(a1[:,0:3] < bbox1[1,:], axis=1) \
                    & np.all(a2[:,0:3] >= bbox2[0,:], axis=1) \
                    & np.all(a2[:,0:3] < bbox2[1,:], axis=1)
                print('Retaining %d/%d intersecting centroids' % (cond.sum(), a1.shape[0]))
                return a1[np.nonzero(cond)[0],:]

            # clip by both bboxes to find intersection
            self.s1 = clip(self.s1, s1inv, bbox1, bbox2)
            self.s2 = clip(self.s2, s2inv, bbox1, bbox2)

        if classifier_override is not None:
            if not isinstance(classifier_override, dict):
                raise ValueError('Classifier override must be a dictionary or null')

            if 'cmin' in classifier_override:
                def cmin(a, q):
                    cond = ((a[:,3] - a[:,4]) / a[:,4]) > q
                    print('Retaining %d/%d centroids for cmin=%f criteria' % (cond.sum(), a.shape[0], q))
                    a2 = a[np.nonzero(cond)[0],:]
                    return a2
                q = np.float32(classifier_override['cmin'])
                self.s1 = cmin(self.s1, q)
                self.s2 = cmin(self.s2, q)

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

def gross_unit_alignment(origin_xyz, yunit_xyz, zinterc_xyz):
    """Return transformation to convert XYZ points into unit space.

       This is a gross anatomical alignment based on three reference
       points chosen consistently in two images.

       Arguments:
          origin_xyz: will become origin (0,0,0)
          yunit_xyz: will become (0,1,0) to form unit vector on Y axis
          zinterc_xyz: will become Z=0 intercept

       Results:
          M: 4x4 transform matrix  image microns -> unit space
          M_inv: 4x4 transform matrix  unit space -> image microns
          length: micron per unit distance for this image

    """
    # scale to have unit length
    length = np.linalg.norm(yunit_xyz-origin_xyz)
    Ms = matrix_scale(1./length)
    Ms_inv = matrix_scale(length/1.)
    origin_xyz, yunit_xyz, zinterc_xyz = transform_points(
        Ms,
        np.stack((origin_xyz, yunit_xyz, zinterc_xyz)),
        np.float64
    )

    # translate to origin
    Mt = matrix_translate(0 - origin_xyz)
    Mt_inv = matrix_translate(0 + origin_xyz)
    origin_xyz, yunit_xyz, zinterc_xyz = transform_points(
        Mt,
        np.stack([origin_xyz, yunit_xyz, zinterc_xyz]),
        np.float64
    )

    # rotate unit vector onto Y axis
    Mr1_axis = np.cross(y_axis, yunit_xyz)
    Mr1_angle = math.acos(
        np.inner(y_axis, yunit_xyz)
        / (np.linalg.norm(y_axis) * np.linalg.norm(yunit_xyz))
    )
    if np.linalg.norm(Mr1_axis) > 0:
        Mr1 = matrix_rotate(Mr1_axis, Mr1_angle)
        Mr1_inv = matrix_rotate(Mr1_axis, 0 - Mr1_angle)
    else:
        Mr1 = matrix_ident()
        Mr1_inv = matrix_ident()
    origin_xyz, yunit_xyz, zinterc_xyz = transform_points(
        Mr1,
        np.stack([origin_xyz, yunit_xyz, zinterc_xyz]),
        np.float64
    )

    # roll on Y axis to set Z=0 intercept
    Mr2_axis = y_axis
    zinterc_in_xz = zinterc_xyz.copy()
    zinterc_in_xz[1] = 0
    Mr2_angle = math.acos(
        np.inner(x_axis, zinterc_in_xz)
        / (np.linalg.norm(x_axis) * np.linalg.norm(zinterc_in_xz))
    )
    xpos = 1 if zinterc_in_xz[0] >= 0 else -1
    zpos = 1 if zinterc_in_xz[2] >= 0 else -1
    if (xpos*zpos) < 0:
        Mr2_angle = 0 - Mr2_angle
    Mr2 = matrix_rotate(Mr2_axis, Mr2_angle)
    Mr2_inv = matrix_rotate(Mr2_axis, 0 - Mr2_angle)
    origin_xyz, yunit_xyz, zinterc_xyz = transform_points(
        Mr2,
        np.stack([origin_xyz, yunit_xyz, zinterc_xyz]),
        np.float64
    )

    # compose stacked transforms as single matrix
    M_inv = np.matmul(np.matmul(np.matmul(Mr2_inv, Mr1_inv), Mt_inv), Ms_inv)
    M = np.matmul(np.matmul(np.matmul(Ms, Mt), Mr1), Mr2)
    return M, M_inv, length

class ImageGrossAlignment (object):
    """Local representation of one image and its alignment data.

    Several computed properties are made available, with priority
    going to explicit alignment matrices stored in the catalog.

    """
    @classmethod
    def from_image_id(cls, ermrest_catalog, image_id, disable_gross_align=False):
        """Instantiate class by finding metadata for a given image_id in ermrest_catalog.

        :param ermrest_catalog: an ErmrestCatalog instance to use for metadata queries
        :param image_id: an ID or RID value to locate one record from the Image table

        """
        r = ermrest_catalog.get(cls.metadata_query_url(image_id))
        r.raise_for_status()
        result = r.json()
        if len(result) != 1:
            raise ValueError('Expected exactly 1 catalog result for %s but found %d.' % (image_id, len(result)))
        return cls(result[0], disable_gross_align=disable_gross_align)

    @staticmethod
    def metadata_query_url(image_id):
        """Build ERMrest query URL returning metadata record needed by class."""
        return (
            '/attributegroup'
            '/I:=Zebrafish:Image/ID=%(id)s;RID=%(id)s'
            '/AS1:=left(I:Alignment%%20Standard)=(Zebrafish:Alignment%%20Standard:RID)'
            '/ASI1:=left(AS1:Image)=(Zebrafish:Image:RID)'
            '/AS2:=left(ASI1:Alignment%%20Standard)=(Zebrafish:Alignment%%20Standard:RID)'
            '/ASI2:=left(AS2:Image)=(Zebrafish:Image:RID)'
            '/$I'
            '/*'
            ';ASI1_obj:=array(ASI1:*)'
            ',AS1_obj:=array(AS1:*)'
            ',ASI2_obj:=array(ASI2:*)'
            ',AS2_obj:=array(AS2:*)'
        ) % {
            'id': urlquote(image_id),
        }

    def __init__(self, metadata, swap_p1_p2=False, disable_gross_align=False):
        """Instantiate with record metadata retrieved from image and related entities.

        :param metadata: single row result from metadata_query_url(image_id)
        :param swap_p1_p2: align Y-axes to P0->P2 vector if True, else P0->P1 (default)
        :param disable_gross_align: fallback to 3-point canonical alignment when False (default)

        """
        self._metadata = metadata
        self.disable_gross_align = disable_gross_align

        grid_zyx = np.array([0.4, 0.26, 0.26], dtype=np.float64)
        def get_align_coord(colname, axis):
            p = self._metadata[colname]
            if isinstance(p, dict):
                if axis in p:
                    return p[axis]
                else:
                    raise ValueError('"%s" lacks field "%s"' % (colname, axis))
            else:
                raise ValueError('"%s" should be an object, not %s' % (colname, type(p)))

        if disable_gross_align:
            return

        p0, p1, p2 = centroids_zx_swap(
            np.array(
                [
                    [
                        get_align_coord(colname, axis)
                        for axis in ['z', 'y', 'x']
                    ]
                    for colname in ['Align P0 ZYX', 'Align P1 ZYX', 'Align P2 ZYX']
                ],
                #  Using np.float32 causes errors on some platforms, so use float64
                dtype=np.float64
            ) * grid_zyx
        )
        self.alignment_points_xyz = np.stack((p0, p1, p2))

        self.swap_p1_p2 = swap_p1_p2
        if swap_p1_p2:
            p1, p2 = p2, p1

        self._M, self._M_inv, self.length = gross_unit_alignment(p0, p1, p2)

    @property
    def RID(self):
        """The RID column of this Image record."""
        return self._metadata['RID']

    def _coalesce_first(self, k):
        a = self._metadata[k]
        if a is not None:
            return a[0]

    @property
    def alignment_standard(self):
        """Alignment Standard record content for this Image, or None."""
        return self._coalesce_first('AS1_obj')

    @property
    def alignment_standard_image(self):
        """Image record content for self.alignment_standard, or None."""
        return self._coalesce_first('ASI1_obj')

    @property
    def alignment_depth(self):
        """Number of hops of Alignment Standard for this image.

           Possible values:
           0: Alignment Standard is not configured
           1: Alignment Standard is canonical
           2: Alignment Standard uses one intermediate image

           Deeper chains are not currently supported and will raise a ValueError.
        """
        if self.alignment_standard is None:
            return 0
        elif self.alignment_standard_image["Alignment Standard"] is None:
            return 1
        ASI2 = self._coalesce_first('ASI2_obj')
        if ASI2["Alignment Standard"] is not None:
            raise ValueError("Alignment Standard %s is non-canonical." % AS2["RID"])
        return 2

    @property
    def has_standard(self):
        """True if self references an Alignment Standard."""
        return self.alignment_depth > 0

    @property
    def canonical_alignment_standard(self):
        try:
            return {
                1: self._coalesce_first('AS1_obj'),
                2: self._coalesce_first('AS2_obj'),
            }[self.alignment_depth]
        except KeyError:
            raise ValueError('Unexpected alignment depth %s' % self.alignment_depth)

    @property
    def canonical_alignment_standard_image(self):
        try:
            return {
                1: self._coalesce_first('ASI1_obj'),
                2: self._coalesce_first('ASI2_obj'),
            }[self.alignment_depth]
        except KeyError:
            raise ValueError('Unexpected alignment depth %s' % self.alignment_depth)

    @property
    def M(self):
        """4x4 transform matrix to move image microns into unit space."""
        return self._M

    @property
    def M_inv(self):
        """4x4 (inverse) transform matrix to move unit space into image microns."""
        return self._M_inv

    @property
    def M_canonical(self):
        """4x4 transform matrix to move image microns into canonical micron space.

           If an explicit "Canonical Alignment" field is populated in
           this Image record in the catalog, that matrix is
           returned. Otherwise, an alignment is computed via the
           "Alignment Standard" which must itself then be a canonical
           alignment.

        """
        # return stored alignment, if present
        if self._metadata['Canonical Alignment']:
            return np.array(self._metadata['Canonical Alignment'], dtype=np.float64)

        if self.alignment_depth == 1 and self._metadata['Alignment']:
            sys.stderr.write('using alignment matrix for image %s -> %s\n' % (
                self.RID,
                self.alignment_standard_image['RID'],
            ))
            return np.array(self._metadata['Alignment'], dtype=np.float64)

        if self.alignment_depth == 2 and self._metadata['Alignment'] \
           and self.alignment_standard_image['Alignment']:
            M0 = np.array(self._metadata['Alignment'], dtype=np.float64)
            M1 = np.array(self.alignment_standard_image['Alignment'], dtype=np.float64)
            sys.stderr.write('using compound alignment matrices for image %s -> %s -> %s\n' % (
                self.RID,
                self.alignment_standard_image['RID'],
                self.canonical_alignment_standard_image['RID'],
            ))
            return np.matmul(M0, M1)

        if self.disable_gross_align:
            raise ValueError('canonical alignment not available for Image %s' % self.RID)

        # compute alignment
        metadata = dict(self.canonical_alignment_standard_image)
        metadata.update({
            'AS1_obj': None,
            'ASI1_obj': None,
            'AS2_obj': None,
            'ASI2_obj': None,
        })
        standard = ImageGrossAlignment(metadata, self.swap_p1_p2)
        sys.stderr.write('using 3-point alignment for image %s -> %s\n' % (
            self.RID,
            standard.RID,
        ))
        return np.matmul(self.M, standard.M_inv)

    @property
    def M_canonical_inv(self):
        """4x4 (inverse) transform matrix to canonical microns into image micron space.

           If an explicit "Canonical Alignment" field is populated in
           this Image record in the catalog, the inverse of that
           matrix is computed. Otherwise, an inverted alignment is
           computed via the "Alignment Standard" which must itself
           then be a canonical alignment.

        """
        # return invert of stored alignment, if present
        if self._metadata['Canonical Alignment']:
            return np.linalg.inv(np.array(self._metadata['Canonical Alignment'], dtype=np.float64))

        if self.alignment_depth == 1 and self._metadata['Alignment']:
            return np.linalg.inv(np.array(self._metadata['Alignment'], dtype=np.float64))

        if self.alignment_depth == 2 and self._metadata['Alignment'] \
           and self.alignment_standard_image['Alignment']:
            M0 = np.array(self._metadata['Alignment'], dtype=np.float64)
            M1 = np.array(self.alignment_standard_image['Alignment'], dtype=np.float64)
            return np.linalg.inv(np.matmul(M0, M1))

        if self.disable_gross_align:
            raise ValueError('canonical alignment not available for Image %s' % self.RID)

        # compute inverted alignment
        metadata = dict(self.canonical_alignment_standard_image)
        metadata.update({
            'AS1_obj': None,
            'ASI1_obj': None,
            'AS2_obj': None,
            'ASI2_obj': None,
        })
        standard = ImageGrossAlignment(metadata, self.swap_p1_p2)
        return np.matmul(standard.M, self.M_inv)
