
#
# Copyright 2014-2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import numpy as np

import datetime

import os
import sys
import math
import csv
import re
import atexit

import volspy.viewer as base
from vispy import app, gloo, visuals

from synspy.analyze.block import BlockedAnalyzerOpt, assign_voxels_opt, compose_3d_kernel, gaussian_kernel, batch_analyze, get_mode_and_footprints
from synspy.analyze.util import load_segment_status_from_csv, dump_segment_info_to_csv

import tifffile
        
_color_uniforms = """
uniform sampler3D u_voxel_class_texture;
uniform sampler3D u_measures_texture;
uniform sampler3D u_status_texture;
uniform int u_numchannels;
uniform float u_gain;
uniform float u_floorlvl;
uniform float u_nuclvl;
uniform float u_msklvl;
uniform float u_zerlvl;
uniform float u_toplvl;
uniform float u_transp;
"""

# R: raw signal
# G: synapse core samples
# B: synapse vicinity samples
# A: auto-fluorescence signal and synapse mask sample


# centroid status values are bit-mapped uint8 using 3 bits
# bit 0: 1 means override, 0 means default class
# bit 1: 1 means synapse, 0 means non-synapse
# bit 2: 1 means clickable, 0 means non-clickable
#
# value 0: default class
# value 1: override non-synapse (non-clickable)
# value 3: override synapse     (non-clickable)
# value 5: override non-synapse (clickable)
# value 7: override synapse     (clickable)
#
# click cycle values: 0 -> 5 -> 7 -> 0

_segment_colorxfer = """
{
    vec4 segment_id;
    float segment_status;

    col_smp = vec4(0,0,0,0);

    // lookup voxel's packed segment ID
    segment_id = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(segment_id.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, segment_id.rgb);
       segment_status = texture3D(u_status_texture, segment_id.rgb).r;

       if ( (segment_status*255) == 5 || (segment_status*255) == 7 ) {
          // segment is clickable and overridden
          col_smp = segment_id;
       }
       else if (col_packed_smp.g > u_nuclvl) { /* pass */ }
       else if (col_packed_smp.r < u_floorlvl) { /* pass */ }
       else if (col_packed_smp.b > u_msklvl) { /* pass */ }
       else if ( (segment_status*255) == 1 || (segment_status*255) == 3 ) { /* pass */ }
       else {
          // segment is clickable default
          col_smp = segment_id;
       }
    }
}

"""

_segment_alpha = """
"""

_linear1_grayxfer1 = """
{
    vec4 segment_id;
    float segment_status;
    float S;

    S = col_smp.r;

    // lookup voxel's packed segment ID
    segment_id = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(segment_id.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, segment_id.rgb);
       segment_status = texture3D(u_status_texture, segment_id.rgb).r;

       if (all(equal(u_picked, segment_id))) {
          // segment is picked via mouse-over

          if ((segment_status*255) == 5) {
             // segment is forced off, clickable
             col_smp.rgb = vec3(0,1,1);
          }
          else if ((segment_status*255) == 7) {
             // segment is forced on, clickable
             col_smp.rgb = vec3(1,1,0);
          }
          else {
             // segment is default, clickable
             col_smp.rgb = vec3(0,1,0);
          }
       }
       else {
          col_smp.rgb = vec3(S);
       }
    }
    else {
       col_smp.rgb = vec3(S);
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
}
"""

_linear1_grayxfer2 = """
{
    vec4 segment_id;
    float segment_status;
    float S;

    S = col_smp.r;

    // lookup voxel's packed segment ID
    segment_id = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(segment_id.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, segment_id.rgb);
       segment_status = texture3D(u_status_texture, segment_id.rgb).r;

       if (all(equal(u_picked, segment_id))) {
          // segment is picked via mouse-over

          if ((segment_status*255) == 5) {
             // segment is forced off, clickable
             col_smp.rgb = vec3(0,1,1);
          }
          else if ((segment_status*255) == 7) {
             // segment is forced on, clickable
             col_smp.rgb = vec3(1,1,0);
          }
          else {
             // segment is default, clickable
             col_smp.rgb = vec3(0,1,0);
          }
       }
       else if ((segment_status*255) == 7 || (segment_status*255) == 3) {
          // segment is forced on
          col_smp.rgb = vec3(S,S,0);
       }
       else {
          col_smp.rgb = vec3(S);
       }
    }
    else {
       col_smp.rgb = vec3(S);
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
}
"""
_linear1_sparse_gray = """
{
    vec4 segment_id;
    float segment_status;
    float S;

    S = col_smp.r;

    // lookup voxel's packed segment ID
    segment_id = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(segment_id.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, segment_id.rgb);
       segment_status = texture3D(u_status_texture, segment_id.rgb).r;

       if (all(equal(u_picked, segment_id))) {
          // segment is picked via mouse-over

          if ((segment_status*255) == 5) {
             // segment is forced off, clickable
             col_smp.rgb = vec3(0,1,1);
          }
          else if ((segment_status*255) == 7) {
             // segment is forced on, clickable
             col_smp.rgb = vec3(1,1,0);
          }
          else {
             // segment is default, clickable
             col_smp.rgb = vec3(0,1,0);
          }
       }
       else if ((segment_status*255) == 7 || (segment_status*255) == 3) {
          // segment is forced on
          col_smp.rgb = vec3(S,S,S);
       }
       else {
          col_smp.rgb = vec3(0);
       }
    }
    else {
       col_smp.rgb = vec3(0);
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
}
"""
_linear1_sparse_yellow = """
{
    vec4 segment_id;
    float segment_status;
    float S;

    S = col_smp.r;

    // lookup voxel's packed segment ID
    segment_id = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(segment_id.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, segment_id.rgb);
       segment_status = texture3D(u_status_texture, segment_id.rgb).r;

       if (all(equal(u_picked, segment_id))) {
          // segment is picked via mouse-over

          if ((segment_status*255) == 5) {
             // segment is forced off, clickable
             col_smp.rgb = vec3(0,1,1);
          }
          else if ((segment_status*255) == 7) {
             // segment is forced on, clickable
             col_smp.rgb = vec3(1,1,0);
          }
          else {
             // segment is default, clickable
             col_smp.rgb = vec3(0,1,0);
          }
       }
       else if ((segment_status*255) == 7 || (segment_status*255) == 3) {
          // segment is forced on
          col_smp.rgb = vec3(S,S,0);
       }
       else {
          col_smp.rgb = vec3(0);
       }
    }
    else {
       col_smp.rgb = vec3(0);
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
}
"""

_linear1_colorxfer = """
{
    vec4 segment_id;
    float segment_status;
    float S;

    S = col_smp.r;

    // lookup voxel's packed segment ID
    segment_id = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(segment_id.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, segment_id.rgb);
       segment_status = texture3D(u_status_texture, segment_id.rgb).r;

       if (all(equal(u_picked, segment_id))) {
          // segment is picked via mouse-over

          if ((segment_status*255) == 5) {
             // segment is forced off, clickable
             col_smp.rgb = vec3(0,1,1);
          }
          else if ((segment_status*255) == 7) {
             // segment is forced on, clickable
             col_smp.rgb = vec3(1);
          }
          else {
             // segment is default, clickable
             col_smp.rgb = vec3(1,1,0);
          }
       }
       else if ((segment_status*255) == 5 || (segment_status*255) == 1) {
          // segment is forced off
          col_smp.rgb = vec3(0,S,S);
       }
       else if ((segment_status*255) == 7 || (segment_status*255) == 3) {
          // segment is forced on
          col_smp.rgb = vec3(S);
       }
       else if (col_packed_smp.g > u_nuclvl) { 
          col_smp.rgb = vec3(0,S,0);
       }
       else if (col_packed_smp.r < u_floorlvl) { 
          col_smp.rgb = vec3(0,S,0);
       }
       else if (col_packed_smp.b > u_msklvl) {
          // segment red over threshold
          col_smp.rgb = vec3(S,0,0);
       }
       else {
          // segment syn and vcn within range
          col_smp.rgb = vec3(S,S,0);
       }
    }
    else {
       col_smp.rgb = vec3(0,S,0);
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
}
"""

_linear_alpha = """
   col_smp.a = max(max(col_smp.r, col_smp.g), col_smp.b) * u_transp;
"""

_binary1_colorxfer = """
{
    vec4 segment_id;
    float segment_status;
    float S;

    S = col_smp.r;

    // lookup voxel's packed segment ID
    segment_id = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(segment_id.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, segment_id.rgb);
       segment_status = texture3D(u_status_texture, segment_id.rgb).r;
       if (all(equal(u_picked, segment_id))) {
          // segment is picked via mouse-over

          if ((segment_status*255) == 5) {
             // segment is forced off, clickable
             col_smp.rgb = vec3(0,1,1);
          }
          else if ((segment_status*255) == 7) {
             // segment is forced on, clickable
             col_smp.rgb = vec3(1);
          }
          else {
             // segment is default, clickable
             col_smp.rgb = vec3(1,1,0);
          }
       }
       else if ((segment_status*255) == 5 || (segment_status*255) == 1) {
          // segment is forced off
          col_smp.rgb = 0.92 * vec3(0,1,1);
       }
       else if ((segment_status*255) == 7 || (segment_status*255) == 3) {
          // segment is forced on
          col_smp.rgb = 0.92 * vec3(1);
       }
       else if (col_packed_smp.g > u_nuclvl) { 
          col_smp.rgb = vec3(0,S,0);
       }
       else if (col_packed_smp.r < u_floorlvl) { 
          col_smp.rgb = vec3(0,S,0);
       }
       else if (col_packed_smp.b > u_msklvl) {
          // segment red over threshold
          col_smp.rgb = vec3(1,0,0);
       }
       else {
          // segment syn and vcn within range
          col_smp.rgb = 0.92 * vec3(1,1,0);
       }
    }
    else {
       col_smp.rgb = vec3(0,S,0);
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
}
"""

_binary_alpha = _linear_alpha

def adjust_level(uniform, attribute, step=0.0005, altstep=None, trace="%(uniform)s level set to %(level).5f", tracenorm=True):
    def helper(origmethod):
        def wrapper(*args):
            self = args[0]
            event = args[1]
            level = getattr(self, attribute)

            if 'Alt' in event.modifiers:
                if altstep is not None:
                    delta = altstep
                else:
                    delta = 0.1 * step
                    
            if 'Shift' in event.modifiers:
                level += step
            else:
                level -= step

            setattr(self, attribute, level)
            
            self.volume_renderer.set_uniform(uniform, level)
            self.update()

            if trace:
                if tracenorm:
                    level = level * (self.data_max - self.data_min) + self.data_min
                print(trace % dict(uniform=uniform, level=level))

        wrapper.__doc__ = origmethod.__doc__
        return wrapper
    return helper

class Canvas(base.Canvas):

    _vol_interp = {
        'nearest': 'nearest',
        'linear': 'linear'
    }.get(os.getenv('VOXEL_SAMPLE', '').lower(), 'linear')

    def splat_centroids(self, reduction, shape, centroids, centroid_measures):
        splat_kern = compose_3d_kernel(list(map(
            lambda d, s, r: gaussian_kernel(d/s/6./r),
            self.synapse_diam_microns,
            self.raw_image.micron_spacing,
            reduction
        )))
        splat_kern /= splat_kern.sum()
        print("segment map splat kernel", splat_kern.shape, splat_kern.sum(), splat_kern.max())
        segment_map = assign_voxels_opt(
            centroid_measures[:,0],
            np.array(centroids, dtype=np.int32) // np.array(reduction, dtype=np.int32),
            shape,
            splat_kern
        )
        return segment_map

    def _pick_rgb_to_segment(self, pick):
        """Convert (R,G,B) encoding of segment pick to 1-based segment ID"""
        return int(sum([ pick[i] * 2**(8*i) for i in range(3) ], 0))

    def _pick_segment_to_rgb(self, index):
        """Convert 1-based segment ID to (R,G,B) encoding of segment pick"""
        return np.array([ (index//2**(8*i)) % 256 for i in range(3) ])

    def _get_centroid_status(self, pick):
        """Get status byte for centroid pick (R,G,B) """
        return self.centroid_status[self._pick_rgb_to_segment(pick)]

    def _set_centroid_status(self, pick, b):
        """Set status byte for centroid pick (R,G,B) """
        self.centroid_status[self._pick_rgb_to_segment(pick)] = b
        self.status_texture.set_data(np.array([[[b]]], dtype=np.uint8), offset=tuple(pick[::-1]), copy=True)

    def retire_centroid_batch(self, event):
        """Expunge manually-classified as non-clickable."""
        for id in self.centroids_batch:
            pick = self._pick_segment_to_rgb(id+1)
            self._set_centroid_status(
                pick[0:3],
                # state transitions for clickable centroids only...
                { 0: 0, 5: 1, 7: 3 }[
                    self._get_centroid_status(pick[0:3])
                ]
            )
        self.centroids_batch.clear()

    def endorse_centroids(self, event):
        """Endorse thresholded centroids as true positives."""
        centroids, measures, status, indices = self.thresholded_segments()
        for i in range(centroids.shape[0]):
            if status[i] == 0:
                self._set_centroid_status(
                    self._pick_segment_to_rgb(indices[i]+1),
                    7
                )
                self.centroids_batch.add(indices[i])

    def endorse_or_expunge(self, event):
        """Endorse ('e') thresholded centroids as true or expunge ('E') centroids as non-clickable."""
        if 'Shift' in event.modifiers:
            self.retire_centroid_batch(event)
        else:
            self.endorse_centroids(event)

    def _reform_image(self, I, meta, view_reduction):
        splits = [(datetime.datetime.now(), None)]
        
        self.raw_image = I
        analyzer, view_image, centroids, centroid_measures = batch_analyze(
            I, self.synapse_diam_microns, self.vicinity_diam_microns, self.redblur_microns, view_reduction
        )
        splits.append((datetime.datetime.now(), 'volume process'))

        centroid_status = np.zeros((256**3,), dtype=np.uint8)

        # get labeled voxels
        assert np.isnan(centroid_measures).sum() == 0
        print("measures range", centroid_measures.min(axis=0), centroid_measures.max(axis=0))
        print("centroids:", centroids.min(axis=0), centroids.max(axis=0))
        print("view_image shape:", view_image.shape)
        print("view_image range", view_image.min(), view_image.max())

        # align 3D textures for opengl?
        assert view_image.shape[3] < 4
        result_shape = tuple(list(map(
            lambda s, m: s + s%m,
            view_image.shape[0:3],
            [1, 1, 4]
        )) + [view_image.shape[3] >= 2 and 3 or 1])
        print("results shape:", result_shape)

        segment_map = self.splat_centroids(view_reduction, result_shape[0:3], centroids, centroid_measures)
        
        if centroid_measures.shape[0] <= (2**8-1):
            nb = 1
        elif centroid_measures.shape[0] <= (2**16-1):
            nb = 2
        else:
            assert centroid_measures.shape[0] <= (2**24-1), "too many segment IDs to RGB-pack"
            nb = 3

        if nb == 1:
            fmt = 'red'
        else:
            fmt = 'rgb'[0:nb]

        print("voxel_class_texture %d segments, %d bytes, %s format" % (centroid_measures.shape[0], nb, fmt))
            
        # pack least significant byte as R, then G, etc.
        segment_map_uint8 = np.zeros(segment_map.shape + (nb,), dtype=np.uint8)
        for i in range(nb):
            segment_map_uint8[:,:,:,i] = (segment_map[:,:,:] // (2**(i*8))) % 2**8

        del segment_map
        self.voxel_class_texture = gloo.Texture3D(segment_map_uint8.shape, format=fmt)
        self.voxel_class_texture.set_data(segment_map_uint8)
        self.voxel_class_texture.interpolation = 'nearest'
        self.voxel_class_texture.wrapping = 'clamp_to_edge'
        del segment_map_uint8
        splits.append((datetime.datetime.now(), 'segment texture'))

        self.data_max = max(
            view_image[:,:,:,0].max(),
            centroid_measures[:,0:2].max()
        )
        self.data_min = min(
            view_image[:,:,:,0].min(),
            centroid_measures[:,0:2].min()
        )

        if view_image.shape[3] > 1:
            self.data_max = max(
                self.data_max,
                view_image[:,:,:,1].max(),
                centroid_measures[:,4].max()
            )
            self.data_min = min(
                self.data_min,
                view_image[:,:,:,1].min(),
                centroid_measures[:,4].min()
            )
        
        # pack segment measures into a 3D grid
        max_classid = centroid_measures.shape[0] + 1
        W = 256
        seg_meas = np.zeros((W, W, W, 3), dtype=np.float32)
        seg_meas_flat = seg_meas.reshape((W**3, 3))
        seg_meas_flat[1:max_classid,0:2] = centroid_measures[:,0:2]
        if centroid_measures.shape[1] > 4:
            seg_meas_flat[1:max_classid,2] = centroid_measures[:,4]

        seg_meas = (seg_meas - self.data_min) / (self.data_max - self.data_min)

        self.measures_texture = gloo.Texture3D(seg_meas.shape, format='rgb', internalformat='rgb16f')
        self.measures_texture.set_data(seg_meas)
        self.measures_texture.interpolation = 'nearest'
        self.measures_texture.wrapping = 'clamp_to_edge'
        self.status_texture = gloo.Texture3D((W,W,W,1), format='red', internalformat='red')
        self.status_texture.set_data(centroid_status.reshape((W,W,W,1)))
        self.status_texture.interpolation = 'nearest'
        self.status_texture.wrapping = 'clamp_to_edge'
        splits.append((datetime.datetime.now(), 'segment measures and status textures'))

        result = np.zeros(result_shape, dtype=np.float32)
        result[0,0,0,0] = self.data_min
        result[0,0,1,0] = self.data_max
        result_box = result[
            0:view_image.shape[0],
            0:view_image.shape[1],
            0:view_image.shape[2],
            :
        ]
        result_box[:,:,:,0] = view_image[:,:,:,0]
            
        splits.append((datetime.datetime.now(), 'scalar volume'))
            
        perf_vector = list(map(lambda t0, t1: ((t1[0]-t0[0]).total_seconds(), t1[1]), splits[0:-1], splits[1:]))
        for elapsed, desc in perf_vector:
            print("%8.2fs %s task time" % (elapsed, desc))
        
        print("measure counts:", centroid_measures.shape)
        print("packed data range: ", self.data_min, self.data_max)

        self.analyzer = analyzer
        self.syn_values = centroid_measures[:,0]
        self.vcn_values = centroid_measures[:,1]
        if centroid_measures.shape[1] > 4:
            self.red_values = centroid_measures[:,2]
        else:
            self.red_values = np.zeros((centroid_measures.shape[0],), dtype=np.float32)
        self.centroids = centroids
        self.centroid_measures = centroid_measures
        self.centroid_status = centroid_status
        #self.widths = widths

        return result

    _frag_glsl_dicts = [
        dict(
            uniforms=_color_uniforms,
            colorxfer=_linear1_colorxfer,
            alphastmt=_linear_alpha,
            desc='Green field with linear colorized segments.'
        ),
        dict(
            uniforms=_color_uniforms,
            colorxfer=_binary1_colorxfer,
            alphastmt=_binary_alpha,
            desc='Green field with binary colorized segments.'
        ),
        dict(
            uniforms=_color_uniforms,
            colorxfer=_linear1_sparse_gray,
            alphastmt=_linear_alpha,
            desc="Void field with linear grayscale manual segments."
        ),
        dict(
            uniforms=_color_uniforms,
            colorxfer=_linear1_grayxfer1,
            alphastmt=_linear_alpha,
            desc="Grayscale field."
        ),
        dict(
            uniforms=_color_uniforms,
            colorxfer=_linear1_grayxfer2,
            alphastmt=_linear_alpha,
            desc="Grayscale field with linear colorized manual segments."
        ),
        dict(
            uniforms=_color_uniforms,
            colorxfer=_linear1_sparse_yellow,
            alphastmt=_linear_alpha,
            desc="Void field with linear colorized manual segments."
        ),
        dict(
            uniforms=_color_uniforms,
            colorxfer=_segment_colorxfer,
            alphastmt=_segment_alpha,
            desc="Voxels colored by RGB-packed segment ID."
        )
        ]
    _pick_glsl_index = 6
    
    def __init__(self, filename1):
        
        self.do_nuclei, footprints = get_mode_and_footprints()
        self.synapse_diam_microns, self.vicinity_diam_microns, self.redblur_microns = footprints

        base.Canvas.__init__(self, filename1)

        try:
            bn = os.path.basename(filename1)
            m = re.match('^(?P<id>.+)[.]ome[.]tif+$', bn)
            self.dump_prefix = './%s' % (m.groupdict()['id'],)
        except:
            # backwards compatible default
            self.dump_prefix = '%s-' % filename1

        self.dump_prefix = os.getenv('DUMP_PREFIX', self.dump_prefix)
        print('Using DUMP_PREFIX="%s"' % self.dump_prefix)

        # textures prepared by self._reform_image() during base init above...
        self.volume_renderer.set_uniform('u_voxel_class_texture', self.voxel_class_texture)
        self.volume_renderer.set_uniform('u_measures_texture', self.measures_texture)
        self.volume_renderer.set_uniform('u_status_texture', self.status_texture)

        self.key_press_handlers['L'] = self.load_classified_segments
        self.key_press_handlers['E'] = self.endorse_or_expunge
        self.key_press_handlers['N'] = self.adjust_nuc_level
        self.key_press_handlers['M'] = self.adjust_msk_level
        self.key_press_handlers['T'] = self.adjust_zer_level
        self.key_press_handlers['U'] = self.adjust_top_level
        self.key_press_handlers['O'] = self.adjust_transp_level
        self.key_press_handlers['D'] = self.dump_params_or_classified
        self.key_press_handlers['H'] = self.dump_segment_heatmap
        self.key_press_handlers['?'] = self.help

        self.auto_dump_load = os.getenv('SYNSPY_AUTO_DUMP_LOAD', 'false').lower() == 'true'
        print('Using SYNSPY_AUTO_DUMP_LOAD=%s' % str(self.auto_dump_load).lower())

        # provide better names for synspy parameters on HUD
        self.hud_display_names['u_floorlvl'] = 'core measure'
        self.hud_display_names['u_nuclvl'] = 'hollow measure'
        self.hud_display_names['u_msklvl'] = 'autofluourescence'
        self.hud_display_names['u_zerlvl'] = 'zero point'
        self.hud_display_names['u_toplvl'] = 'saturation point'
        self.hud_display_names['u_transp'] = 'opacity'

        self.user_notices = []
        if os.getenv('USER_NOTICES_FILE'):
            f = open(os.getenv('USER_NOTICES_FILE'))
            self.user_notices = []
            for line in f.readlines():
                line = line.strip()
                parts = line.split(',')
                try:
                    num = int(parts[0])
                    self.user_notices.append((num, ','.join(parts[1:])))
                except:
                    self.user_notices.append((line, None))
            if self.user_notices[-1] == ('', None):
                del self.user_notices[-1]
                
        assert len(self.user_notices) <= 12
        for i in range(len(self.user_notices)):
            self.key_press_handlers['F%d' % (i + 1)] = self.emit_notice
        
        # provide better value display for HUD
        def value_denorm(v):
            return "%.1f" % (v * (self.data_max - self.data_min) + self.data_min)
        for uniform in ['u_floorlvl', 'u_nuclvl', 'u_msklvl', 'u_zerlvl', 'u_toplvl']:
            self.hud_value_rewrite[uniform] = value_denorm
        
        self.pick_click = False
        self.centroids_batch = set() # store 0-based centroid IDs here...

        self.text_overlay = visuals.TextVisual('DUMMY', color="white", font_size=12)
        if not hasattr(self.text_overlay, 'transforms'):
            # temporary backwards compatibility
            self.text_overlay_transform = visuals.transforms.TransformSystem(self)

        try:
            self.size = tuple([ int(x) for x in os.getenv('WINDOW_SIZE', '').split('x') ])
            assert len(self.size) == 2, 'WINDOW_SIZE must have form WxH'
        except:
            print('Using default WINDOW_SIZE=512x512')
            self.size = 512, 512

        self.auto_dumped = False
        if self.auto_dump_load:
            try:
                self.load_classified_segments(None)
            except IOError as e:
                print('Skipping auto-load of segment status on error: %s' % e)

            @atexit.register
            def shutdown():
                if not self.auto_dumped:
                    self.on_close()

    def emit_notice(self, event):
        """Emit user-defined notices to heads-up display."""
        k = event.key.name
        n = int(k[re.search('[0-9]', k).start():])
        n = n - 1
        assert n >= 0
        assert n < len(self.user_notices)
        key, value = self.user_notices[n]
        self.volume_renderer.uniform_changes[key] = value
            
    def on_resize(self, event):
        base.Canvas.on_resize(self, event)
        if hasattr(self.text_overlay, 'transforms'):
            self.text_overlay.transforms.configure(canvas=self, viewport=self.viewport1)
        else:
            # temporary backwards compatibility
            self.text_overlay_transform = visuals.transforms.TransformSystem(self)
        
    def reload_data(self):
        base.Canvas.reload_data(self)

    def reset_ui(self, event=None):
        """Reset UI controls to startup state."""
        if self.do_nuclei:
            self.nuclvl = (self.vcn_values.max()-self.data_min) / (self.data_max-self.data_min)
            self.msklvl = 1.0
            self.floorlvl = (0.9*self.syn_values.max()-self.data_min) / (self.data_max - self.data_min)
        else:
            self.nuclvl = (1.2*self.vcn_values.mean()-self.data_min) / (self.data_max-self.data_min)
            self.msklvl = (self.red_values.max()-self.data_min) / (self.data_max-self.data_min) or 1.0
            self.floorlvl = (0.9*self.syn_values.mean()-self.data_min) / (self.data_max-self.data_min)

        self.zerlvl = (0.0 - self.data_min) / (self.data_max-self.data_min)
        self.toplvl = self.zerlvl + (0.5 * self.data_max) / (self.data_max-self.data_min)
        self.transp = 0.8
        self.pick_pos = None

        self.volume_renderer.set_uniform('u_nuclvl', self.nuclvl)
        self.volume_renderer.set_uniform('u_msklvl', self.msklvl)
        self.volume_renderer.set_uniform('u_zerlvl', self.zerlvl)
        self.volume_renderer.set_uniform('u_toplvl', self.toplvl)
        self.volume_renderer.set_uniform('u_transp', self.transp)
        base.Canvas.reset_ui(self, event)
        self.volume_renderer.set_uniform('u_floorlvl', self.floorlvl)

    def dump_parameters(self, event):
        """Dump current parameters."""
        print("""
gain: %f
zoom: %f
color mode: %d %s

small feature threshold: %f
nuclear feature threshold: %f
red mask threshold: %f
zero crossing threshold: %f
upper clipping threshold: %f
transparency factor: %f
""" % (
            self.gain,
            self.zoom,
            self.volume_renderer.color_mode, 
            self._frag_glsl_dicts[self.volume_renderer.color_mode].get('desc', 'UNKNOWN'),
            self.floorlvl * (self.data_max - self.data_min) + self.data_min,
            self.nuclvl * (self.data_max - self.data_min) + self.data_min,
            self.msklvl * (self.data_max - self.data_min) + self.data_min,
            self.zerlvl * (self.data_max - self.data_min) + self.data_min,
            self.toplvl * (self.data_max - self.data_min) + self.data_min,
            self.transp
            ))

    def dump_params_or_classified(self, event):
        """Dump current parameters ('d') or voxel classification ('D')."""
        if 'Shift' in event.modifiers:
            self.dump_classified_voxels(event)
        else:
            self.dump_parameters(event)

    def _csv_dump_filename(self):
        if self.do_nuclei:
            return '%s_nucleic_only.csv' % self.dump_prefix
        else:
            return '%s_synaptic_only.csv' % self.dump_prefix

    def dump_classified_voxels(self, event=None):
        """Dump a segment list."""

        csv_name = self._csv_dump_filename()
        saved_params = {
            'Z': 'saved',
            'Y': 'params',
            'X': ('(core, vicinity, zerolvl, toplvl,'
                  + ((self.centroid_measures.shape[1] == 5) and  'autofl' or '')
                  + 'transp):'),
            'raw core': self.floorlvl * (self.data_max - self.data_min) + self.data_min,
            'raw hollow': self.nuclvl * (self.data_max - self.data_min) + self.data_min,
            'DoG core': self.zerlvl * (self.data_max - self.data_min) + self.data_min,
            'DoG hollow': self.toplvl * (self.data_max - self.data_min) + self.data_min,
            'override': (self.transp),
        }
        if self.centroid_measures.shape[1] == 5:
            saved_params['red'] = (self.msklvl * (self.data_max - self.data_min) + self.data_min)

        dump_segment_info_to_csv(self.centroids, self.centroid_measures[1:self.centroid_measures.shape[0]+1], self.centroid_status[1:self.centroid_measures.shape[0]+1], self.vol_cropper.slice_origin, csv_name, saved_params=saved_params, all_segments=False)

        msg = "%s dumped" % csv_name
        if self.hud_enable:
            self.volume_renderer.uniform_changes[msg] = None
        print(msg)

    def load_classified_segments(self, event):
        """Load a segment list with manual override status values."""
        # assume that dump is ordered subset of current analysis
        csv_name = self._csv_dump_filename()
        status, saved_params = load_segment_status_from_csv(self.centroids, self.vol_cropper.slice_origin, csv_name)

        if saved_params is not None:
            ignore1, measures, ignore2, ignore3 = self.thresholded_segments()

            self.floorlvl = (float(saved_params['raw core']) - self.data_min) / (self.data_max - self.data_min)
            self.nuclvl = (float(saved_params['raw hollow']) - self.data_min) / (self.data_max - self.data_min)
            self.zerlvl = (float(saved_params['DoG core']) - self.data_min) / (self.data_max - self.data_min)
            self.toplvl = (float(saved_params['DoG hollow']) - self.data_min) / (self.data_max - self.data_min)

            if measures.shape[1] == 5:
                self.msklvl = (float(saved_params['red']) - self.data_min) / (self.data_max - self.data_min)

            self.transp = float(saved_params['override'])

            self.volume_renderer.set_uniform('u_floorlvl', self.floorlvl)
            self.volume_renderer.set_uniform('u_nuclvl', self.nuclvl)
            self.volume_renderer.set_uniform('u_zerlvl', self.zerlvl)
            self.volume_renderer.set_uniform('u_toplvl', self.toplvl)
            self.volume_renderer.set_uniform('u_msklvl', self.msklvl)
            self.volume_renderer.set_uniform('u_transp', self.transp)
            self.update()

        self.centroids_batch.clear()
        override_indices = (status > 0).nonzero()[0]
        for i in override_indices:
            self._set_centroid_status(
                self._pick_segment_to_rgb(i+1),
                {1: 5, 3: 7, 5: 5, 7: 7}[status[i]]
            )
            self.centroids_batch.add(i)

        msg = '%s loaded' % csv_name
        if self.hud_enable:
            self.volume_renderer.uniform_changes[msg] = None
        print(msg)

    def thresholded_segments(self):
        """Return subset of centroid data where centroids match thresholds."""
        # get thresholds from OpenGL back to absolute values
        floorlvl, nuclvl, msklvl = [x * (self.data_max - self.data_min) + self.data_min for x in [self.floorlvl, self.nuclvl, self.msklvl]]

        # a 1D bitmap of centroid inclusion
        matches = (
            (self.centroid_measures[:,0] >= floorlvl)
            * (self.centroid_measures[:,1] <= nuclvl)
        ) + (self.centroid_status[1:self.centroid_measures.shape[0]+1] > 0)
        
        if self.centroid_measures.shape[1] > 4:
            matches *= self.centroid_measures[:,4] <= msklvl

        return self.centroids[matches], \
            self.centroid_measures[matches], \
            self.centroid_status[1:self.centroid_measures.shape[0]+1][matches], \
            np.arange(0, self.centroids.shape[0])[matches]
    
        
    def dump_segment_heatmap(self, event):
        """Dump a heatmap image with current thresholds."""

        IMAGE_SIZE = 512
        heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

        def render(v0, v1, ch):
            print("rendering %d points to heatmap ch%d (%f ... %f) x (%f ... %f)" % (
                len(v0), ch,
                np.nanmin(v0), np.nanmax(v0),
                np.nanmin(v1), np.nanmax(v1)
            ))
            for i in range(len(v0)):
                x = (IMAGE_SIZE-1) * v0[i]
                y = (IMAGE_SIZE-1) * v1[i]
                assert x < IMAGE_SIZE and x >= 0 and y < IMAGE_SIZE and y >= 0, 'x,y = %s,%s' % (x, y)
                heatmap[int(y),int(x),int(ch)] += 1

            hmax = heatmap[:,:,ch].max()
            max_bins = np.argwhere(heatmap[:,:,ch] == hmax)[0:20]
            print("heatmap ch%d max: %s" % (ch, hmax))
            print("heatmap max %s at XY bins %s" % (hmax, max_bins))

        def normalize(m):
            m = np.clip(m, 0, np.inf)
            m = np.log1p(m)
            return m
            
        # render all peaks
        m = normalize(self.centroid_measures[self.centroid_measures[:,2] >= 0])
        hmax = m.max()
        m /= hmax
        render(m[:,0], m[:,1], 0)

        # render thresholded peaks
        c, m, ignore1, ignore2 = self.thresholded_segments()
        print(c.shape, m.shape)
        m = normalize(m)
        m /= hmax
        render(m[:,0], m[:,1], 1)
        
        render(m[:,2], m[:,1], 1)

        render(m[:,0], m[:,3], 1)
        render(m[:,0], m[:,3], 2)
            
        heatmap = np.log1p(heatmap)
        heatmap = np.clip(255.0 * 4 * heatmap / heatmap.max(), 0, 255)

        tifffile.imsave('%sheatmap.tiff' % self.dump_prefix, heatmap.astype(np.uint8)[slice(None,None,-1),:,:])

    def on_mouse_press(self, event):
        self.pick_pos = event.pos
        self.update()
        
    def on_mouse_release(self, event):
        base.Canvas.on_mouse_release(self, event)

        if (event.pos - event.press_event.pos).max() == 0:
            self.pick_click = True
        else:
            self.pick_click = False
        self.update()
           
    def on_mouse_move(self, event):
        base.Canvas.on_mouse_move(self, event)

        if not event.is_dragging:
            X, Y, W, H = self.viewport1
            
            self.pick_pos = event.pos
            pos = np.array(event.pos)
            pos += np.array((0, -40))
            border = np.array((50,20))
            pos = np.minimum(
                np.array(self.size) - border,
                np.maximum( border, pos)
            )
            self.text_overlay.pos = [
                pos,
                pos + np.array((0, 15))
            ]
            self.update()
        else:
            self.pick_pos = None
            self.pick_click = False

    def on_close(self, event=None):
        if self.auto_dump_load:
            print('Dumping state...')
            self.dump_classified_voxels(event)
            self.auto_dumped = True

    @adjust_level('u_floorlvl', 'floorlvl', trace="feature threshold set to %(level).5f")
    def adjust_floor_level(self, event):
        """Increase ('F') or decrease ('f') small feature threshold level."""
        pass
    
    @adjust_level('u_nuclvl', 'nuclvl', trace="negative vicinity threshold set to %(level).5f")
    def adjust_nuc_level(self, event):
        """Increase ('N') or decrease ('n') nuclei-scale feature threshold level."""
        pass

    @adjust_level('u_msklvl', 'msklvl', 0.005, trace="red mask level set to %(level).5f")
    def adjust_msk_level(self, event):
        """Increase ('M') or descrease ('m') red-channel mask threshold level."""
        pass

    @adjust_level('u_zerlvl', 'zerlvl', trace="zero-crossing level set to %(level).5f")
    def adjust_zer_level(self, event):
        """Increase ('T') or decrease ('t') transparency zero-crossing level."""
        pass

    @adjust_level('u_toplvl', 'toplvl', trace="upper clipping level set to %(level).5f")
    def adjust_top_level(self, event):
        """Increase ('U') or decrease ('u') upper clipping level."""
        pass

    @adjust_level('u_transp', 'transp', 0.005, trace="opacity factor set to %(level).5f", tracenorm=False)
    def adjust_transp_level(self, event):
        """Increase ('O') or decrease ('o') opacity factor."""
        pass

    def on_draw(self, event, color_mask=(True, True, True, True)):
        def on_pick(picked):
            """Work to handle segment-click action within multi-pass rendering cycle."""
            if self.pick_click and picked[0:3].max() > 0:
                self._set_centroid_status(
                    picked[0:3],
                    # state transitions for clickable centroids only...
                    { 0: 5, 5: 7, 7: 0 }[
                        self._get_centroid_status(picked[0:3])
                    ]
                )
                self.centroids_batch.add(
                    # change 1-based to 0-based
                    self._pick_rgb_to_segment(picked[0:3]) - 1
                )
                # UGLY: click has been handled so clear flag!
                self.pick_click = False
                
        picked = base.Canvas.on_draw(self, event, color_mask, pick=self.pick_pos, on_pick=on_pick)
        if picked is not None and picked[0:3].max() > 0 and False:
            segment_id = (
                picked[0]
                + picked[1] * 2**8
                + picked[2] * 2**16
            ) - 1

            gloo.set_state(cull_face=False)
            gloo.set_viewport((0, 0)+ self.size)
            self.text_overlay.text = [
                ' '.join(map(str, self.centroids[segment_id, :])),
                '%0.1f %0.1f' % (
                    self.centroid_measures[segment_id, 0],
                    self.centroid_measures[segment_id, 1],
                )
            ]
            if hasattr(self.text_overlay, 'transforms'):
                self.text_overlay.draw()
            else:
                self.text_overlay.draw(self.text_overlay_transform)


def main():
    c = Canvas(sys.argv[1])
    c.show()
    app.run()

if __name__ == '__main__':
    sys.exit(main())
