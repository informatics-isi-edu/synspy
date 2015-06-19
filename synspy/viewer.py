
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import numpy as np

import datetime

import os
import math
import csv

import volspy.viewer as base
from vispy import gloo

from analyze import BlockedAnalyzerOpt, assign_voxels_opt, compose_3d_kernel, gaussian_kernel

import tifffile
        
_color_uniforms = """
uniform sampler3D u_voxel_class_texture;
uniform sampler3D u_measures_texture;
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

_linear1_colorxfer = """
    col_smp.b = col_smp.r;
    col_smp.r = 0;
    col_smp.g = 0;

    // lookup voxel's packed segment ID
    col_packed_smp = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(col_packed_smp.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, col_packed_smp.rgb);

       if (col_packed_smp.g > u_nuclvl) { /* pass */ }
       else if (col_packed_smp.r < u_floorlvl) { /* pass */ }
       else if (col_packed_smp.b > u_msklvl) {
          // segment red over threshold so mark as red
          col_smp.r = col_smp.b;
          col_smp.b = 0;
       }
       else {
          // segment syn and vcn within range so mark as cyan
          col_smp.g = col_smp.b;
          col_smp.b = 0;
       }
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
"""

_linear_alpha = """
   col_smp.a = max(max(col_smp.r, col_smp.g), col_smp.b) * u_transp;
"""

_binary1_colorxfer = """
    col_smp.b = col_smp.r;
    col_smp.r = 0;
    col_smp.g = 0;

    // lookup voxel's packed segment ID
    col_packed_smp = texture3D(u_voxel_class_texture, texcoord.xyz / texcoord.w);
    
    if ( any(greaterThan(col_packed_smp.rgb, vec3(0))) )  {
       // measures are packed as R=syn, G=vcn, B=redmask
       col_packed_smp = texture3D(u_measures_texture, col_packed_smp.rgb);

       if (col_packed_smp.g > u_nuclvl) { /* pass */ }
       else if (col_packed_smp.r < u_floorlvl) { /* pass */ }
       else if (col_packed_smp.b > u_msklvl) {
          // segment red over threshold so mark as red
          col_smp.r = 1;
          col_smp.b = 0;
       }
       else {
          // segment syn and vcn within range so mark as cyan
          col_smp.g = 1;
          col_smp.b = 0;
       }
    }

    // apply interactive range clipping  [zerlvl, toplvl]
    col_smp = (col_smp - u_zerlvl) / (u_toplvl - u_zerlvl);
    col_smp = clamp( u_gain * col_smp, 0.0, 1.0);

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
                print trace % dict(uniform=uniform, level=level)

        return wrapper
    return helper

class Canvas(base.Canvas):

    #_vol_interp = 'nearest'
    _vol_interp = 'linear'
    
    def _reform_image(self, I, meta, view_reduction):
        analyzer = BlockedAnalyzerOpt(I, self.synapse_diam_microns, self.vicinity_diam_microns, self.redblur_microns, view_reduction)
        self.raw_image = I

        splits = [(datetime.datetime.now(), None)]
        
        view_image, centroids, centroid_measures = analyzer.volume_process()
        splits.append((datetime.datetime.now(), 'volume process'))

        # get labeled voxels
        assert np.isnan(centroid_measures).sum() == 0
        print "measures range", centroid_measures.min(axis=0), centroid_measures.max(axis=0)
        centroids2 = np.array(centroids, dtype=np.int32) / np.array(analyzer.view_reduction, dtype=np.int32)
        print "centroids:", centroids.min(axis=0), centroids.max(axis=0)
        print "centroids2:", centroids2.min(axis=0), centroids2.max(axis=0)
        print "view_image shape:", view_image.shape

        # align 3D textures for opengl?
        assert view_image.shape[3] < 4
        result_shape = tuple(map(
            lambda s, m: s + s%m,
            view_image.shape[0:3],
            [1, 1, 4]
        ) + [view_image.shape[3] >= 2 and 3 or 1])
        print "results shape:", result_shape
        
        splat_kern = compose_3d_kernel(map(
            lambda d, s, r: gaussian_kernel(d/s/6./r),
            self.synapse_diam_microns,
            I.micron_spacing,
            analyzer.view_reduction
        ))
        splat_kern /= splat_kern.sum()
        print "segment map splat kernel", splat_kern.shape, splat_kern.sum(), splat_kern.max()
        segment_map = assign_voxels_opt(
            centroid_measures[:,0],
            centroids2,
            result_shape[0:3],
            splat_kern
        )

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

        print "voxel_class_texture %d segments, %d bytes, %s format" % (centroid_measures.shape[0], nb, fmt)
            
        # pack least significant byte as R, then G, etc.
        segment_map_uint8 = np.zeros(segment_map.shape + (nb,), dtype=np.uint8)
        for i in range(nb):
            segment_map_uint8[:,:,:,i] = (segment_map[:,:,:] / (2**(i*8))) % 2**8

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
        splits.append((datetime.datetime.now(), 'segment measures texture'))

        result = np.zeros(result_shape, dtype=I.dtype)
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
            
        perf_vector = map(lambda t0, t1: ((t1[0]-t0[0]).total_seconds(), t1[1]), splits[0:-1], splits[1:])
        for elapsed, desc in perf_vector:
            print "%8.2fs %s task time" % (elapsed, desc)
        
        print "measure counts:", centroid_measures.shape
        print "packed data range: ", self.data_min, self.data_max

        self.analyzer = analyzer
        self.syn_values = centroid_measures[:,0]
        self.vcn_values = centroid_measures[:,1]
        if centroid_measures.shape[1] > 4:
            self.red_values = centroid_measures[:,2]
        else:
            self.red_values = np.zeros((centroid_measures.shape[0],), dtype=np.float32)
        self.centroids = centroids
        self.centroid_measures = centroid_measures
        #self.widths = widths

        return result

    _frag_glsl_dicts = [
        dict(
            uniforms=_color_uniforms,
            colorxfer=_linear1_colorxfer,
            alphastmt=_linear_alpha,
            desc='White-linear segments, blue-linear background, and red-linear mask channel.'
            ),
        dict(
            uniforms=_color_uniforms,
            colorxfer=_binary1_colorxfer,
            alphastmt=_binary_alpha,
            desc='White-boolean segments, blue-linear background, and red-boolean mask channel.'
            )
        ]

    def __init__(self, filename1):
        
        # TODO: put these under UI control?
        self.synapse_diam_microns = (2.75, 1.5, 1.5)
        self.vicinity_diam_microns = (4.0, 2.75, 2.75)

        self.redblur_microns = (3.0, 3.0, 3.0)

        base.Canvas.__init__(self, filename1)
        # textures prepared by self._reform_image() during base init above...
        self.volume_renderer.set_uniform('u_voxel_class_texture', self.voxel_class_texture)
        self.volume_renderer.set_uniform('u_measures_texture', self.measures_texture)

        self.key_press_handlers['N'] = self.adjust_nuc_level
        self.key_press_handlers['M'] = self.adjust_msk_level
        self.key_press_handlers['T'] = self.adjust_zer_level
        self.key_press_handlers['U'] = self.adjust_top_level
        self.key_press_handlers['O'] = self.adjust_transp_level
        self.key_press_handlers['D'] = self.dump_params_or_classified
        self.key_press_handlers['H'] = self.dump_segment_heatmap
        self.key_press_handlers['?'] = self.help

        self.size = 512, 512

    def reload_data(self):
        base.Canvas.reload_data(self)

    def reset_ui(self, event=None):
        """Reset UI controls to startup state."""
        self.nuclvl = (1.2*self.vcn_values.mean()-self.data_min) / (self.data_max-self.data_min)
        self.msklvl = (self.red_values.max()-self.data_min) / (self.data_max-self.data_min) or 1.0
        self.zerlvl = 0.28
        self.toplvl = 0.4
        self.transp = 0.8

        self.volume_renderer.set_uniform('u_nuclvl', self.nuclvl)
        self.volume_renderer.set_uniform('u_msklvl', self.msklvl)
        self.volume_renderer.set_uniform('u_zerlvl', self.zerlvl)
        self.volume_renderer.set_uniform('u_toplvl', self.toplvl)
        self.volume_renderer.set_uniform('u_transp', self.transp)
        base.Canvas.reset_ui(self, event)
        self.floorlvl = (0.9*self.syn_values.mean()-self.data_min) / (self.data_max-self.data_min)
        self.volume_renderer.set_uniform('u_floorlvl', self.floorlvl)

    def dump_parameters(self, event):
        """Dump current parameters."""
        print """
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
            )

    def dump_params_or_classified(self, event):
        """Dump current parameters ('d') or voxel classification ('D')."""
        if 'Shift' in event.modifiers:
            self.dump_classified_voxels(event)
        else:
            self.dump_parameters(event)

    def dump_classified_voxels(self, event):
        """Dump a volume image with centroid markers."""
        dtype = np.uint16
        dmax = 1./self.raw_image.max() * (2**16-1)

        result = np.zeros( self.raw_image.shape[0:3] + (3,), dtype ) # RGB debug image
        result[:,:,:,2] = self.raw_image[:,:,:,0] * dmax

        for centroid in self.centroids:
            result[tuple(
                slice(c, c+1)
                for c in centroid
            ) + (1,)] = (2**16-1)

        result = np.sqrt(result)
        result = (result * ((2**8-1)/result.max())).astype(np.uint8)
            
        tifffile.imsave('/scratch/debug.tiff', result)
        print "/scratch/debug.tiff dumped"

        csvfile = open('/scratch/segments.csv', 'w')
        writer = csv.writer(csvfile)
        writer.writerow(
            ('Z', 'Y', 'X', 'raw core', 'raw hollow', 'DoG core', 'DoG hollow')
            + ((self.centroid_measures.shape[1] == 5) and ('red',) or ())
        )
        for i in range(self.centroid_measures.shape[0]):
            Z, Y, X = self.centroids[i]
            writer.writerow( 
                (Z, Y, X) + tuple(self.centroid_measures[i,m] for m in range(self.centroid_measures.shape[1]))
            )
        del writer
        csvfile.close()
        print "/scratch/segments.csv dumped"

    def dump_segment_heatmap(self, event):
        """Dump a heatmap image with current thresholds."""

        IMAGE_SIZE = 512
        heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

        def render(v0, v1, ch):
            print "rendering %d points to heatmap ch%d (%f ... %f) x (%f ... %f)" % (
                len(v0), ch,
                np.nanmin(v0), np.nanmax(v0),
                np.nanmin(v1), np.nanmax(v1)
            )
            for i in range(len(v0)):
                x = (IMAGE_SIZE-1) * v0[i]
                y = (IMAGE_SIZE-1) * v1[i]
                assert x < IMAGE_SIZE and x >= 0 and y < IMAGE_SIZE and y >= 0, 'x,y = %s,%s' % (x, y)
                heatmap[y,x,ch] += 1

            hmax = heatmap[:,:,ch].max()
            max_bins = np.argwhere(heatmap[:,:,ch] == hmax)[0:20]
            print "heatmap ch%d max: %s" % (ch, hmax)
            print "heatmap max %s at XY bins %s" % (hmax, max_bins)

        m = self.centroid_measures[self.centroid_measures[:,2] >= 0]
        m = np.clip(m, 0, np.inf)
        m = np.log1p(m)
        m /= m.max()

        render(m[:,1], m[:,0], 0)
        render(m[:,1], m[:,2], 1)
        render(m[:,1], m[:,3], 2)
            
        heatmap = np.log1p(heatmap)
        heatmap = np.clip(255.0 * 3 * heatmap / heatmap.max(), 0, 255)

        tifffile.imsave('/scratch/heatmap.tiff', heatmap.astype(np.uint8)[slice(None,None,-1),:,:])
        
    
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

