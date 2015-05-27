
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

from analyze import BlockedAnalyzerOpt, assign_voxels_opt

import tifffile
        
_color_uniforms = """
uniform int u_numchannels;
uniform float u_gain;
uniform float u_floorlvl;
uniform float u_nuclvl;
uniform float u_msklvl;
"""

# R: raw signal
# G: synapse core samples
# B: synapse vicinity samples
# A: mskblur gaussian blurred auto-fluorescence signal

_linear1_colorxfer = """
       if (col_smp.a > u_msklvl) {
          col_smp.r = col_smp.a;
          col_smp.g = 0.0;
          col_smp.b = 0.0;
       }
       else if (col_smp.b < u_nuclvl && col_smp.g > u_floorlvl) {
          col_smp.g = col_smp.r;
          col_smp.a = col_smp.r;
          col_smp.b = 0.0;
          col_smp.r = 0.0;
       }
       else {
          col_smp.b = col_smp.r;
          col_smp.a = col_smp.r;
          col_smp.r = 0.0;
          col_smp.g = 0.0;
       }

       col_smp = clamp( u_gain * col_smp, 0.0, 1.0);
"""

_linear_alpha = """
       col_smp.a = clamp(col_smp.a, 0.0, 1.0);
"""

_binary1_colorxfer = """
       if (col_smp.a > u_msklvl) {
          col_smp.r = 1.0;
          col_smp.g = 0.0;
          col_smp.b = 0.0;
          col_smp.a = 0.05;
       }
       else if (col_smp.b < u_nuclvl && col_smp.g > u_floorlvl) {
          col_smp.g = 1.0;
          col_smp.a = 1.0;
          col_smp.b = 0.0;
          col_smp.r = 0.0;
       }
       else {
          col_smp.b = col_smp.r;
          col_smp.a = clamp(col_smp.r, 0.0, 0.7);
          col_smp.r = 0.0;
          col_smp.g = 0.0;
       }

       col_smp.rgb = clamp( u_gain * col_smp.rgb, 0.0, 1.0);
       if (col_smp.a > 0.9) {
          // leave alpha alone for segment markers
       }
       else {
          // don't let voxel alpha saturate
          col_smp.a = clamp( u_gain * col_smp.a, 0.0, 0.5);
       }

"""

_binary_alpha = ""

class Canvas(base.Canvas):

    _vol_interp = 'nearest' # 'linear'
    
    def _reform_image(self, I, meta):
        raw_channel = I[:,:,:,0]

        try:
            red_channel = I[:,:,:,1]
        except IndexError:
            red_channel = np.zeros(I.shape[0:3], I.dtype)

        analyzer = BlockedAnalyzerOpt(raw_channel, red_channel, meta, self.synapse_diam_microns, self.vicinity_diam_microns, self.redblur_microns)

        self.raw_shape = I.shape

        t0 = datetime.datetime.now()
        t00 = t0
        raw_channel, syn_channel, pks_channel, msk_channel = analyzer.volume_process()
        t1 = datetime.datetime.now()
        print "\nvolume_process took %s seconds\n" % (t1-t0).total_seconds()

        # get per-core measurements
        t0 = datetime.datetime.now()
        syn_values, vcn_values, centroids, widths = analyzer.analyze(syn_channel, pks_channel)
        t1 = datetime.datetime.now()
        print "\nanalyze took %s seconds" % (t1-t0).total_seconds()

        # get labeled voxels
        t0 = datetime.datetime.now()
        segment_map = assign_voxels_opt(syn_values, centroids, syn_channel.shape, analyzer.kernels_3d[0])
        t1 = datetime.datetime.now()
        print "assign_voxels took %s seconds" % (t1-t0).total_seconds()

        # fill segmented voxels w/ per-segment measurements
        t0 = datetime.datetime.now()
        syn_val_map = np.array([0] + list(syn_values)).astype(I.dtype)[segment_map]
        vcn_val_map = np.array([0] + list(vcn_values)).astype(I.dtype)[segment_map]
        t1 = datetime.datetime.now()
        print "taking assigned voxels took %s seconds" % (t1-t0).total_seconds()
        print "total image processing was %s seconds\n" % (t1-t00).total_seconds()

        result = np.zeros( syn_channel.shape + (4,), dtype=I.dtype )
        result[:,:,:,0] = raw_channel
        result[:,:,:,1] = syn_val_map
        result[:,:,:,2] = vcn_val_map
        result[:,:,:,3] = msk_channel

        self.data_max = result.max()
        self.data_min = result.min()

        print "measure counts:", len(syn_values), len(vcn_values), len(centroids)
        print "map range:", segment_map.min(), segment_map.max()
        print "packed data range: ", self.data_min, self.data_max

        segment_map = None

        self._all_segments = (syn_channel, pks_channel)
        self.analyzer = analyzer
        self.syn_values = syn_values
        self.vcn_values = vcn_values
        self.centroids = centroids
        self.widths = widths

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

        #self.synapse_diam_microns = (2.3, 1.2, 1.2)
        #self.vicinity_diam_microns = (4.75, 2.5, 2.5)

        #self.synapse_diam_microns = (2.4, 1.3, 1.3)
        #self.vicinity_diam_microns = (8.0, 4.0, 4.0)

        # tweak for dataset 19 image 03b exploration...
        self.synapse_diam_microns = (3.0, 1.5, 1.5)
        self.vicinity_diam_microns = (6.0, 3.0, 3.0)

        self.redblur_microns = (3.0, 3.0, 3.0)

        base.Canvas.__init__(self, filename1)

        self.key_press_handlers['N'] = self.adjust_nuc_level
        self.key_press_handlers['M'] = self.adjust_msk_level
        self.key_press_handlers['D'] = self.dump_params_or_classified
        self.key_press_handlers['H'] = self.dump_segment_heatmap
        self.key_press_handlers['?'] = self.help

        self.size = 512, 512

    def reload_data(self):
        base.Canvas.reload_data(self)

    def reset_ui(self, event=None):
        """Reset UI controls to startup state."""
        #self.nuclvl = 0.02
        #self.msklvl = 0.12
        self.nuclvl = 0.2 #1268.0 / (self.data_max - self.data_min)
        self.msklvl = 4531.0 / (self.data_max - self.data_min)
        self.volume_renderer.set_uniform('u_nuclvl', self.nuclvl)
        self.volume_renderer.set_uniform('u_msklvl', self.msklvl)
        base.Canvas.reset_ui(self, event)
        #self.floorlvl = 0.01
        self.floorlvl = 0.02 #1449.0 / (self.data_max - self.data_min)
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
""" % (
            self.gain,
            self.zoom,
            self.volume_renderer.color_mode, 
            self._frag_glsl_dicts[self.volume_renderer.color_mode].get('desc', ''),
            self.floorlvl * (self.data_max - self.data_min),
            self.nuclvl * (self.data_max - self.data_min),
            self.msklvl * (self.data_max - self.data_min)
            )

    def dump_params_or_classified(self, event):
        """Dump current parameters ('d') or voxel classification ('D')."""
        if 'Shift' in event.modifiers:
            self.dump_classified_voxels(event)
        else:
            self.dump_parameters(event)

    def dump_classified_voxels(self, event):
        """Dump a volume image with classified voxels at current thresholds."""
        syn_lvl = self.floorlvl * (self.data_max - self.data_min)
        vcn_lvl = self.nuclvl * (self.data_max - self.data_min)
        msk_lvl = self.msklvl * (self.data_max - self.data_min)

        data = self.vol_cropper.pyramid[0]

        dtype = np.uint16
        dmax = 1./data.max() * (2**16-1)

        print data.min(), data.max()

        print "raw shape: %s" % (self.raw_shape,)
        print "processed shape: %s" % (data.shape,)

        result = np.empty( self.raw_shape[0:3] + (3,), dtype ) # RGB debug image

        zpad = (self.raw_shape[0] - data.shape[0])/2
        ypad = (self.raw_shape[1] - data.shape[1])/2
        xpad = (self.raw_shape[2] - data.shape[2])/2

        print "analysis results padded by (%s,%s,%s) in (Z,Y,X)" % (zpad,ypad,xpad)

        zslc = slice(zpad,-zpad)
        yslc = slice(ypad,-ypad)
        xslc = slice(xpad,-xpad)

        result[zslc,yslc,xslc,0] = (data[:,:,:,3] * dmax) * (data[:,:,:,3] > msk_lvl)

        result[zslc,yslc,xslc,1] = (data[:,:,:,0] * dmax) * (
            (data[:,:,:,3] <= msk_lvl)
            & (data[:,:,:,1] >= syn_lvl) 
            & (data[:,:,:,2] <= vcn_lvl)
        )

        result[zslc,yslc,xslc,2] = (data[:,:,:,0] * dmax) * (
            (data[:,:,:,3] > msk_lvl)
            | (data[:,:,:,1] < syn_lvl) 
            | (data[:,:,:,2] > vcn_lvl)
        )

        tifffile.imsave('/scratch/debug.tiff', result)
        print "/scratch/debug.tiff dumped"

        csvfile = open('/scratch/segments.csv', 'w')
        writer = csv.writer(csvfile)
        writer.writerow( ('Z', 'Y', 'X', 'D', 'H', 'W', 'central value', 'vicinity value', 'red value') )
        for i in range(len(self.syn_values)):
            Z, Y, X = self.centroids[i]
            D, H, W = self.widths[i]
            red = data[Z,Y,X,3]
            Z += zpad
            Y += ypad
            X += xpad
            writer.writerow( 
                (Z, Y, X, D, H, W, self.syn_values[i], self.vcn_values[i], red) 
            )
        del writer
        csvfile.close()
        print "/scratch/segments.csv dumped"

    def dump_segment_heatmap(self, event):
        """Dump a heatmap image with current thresholds."""

        IMAGE_SIZE = 1024
        HM_GAIN = 64

        MAX_VAL = max(self._all_segments[0].max(), self._all_segments[1].max())
        MAX_SYN = 80000
        MAX_VCN = 80000

        # correct for OpenGL texel normalization
        syn_lvl = self.floorlvl * (self.data_max - self.data_min) + self.data_min
        vcn_lvl = self.nuclvl * (self.data_max - self.data_min) + self.data_min

        heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

        def reanalyze():
            return self.analyzer.analyze(
                self._all_segments[0],
                self._all_segments[1],
                syn_lvl,
                vcn_lvl
            )

        def render(syn_vals, vcn_vals, ch):
            print "rendering %d segments to heatmap" % len(syn_vals)
            for i in range(len(syn_vals)):
                if syn_vals[i] > 0:
                    x = min(IMAGE_SIZE-1,math.log(syn_vals[i]) * IMAGE_SIZE / math.log(MAX_SYN))
                    #x = min(IMAGE_SIZE-1,syn_vals[i] * IMAGE_SIZE / MAX_SYN)
                    if vcn_vals[i] > 0:
                        y = min(IMAGE_SIZE-1,math.log(vcn_vals[i]) * IMAGE_SIZE / math.log(MAX_VCN))
                        #y = min(IMAGE_SIZE-1,vcn_vals[i] * IMAGE_SIZE / MAX_VCN)
                    else:
                        y = 0
                    heatmap[y,x,ch] += 1
                    #heatmap = np.sqrt(heatmap)

            histo = []
            hmax = heatmap.max()
            print "heatmap max:", hmax
            for i in range(hmax):
                s = np.sum((heatmap > i) & (heatmap <= (i+1)))
                histo.append(s)

            max_bins = []
            for j in range(IMAGE_SIZE):
                for i in range(IMAGE_SIZE):
                    if heatmap[j,i,0] == hmax:
                        max_bins.append( (i, j) )

            print "heatmap max %s at XY bins %s" % (hmax, max_bins)
            print "heatmap max %s at syn,vcn bins %s" % (
                hmax, 
                map(
                    lambda p: (
                        math.exp(p[0] * math.log(MAX_SYN) / IMAGE_SIZE), 
                        math.exp(p[1] * math.log(MAX_SYN) / IMAGE_SIZE)
                    ), 
                    max_bins
                )
            )
                
            syn_vals = list(syn_vals)
            vcn_vals = list(vcn_vals)
            syn_vals.sort()
            vcn_vals.sort()
            try:
                print "median syn %s vcn %s" % (syn_vals[len(syn_vals)/2], vcn_vals[len(vcn_vals)/2])
            except:
                pass

            print "histo: %s" % histo

        # render all centroids
        render(self.syn_values, self.vcn_values, 0)

        # render matching centroids
        syn_values, vcn_values, centroids = reanalyze()
        render(syn_values, vcn_values, 1)

        #heatmap = np.log(heatmap)
        #heatmap = 255.0 * heatmap / heatmap.max()
        
        heatmap = heatmap * HM_GAIN
        heatmap = heatmap * (heatmap <= 255.0) + 255.0 * (heatmap > 255.0)

        tifffile.imsave('heatmap.tiff', heatmap.astype(np.uint8))
        

    def adjust_floor_level(self, event):
        """Increase ('F') or decrease ('f') small feature threshold level."""
        if 'Alt' in event.modifiers:
            step = 0.00005
        else:
            step = 0.0005

        if 'Shift' in event.modifiers:
            self.floorlvl += step
        else:
            self.floorlvl -= step
        self.volume_renderer.set_uniform('u_floorlvl', self.floorlvl)
        self.update()
        print 'small feature level set to %.5f' % (self.floorlvl * self.data_max)

    def adjust_nuc_level(self, event):
        """Increase ('N') or decrease ('n') nuclei-scale feature threshold level."""
        if 'Alt' in event.modifiers:
            step = 0.00005
        else:
            step = 0.0005

        if 'Shift' in event.modifiers:
            self.nuclvl += step
        else:
            self.nuclvl -= step
        self.volume_renderer.set_uniform('u_nuclvl', self.nuclvl)
        self.update()
        print 'nucleus level set to %.5f' % (self.nuclvl * self.data_max)

    def adjust_msk_level(self, event):
        """Increase ('M') or descrease ('m') red-channel mask threshold level."""
        if 'Alt' in event.modifiers:
            step = 0.0005
        else:
            step = 0.005

        if 'Shift' in event.modifiers:
            self.msklvl += step
        else:
            self.msklvl -= step
        self.volume_renderer.set_uniform('u_msklvl', self.msklvl)
        self.update()
        print 'red mask level set to %.5f' % (self.msklvl * self.data_max)

