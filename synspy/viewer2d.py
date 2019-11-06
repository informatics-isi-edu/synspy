
#
# Copyright 2014-2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import sys
import os
import numpy as np
import json
import atexit
import traceback

import time
import datetime
from functools import reduce

from vispy import gloo
from vispy import app
from vispy import visuals

from synspy.analyze.util import gaussian_kernel, load_segment_status_from_csv, dump_segment_info_to_csv
from synspy.analyze.np import assign_voxels
from volspy.util import clamp, ImageMetadata

#gloo.gl.use_gl('pyopengl debug')

def keydoc(details):
    def helper(original_method):
        original_method._keydocs = details
        return original_method
    return helper

class SynspyImageManager (object):

    def __init__(self, filename):
        with np.load(filename) as parts:
            self.properties = json.loads(parts['properties'].tostring().decode('utf8'))
            self.data = parts['voxels'].astype(np.float32) * np.float32(self.properties['voxel_divisor'])
            self.measures = parts['measures'].astype(np.float32) * np.float32(self.properties['measures_divisor'])
            self.centroids = parts['centroids'].astype(np.int32)
            if 'statuses' in parts:
                self.statuses = parts['statuses']
            else:
                self.statuses = None
            self.slice_origin = np.array(self.properties['slice_origin'], dtype=np.int32)
            self.meta = ImageMetadata(
                self.properties['image_grid'][2],
                self.properties['image_grid'][1],
                self.properties['image_grid'][0],
                'ZYXC'
            )

        D, H, W = self.data.shape[0:3]

        assert self.properties['core_diam_microns'][1] == self.properties['core_diam_microns'][2]

        self.z_radius = int(
            self.properties['core_diam_microns'][0]
            / self.properties['image_grid'][0]
            / 2.0
        )
        self.xy_radius = int(
            self.properties['core_diam_microns'][1]
            / self.properties['image_grid'][1]
            / 2.0
        )
        #print 'kernel radii %d, %d, %d' % (self.z_radius, self.xy_radius, self.xy_radius)

        def dstslice(offset, limit):
            if offset < 0:
                return slice(0, limit+offset)
            else:
                return slice(offset, limit)

        def srcslice(offset, limit):
            if offset < 0:
                return slice(0-offset, limit)
            else:
                return slice(0, limit-offset)

        # we will pack segment indices into 3 bytes as RGB uint8
        # so we can use them as 3D texture coordinates
        assert self.centroids.shape[0] < (2**24 - 1)

        self.kernel_slices = [
            (yoff, xoff)
            for yoff, xoff in [
                    (yoff, xoff)
                    for yoff in range(-self.xy_radius-1, self.xy_radius+2)
                    for xoff in range(-self.xy_radius-1, self.xy_radius+2)
            ]
            if ((xoff*xoff + yoff*yoff) < (self.xy_radius*self.xy_radius))
        ]
        # sort so we can splat centroids from center outward to assign texels
        self.kernel_slices.sort(key=lambda p: p[0]**2 + p[1]**2)

        self.kernel_radius = reduce(
            max,
            [
                max(abs(yoff), abs(xoff))
                for yoff, xoff in self.kernel_slices
            ],
            0
        )
        self.kernel_splat = np.zeros((2*self.kernel_radius+1, 2*self.kernel_radius+1), dtype=np.uint8)
        for yoff, xoff in self.kernel_slices:
            self.kernel_splat[self.kernel_radius+yoff, self.kernel_radius+xoff] = 1

        self.kernel_slices = [
            (dstslice(yoff, H), dstslice(xoff, W),
             srcslice(yoff, H), srcslice(xoff, W))
            for yoff, xoff in self.kernel_slices
        ]

        self.textures = None
        self.minval = self.data.min()
        self.maxval = self.data.max()
        self.value_norm = max(abs(self.minval), abs(self.maxval))
        self.last_Z = None
        self.last_channels = None
        self.channels = None
        self.set_view()

    def set_view(self, channels=None):
        if channels is not None:
            # use caller-specified sequence of channels
            assert type(channels) is tuple
            assert len(channels) <= 4
            self.channels = channels
        else:
            # default to first N channels u to 4 for RGBA direct mapping
            self.channels = tuple(range(0, min(self.data.shape[3], 4)))
        for c in self.channels:
            assert c >= 0
            assert c < self.data.shape[3]

    def _get_texture_format(self, nc, bps):
        return {
            (1,1): ('luminance', 'red'),
            (1,2): ('luminance', 'r16f'),
            (1,4): ('luminance', 'r16f'),
            (2,1): ('rg', 'rg'),
            (2,2): ('rg', 'rg32f'),
            (2,4): ('rg', 'rg32f'),
            (3,1): ('rgb', 'rgb'),
            (3,2): ('rgb', 'rgb16f'),
            (3,4): ('rgb', 'rgb16f'),
            (4,1): ('rgba', 'rgba'),
            (4,2): ('rgba', 'rgba16f'),
            (4,4): ('rgba', 'rgba16f')
        }[(nc, bps)]

    def get_textures(self, Z):
        """Returns ((image_texture, map_texture, measures_texture), map_ndarray, status3d_flat)"""
        I0 = self.data

        # choose size for texture data
        D, H, W = self.data.shape[0:3]
        C = len(self.channels)

        if self.textures is None:
            format, internalformat = self._get_texture_format(len(self.channels), 2)
            #print 'allocating textures...'
            self.textures = [
                gloo.Texture2D(shape=(H, W, C), format=format, internalformat=internalformat),
                gloo.Texture2D(shape=(H, W, 3), format='rgb', internalformat='rgb'),
                gloo.Texture3D(shape=(256,256,256,3), format='rgb', internalformat='rgb16f'),
                gloo.Texture3D(shape=(256,256,256), format='luminance', internalformat='red'),
            ]
            self.map_ndarray = np.zeros((H, W, 4), dtype=np.uint8)
            self.measures3d = np.zeros((256,256,256,3), dtype=np.float32)
            self.status3d = np.zeros((256,256,256), dtype=np.uint8)
            for texture in self.textures:
                texture.interpolation = 'nearest'
                texture.wrapping = 'clamp_to_edge'

            if self.statuses is not None:
                # if NPZ included initial status, load it here!
                status3d_flat = self.status3d.reshape((256**3,))
                status3d_flat[1:self.statuses.shape[0]+1] = self.statuses[:]

        elif self.last_channels == self.channels and self.last_Z == Z:
            #print 'reusing textures'
            return self.textures
        else:
            #print 'regenerating texture'
            pass

        # normalize image data for OpenGL [0,1.0] or [0,2**N-1] and zero black-level
        scale = 1.0/self.value_norm
        if I0.dtype == np.uint8 or I0.dtype == np.int8:
            tmpout = np.zeros((H, W, C), dtype=np.uint8)
            scale *= float(2**8-1)
        else:
            assert I0.dtype == np.float16 or I0.dtype == np.float32 or I0.dtype == np.uint16 or I0.dtype == np.int16
            tmpout = np.zeros((H, W, C), dtype=np.uint16 )
            scale *= (2.0**16-1)

        # pack selected channels into texture
        for i in range(C):
            tmpout[:,:,i] = (I0[Z,:,:,self.channels[i]].astype(np.float32) - self.minval) * scale

        self.last_channels = self.channels
        self.last_Z = Z
        self.textures[0].set_data(tmpout)

        # splat intersecting segment IDs into map texture
        tmpout = np.zeros((H, W, 4), dtype=np.uint8)
        self.map_ndarray[:,:,:] = 0

        indices = ((self.centroids[:,0] >= Z - self.z_radius) * (self.centroids[:,0] <= Z + self.z_radius)).nonzero()[0]

        if indices.shape[0] > len(self.kernel_slices):
            # optimized for a bunch of points spread around a little
            for i in range(indices.shape[0]):
                idx = indices[i]
                y, x = self.centroids[idx,1:3]
                idx += 1 # offset indices 0..N-1 as 1..N
                tmpout[y, x, 0] = idx % 2**8
                tmpout[y, x, 1] = (idx // 2**8) % 2**8
                tmpout[y, x, 2] = (idx // 2**16) % 2**8

            for dyslc, dxslc, syslc, sxslc in self.kernel_slices:
                dst = self.map_ndarray[dyslc, dxslc, :]
                dst_not_filled = (
                    (dst[:,:,0] == 0)
                    * (dst[:,:,1] == 0)
                    * (dst[:,:,2] == 0)
                )[:,:,None]
                dst += tmpout[ syslc, sxslc, :] * dst_not_filled
            # end loop (synspy#33 was indentation regression on previous line!)
        else:
            # optimized for fewer points spread around a lot
            for i in range(indices.shape[0]):
                idx = indices[i]
                y, x = self.centroids[idx,1:3]
                idx += 1 # offset 0..N-1 as 1..N
                r = idx % 2**8
                g = (idx // 2**8) % 2**8
                b = (idx // 2**16) % 2**8
                if self.kernel_radius < y < (self.map_ndarray.shape[0] - self.kernel_radius) \
                   and self.kernel_radius < x < (self.map_ndarray.shape[1] - self.kernel_radius):
                    dslc = (
                        slice(y-self.kernel_radius, y+self.kernel_radius+1),
                        slice(x-self.kernel_radius, x+self.kernel_radius+1),
                    )
                    dst_not_filled = (
                        (self.map_ndarray[dslc + (0,)] == 0)
                        * (self.map_ndarray[dslc + (1,)] == 0)
                        * (self.map_ndarray[dslc + (2,)] == 0)
                    )
                    self.map_ndarray[dslc + (0,)] = r * dst_not_filled * self.kernel_splat + self.map_ndarray[dslc + (0,)]
                    self.map_ndarray[dslc + (1,)] = g * dst_not_filled * self.kernel_splat + self.map_ndarray[dslc + (1,)]
                    self.map_ndarray[dslc + (2,)] = b * dst_not_filled * self.kernel_splat + self.map_ndarray[dslc + (2,)]

        # pack measures into 3D grid
        meas3d_flat = self.measures3d.reshape((256**3, 3))
        meas3d_flat[1:self.measures.shape[0]+1,0:2] = self.measures[:,0:2]
        if self.measures.shape[1] > 4:
            meas3d_flat[1:self.measures.shape[0]+1,2] = self.measures[:,4]

        # renormalize to [0,1]
        self.measures3d[:,:] = self.measures3d[:,:] * (1.0/self.value_norm)

        # pack statuses into 3D grid
        status3d_flat = self.status3d.reshape((256**3,))

        self.textures[1].set_data(self.map_ndarray)
        self.textures[2].set_data(self.measures3d)
        self.textures[3].set_data(self.status3d)
        
        return self.textures, self.map_ndarray, status3d_flat

# prepare a simple quad to cover the viewport
quad = np.zeros(4, dtype=[
    ('a_position', np.float32, 2),
    ('a_texcoord', np.float32, 2)
])
quad['a_position'] = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
quad['a_texcoord'] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

vert_shader = """
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main()
{
   v_texcoord = a_texcoord;
   gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

def frag_shader(**kwargs):
    return """
uniform sampler2D u_image_texture;
uniform sampler2D u_map_texture;
uniform sampler3D u_measures_cube;
uniform sampler3D u_status_cube;
uniform float u_gain;
uniform float u_drag_button;
uniform float u_feature_level;
uniform float u_neighbor_level;
uniform float u_black_level;
uniform vec3 u_pick;
uniform vec2 u_paint_center;
uniform vec2 u_paint_radii2_inv;
varying vec2 v_texcoord;

void main()
{
   vec4 pixel = texture2D(u_image_texture, v_texcoord);
   vec4 segment = texture2D(u_map_texture, v_texcoord);
   vec4 picked;
   vec4 measures;
   vec4 result;
   vec2 ellipse_test;
   float status;

   picked = vec4(u_pick.r, u_pick.g, u_pick.b, 0.0); // picked segment ID

   pixel.rgba = clamp(pixel.rgba - u_black_level, 0.0, 1.0);
   result.rgba = %(colorxfer)s;

   ellipse_test = v_texcoord.xy - u_paint_center;
   ellipse_test = ellipse_test * ellipse_test;
   ellipse_test = ellipse_test * u_paint_radii2_inv;

   if ( any(greaterThan(segment.rgb, vec3(0))) ) {
     // non-zero segment ID means we are in a segment...
     measures = texture3D(u_measures_cube, segment.rgb);
     status = texture3D(u_status_cube, segment.rgb).r * 255;

     if ( all(equal(segment.rgb, picked.rgb)) ) {
       // voxel is part of segment currently picked by user
       if (status == 5) {
         // segment is forced off, clickable
         result.rgb = %(pick_off)s;
       }
       else if (status == 7) {
         // segment is forced on, clickable
         result.rgb = %(pick_on)s;
       }
       else {
         result.rgb = %(pick_def)s;
       }
     }
     else {
       // voxel is part of segment not currently picked
       if (status == 5) {
         // segment is forced off, clickable
         result.rgb = %(off)s;
       }
       else if (status == 7) {
         // segment is forced on, clickable
         result.rgb = %(on)s;
       }
       else if ( measures.r >= u_feature_level && measures.g <= u_neighbor_level ) {
         // segment is within range
         result.rgb = %(inrange)s;
       }

     }
   }
   else {
     if ( u_drag_button > 0 && (ellipse_test.x + ellipse_test.y) < 1.0 ) {
       result.rgb = vec3(1,1,1) - result.rgb;
     }
   }

   gl_FragColor = result;
}
""" % kwargs

def adjust_level(uniform, attribute, step=0.0005, altstep=None, tracenorm=True):
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
            
            self.program[uniform] = level

            if tracenorm:
                level = level * self.vol_slicer.value_norm
            self.trace(attribute, level)

            self.update()

        wrapper.__doc__ = origmethod.__doc__
        return wrapper
    return helper

class Canvas(app.Canvas):

    def __init__(self, filename):
        self.basename = os.path.basename(filename)
        assert self.basename.endswith('.npz')
        self.accession_id = self.basename[0:-4]
        self.dump_prefix = './%s.' % self.accession_id
        self.dump_prefix = os.getenv('DUMP_PREFIX', self.dump_prefix)
        
        self.vol_slicer = SynspyImageManager(filename)
        self.vol_slicer.set_view()

        D, H, W, Nc = self.vol_slicer.data.shape

        app.Canvas.__init__(self, size=(min(800, W), min(800, H)), keys='interactive')
        self._hud_timer = None
        self.hud_items = []
        
        self.program = gloo.Program()
        self.program.bind(gloo.VertexBuffer(quad))
        self.textures, self.segment_map, self.segment_status = self.vol_slicer.get_textures(0)
        self.program['u_image_texture'] = self.textures[0]
        self.program['u_map_texture'] = self.textures[1]
        self.program['u_measures_cube'] = self.textures[2]
        self.program['u_status_cube'] = self.textures[3]

        self.mouse_button_offset = 0

        self.key_press_handlers = {
            'B': self.toggle_blend,
            #'D': self.dump_or_report,
            'D': self.report,
            'E': self.toggle_erase,
            'F': self.adjust_feature_level,
            'G': self.adjust_gain,
            'H': self.help,
            #'L': self.load_csv,
            'N': self.adjust_neighbor_level,
            'P': self.toggle_paint,
            'R': self.reset,
            'T': self.adjust_black_level,
            'Up': self.adjust_depth,
            'Down': self.adjust_depth,
            'Left': self.adjust_paint_zoom,
            'Right': self.adjust_paint_zoom,
        }

        self.frag_shaders = [
            # green linear with binary segments (picked is brighter)
            (
                'green linear with binary segments',
                frag_shader(
                    colorxfer='vec4(0.0, pixel.r * u_gain, 0.0, 1.0)',
                    pick_off= 'vec3(0.2, 1, 1)',
                    pick_on=  'vec3(1, 0.2, 1)',
                    pick_def= 'vec3(1, 1, 0.2)',
                    off=      'vec3(0.2, 0.6, 0.6)',
                    on=       'vec3(0.6, 0.2, 0.6)',
                    inrange=  'vec3(0.6, 0.6, 0.2)',
                )
            ),
            (
                'gray intensity only',
                frag_shader(
                    colorxfer='vec4(pixel.r, pixel.r, pixel.r, 1.0/u_gain) * u_gain',
                    pick_off= 'vec3(0.2, 1, 1)',
                    pick_on=  'vec3(1, 0.2, 1)',
                    pick_def= 'vec3(1, 1, 1)',
                    off=      'result.rgb',
                    on=       'result.rgb',
                    inrange=  'result.rgb',
                )
            ),
            (
                'gray intensity with magenta=ON synapses',
                frag_shader(
                    colorxfer='vec4(pixel.r, pixel.r, pixel.r, 1.0/u_gain) * u_gain',
                    pick_off= 'vec3(0.2, 1, 1)',
                    pick_on=  'vec3(1, 0.2, 1)',
                    pick_def= 'vec3(1, 1, 1)',
                    off=      'result.rgb',
                    on=       'vec3(result.g, 0.2 * result.g, result.g) * 1.2',
                    inrange=  'result.rgb',
                )
            ),
            (
                'gray intensity with magenta=ON cyan=OFF yellow=neither synapses',
                frag_shader(
                    colorxfer='vec4(pixel.r, pixel.r, pixel.r, 1.0/u_gain) * u_gain',
                    pick_off= 'vec3(0.2, 1, 1)',
                    pick_on=  'vec3(1, 0.2, 1)',
                    pick_def= 'vec3(1, 1, 1)',
                    off=      'vec3(0.2 * result.g, result.g, result.g) * 1.2',
                    on=       'vec3(result.g, 0.2 * result.g, result.g) * 1.2',
                    inrange=  'vec3(result.g, result.g, 0.2 * result.g) * 1.2',
                )
            )
        ]

        self.reset()

        self.text_hud = visuals.TextVisual('', color="white", font_size=12 * self.font_scale, anchor_x="left", bold=True)
        if not hasattr(self.text_hud, 'transforms'):
            # temporary backwards compatibility
            self.text_hud_transform = visuals.transforms.TransformSystem(self)

        self.prev_size = None
        self.set_viewport1((min(800, W), min(800, H)))
        
        self.show()

        # auto-load
        try:
            self.load_csv()
        except:
            pass

        # auto-dump
        self.auto_dumped = False
        @atexit.register
        def shutdown():
            if not self.auto_dumped:
                sys.stderr.write('caught exit... dumping CSV...')
                self.dump_csv()
                sys.stderr.write('done.\n')

    def on_close(self, event):
        self.trace('Window close', 'dumping state...')
        self.on_draw()
        self.dump_csv()
        self.auto_dumped = True
        self.trace('Window close', 'closing...')
        self.on_draw()
        self.hud_drain(drain_all=True)

    def trace(self, attribute, value, mesg=None):
        if type(value) is float:
            value = '%.2f' % value
        self.hud_items.append(
            (attribute, mesg if mesg is not None else "%s: %s" % (attribute, value), datetime.datetime.now())
        )

    def hud_drain(self, drain_all=False):
        # remove duplicate keys retaining newest item (latest in list)
        hud_items = []
        hud_keys = set()
        for k, mesg, ts in self.hud_items[::-1]:
            if k not in hud_keys:
                hud_keys.add(k)
                hud_items.append((k, mesg, ts))
        hud_items.reverse()

        # expire old content
        now = datetime.datetime.now()
        items = [
            (k, mesg, ts, ((now - ts).total_seconds() <= self.hud_age_s) and not drain_all)
            for k, mesg, ts in hud_items
        ]
        self.hud_items = [
            (k, mesg, ts)
            for k, mesg, ts, retain in items
            if retain
        ]
        for k, mesg, ts, retain in items:
            if not retain:
                print(mesg)
        
    def help(self, event):
        """Show help information (H)."""
        self.hud_drain(drain_all=True)
        handlers = list(dict([ (handler, key) for key, handler in list(self.key_press_handlers.items()) ]).items())
        handlers.sort(key=lambda p: (len(p[1]), p[1]))

        for mesg in [
                'Click segments with mouse to toggle classification status.',
#                'Drag primary (i.e. left) button to set "true" segments en masse.',
 #               'Drag secondary (i.e. right) button to clear segments en masse.',
                'Keyboard commands:',
                '  Exit (ESC).',
        ] + [ '  ' + p[0].__doc__ for p in handlers ]:
            self.trace(mesg, '', mesg)

        self.update()

    def get_hud_text_pos_lists(self):
        self.hud_drain()

        # build up display content
        hud_text = []
        hud_pos = []
        for k, mesg, ts in self.hud_items:
            hud_text.append(mesg)
            hud_pos.append( np.array((5 * self.font_scale, (12 + len(hud_text) * 15) * self.font_scale)) )

        return hud_text, hud_pos

    def dump_or_report(self, event):
        """Dump current parameters (d) or segments CSV file (D)."""
        if 'Shift' in event.modifiers:
            self.dump_csv(event)
        else:
            self.report(event)
            
    def report(self, event=None):
        """Dump (d) current parameters."""
        self.trace('Z', '%d of %d' % (self.vol_slicer.last_Z, self.vol_slicer.data.shape[0]))
        self.trace('blend mode', self.frag_shaders[self.current_shader][0])
        
        for attribute, value in [
                ('gain', self.gain),
                ('black_level', self.black_level * self.vol_slicer.value_norm),
                ('feature_level', self.feature_level * self.vol_slicer.value_norm),
                ('neighbor_level', self.neighbor_level * self.vol_slicer.value_norm),
                ('paint_zoom', self.paint_zoom),
        ]:
            self.trace(attribute, value)

        if event is not None:
            self.update()
        
    def reset(self, event=None):
        """Reset (r) rendering mode and thresholds."""
        self.hud_drain(drain_all=True)
        self.hud_age_s = 10

        self.sticky_drag = False
        self.prev_paint_center = (-1000, -1000)
        self.paint_center = self.prev_paint_center

        self.prev_pick_idx = 0
        self.pick_idx = 0
        self.drag_button = 0
        self.paint_zoom = 1.0

        self.gain = 1.0
        self.feature_level = 0.0 if self.vol_slicer.statuses is None else self.vol_slicer.measures[:,0].min() / self.vol_slicer.value_norm
        self.neighbor_level = 0.0 if self.vol_slicer.statuses is None else self.vol_slicer.measures[:,1].max() / self.vol_slicer.value_norm
        self.black_level = 0.0
        # for compatibility with 3d viewer, save these unused values
        self.saved_opacity = 0.8
        self.saved_redlvl = 0.0
        self.saved_toplvl = self.vol_slicer.maxval

        self.current_shader = 0
        self.program.set_shaders(vert_shader, self.frag_shaders[self.current_shader][1])

        self.program['u_gain'] = self.gain
        self.program['u_black_level'] = self.black_level
        self.program['u_feature_level'] = self.feature_level
        self.program['u_neighbor_level'] = self.neighbor_level
        self.program['u_paint_center'] = self.paint_center

        self.font_scale = 1

        self.report()
        
        if self._hud_timer is not None:
            self._hud_timer.stop()
            self._hud_timer = None

        if event is not None:
            self.update()

    def set_viewport1(self, window_size):
        if self.prev_size == window_size:
            return

        self.prev_size = window_size

        ww, wh = list(map(float, window_size))
        dh, dw = list(map(float, self.vol_slicer.data.shape[1:3]))

        self.paint_radii_tex = (
            2.5 / self.vol_slicer.properties['image_grid'][2],
            2.5 / self.vol_slicer.properties['image_grid'][1]
        )

        self.paint_radii_texn = (
            self.paint_radii_tex[0] / dw,
            self.paint_radii_tex[1] / dh,
        )

        # defer this to on_mouse_press so we can scale differently for each button
        self.program['u_paint_radii2_inv'] = (float("inf"), float("inf"))

        daspect = dw/dh
        if ww/wh > daspect:
            self.viewport1 = (ww - wh*daspect)/2, 0, wh*daspect, wh
        else:
            self.viewport1 = 0, (wh - ww/daspect)/2, ww, ww/daspect

        gloo.set_viewport(*self.viewport1)

    def adjust_gain(self, event):
        """Increase (G) or decrease (g) image intensity gain."""
        if event.key == 'G':
            if 'Shift' in event.modifiers:
                self.gain *= 1.25
            else:
                self.gain *= 1./1.25

        self.trace('gain', self.gain)
        self.program['u_gain'] = self.gain
        self.update()

    def toggle_blend(self, event):
        """Cycle forward (b) or backward (B) through blending modes."""
        if 'Shift' in event.modifiers:
            sign = -1
        else:
            sign = 1
        self.current_shader = (self.current_shader + 1 * sign) % len(self.frag_shaders)
        self.program.set_shaders(vert_shader, self.frag_shaders[self.current_shader][1])
        self.trace('blend mode', self.frag_shaders[self.current_shader][0])
        self.update()

    @adjust_level('u_feature_level', 'feature_level')
    def adjust_feature_level(self, event):
        """Increase (F) or decrease (f) feature threshold."""
        pass
        
    @adjust_level('u_neighbor_level', 'neighbor_level')
    def adjust_neighbor_level(self, event):
        """Increase (N) or decrease (n) neighborhood threshold."""
        pass

    @adjust_level('u_black_level', 'black_level')
    def adjust_black_level(self, event):
        """Increase (T) or decrease (t) black level aka transparency zero point."""
        pass

    def on_resize(self, event):
        self.set_viewport1(event.physical_size)

        if hasattr(self.text_hud, 'transforms'):
            self.text_hud.transforms.configure(canvas=self, viewport=(0, 0) + event.physical_size)
        else:
            # temporary backwards compatibility
            self.text_hud_transform = visuals.transforms.TransformSystem(self)

        self.update()

    def on_key_press(self, event):
        handler = self.key_press_handlers.get(event.key)
        if handler:
            handler(event)
        elif event.key in ['Shift', 'Escape', 'Alt', 'Control']:
            pass
        else:
            print('no handler for key %s' % event.key)

    def on_mouse_wheel(self, event):
        Z = self.vol_slicer.last_Z - event.delta[1]
        Z = clamp(Z, 0, self.vol_slicer.data.shape[0] - 1)
        if Z != self.vol_slicer.last_Z:
            self.vol_slicer.get_textures(int(Z))
            self.trace('Z', '%d of %d' % (int(Z), self.vol_slicer.data.shape[0]))
            self.update()

    def adjust_depth(self, event):
        """Increase (up-arrow) or decrease (down-arrow) Z slice position."""
        if event.key == 'Up':
            delta = -1
        else:
            delta = 1

        Z = self.vol_slicer.last_Z + delta
        Z = clamp(Z, 0, self.vol_slicer.data.shape[0] - 1)
            
        self.vol_slicer.get_textures(Z)
        self.trace('Z', '%d of %d' % (int(Z), self.vol_slicer.data.shape[0]))
        self.update()

    def adjust_paint_zoom(self, event):
        """Increase (right-arrow) or decrease (left-arrow) paint brush zoom factor."""
        if event.key == 'Right':
            self.paint_zoom *= 1.1
        else:
            self.paint_zoom *= 1/1.1
        if self.drag_button:
            self.program['u_paint_radii2_inv'] = tuple([
                # scale radii by drag_button which is 1 or 2
                1.0 / (self.drag_button * self.paint_zoom * x)**2
                for x in self.paint_radii_texn
            ])
        self.trace('paint_zoom', self.paint_zoom)
        self.update()

    def find_paint_center(self, event):
        X0, Y0, W, H = [ float(x) for x in self.viewport1 ]
        dh, dw = [ float(x) for x in self.segment_map.shape[0:2] ]
        x, y = event.pos
        x = (x * self.pixel_scale - X0) * (1.0/W)
        y = (y * self.pixel_scale - Y0) * (1.0/H)
        y = 1.0 - y

        self.paint_center = (x, y)
        self.program['u_paint_center'] = self.paint_center

        #if event.button and not self.sticky_drag:
        #    self.drag_button = event.button + self.mouse_button_offset
        self.pick_idx = 0

        if self.paint_center != self.prev_paint_center \
           or self.pick_idx != self.prev_pick_idx:
            self.update()
            self.prev_paint_center = self.paint_center
            self.prev_pick_idx = self.pick_idx

    def paint_segments(self, event):
        X0, Y0, W, H = self.viewport1
        dh, dw = list(map(float, self.segment_map.shape[0:2]))
        x, y = event.pos
        x = int((x * self.pixel_scale - X0) * (dw/W))
        y = int((y * self.pixel_scale - Y0) * (dh/H))
        y = int(dh) - y # flip y to match norm. device coord system
        xr, yr = self.paint_radii_tex

        # scale radii by button number 1 or 2
        xr *= self.drag_button * self.paint_zoom
        yr *= self.drag_button * self.paint_zoom

        def mkslc(c, r, w):
            return slice(
                int(max(c - r - 1, 0)),
                int(min(c + r + 2, w))
            )

        # slice out bounding box for paint brush circle (ellipse in texture space)
        map_slc = (
            mkslc(y, yr, dh),
            mkslc(x, xr, dw),
            slice(0, 3)
        )
        smap = self.segment_map[map_slc]

        # create map of X, Y coords for each pixel in segment map bounding box
        coords = np.ones( (map_slc[0].stop - map_slc[0].start, map_slc[1].stop - map_slc[1].start, 2), np.float32)
        coords[:,:,0] *= np.arange(map_slc[0].start, map_slc[0].stop)[:,None] # Y coord range
        coords[:,:,1] *= np.arange(map_slc[1].start, map_slc[1].stop)[None,:] # X coord range

        # turn it into a 1D problem
        coords_y = coords[:,:,0].flatten()
        coords_x = coords[:,:,1].flatten()
        smap = smap.reshape(-1,3)

        # efficiently solve ellipse equation for area under brush
        coords_y = coords_y - np.float32(y)
        coords_y2 = coords_y * coords_y
        coords_x = coords_x - np.float32(x)
        coords_x2 = coords_x * coords_x
        in_ellipse = (
            (coords_y2 * np.float32(1.0/yr**2)
             + coords_x2 * np.float32(1.0/xr**2))
            < 1.0
        )
        in_ellipse_segmented = (
            in_ellipse
            * ((smap[:,0] > 0)
               + (smap[:,1] > 0)
               + (smap[:,2] > 0))
        ).nonzero()[0]

        # extract segment IDs under brush
        paint_rgb = smap[in_ellipse_segmented,:]
        paint_idx = paint_rgb[:,0] + paint_rgb[:,1] * 2**8 + paint_rgb[:,2] * 2**16
        paint_idx, unique_idx = np.unique(paint_idx, return_index=True)
        paint_rgb = paint_rgb[(unique_idx, slice(None))]

        # set new segment status for affected segment IDs
        newval = {1: 7, 2: 0}[self.drag_button]
        anychanged = False
        for i in range(paint_idx.shape[0]):
            core = self.vol_slicer.measures[paint_idx[i]-1,0] / self.vol_slicer.properties['measures_divisor']
            hollow = self.vol_slicer.measures[paint_idx[i]-1,1] / self.vol_slicer.properties['measures_divisor']
            if newval == 7 and (core < self.feature_level or hollow > self.neighbor_level):
                # don't paint segments out of range
                continue
            if self.segment_status[paint_idx[i]] != newval:
                self.segment_status[paint_idx[i]] = newval
                self.textures[3].set_data(
                    np.array([[[newval]]], dtype=np.uint8),
                    offset=tuple(paint_rgb[i, ::-1]),
                    copy=True
                )
                anychanged = True

        if anychanged:
            self.update()
 
    def find_pick_idx(self, event):
        X0, Y0, W, H = self.viewport1
        dh, dw = list(map(float, self.segment_map.shape[0:2]))
        x, y = event.pos
        x = int((x * self.pixel_scale - X0) * (dw/W))
        y = int((y * self.pixel_scale - Y0) * (dh/H))
        y = int(dh) - y # flip y to match norm. device coord system

        if 0 <= x < dw and 0 <= y < dh:
            self.pick_rgb = self.segment_map[y, x, 0:3]
            self.pick_idx = int(
                self.segment_map[y, x, 0]
                + self.segment_map[y, x, 1] * 2**8
                + self.segment_map[y, x, 2] * 2**16
            )
        else:
            self.pick_rgb = self.segment_map[0, 0, 0:3]
            self.pick_idx = 0

        if self.pick_idx != self.prev_pick_idx:
            self.update()
            self.prev_pick_idx = self.pick_idx

    def on_mouse_press(self, event):
        self.find_pick_idx(event)
        if event.button == 0:
            self.mouse_button_offset = 1

    def toggle_paint(self, event):
        """Start (P) or stop (p) paint mode."""
        if 'Shift' in event.modifiers:
            self.start_drag(1)
            self.sticky_drag = True
        else:
            self.stop_drag()
        self.update()

    def toggle_erase(self, event):
        """Start (E) or stop (e) erase mode."""
        if 'Shift' in event.modifiers:
            self.start_drag(2)
            self.sticky_drag = True
        else:
            self.stop_drag()
        self.update()

    def start_drag(self, button):
        self.drag_button = button + self.mouse_button_offset
        self.program['u_paint_radii2_inv'] = tuple([
            # scale radii by drag_button which is 1 or 2
            1.0 / ((button + self.mouse_button_offset) * self.paint_zoom * x)**2
            for x in self.paint_radii_texn
        ])

    def on_mouse_release(self, event):
        if self.drag_button == 0:
            if self.pick_idx > 0:
                b = {
                    # state-transitions for clicking centroids
                    0: 7, 5: 7, 7: 0
                }[self.segment_status[self.pick_idx]]
                self.segment_status[self.pick_idx] = b # track state for ourselves
                self.textures[3].set_data( # poke into status_cube texture for renderer
                    np.array([[[b]]], dtype=np.uint8),
                    offset=tuple(self.pick_rgb[::-1]),
                    copy=True
                )
        self.stop_drag()
        self.update()

    def stop_drag(self):
        self.drag_button = 0
        self.sticky_drag = False

    def csv_file_name(self):
        if self.vol_slicer.properties['synspy_nuclei_mode']:
            return self.dump_prefix + '_nucleic_only.csv'
        else:
            return self.dump_prefix + '_synaptic_only.csv'

    def load_csv(self, event=None):
        """Load (L) segment classification from CSV file."""
        if self.vol_slicer.statuses is not None:
            return
        csvfile = self.csv_file_name()
        try:
            status, saved_params = load_segment_status_from_csv(
                self.vol_slicer.centroids,
                self.vol_slicer.slice_origin,
                csvfile
            )
            # shift everything to index+1
            self.segment_status[1:self.vol_slicer.centroids.shape[0]+1] = status[:]
            self.textures[3].set_data(self.segment_status.reshape((256,256,256)))
            self.trace(
                csvfile, 'loaded with %d/%d segments overridden' % (
                    (self.segment_status > 0).nonzero()[0].shape[0],
                    self.vol_slicer.centroids.shape[0]
                )
            )
            if saved_params is not None:
                for attr, col in [
                        ("feature_level", "raw core"),
                        ("neighbor_level", "raw hollow"),
                        ("black_level", "DoG core"),
                ]:
                    value = float(saved_params[col])
                    self.trace(attr, value)
                    value = value / self.vol_slicer.properties['measures_divisor']
                    setattr(self, attr, value)
                    self.program['u_%s' % attr] = value

                self.saved_toplvl = float(saved_params['DoG hollow'])
                if self.vol_slicer.measures.shape[1] == 5:
                    self.saved_redlvl = float(saved_params['red'])
                self.saved_opacity = float(saved_params['override'])
        except Exception as e:
            self.trace(csvfile, 'load failed: ' + str(e))
            et, ev, tb = sys.exc_info()
            print(traceback.format_exception(et, ev, tb))
            raise
        self.update()

    def dump_csv(self, event=None):
        """Dump (D) segment CSV file."""
        if self.vol_slicer.statuses is not None:
            return
        csvfile = self.csv_file_name()
        try:
            saved_params = {
                'Z': 'saved',
                'Y': 'params',
                'X': ('(core, vicinity, zerolvl, toplvl,'
                      + (' autfl,' if self.vol_slicer.measures.shape[1] == 5 else '')
                      + ' transp):'),
                'raw core': self.feature_level * self.vol_slicer.properties['measures_divisor'],
                'raw hollow': self.neighbor_level * self.vol_slicer.properties['measures_divisor'],
                'DoG core': self.black_level * self.vol_slicer.properties['measures_divisor'],
                'DoG hollow': self.saved_toplvl,
                'override': self.saved_opacity,
            }
            if self.vol_slicer.measures.shape[1] == 5:
                saved_params['red'] = self.saved_redlvl

            dump_segment_info_to_csv(
                self.vol_slicer.centroids,
                self.vol_slicer.measures,
                self.segment_status[1:self.vol_slicer.centroids.shape[0]+1],
                self.vol_slicer.slice_origin,
                csvfile,
                saved_params=saved_params,
                all_segments=False
            )
            self.trace(
                csvfile, 'dumped with %d/%d segments overridden' % (
                    (self.segment_status > 0).nonzero()[0].shape[0],
                    self.vol_slicer.centroids.shape[0]
                )
            )
        except Exception as e:
            self.trace(csvfile, 'dump failed: ' + str(e))
        self.update()
            
    def on_mouse_move(self, event):
        # drag-based paint and erase disabled at request of BDemps.
        if event.is_dragging and False:
            if event.button == 0:
                self.mouse_button_offset = 1
            self.start_drag(event.button)

        self.find_paint_center(event)
        if self.drag_button != 0:
            self.paint_segments(event)
        else:
            self.find_pick_idx(event)

    def on_timer(self, event):
        self.update()
            
    def on_draw(self, event=None):
        self.program['u_pick'] = (
            float(self.pick_idx % 256) / 255.0,
            float(self.pick_idx // 2**8 % 256) / 255.0,
            float(self.pick_idx // 2**16 % 256) / 255.0
        )
        self.program['u_drag_button'] = self.drag_button

        # draw image slice
        gloo.set_clear_color((0, 0, 0), 1.0)
        gloo.clear(color=True, depth=True)
        gloo.set_viewport(*self.viewport1)
        self.program.draw('triangle_strip')

        # draw HUD
        hud_lists = self.get_hud_text_pos_lists()
        if hud_lists[0]:
            gloo.set_viewport((0, 0) + self.physical_size)
            
            self.text_hud.text, self.text_hud.pos = hud_lists

            if hasattr(self.text_hud, 'transforms'):
                self.text_hud.draw()
            else:
                self.text_hud.draw(self.text_hud_transform)

            if self._hud_timer is not None:
                self._hud_timer.stop()
                self._hud_timer = None

            now = datetime.datetime.now()
            self._hud_timer = app.Timer(
                interval=(now - self.hud_items[0][2]).total_seconds(),
                iterations=1,
                start=True,
                app=self.app,
                connect=self.on_timer
            )

        if event is None:
            self.swap_buffers()


def main():
    c = Canvas(sys.argv[1])
    c.show()
    app.run()

if __name__ == '__main__':
    sys.exit(main())
