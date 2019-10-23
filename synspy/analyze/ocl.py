
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import os
from numpy import array, concatenate, float32, int32, empty, uint32
import numpy

import pyopencl as cl
import pyopencl.array as cl_array
from functools import reduce

# ugly hack: try to choose context automatically
#
# 1. prefer Intel so we don't hit GPU RAM limit as easily,
# 2. fallback to NVIDIA if available
# 3. fallback to PyOpenCL automatic mode if we don't know what to do
platforms = cl.get_platforms()

platform_devices = [
    [ d for d in p.get_devices() if d.available ]
    for p in platforms
]

vendor_platform_idx = {
    platform_devices[i][0].vendor_id: i
    for i in range(len(platform_devices))
    if platform_devices[i]
}

idx = vendor_platform_idx.get(
    32902, # prefer Intel
    vendor_platform_idx.get(4318) # or NVIDIA
)
if idx is not None:
    answers = ['%d' % idx]
    print('synspy: choosing %s' % platforms[idx])
else:
    answers = None

ctx = cl.create_some_context(answers=answers)

def product(terms):
    return reduce(lambda a, b: a*b, terms, 1)

def filter_dev_float32(clq, src_dev, kernel_dev, dst_dev, axis=0, reducer="ADDER"):
    """Filter ND device source by kernel along designated axis.

       All device buffers are float32 typed.
    """
    ndim = len(src_dev.shape)
    assert ndim > 1, \
        "data must have 2 or more dimensions"
    kw = kernel_dev.shape[0]
    assert src_dev.shape[axis] == (dst_dev.shape[axis] + kw - 1), \
        "destination shape must be smaller than source shape by kernel width-1 on convolution axis"
    assert (kw % 2) == 1, "kernel length must be odd"
    assert (src_dev.shape[0:axis] + src_dev.shape[axis+1:]) \
        == (dst_dev.shape[0:axis] + dst_dev.shape[axis+1:]), \
        "destination shape must match source shape on all other axes"

    ndim = len(src_dev.shape)

    work_shape = array(src_dev.shape, dtype=int32)
    work_shape[axis] = 1
    nitems = product(work_shape)

    strides = array([ s / src_dev.dtype.itemsize for s in src_dev.strides ], dtype=int32)
    dst_strides = array([ s / dst_dev.dtype.itemsize for s in dst_dev.strides ], dtype=int32)

    remainder = 0
    vecsize = 16

    # compile-in static problem shape for best performance
    if axis == (ndim-1):
        CL = """
    #define N_DIM %(ndim)d
    #define AXIS %(axis)d
    #define STRIDE %(stride)d
    #define N_WEIGHTS %(kw)d
    #define N_INPUTS %(width)d

    #define ADDER(a,b) (a) + (b)

    __kernel void conv1d(
       __global float*            dst,
       __global const float*      src,
       __constant float*          weights,
       __constant int*            work_shape,
       __constant int*            work_strides,
       __constant int*            dst_strides)
    {
       float sbuf[N_WEIGHTS];
       int coord[N_DIM];
       float accum;
       int spos = 0; // scans through src
       int dpos = 0; // scans through dst
       int i, j;
       int index = get_global_id(0);

       // find work item start position
       // dimension order is "C" style row-major
       for (i=N_DIM-1; i>=0; i--)
       {
          coord[i] = index %% work_shape[i];
          index = index / work_shape[i];
       }
       for (i=N_DIM-1; i>=0; i--)
       {
          spos += coord[i] * work_strides[i];
          dpos += coord[i] * dst_strides[i];
       }

       // pre-roll input buffer
       i=0;
       while (i<N_WEIGHTS-1)
       {
          sbuf[i] = src[spos];
          spos += STRIDE;
          i++;
       }

       // pipelined convolution kernel
       // retires one output for each input step
       while (i<N_INPUTS)
       {
          // fill out input buffer
          sbuf[i%%N_WEIGHTS] = src[spos];

          i++;

          // compute one output value using buffered inputs and weights
          accum = weights[0] * sbuf[(0 + i)%%N_WEIGHTS];
          for (j=1; j<N_WEIGHTS; j++)
          {
             // i also acts as start offset for sbuf modular FIFO!
             accum = %(reducer)s(accum, weights[j] * sbuf[(j + i)%%N_WEIGHTS]);
          }
          dst[dpos] = accum;

          spos += STRIDE;
          dpos += STRIDE;
       }
    }
        """
    else:
        # vectorize vecsize scalar problems on major axis
        remainder = work_shape[ndim-1] % vecsize
        work_shape[ndim-1] /= vecsize

        if remainder:
            work_shape[ndim-1] += 1
            nitems = product(work_shape)
        
        CL = """
    #define N_DIM %(ndim)d
    #define AXIS %(axis)d
    #define STRIDE %(stride)d
    #define N_WEIGHTS %(kw)d
    #define N_INPUTS %(width)d
    #define PARTIAL %(remainder)d
    #define VECSIZE %(vecsize)d

    #define ADDER(a,b) (a) + (b)

    // vectorized code for axis other than major storage axis
    // i.e. we can process multiple axial convolutions in a vector

    // handle partial or whole vector load and store... this code will in-line
    float%(vecsize)d pvload%(vecsize)d(
       int                   is_partial,
       __global const float* src)
    {
       if (is_partial) {
          float pbuf[%(vecsize)d];
          int i;

          for (i=0; i<PARTIAL; i++)
             pbuf[i] = src[i];
        
          for (i=PARTIAL; i<%(vecsize)d; i++)
             pbuf[i] = 0;

          return vload%(vecsize)d(0, pbuf);
       }
       else {
          return vload%(vecsize)d(0, src);
       }
    }

    void pvstore%(vecsize)d(
       int              is_partial,
       float%(vecsize)d vec,
       __global float*  dst)
    {
       if (is_partial) {
          float pbuf[%(vecsize)d];
          int i;

          vstore%(vecsize)d(vec, 0, pbuf);

          for (i=0; i<PARTIAL; i++)
             dst[i] = pbuf[i];
       }
       else {
          vstore%(vecsize)d(vec, 0, dst);
       }
    }

    __kernel void conv1d(
       __global float*            dst,
       __global const float*      src,
       __constant float*          weights,
       __constant int*            work_shape,
       __constant int*            work_strides,
       __constant int*            dst_strides)
    {
       float%(vecsize)d sbuf[N_WEIGHTS];  // each vector component is a parallel problem
       int coord[N_DIM];
       float%(vecsize)d accum;
       int spos = 0; // scans through src
       int dpos = 0; // scans through dst
       int i, j;
       int index = get_global_id(0);
       int partial = 0;

       // find work item start position
       // dimension order is "C" style row-major
       for (i=N_DIM-1; i>=0; i--)
       {
          coord[i] = index %% work_shape[i];
          index = index / work_shape[i];
       }

       if ( (PARTIAL > 0) && (coord[N_DIM-1] == (work_shape[N_DIM-1] - 1)) )
       {
          // this work item is a partially filled vector
          partial = 1;
       }

       // each work unit does %(vecsize)d scalar problems so seek further over
       coord[N_DIM-1] *= %(vecsize)d;

       for (i=N_DIM-1; i>=0; i--)
       {
          spos += coord[i] * work_strides[i];
          dpos += coord[i] * dst_strides[i];
       }

       // pre-roll input buffer
       i=0;
       while (i<N_WEIGHTS - 1)
       {
          sbuf[i] = pvload%(vecsize)d(partial, src + spos);
          spos += STRIDE;
          i++;
       }

       // pipelined convolution kernel
       // retires one output vector for each input vector
       while (i<N_INPUTS)
       {
          // fill out input buffer
          sbuf[i%%N_WEIGHTS] = pvload%(vecsize)d(partial, src + spos);

          i++;

          // compute one output value using buffered inputs and weights
          accum = sbuf[(0 + i)%%N_WEIGHTS] * weights[0];
          for (j=1; j<N_WEIGHTS; j++)
          {
             // i also acts as start offset for sbuf modular FIFO!
             accum = %(reducer)s(accum, sbuf[(j + i)%%N_WEIGHTS] * weights[j]);
          }
          pvstore%(vecsize)d(partial, accum, dst + dpos);

          spos += STRIDE;
          dpos += STRIDE;
       }
    }
    """

    CL = CL % dict(
        axis=axis,
        ndim=ndim,
        reducer=reducer,
        kw=kw, 
        width=src_dev.shape[axis],
        stride=strides[axis],
        remainder=remainder,
        vecsize=vecsize,
        veczeros=",".join([ "0" for i in range(vecsize) ])
    )
    #print CL

    program = cl.Program(ctx, CL).build()

    #print "input %s using %d opencl work items in work-shape %s work strides %s dest strides %s steps %s stride %s" % (src_dev.shape, nitems, work_shape, strides, dst_strides, src_dev.shape[axis], strides[axis])

    shape_dev = cl_array.to_device(clq, work_shape)
    strides_dev = cl_array.to_device(clq, strides)
    dst_strides_dev = cl_array.to_device(clq, dst_strides)

    program.conv1d(clq, (nitems,), None, dst_dev.data, src_dev.data, kernel_dev.data, shape_dev.data, strides_dev.data, dst_strides_dev.data)

    return

def filterNx1d(src, kernels, reducer="ADDER", clq=None):
    """ND filter using 1D kernels

       Trims borders by filter kernel width in each dimension.

       if a command queue is passed as clq parameter, device results
       are returned so that the caller can compose a more complex
       on-device calculation before extracting results back to the
       host.

    """
    assert len(kernels) == len(src.shape)

    if clq is None:
        return_dev = False
        clq = cl.CommandQueue(ctx)
        
        if True:
            src = src.astype(float32, copy=False)
            src_dev = cl_array.empty( clq, src.shape, float32 )
            src_tmp = src_dev.map_to_host()
            src_tmp[...] = src[...] # reforms as contiguous
            del src_tmp
        else:
            src = src.astype(float32, copy=True)
            src_dev = cl_array.to_device(clq, src)
    else:
        return_dev = True
        src_dev = src

    for d in range(len(kernels)-1, -1, -1):
        kernel = array(kernels[d], dtype=float32)
        kernel_dev = cl_array.to_device(clq, kernel)
        assert kernel.ndim == 1
        assert (kernel.shape[0] % 2) == 1

        dst_dev = cl_array.empty(
            clq,
            src_dev.shape[0:d] + (src_dev.shape[d] - kernel.shape[0] + 1,) + src_dev.shape[d+1:],
            dtype=float32
            )

        filter_dev_float32(clq, src_dev, kernel_dev, dst_dev, axis=d, reducer=reducer)
        
        #dst_tmp = dst_dev.map_to_host(clq)
        #print dst_tmp
        #print "--"

        src_dev = dst_dev

    if return_dev:
        return dst_dev
    else:
        dst_tmp = dst_dev.map_to_host(clq)
        clq.finish()
        #dst = empty( dst_tmp.shape, float32 )
        #dst[...] = dst_tmp[...]
        return dst_tmp


def convNx1d(src, kernels, clq=None):
    """ND convolution using 1D kernels

       Trims borders by filter kernel width in each dimension.

    """
    return filterNx1d(src, kernels, reducer="ADDER", clq=clq)

def maxNx1d(src, lengths, clq=None):
    kernels = [ [ 1 for k in range(length) ] for length in lengths ]
    return filterNx1d(src, kernels, reducer="max", clq=clq)


def sum_labeled_dev(clq, src_dev, labels_dev, tmp_dev, dst_dev, min_label=1):
    """Sum source values for each label and write sums to destination array.

       tmp_dev shape (T, N) permits T parallel tasks to work on sums
       for up to N labels in range 0..N-1.

       tmp_dev must be zeroed by caller prior to call.

       final result will be in dst_dev.

    """
    assert src_dev.shape == labels_dev.shape
    assert len(tmp_dev.shape) == 2
    assert len(dst_dev.shape) == 1
    assert tmp_dev.shape[1] == dst_dev.shape[0]

    ncells = product(src_dev.shape)

    nitems = tmp_dev.shape[0] - 1
    nlabels = tmp_dev.shape[1]

    if nitems > ncells:
        # cannot have more workers than source data cells!
        nitems = ncells

    worksize = ncells / nitems
    workrem = ncells % nitems

    if workrem:
        nitems += 1

    CL = """
    __kernel void sum_cells(
       __global float*                dst,
       __global const %(data_type)s*  src,
       __global const %(label_type)s* labels,
       unsigned int                   num_labels,
       unsigned int                   min_label,
       unsigned int                   num_items,
       unsigned int                   work_size,
       unsigned int                   work_remains)
    {
       %(label_type)s cell_label;
       int index = get_global_id(0);
       int i, limit;
       
       if (work_remains > 0 && index == (num_items-1))
          limit = work_remains;
       else
          limit = work_size;

       for (i=0; i<limit; i++) {
          cell_label = labels[index*work_size + i];
          if (cell_label >= min_label && cell_label < num_labels) {
             dst[index*num_labels + cell_label] += src[index*work_size + i];
          }
       }
    }
    
    __kernel void sum_labels(
       __global float*        dst,
       __global const float*  src,
       unsigned int           num_labels,
       unsigned int           num_vals)
    {
       int index = get_global_id(0);
       int i;
       float accum = 0;
       
       for (i=0; i<num_vals; i++) {
          accum += src[i*num_labels + index];
       }

       dst[index] = accum;
    }
    """ % dict(
        data_type=cl.tools.dtype_to_ctype(src_dev.dtype),
        label_type=cl.tools.dtype_to_ctype(labels_dev.dtype)
    )

    program = cl.Program(ctx, CL).build()

    program.sum_cells(
        clq, (nitems,), None, 
        tmp_dev.data, src_dev.data, labels_dev.data, uint32(nlabels), uint32(min_label), uint32(nitems), uint32(worksize), uint32(workrem)
    )

    clq.flush()

    program.sum_labels(
        clq, (nlabels,), None, 
        dst_dev.data, tmp_dev.data, uint32(nlabels), uint32(nitems)
    )

    clq.flush()

TOTAL_ITEMS=2000


def sum_labeled(src, labels, n=None, clq=None):
    if clq is None:
        clq = cl.CommandQueue(ctx)
        return_dev = False
        if src.dtype == numpy.bool:
            src = src.astype(numpy.uint8)
        src_dev = cl_array.to_device(clq, src)
        labels_dev = cl_array.to_device(clq, labels)
    else:
        return_dev = True
        src_dev = src
        labels_dev = labels
        
    if n is None:
        n = labels_dev.max() + 1
    tmp_dev = cl_array.zeros(clq, (TOTAL_ITEMS, n), float32)
    dst_dev = cl_array.zeros(clq, (n,), float32)

    sum_labeled_dev(clq, src_dev, labels_dev, tmp_dev, dst_dev)

    if return_dev:
        return dst_dev
    else:
        result = dst_dev.map_to_host()
        clq.finish()
        return result

def assign_voxels_dev(clq, values_dev, centroids_dev, kernel_dev, weighted_dev, labeled_dev):
    
    CL = """
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

    // compute dst-based bbox for splat/dst intersection
    // splat_shape box centered at centroid
    // but clamped by dst_shape boundaries
    void dst_slice(
       const int3 splat_shape,
       const int3 dst_shape,
       const int3 centroid,
       int3*      lower_out,
       int3*      upper_out)
    {
       *lower_out = max(
          centroid - splat_shape/((int3) (2,2,2)),
          (int3) (0,0,0)
       );

       *upper_out = min(
          centroid + splat_shape/((int3) (2,2,2)) + (int3) (1,1,1),
          dst_shape
       );
    }

    // compute splat-based bbox for splat/dst intersection
    // same clamped shape as dst_slice
    // but re-centered in original splat_shape coordinates
    void splat_slice(
       const int3 splat_shape,
       const int3 dst_shape,
       const int3 centroid,
       int3*      lower_out,
       int3*      upper_out)
    {
       dst_slice(splat_shape, dst_shape, centroid, lower_out, upper_out);
       *lower_out = *lower_out - centroid + splat_shape / ((int3) (2,2,2));
       *upper_out = *upper_out - centroid + splat_shape / ((int3) (2,2,2));
    }

    __kernel void fill(
       __global const float*  values,
       __global const int*    centroids,
       const unsigned int     num_segments,
       __global const float*  weights,
       const int3             weights_shape,
       __global float*        weighted_dst,
       __global %(lab_type)s* labeled_dst,
       const int3             dst_shape,
       const int              pass_number)
    {
       // vectors are in Z,Y,X component order
       int3 splat_stride, dst_stride;
       int3 splat_lower, splat_upper;
       int3 dst_lower, dst_upper;
       int3 centroid;
       float seg_value;
       unsigned int seg_id = get_global_id(0);
       int i,j,k; // splat coords
       int x,y,z; // dst coords

       // get array layouts
       splat_stride = (int3) (weights_shape.s1 * weights_shape.s2, weights_shape.s2, 1);
       dst_stride = (int3) (dst_shape.s1 * dst_shape.s2, dst_shape.s2, 1);

       // get centroid properties
       centroid = vload3(seg_id, centroids);
       seg_value = values[seg_id];

       // compute intersecting bounding box in splat and dst coordinate spaces
       splat_slice(weights_shape, dst_shape, centroid, &splat_lower, &splat_upper);
       dst_slice(weights_shape, dst_shape, centroid, &dst_lower, &dst_upper);

       // process each voxel in bounding box
       k = splat_lower.s0;  z = dst_lower.s0;
       while (k < splat_upper.s0) {
         j = splat_lower.s1;  y = dst_lower.s1;
         while (j < splat_upper.s1) {
            i = splat_lower.s2;  x = dst_lower.s2;
            while (i < splat_upper.s2) {
               union {
                 int ival;
                 float fval;
               } splat, old;
               __global float* voxel;

               splat.fval = seg_value * weights[
                  splat_stride.s0*k + splat_stride.s1*j + i
               ];

               voxel = weighted_dst + dst_stride.s0*z + dst_stride.s1*y + x;

               if ((pass_number == 1) && (splat.fval > 0.0)) {
                  // splat our value
                  old.ival = atom_xchg((__global int*)voxel, splat.ival);
                  // but restore old value if it is larger
                  while (old.fval > splat.fval) {
                     splat.fval = old.fval;
                     old.ival = atom_xchg((__global int*)voxel, splat.ival);
                  }
               }
               else if ((pass_number == 2) && (splat.fval > 0.0)) {
                  // splat our label if our value matches
                  int ival = * (__global int*) voxel;

                  if (ival == splat.ival) {
                     %(lab_type)s old, label;
                     __global %(lab_type)s* label_voxel;

                     label_voxel = labeled_dst + dst_stride.s0*z + dst_stride.s1*y + x;
                     label = seg_id + 1;
                     old = atom_xchg(label_voxel, label);
                  }
               }
               i++;
               x++;
            }
            j++;
            y++;
         }
         k++;
         z++;
       }
    }
    """ % dict(
        lab_type=cl.tools.dtype_to_ctype(labeled_dev.dtype)
    )

    program = cl.Program(ctx, CL).build()

    # first we fill in weighted voxel values of all segments,
    # with maximum value winning in overlapped voxels
    program.fill(
        clq, (values_dev.shape[0],), None,
        values_dev.data, centroids_dev.data, uint32(values_dev.shape[0]),
        kernel_dev.data, cl_array.vec.make_int3(*kernel_dev.shape),
        weighted_dev.data, labeled_dev.data, cl_array.vec.make_int3(*weighted_dev.shape),
        int32(1)
    )

    clq.flush()

    # then we can fill in labeled voxels where voxels have matching 
    # weighted value, with arbitrary winner in overlapped segments of equal value
    program.fill(
        clq, (values_dev.shape[0],), None,
        values_dev.data, centroids_dev.data, uint32(values_dev.shape[0]),
        kernel_dev.data, cl_array.vec.make_int3(*kernel_dev.shape),
        weighted_dev.data, labeled_dev.data, cl_array.vec.make_int3(*weighted_dev.shape),
        int32(2)
    )

    clq.flush()


def assign_voxels(syn_values, centroids, valid_shape, syn_kernel_3d, gridsize=None):
    assert len(centroids) == len(syn_values)

    N = len(syn_values)
    dtype = numpy.uint32
    if N > (2**32-2):
        raise NotImplementedError("Absurdly large segment count %s" % N)

    syn_values = numpy.array(syn_values, float32)
    centroids = numpy.array(centroids, int32)

    # use a slight subset as the splatting body
    body_shape = syn_kernel_3d.shape
    D, H, W = body_shape
    radial_fraction = numpy.clip(float(os.getenv('SYNSPY_SPLAT_SIZE', '1.0')), 0, 2)
    limits = (
        syn_kernel_3d[D//2-1,0,W//2-1],
        syn_kernel_3d[D//2,H//2,W//2]
    )
    limit = limits[1] - (limits[1] - limits[0]) * radial_fraction
    mask_3d = syn_kernel_3d >= limit
    mask_3d[tuple([w//2 for w in mask_3d.shape])] = 1 # fill at least central voxel
    print("SPLAT BOX SHAPE %s   USER COEFFICIENT %f   MASK VOXELS %d" % (
        body_shape,
        radial_fraction,
        mask_3d.sum()
    ))
    kernel = syn_kernel_3d * mask_3d

    # trim off zero-padded border to reduce per-feature work size
    def trim(data):
        assert data.min() >= 0.0
        for axis in range(data.ndim):
            while data.shape[axis] > 1 and data[tuple(
                    [ slice(None) for i in range(axis) ]
                    + [ 0 ]
                    + [ slice(None) for i in range(axis + 1, data.ndim) ]
            )].max() == 0.0:
                data = data[tuple(
                    [ slice(None) for i in range(axis) ]
                    + [ slice(1, None) ]
                    + [ slice(None) for i in range(axis + 1, data.ndim) ]
                )]
            while data.shape[axis] > 1 and data[tuple(
                    [ slice(None) for i in range(axis) ]
                    + [ data.shape[axis]-1 ]
                    + [ slice(None) for i in range(axis + 1, data.ndim) ]
            )].max() == 0.0:
                data = data[tuple(
                    [ slice(None) for i in range(axis) ]
                    + [ slice(0, data.shape[axis]-1) ]
                    + [ slice(None) for i in range(axis + 1, data.ndim) ]
                )]
        return numpy.array(data, dtype=data.dtype)

    kernel = trim(kernel)
    if gridsize:
        print("assigning voxels with %fx%fx%f micron core bbox" % tuple(map(
            lambda v, r: v * r,
            kernel.shape,
            gridsize
        )))
    
    assert syn_values.ndim == 1
    assert centroids.ndim == 2
    assert centroids.shape[1] == 3
    assert syn_values.shape[0] == centroids.shape[0]
    assert kernel.ndim == 3

    clq = cl.CommandQueue(ctx)

    values_dev = cl_array.to_device(clq, syn_values)
    centroids_dev = cl_array.to_device(clq, centroids)
    kernel_dev = cl_array.to_device(clq, kernel)

    weighted_dev = cl_array.zeros(clq, valid_shape, float32)
    labeled_dev = cl_array.zeros(clq, valid_shape, dtype)

    assign_voxels_dev(clq, values_dev, centroids_dev, kernel_dev, weighted_dev, labeled_dev)

    #weighted = weighted_dev.map_to_host()
    labeled = labeled_dev.map_to_host()
    clq.finish()

    return labeled

def fwhm_estimate_dev(clq, synapse_dev, centroids_dev, syn_val_dev, vcn_val_dev, noise, voxel_size, widths_dev):
    
    CL = """
    float measure_1d(
       __global const float* synapse,
       size_t                center,
       float                 half_magnitude,
       int                   min_offset,
       int                   max_offset,
       int                   stride)
    {
       float lower, upper;
       int offset0, offset1;
       float value0, value1;

       value1 = synapse[center];
       offset1 = 0;
       for (offset1=0; offset1>=min_offset; offset1--) {
          value0 = value1;
          offset0 = offset1;
          value1 = synapse[center + offset1 * stride];
          if (value1 <= half_magnitude) break;
       }
       lower = (float) offset1 + smoothstep((float) value1, (float) value0, half_magnitude);

       value1 = synapse[center];
       offset1 = 0;
       for (offset1=0; offset1<=max_offset; offset1++) {
          value0 = value1;
          offset0 = offset1;
          value1 = synapse[center + offset1 * stride];
          if (value1 <= half_magnitude) break;
       }
       upper = (float) offset0 + smoothstep((float) value0, (float) value1, half_magnitude);

       return (upper - lower);
    }

    __kernel void measure(
       __global const float*  synapse,
       const int3             synapse_shape,
       __global const float*  syn_val,
       __global const float*  vcn_val,
       __global const int*    centroids,
       const unsigned int     num_segments,
       float                  noise,
       float3                 voxel_size,
       __global float*        widths_dst)
    {
       unsigned int seg_id = get_global_id(0);
       int3 stride, centroid;
       float3 width;
       int x,y,z;
       float floor_value, full_magnitude, half_magnitude;
       size_t center;
       
       floor_value = max(vcn_val[seg_id] * (float) 1.5, noise);
       full_magnitude = max(syn_val[seg_id] - floor_value, (float) 0.0);
       half_magnitude = full_magnitude / 2.0 + floor_value;

       stride = (int3) (synapse_shape.s1 * synapse_shape.s2, synapse_shape.s2, 1);
       centroid = vload3(seg_id, centroids);
       center = centroid.s0 * stride.s0 + centroid.s1 * stride.s1 + centroid.s2;
      
       width.s0 = measure_1d(synapse, center, half_magnitude, 0 - centroid.s0, synapse_shape.s0 - centroid.s0 - 1, stride.s0);
       width.s1 = measure_1d(synapse, center, half_magnitude, 0 - centroid.s1, synapse_shape.s1 - centroid.s1 - 1, stride.s1);
       width.s2 = measure_1d(synapse, center, half_magnitude, 0 - centroid.s2, synapse_shape.s2 - centroid.s2 - 1, stride.s2);

       width = width * voxel_size;
       vstore3(width, seg_id, widths_dst);
    }
    """
    
    program = cl.Program(ctx, CL).build()

    program.measure(
        clq, (syn_val_dev.shape[0],), None,
        synapse_dev.data, cl_array.vec.make_int3(*synapse_dev.shape),
        syn_val_dev.data, vcn_val_dev.data, centroids_dev.data, uint32(syn_val_dev.shape[0]),
        float32(noise),
        cl_array.vec.make_float3(*voxel_size),
        widths_dev.data
    )

    clq.flush()

def fwhm_estimate(synapse, centroids, syn_values, vcn_values, noise, voxel_size):
    
    syn_values = numpy.array(syn_values, float32)
    vcn_values = numpy.array(vcn_values, float32)
    centroids = numpy.array(centroids, int32)

    assert syn_values.ndim == 1
    assert vcn_values.shape == syn_values.shape
    assert centroids.ndim == 2
    assert centroids.shape[1] == 3
    assert syn_values.shape[0] == centroids.shape[0]

    clq = cl.CommandQueue(ctx)

    synapse_dev = cl_array.to_device(clq, synapse)
    syn_val_dev = cl_array.to_device(clq, syn_values)
    vcn_val_dev = cl_array.to_device(clq, vcn_values)
    centroids_dev = cl_array.to_device(clq, centroids)

    widths_dev = cl_array.zeros(clq, (syn_values.shape[0], 3), float32)

    fwhm_estimate_dev(clq, synapse_dev, centroids_dev, syn_val_dev, vcn_val_dev, noise, voxel_size, widths_dev)

    widths = widths_dev.map_to_host()
    clq.finish()

    return widths

def weighted_measure_dev(clq, data_dev, centroids_dev, kernel_dev, measures_dev):
    
    CL = """
    __kernel void measure(
       __global const float*  data,
       const int3             data_shape,
       __global const int*    centroids,
       __global const float*  kern,
       const int3             kern_shape,
       __global float*        measures_dst)
    {
       unsigned int seg_id = get_global_id(0);
       int3 centroid, kstride, dstride, D0, D1, K0;
       int x,y,z, i,j,k;
       float measure = 0.0;
       
       dstride = (int3) (data_shape.s1 * data_shape.s2, data_shape.s2, 1);
       kstride = (int3) (kern_shape.s1 * kern_shape.s2, kern_shape.s2, 1);

       centroid = vload3(seg_id, centroids);

       // data sampling bounding box corners intersect data and kernel
       D0 = max((int3) (0,0,0), centroid - kern_shape/2);
       D1 = min(centroid + kern_shape/2 + 1, data_shape);

       // kernel bounding box corner is offset by centroid and kernel radius
       K0 = D0 + kern_shape/2 - centroid;

       for (z=D0.s0, k=K0.s0; z<D1.s0; z++, k++) {
          for (y=D0.s1, j=K0.s1; y<D1.s1; y++, j++) {
             for (x=D0.s2, i=K0.s2; x<D1.s2; x++, i++) {
                float s = kern[kstride.s0 * k + kstride.s1 * j + i];
                measure += data[dstride.s0 * z + dstride.s1 * y + x] * s;
             }
          }
       }

       measures_dst[seg_id] = measure;
    }
    """
    
    program = cl.Program(ctx, CL).build()

    program.measure(
        clq, (centroids_dev.shape[0],), None,
        data_dev.data, cl_array.vec.make_int3(*data_dev.shape),
        centroids_dev.data,
        kernel_dev.data, cl_array.vec.make_int3(*kernel_dev.shape),
        measures_dev.data
    )

    clq.flush()

def weighted_measure(data, centroids, kernel, clq=None):
    assert len(data.shape) == 3
    assert len(kernel.shape) == 3
    assert len(centroids.shape) == 2
    assert centroids.shape[1] == 3

    if clq is None:
        clq = cl.CommandQueue(ctx)
        data_dev = cl_array.to_device(clq, data)
        centroids_dev = cl_array.to_device(clq, centroids)
        kernel_dev = cl_array.to_device(clq, kernel)
        return_dev = False
    else:
        data_dev = data
        centroids_dev = centroids
        kernel_dev = kernel
        return_dev = True
        
    measures_dev = cl_array.zeros(clq, (centroids.shape[0],), float32)
    weighted_measure_dev(clq, data_dev, centroids_dev, kernel_dev, measures_dev)

    if return_dev:
        return measures_dev
    else:
        measures = measures_dev.map_to_host()
        clq.finish()
        return measures

def nd_arange_dev(clq, out_dev, axis, start, step):

    CL = """
    __kernel void nd_arange(
       __global       float* dst,
       __global const int* data_strides,
       const int  axis,
       const float start,
       const float step,
       const int  n_steps)
    {
       int a;
       int dpos = 0;
       int i;
       float x = start;
       
       for (i=0; i<%(NDIM)d; i++) {
          dpos += data_strides[i] * get_global_id(i);
       }

       for (i=0; i<n_steps; i++) {
          dst[dpos] = x;
          x += step;
          dpos += data_strides[axis];
       }
    }
    """ % dict(NDIM=len(out_dev.shape))

    program = cl.Program(ctx, CL).build()

    work_shape = (
        tuple(out_dev.shape[i] for i in range(axis)) 
        + (1,)
        + tuple(out_dev.shape[i] for i in range(axis+1, len(out_dev.shape)))
    )
    strides = array([s/out_dev.dtype.itemsize for s in out_dev.strides], dtype=int32)
    program.nd_arange(
        clq, work_shape, None,
        out_dev.data, cl_array.to_device(clq, strides).data, uint32(axis),
        float32(start), float32(step), uint32(out_dev.shape[axis])
    )
    clq.flush()

def nd_arange(shape, axis=0, start=0, step=1, clq=None):
    """Fill an ND-array along one axis with a stepped range.

       nd_arange((Z, Y, X), axis=2, start=A, step=B) is functionally
       equivalent to:

         np.arange(A, A+X*B, B)[None,None,:] * np.ones((Z, Y, X), np.float32)

       but does the work on the OpenCL device and without relying on
       array-broadcasting which is not supported in PyOpenCL.

    """
    assert axis >=0
    assert axis < len(shape)
    
    if clq is None:
        clq = cl.CommandQueue(ctx)
        return_dev = False
    else:
        return_dev = True

    out_dev = cl_array.empty(clq, shape, float32)
    nd_arange_dev(clq, out_dev, axis, start, step)
    
    if return_dev:
        return out_dev
    else:
        out = out_dev.map_to_host()
        clq.finish()
        return out

