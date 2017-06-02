
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from numpy import array, concatenate, float32, empty
import scipy
from scipy import ndimage

import numexpr
from numexpr import evaluate as neval

#numexpr.set_num_threads(numexpr.ncores)

def convNx1d(src, kernels):
    """ND convolution using 1D kernels

       Trims borders by filter kernel width in each dimension.

       This version uses numerexpr and produces float32
       intermediate and final results regardless of input type.
    """
    accum = src.astype('float32', copy=False)
    
    for d in range(len(kernels)):
        L = accum.shape[d]
        kernel = kernels[d]
        kernel_width = len(kernel)
        if (kernel_width % 2) != 1:
            raise NotImplementedError('convNx1d on even-length kernel')
        kernel_radius = kernel_width/2

        if kernel_radius < 1:
            print("warning: dimension %d kernel %d is too small, has no effect" % (d, kernel_width))
            continue
        elif kernel_radius > L:
            raise ValueError("dimension %d length %d too small for kernel %d" % (d, L, kernel_width))

        src1d = accum

        sum_dict = dict(
            [
                ("a%d" % i, src1d[ tuple([ slice(None) for j in range(d)] + [ slice(i,i+L-kernel_width+1) ] + [ Ellipsis ]) ])
                for i in range(kernel_width)
            ] + [
                ("s%d" % i, float32(kernel[i])) for i in range(kernel_width)
            ]
        )

        sum_terms = [
            "a%d * s%d" % (i, i)
            for i in range(kernel_width)
        ]

        # numexpr cannot handle a large number of input variables
        K = 12
        tmp_accum = None
        while sum_terms:
            sum_expr = " + ".join(sum_terms[0:K])
            sum_terms = sum_terms[K:]
            if tmp_accum is not None:
                sum_dict['accum'] = tmp_accum
                sum_expr = "accum + " + sum_expr
            tmp_accum = neval(sum_expr, local_dict=sum_dict)

        accum = tmp_accum

    return accum
        

def array_mult(a1, a2):
    return neval("a1 * a2")

