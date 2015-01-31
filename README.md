# Synspy: Synaptic image segmentation using numpy, opencl, volspy, and vispy

[Synspy](http://github.com/informatics-isi-edu/volspy) is an
interactive volume segmentation tool. Synspy is being developed to
support research involving 3D fluorescence microscopy of live
Zebrafish synapses.

## Status

Synspy is experimental software that is subject to frequent changes in
direction depending on the needs of the authors and their scientific
collaborators.

## Using Synspy

Synspy has three usage scenarios:

1. A framework for developing batch image analysis tools that
  can segment and measure very large images.
2. A framework for image segmentation tools, where a basic
  interactive volume rendering capability can complement custom
  data-processing tools.
3. A standalone application for interactively segmenting images with
  our current segmentation algorithm.

### Prerequisites

Synspy is developed primarily on Linux with Python 2.7 but also tested
on Mac OSX. It has several requirements:

- [Volspy](http://github.com/informatics-isi-edu/volspy) volume
  rendering framework.
- [Vispy](http://vispy.org) visualization library to access OpenGL
  GPUs.  A recent development version is needed, including
  high-precision texture format features merged into the
  [vispy/master](https://github.com/vispy/vispy) branch on 2015-01-30.
- [Numpy](http://www.numpy.org) numerical library to process
  N-dimensional data.
- [Numexpr](https://github.com/pydata/numexpr) numerical expression
  evaluator to accelerate some Numpy operations.
- [PyOpenCL](http://mathema.tician.de/software/pyopencl/) to access
  OpenCL parallel computing platforms.
- [Tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) for
  access to OME-TIFF and LSM microscopy file formats.
- [NiBabel](http://nipy.org/nibabel) for access to additional
  neuroimaging file formats such as NifTI.

The image processing part of Synspy can tolerate a missing Numerexpr
or PyOpenCL library by falling back to slower standard Numpy code
paths.

The file-reading part of Synspy can tolerate a missing Tifffile or
NiBabel prerequisite if you do not need to read those types of files.

### Installation

0. Install all third-party prerequisites.
1. Check out the development code from GitHub for Synspy and Volspy.
2. Install with `python setup.py install` for each of Synspy and Volspy.

### Segmenting an Image

1. Obtain a sample 2 channel 3D TIFF image such as:
   http://www.isi.edu/~karlcz/sample-data/zebra-d19-03b-D.ome.tiff.gz
   **Warning: this is a large 886 MB file!**
2. Launch the viewer `synspy-viewer zebra-d19-03b-D.ome.tiff`
3. Interact with the viewer:
  - Press `ESC` when you have had enough.
  - Press `?` to print UI help text to console output.
  - The `f` and `F` keys control the minimum *feature* intensity
    threshold for classifying blobs as synapses.
  - The `n` and `N` keys control the maximum *noise* threshold to
    reject blobs that are not in a dark background.
  - The `m` and `M` keys control the maximum *mask* threshold to
    reject blobs that are in autofluorescing regions according to the
    red-channel mask.
  - The number keys `0` to `9` with and without shift modifier control
    the intensity gain of the on-screen rendering (but do not affect
    image analysis).
  - The `c` key cycles through feature color interpretation for
    on-screen rendering, where unclassified voxels are rendered in
    blue:
    1. Linear intensity mapping for voxels within classified segments
      (green for synapses, red for auto-fluorescence).
    2. Full intensity fill for voxels within classified segments
      (green for synapses, red for auto-fluorescence).
  - The 'd' key *dumps* current parameters to the console output.
  - The 'D' key writes out a *debug* image with classified voxels and
    also writes out the list of segmented blobs as a CSV file.
  - The 'h' key writes out a 2D histogram of all blobs using the
    feature intensity and background noise intensity measures as the
    two plotting axes.

Do not be alarmed by the copious diagnostic outputs streaming out on
the console. Did we mention this is experimental code?

## Help and Contact

Please direct questions and comments to the [project issue
tracker](https://github.com/informatics-isi-edu/synspy/issues) at
GitHub.

## License

Synspy is made available as open source under the (new) BSD
License. Please see the [LICENSE
file](https://github.com/informatics-isi-edu/synspy/blob/master/LICENSE)
for more information.

## About Us

Synspy and Volspy are developed in the [Informatics
group](http://www.isi.edu/research_groups/informatics/home) at the
[USC Information Sciences Institute](http://www.isi.edu).  The
computer science researchers involved are:

* Karl Czajkowski
* Carl Kesselman

