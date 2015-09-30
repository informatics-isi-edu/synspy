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
  GPUs.  A recent development version is needed, and it is tested with
  the fork [karlcz/vispy](https://github.com/vispy/vispy) which is
  periodically synchronized with the upstream project.
- [Numpy](http://www.numpy.org) numerical library to process
  N-dimensional data.
- [Numexpr](https://github.com/pydata/numexpr) numerical expression
  evaluator to accelerate some Numpy operations.
- [PyOpenCL](http://mathema.tician.de/software/pyopencl/) to access
  OpenCL parallel computing platforms.
- [Tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) for
  access to OME-TIFF and LSM microscopy file formats.
- [MDAnalysis](http://pypi.python.org/pypi/MDAnalysis/) for basic
  transformation matrix utilities.
- [Python-PCL](https://github.com/strawlab/python-pcl) bindings for
  point-cloud registration algorithms.
  - [Point Cloud Library (PCL)](http://pointclouds.org/) is the 
    actual library. We test with `pcl-devel` on Fedora Linux.
  - PCL is optional and only used by the auxiliary `synspy-register`
    script, used to determine alignment between images using a
    set of common registration features detected in both.

### Installation

0. Install all third-party prerequisites.
1. Check out the development code from GitHub for Synspy and Volspy.
2. Install with `python setup.py install` for each of Synspy and Volspy.

### Detecting Synapses

1. Obtain a sample 2 channel 3D TIFF image such as:
   http://www.isi.edu/~karlcz/sample-data/zebra-d19-03b-D.ome.tiff.gz
   **Warning: this is a large 886 MB file!**
2. Launch the viewer `synspy-viewer zebra-d19-03b-D.ome.tiff`
3. Interact with the viewer:
  - Press `ESC` when you have had enough.
  - Press `?` to print UI help text to console output.
  - The mouse can be used to control the view or interact with segments.
	1. Mouse-over on segments will display some pop-up diagnostics.
	2. Dragging *button 1* will rotate the image
	3. Dragging *button 2* or *button 3* will enable temporary slicing
      and cropping modes where the vertical axis controls the depth of
      a cutting plane.
	4. Clicking a segment will toggle its manual classification state.
  - The `f` and `F` keys control the minimum *feature* intensity
    threshold for classifying blobs as synapses.
  - The `n` and `N` keys control the maximum *neighborhood* threshold to
    reject blobs with bright background.
  - The `m` and `M` keys control the maximum *mask* threshold to
    reject blobs that are in autofluorescing regions according to the
    red-channel mask.
  - The `z` and `Z` keys control the *zoom* level.
  - The `g` and `G` keys control the *gain* level.	
  - The number keys `0` to `9` with and without shift modifier set
    *gain*.
  - The `t` and `T` keys control the *transparency* floor, the voxel
    intensity at which opacity is 0.0.
  - The `u` and `U` keys control the transparency *upper bound*, the
    voxel intensity at which opacity is 1.0.
  - The `o` and `O` keys control the *opacity* factor which is applied
    after the linear ramp from floor to upper bound.
  - The `b` key cycles through feature blending mode for
    on-screen rendering, where unclassified voxels are rendered in
    blue:
    1. Linear intensity mapping for voxels within classified segments
      (green for synapses, red for auto-fluorescence).
    2. Full intensity fill for voxels within classified segments
      (green for synapses, red for auto-fluorescence).
  - The `d` key *dumps* current parameters to the console output.
  - The `D` key *dumps* a voxel classification TIFF image and a
    segment list CSV file.
  - The `l` key *loads* a previously dumped segment list CSV file to
    continue making manual classification decisions in a new session.
  - The `h` key writes out a 2D histogram of all blobs using the
    feature intensity and background noise intensity measures as the
    two plotting axes.
  - The `e` key *expunges* manually classified segments, making those
    segments unclickable so that other adjacent segments are easier to
    click.

Do not be alarmed by the copious diagnostic outputs streaming out on
the console. Did we mention this is experimental code?

### Detecting Nuclei

Follow the same procedure as above, except set the environment
variable `SYNSPY_DETECT_NUCLEI=true` when launching the
`synspy-viewer` application. This changes the size of the footprints
used for scale-sensitive blob detection, so that large nuclei scale
blobs are detected and classified instead of small synapse-scale
blobs.

### Registering Two Images

We have only rudimentary support for registering multiple images for expert users:

1. Acquire two images, such as a *before* and *after* image of the same tissue.
2. Detect nuclei using the previously described method and dump two segment lists such as `nuclei-before.csv` and `nuclei-after.csv` with manually classified features.
3. Filter the dumped lists to only include manually classified features, e.g. those with `override` column values of `7`.
4. Run the `synspy-register` tool
  - `synspy-register nuclei-before.csv nuclei-after.csv`
5. View the resulting alignment and judge quality.
6. Capture the transformation matrix printed to standard output if desired.

The transformation matrix can be applied the second image or its
features (the *after* image in this example). We do not currently
provide any tools to do this transformation since there are too many
choices depending on your goals. We do not in general recommend
resampling the source image grid since a typical anisotropic
microscope image will be greatly degraded by a rotation.

### Environment Parameters

Several environment variables can be set to modify the behavior of the `synspy-viewer` tool on a run-by-run basis, most of which are in common with the `volspy-viewer`:

- `PEAKS_DIAM_FACTOR` (default `0.75`) controls the diameter of a gaussian low-pass filter that is used to smooth the image before doing local maxima detection. This factor adjusts the diameter relative to the built-in synapse-diameter that is used to model the core intensity distribution of synapse candidates. A smaller diameter will allow the detection of more closely spaced local maxima but may introduce errors as pixel-level noise begins to dominate.
- `DUMP_PREFIX` controls how dumped files are named. Defaults to input filename plus a hyphen, e.g. when running on `dir1/image2.ome.tiff` the default behaves as if you specified `DUMP_PREFIX=dir1/image2.ome.tiff-` and will name dump files such as `dir1/image2.ome.tiff-segments.csv`.
- `VOXEL_SAMPLE` selects volume rendering texture sampling modes from `nearest` or `linear` (default for unspecified or unrecognized values).
- `VIEW_MODE` changes the scalar field that is volume-rendered:
  - `raw` renders the raw data (default)
  - `dog` renders a difference-of-gaussians transform to emphasize synapse-scale changes
- `ZYX_SLICE` selects a grid-aligned region of interest to view from the original image grid, e.g. `0:10,100:200,50:800` selects a region of interest where Z<10, 100<=Y<200, and 50<=X<800. (Default slice contains the whole image.)
- `ZYX_VIEW_GRID` changes the desired rendering grid spacing. Set a preferred ZYX micron spacing, e.g. `0.5,0.5,0.5` which the program will try to approximate using integer bin-averaging of source voxels but it will only reduce grid resolution and never increase it. NOTE: Y and X values should be equal to avoid artifacts with current renderer. (Default grid is 0.25, 0.25, 0.25 micron.)
- `ZYX_BLOCK_SIZE` changes the desired sub-block work unit size for decomposing large images to control Numpy or OpenCL working set size. Set a preferred ZYX voxel count, e.g. `256,384,512` which the program will try to approximate to find an evenly divisible block layout.
- `MAX_3D_TEXTURE_WIDTH` sets a limit to the per-dimension size of the volume cube loaded into an OpenGL texture. If the viewing grid is too large to fit, it will be bin-averaged by factors of 2 into a multi-resolution pyramid with limited pan/zoom control in the viewing application to load different subsets of data onto the GPU. (Default is `768`.)
- `ZNOISE_PERCENTILE` enables a sensor noise estimation by calculating the Nth percentile value along the Z axis, e.g. `ZNOISE_PERCENTILE_5` estimates a 2D noise image as the 5th percentile value across the Z stack, and subtracts that noise image from every slice in the stack as a pre-filtering step. *WARNING*: use of this feature causes the entire image to be loaded into RAM, causing a significantly higher minimum RAM size for runs with large input images. (Default is no noise estimate.) 
  - `ZNOISE_ZERO_LEVEL` controls a lower value clamp for the pre-filtered data when percentile filtering is enabled. (Default is `0`.)

The `ZYX_SLICE`, `ZYX_VIEW_GRID`, and `MAX_3D_TEXTURE_WIDTH` parameters have different but inter-related effects on the scope of the volumetric visualization.

1. The `ZYX_VIEW_GRID` can control down-sampling of voxels in arbitrary integer ratios, e.g. to set a preferred grid resolution that can differentiate features of a given size without wasting additional storage space on irrelevant small-scale details. This can save overall RAM required to store the processed volume data by reducing the global image size. The down-sampling occurs incrementally as each sub-block is processed by the block-decomposed processing pipeline.
1. The `ZYX_SLICE` can arbitrarily discard voxels and thus reduce the final volume size, though discarded voxels may be temporarily present in RAM and require additional memory allocation at that time.
1. The `MAX_3D_TEXTURE_WIDTH` can avoid allocating oversized OpenGL 3D textures which would either cause a runtime error or unacceptable performance on a given hardware implementation. This can save overall texture RAM required to store the volume data on the GPU, but actually increases the host RAM requirements since it generates a multi-resolution pyramid on the host from which different 3D texture blocks are retrieved dynamically.

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

