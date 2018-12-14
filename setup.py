
#
# Copyright 2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from setuptools import setup, find_packages
from synspy import __version__

setup(
    name="synspy",
    description="synaptic image segmentation",
    version=__version__,
    packages=find_packages(),
    scripts=[
        "bin/synspy-analyze",
        "bin/synspy-download-images",
        "bin/synspy-pair-npz",
        "bin/synspy-reclassify",
        "bin/synspy-register",
        "bin/synspy-register-batch",
        "bin/synspy_worker",
    ],
    entry_points={
        'console_scripts': [
            'synspy-viewer = synspy.viewer:main',
            'synspy-viewer2d = synspy.viewer2d:main'
        ]
    },
    requires=["volspy", "vispy", "numpy", "pyopencl", "tifffile"],
    maintainer_email="support@misd.isi.edu",
    license='(new) BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ])
