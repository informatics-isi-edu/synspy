#
# Copyright 2017 University of Southern California
# Distributed under the GNU GPL 3.0 license. See LICENSE for more info.
#

""" Installation script for synspy launcher
"""

from setuptools import setup, find_packages
from synspy import __version__

setup(
    name="synspy-launcher",
    description="Synspy launcher GUI",
    url='https://github.com/informatics-isi-edu/synspy/launcher',
    maintainer='USC Information Sciences Institute ISR Division',
    maintainer_email='misd-support@isi.edu',
    version=__version__,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'synspy-launcher = launcher.__main__:main'
        ]
    },
    requires=[
        'os',
        'sys',
        'errno',
        'logging',
        'tempfile',
        'shutil',
        'subprocess',
        'pytz',
        'deriva.core',
        'deriva.qt',
        'PyQt5'],
    license='GNU GPL 3.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License',
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ]
)

