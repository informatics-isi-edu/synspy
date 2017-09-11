#!/usr/bin/python
#
# Copyright 2015-2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from deriva_common import PollingErmrestCatalog, HatracStore, urlquote
from volspy.util import load_image
import sys
import traceback
import platform
import atexit
import shutil
import tempfile
import os
import subprocess
import re
import csv
import json
import numpy as np
from . import register
from .analyze import util

def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg

class WorkerRuntimeError (RuntimeError):
    pass

class WorkerNotReadyError (RuntimeError):
    pass

class WorkerBadDataError (RuntimeError):
    pass

# handle to /dev/null we can use in Popen() calls...
fnull = open(os.devnull, 'r+b')

class Worker (object):
    # server to talk to... defaults to our own FQDN
    servername = os.getenv('SYNSPY_SERVER', platform.uname()[1])

    # secret session cookie
    credfile = os.getenv('SYNSPY_CREDENTIALS', 'credentials.json')
    credentials = json.load(open(credfile))

    # remember where we started
    startup_working_dir = os.getcwd()

    tmpdir = os.getenv('TMPDIR', '/var/tmp')
    
    # track per-instance working dirs
    working_dirs = dict()

    # these are peristent/logical connections so we create once and reuse
    # they can retain state and manage an actual HTTP connection-pool
    catalog = PollingErmrestCatalog(
        'https', 
        servername,
        '1',
        credentials
    )

    store = HatracStore(
        'https', 
        servername,
        credentials
    )

    # for state-tracking across look_for_work() iterations
    idle_etag = None

    def __init__(self, row):
        sys.stderr.write('Claimed job %s.\n' % row['ID'])

        self.row = row
        self.subject_path = '/hatrac/Zf/%s' % row['Subject']

        # we want a temporary work space for our working files
        self.working_dir = tempfile.mkdtemp(dir=self.tmpdir)
        self.working_dirs[self.working_dir] = self.working_dir
        os.chdir(self.working_dir)
        sys.stderr.write('Using working directory %s.\n' % self.working_dir)

    @staticmethod
    def cleanup_working_dir(dirname):
        sys.stderr.write('Purging working directory %s... ' % dirname)
        shutil.rmtree(dirname)
        sys.stderr.write('done.\n')

    def cleanup(self):
        sys.stderr.write('\n')
        os.chdir(self.startup_working_dir)
        if self.working_dir:
            self.cleanup_working_dir(self.working_dir)
            del self.working_dirs[self.working_dir]
            self.working_dir = None

    def get_file(self, url):
        """Download file from URL returning local file name"""
        # short-cut, read file directly out of local hatrac
        filename = '/var/www' + url
        if os.path.isfile(filename):
            return filename
        else:
            # but fall back to HTTPS for remote workers...
            m = re.match('^(?P<basename>[^:]+)(?P<v>[:][0-9A-Z]+)?$', os.path.basename(url))
            filename = m.groupdict()['basename']
            self.store.get_obj(url, destfilename=filename)
            return filename

    def get_image_info(self, img_filename):
        """Extract image resolution and shape."""
        try:
            I, md = load_image(str(img_filename))
        except Exception as e:
            raise WorkerBadDataError('Image %s could not be loaded... is it the wrong format?' % img_filename)
        if not hasattr(I, 'micron_spacing'):
            raise WorkerBadDataError('Image %s lacks expected micron_spacing attribute.' % img_filename)
        return I.micron_spacing, I.shape

    def preprocess_roi(self, img_filename, zyx_slice):
        """Analyze ROI and upload resulting NPZ file, returning NPZ URL."""
        command = [ 'synspy-analyze', img_filename ]
        env = {
            'ZYX_SLICE': zyx_slice,
            'ZYX_IMAGE_GRID': '0.4,0.26,0.26',
            'SYNSPY_DETECT_NUCLEI': dict(nucleic='true').get(self.row['Segmentation Mode'], 'false'),
            'DUMP_PREFIX': './%s' % self.row['ID'],
        }
        sys.stderr.write('Using analysis environment %r\n' % (env,))
        analysis = subprocess.Popen(command, stdin=fnull, env=env)
        code = analysis.wait()
        del analysis
        if code != 0:
            raise WorkerRuntimeError('Non-zero analysis exit status %s!' % code)

        return self.store.put_loc(
            '%s/%s.npz' % (self.subject_path, self.row['ID']),
            '%s.npz' % self.row['ID'],
            headers={'Content-Type': 'application/octet-stream'}
        )

    def filter_synspy_csv(self, csv_url):
        """Process input CSV URL and upload filtered CSV, returning CSV URL."""
        # this should really be dead code in practice... current launcher uploads filtered csv directly
        m = re.match('^(?P<basename>.+)[.]csv(?P<v>[:][0-9A-Z]+)?$', os.path.basename(csv_url))
        base = m.groupdict()['basename']
        csv_filename = '%s.csv' % base

        # download the content to temp dir
        self.store.get_obj(csv_url, destfilename=csv_filename)

        # prepare to read CSV content from temp dir
        csv_file = open(csv_filename, 'r')
        reader = csv.DictReader(csv_file)

        # prepare to write filtered CSV to temp dir
        filtered_filename = '%s-only.csv' % base
        filtered_file = open(filtered_filename, 'w')
        writer = csv.writer(filtered_file)

        # write header
        writer.writerow(
            ('Z', 'Y', 'X', 'raw core', 'raw hollow', 'DoG core', 'DoG hollow')
            + ( ('red',) if 'red' in reader.fieldnames else ())
            + ('override',)
        )

        # copy w/ filtering
        for row in reader:
            if row['Z'] == 'saved' and row['Y'] == 'parameters' \
               or row['override'] and int(row['override']) == 7:
                writer.writerow(
                    (row['Z'], row['Y'], row['X'], row['raw core'], row['raw hollow'], row['DoG core'], row['DoG hollow'])
                    + ( (row['red'],) if 'red' in reader.fieldnames else ())
                    + (row['override'],)
                )

        del reader
        csv_file.close()
        del writer
        filtered_file.close()

        return self.store.put_loc(
            '%s/%s' % (self.subject_path, segments_filtered_file),
            segments_filtered_file,
            headers={'Content-Type': 'text/csv'}
        )

    def register_nuclei(self, n1_url, n2_url, zyx_scale=(0.4,0.26,0.26), filter_status=(3,7)):
        """Register nuclei files returning alignment matrix and processed and uploaded pointcloud URLs.

           Returns:
             M, n1_url, n2_url
        """
        n1_filename = self.get_file(n1_url)
        n2_filename = self.get_file(n2_url)
        nuc1cmsp = util.load_segment_info_from_csv(n1_filename, zyx_scale, filter_status=filter_status)
        nuc2cmsp = util.load_segment_info_from_csv(n2_filename, zyx_scale, filter_status=filter_status)
        M, angles = register.align_centroids(nuc1cmsp[0], nuc2cmsp[0])
        nuc2cmsp = (register.transform_centroids(M, nuc2cmsp[0]),) + nuc2cmsp[1:]
        n1_outfile = '%s-n1-registered.csv' % self.row['ID']
        n2_outfile = '%s-n2-registered.csv' % self.row['ID']
        register.dump_registered_file_pair(
            (n1_outfile, n2_outfile),
            (nuc1cmsp, nuc2cmsp)
        )
        n1_url = self.store.put_loc(
            '%s/%s' % (self.subject_path, n1_outfile),
            n1_outfile,
            headers={'Content-Type': 'text/csv'}
        )
        n2_url = self.store.put_loc(
            '%s/%s' % (self.subject_path, n2_outfile),
            n2_outfile,
            headers={'Content-Type': 'text/csv'}
        )
        return M, n1_url, n2_url

    def matrix_to_prejson(self, M):
        return [
            [
                float(M[i,j])
                for j in range(4)
            ]
            for i in range(4)
        ]

    def register_synapses(self, s1_url, s2_url, zyx_scale=(0.4,0.26,0.26), filter_status=(3,7)):
        """Register synaptic files using image pair alignment, returning URLs of processed and uploaded pointcloud URLs.

           Returns:
             s1_url, s2_url
        """
        s1_filename = self.get_file(s1_url)
        s2_filename = self.get_file(s2_url)
        syn1cmsp = util.load_segment_info_from_csv(s1_filename, zyx_scale, filter_status=filter_status)
        syn2cmsp = util.load_segment_info_from_csv(s2_filename, zyx_scale, filter_status=filter_status)
        M = np.array(self.row['Alignment'], dtype=np.float64)
        syn2cmsp = (register.transform_centroids(M, syn2cmsp[0]),) + syn2cmsp[1:]
        s1_outfile = '%s-s1-registered.csv' % self.row['ID']
        s2_outfile = '%s-s2-registered.csv' % self.row['ID']
        register.dump_registered_file_pair(
            (s1_outfile, s2_outfile),
            (syn1cmsp, syn2cmsp)
        )
        s1_url = self.store.put_loc(
            '%s/%s' % (self.subject_path, s1_outfile),
            s1_outfile,
            headers={'Content-Type': 'text/csv'}
        )
        s2_url = self.store.put_loc(
            '%s/%s' % (self.subject_path, s2_outfile),
            s2_outfile,
            headers={'Content-Type': 'text/csv'}
        )
        return s1_url, s2_url

    get_claimable_work_url = None # GET URL returning row(s) to claim
    put_claim_url = None  # PUT URL to claim work

    @staticmethod
    def claim_input_data(row):
        return {'ID': row['ID'], 'Status': "pre-processing..."}

    @staticmethod
    def failure_input_data(row, e):
        return {'ID': row['ID'], 'Status': {"failed": "%s" % e}}

    put_row_update_baseurl = '/attributegroup/Zebrafish:Image/ID'
    
    def put_row_update(self, update_row):
        self.catalog.put(
            '%s;%s' % (
                self.put_row_update_baseurl,
                ','.join([
                    urlquote(col, safe='')
                    for col in list(update_row.keys())
                    if col != 'ID'
                ])
            ),
            json=[update_row]
        )
        sys.stderr.write('\nupdated in ERMrest: %s' % json.dumps(update_row, indent=2))
    
    @classmethod
    def look_for_work(cls):
        """Find, claim, and process one record.

        1. Find row with actionable state (partial data and Status="analysis complete")
        2. Claim by setting Status="in progress"
        3. Download and process data as necessary
        4. Upload processing results to Hatrac
        5. Update ERMrest w/ processing result URLs and Status="processed"

        Do find/claim with HTTP opportunistic concurrency control and
        caching for efficient polling and quiescencs.

        On error, set Status="failed: reason"

        Result:
         true: there might be more work to claim
         false: we failed to find any work
        """
        # this handled concurrent update for us to safely and efficiently claim a record
        cls.idle_etag, batch = cls.catalog.state_change_once(
            cls.get_claimable_work_url,
            cls.put_claim_url,
            cls.claim_input_data,
            cls.idle_etag
        )
        # we used a batch size of 1 due to ?limit=1 above...
        for row, claim in batch:
            try:
                handler = cls(row)
                handler.run_row_job()
            except WorkerBadDataError as e:
                sys.stderr.write("Aborting task %s on data error: %s\n" % (row["ID"], e))
                cls.catalog.put(cls.put_claim_url, json=[cls.failure_input_data(row, e)])
                # continue with next task...?
            except Exception as e:
                # TODO: eat some exceptions and return True to continue?
                cls.catalog.put(cls.put_claim_url, json=[cls.failure_input_data(row, e)])
                raise

            return True
        else:
            return False

    @classmethod
    def blocking_poll(cls):
        return cls.catalog.blocking_poll(cls.look_for_work)

@atexit.register
def _atexit_cleanup():
    os.chdir(Worker.startup_working_dir)
    for dirname in Worker.working_dirs:
        Worker.cleanup_working_dir(dirname)
        del Worker.working_dirs[dirname]

