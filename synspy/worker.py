
#
# Copyright 2015-2018 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from deriva.core import PollingErmrestCatalog, HatracStore, urlquote, get_credential, DEFAULT_CREDENTIAL_FILE
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

class WorkUnit (object):
    def __init__(
            self,
            get_claimable_url,
            put_claim_url,
            put_update_baseurl,
            run_row_job,
            claim_input_data=lambda row: {'RID': row['RID'], 'Status': "pre-processing..."},
            failure_input_data=lambda row, e: {'RID': row['RID'], 'Status': {"failed": "%s" % e}},
    ):
        self.get_claimable_url = get_claimable_url
        self.put_claim_url = put_claim_url
        self.put_update_baseurl = put_update_baseurl
        self.run_row_job = run_row_job
        self.claim_input_data = claim_input_data
        self.failure_input_data = failure_input_data
        self.idle_etag = None

_work_units = []

def image_row_job(handler):
    # Image state machine:
    # Status is NULL and image URL is not NULL
    # -> Status="pre-processing..." (claimed)
    #    -> Status="ready"
    #       ZYX Spacing=[0.4,0.26,0.26]
    #       CZYX Shape=[C,D,H,W]
    #    -> Status={"failed": "reason"}
    assert handler.row['URL'], handler.row
    img_filename = handler.get_file(handler.row['URL'])
    zyx_spacing, czyx_shape = handler.get_image_info(img_filename)
    zyx_spacing = {
        axis: spacing
        for axis, spacing in zip(['z','y','x'], zyx_spacing)
    }
    czyx_shape = {
        axis: span
        for axis, span in zip(['c','z','y','x'], czyx_shape)
    }
    handler.put_row_update({
        'RID': handler.row['RID'],
        'ZYX Spacing': zyx_spacing,
        'CZYX Shape': czyx_shape,
        'Status': "ready",
    })
    sys.stderr.write('Image %s processing complete.\n' % handler.row['RID'])

_work_units.append(
    WorkUnit(
        '/entity/Zebrafish:Image/Status::null::;Status=null/!URL::null::?limit=1',
        '/attributegroup/Zebrafish:Image/RID;Status',
        '/attributegroup/Zebrafish:Image/RID',
        image_row_job
    )
)

def region_row_job(handler):
    # ImageRegion state machine:
    # ImageStatus="ready" and Status is NULL and Classifier is not NULL
    # -> Status="pre-processing..." (claimed to generate NPZ)
    #    -> Status="analysis pending" (will be available to launcher)
    #       ZYX Slice is set
    #       Npz URL set
    #       -> Status="analysis complete" (launcher has declared success)
    #          Segments URL and/or Segments Filtered URL set
    #          -> Status="pre-processing..." (claimed to filter CSV)
    #             -> Status="processed"
    #                Segments Filtered URL set
    # *-> Status={"failed": "reason"}
    updated_row = {}

    zyx_slice = handler.row['ZYX Slice']
    if zyx_slice is None:
        # calculate the actual slice using the alternate data input fields...
        czyx_shape = handler.row['CZYX Shape']

        z_lower, z_upper, y_lower, y_span, x_lower, x_span = [
            int(handler.row[k])
            for k in ['Z lower', 'Z upper', 'Y lower', 'Y span', 'X lower', 'X span']
        ]

        try:
            # convert Y and X spans to ZYX Slice bounds
            y_upper = y_lower + y_span
            x_upper = x_lower + x_span

            zyx_slice = ','.join([
                '%d:%d' % (b0, b1)
                for b0, b1 in [(z_lower, z_upper), (y_lower, y_upper), (x_lower, x_upper)]
            ])

            updated_row['ZYX Slice'] = zyx_slice
        except TypeError:
            pass

    if handler.row['Status'] is None:
        if handler.row['Npz URL'] is None and zyx_slice is not None:
            img_filename = handler.get_file(handler.row['URL'])
            updated_row['Npz URL'] = handler.preprocess_roi(
                img_filename,
                zyx_slice,
                omit_voxels=handler.row['Segmentation Mode'] == 'nucleic'
            )
            updated_row['Status'] = "analysis pending"
        elif zyx_slice is None:
            raise WorkerRuntimeError('Classifier is set, but ZYX Slice could not be determined.')
        else:
            # let the user try again if they cleared status manually on record...
            updated_row['Status'] = "analysis pending"

    if handler.row['Segments URL'] is not None and handler.row['Segments Filtered URL'] is None:
        updated_row['Segments Filtered URL'] = handler.filter_synspy_csv(handler.row['Segments URL'])
        updated_row.update(handler.compute_synspy_stats(updated_row['Segments Filtered URL'], handler.row))
        updated_row['Status'] = "processed"
    elif handler.row['Segments Filtered URL']:
        updated_row.update(handler.compute_synspy_stats(handler.row['Segments Filtered URL'], handler.row))
        updated_row['Status'] = "processed"

    if updated_row:
        updated_row['RID'] = handler.row['RID']
        handler.put_row_update(updated_row)
    else:
        raise WorkerRuntimeError('row had no work pending %s' % (handler.row,))

    sys.stderr.write('Region %s processing complete.\n' % handler.row['RID'])

_work_units.append(
    WorkUnit(
        '/attribute/I:=Zebrafish:Image/Status=%22ready%22/Zebrafish:Image%20Region/!Classifier::null::/Status::null::;Status=null;Status=%22analysis%20complete%22/*,I:URL,I:CZYX%20Shape?limit=1',
        '/attributegroup/Zebrafish:Image%20Region/RID;Status',
        '/attributegroup/Zebrafish:Image%20Region/RID',
        region_row_job
    )
)

def image_pair_row_job(handler):
    # Image Pair Study state machine:
    # -> Status NULL and N1.Status="processed" and N2.Status="processed"
    # -> Status="processing..." (claimed to generate alignment)
    #    -> Status="aligned" (alignment complete)
    #       Alignment set
    #       Region 1 URL set
    #       Region 2 URL set
    # *-> Status={"failed": "reason"}
    M, n1_url, n2_url = handler.register_nuclei(handler.row['N1_URL'], handler.row['N2_URL'])
    updated_row = {
        'RID': handler.row['RID'],
        'Alignment': handler.matrix_to_prejson(M),
        'Region 1 URL': n1_url,
        'Region 2 URL': n2_url,
        'Status': "aligned"
    }
    handler.put_row_update(updated_row)
    sys.stderr.write('Image pair %s processing complete.\n' % handler.row['RID'])

_work_units.append(
    WorkUnit(
        '/attribute/S:=Image%20Pair%20Study/Status::null::;Status=null/N1:=(Nucleic%20Region%201)/Status=%22processed%22/$S/N2:=(Nucleic%20Region%202)/Status=%22processed%22/$S/*,N1_URL:=N1:Segments%20Filtered%20URL,N2_URL:=N2:Segments%20Filtered%20URL?limit=1',
        '/attributegroup/Zebrafish:Image%20Pair%20Study/RID;Status',
        '/attributegroup/Zebrafish:Image%20Pair%20Study/RID',
        image_pair_row_job
    )
)

def synaptic_pair_row_job(handler):
    # Image Pair Study state machine:
    # -> Status NULL and IP.Status="aligned" and S1.Status="processed" and S2.Status="processed"
    # -> Status="processing..." (claimed to generate aligned files)
    #    -> Status="aligned" (alignment complete)
    #       Region 1 URL set
    #       Region 2 URL set
    # *-> Status={"failed": "reason"}
    s1_url, s2_url = handler.register_synapses(handler.row['S1_URL'], handler.row['S2_URL'])
    updated_row = {
        'RID': handler.row['RID'],
        'Region 1 URL': s1_url,
        'Region 2 URL': s2_url,
        'Status': "aligned"
    }
    handler.put_row_update(updated_row)
    sys.stderr.write('Synaptic pair %s processing complete.\n' % handler.row['RID'])

_work_units.append(
    WorkUnit(
        '/attribute/S:=Synaptic%20Pair%20Study/Status::null::;Status=null/IP:=Image%20Pair%20Study/Status=%22aligned%22/$S/S1:=(Synaptic%20Region%201)/Status=%22processed%22/$S/S2:=(Synaptic%20Region%202)/Status=%22processed%22/$S/*,Alignment:=IP:Alignment,S1_URL:=S1:Segments%20Filtered%20URL,S2_URL:=S2:Segments%20Filtered%20URL?limit=1',
        '/attributegroup/Zebrafish:Synaptic%20Pair%20Study/RID;Status',
        '/attributegroup/Zebrafish:Synaptic%20Pair%20Study/RID',
        synaptic_pair_row_job
    )
)

def nucleic_pair_row_job(handler):
    # Image Pair Study state machine:
    # -> Status NULL and IP.Status="aligned" and S1.Status="processed" and S2.Status="processed"
    # -> Status="processing..." (claimed to generate aligned files)
    #    -> Status="aligned" (alignment complete)
    #       Region 1 URL set
    #       Region 2 URL set
    # *-> Status={"failed": "reason"}
    r1_url, r2_url = handler.register_synapses(handler.row['r1_URL'], handler.row['r2_URL'])
    updated_row = {
        'RID': handler.row['RID'],
        'Region 1 URL': r1_url,
        'Region 2 URL': r2_url,
        'Status': "aligned"
    }
    handler.put_row_update(updated_row)
    sys.stderr.write('Nucleic pair %s processing complete.\n' % handler.row['RID'])

_work_units.append(
    WorkUnit(
        '/attribute/N:=Nucleic%20Pair%20Study/Status::null::;Status=null/IP:=(Study)/Status=%22aligned%22/$N/R1:=(Nucleic%20Region%201)/Status=%22processed%22/$N/R2:=(Nucleic%20Region%202)/Status=%22processed%22/$N/*,Subject:=IP:Subject,Alignment:=IP:Alignment,r1_URL:=R1:Segments%20Filtered%20URL,r2_URL:=R2:Segments%20Filtered%20URL?limit=1',
        '/attributegroup/Zebrafish:Nucleic%20Pair%20Study/RID;Status',
        '/attributegroup/Zebrafish:Nucleic%20Pair%20Study/RID',
        nucleic_pair_row_job,
        lambda row: {'RID': row['RID'], 'Status': "pre-processing..."},
        lambda row, e: {'RID': row['RID'], 'Status': {"failed": "%s" % e}}
    )
)

class Worker (object):
    # server to talk to... defaults to our own FQDN
    servername = os.getenv('SYNSPY_SERVER', platform.uname()[1])

    # secret session cookie
    credentials = get_credential(
        servername,
        credential_file=os.getenv('SYNSPY_CREDENTIALS', DEFAULT_CREDENTIAL_FILE)
    )

    poll_seconds = int(os.getenv('SYNSPY_POLL_SECONDS', '600'))

    scriptdir = os.getenv('SYNSPY_PATH')
    scriptdir = '%s/' % scriptdir if scriptdir else ''

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

    def __init__(self, row, unit):
        sys.stderr.write('Claimed job %s.\n' % row.get('RID'))

        self.row = row
        self.unit = unit
        self.subject_path = '/hatrac/Zf/Zf_%s' % row['Subject']

        self.working_dir = None
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
            raise WorkerBadDataError('Image %s could not be loaded... is it the wrong format? %s' % (img_filename, e))
        if not hasattr(I, 'micron_spacing'):
            raise WorkerBadDataError('Image %s lacks expected micron_spacing attribute.' % img_filename)
        return I.micron_spacing, I.shape

    def preprocess_roi(self, img_filename, zyx_slice, omit_voxels=False):
        """Analyze ROI and upload resulting NPZ file, returning NPZ URL."""
        command = [ self.scriptdir + 'synspy-analyze', img_filename ]
        env = {
            'ZYX_SLICE': zyx_slice,
            'ZYX_IMAGE_GRID': '0.4,0.26,0.26',
            'SYNSPY_DETECT_NUCLEI': str(self.row['Segmentation Mode'].lower() == 'nucleic'),
            'DUMP_PREFIX': './ROI_%s' % self.row['RID'],
            'OMIT_VOXELS': str(omit_voxels).lower(),
        }
        sys.stderr.write('Using analysis environment %r\n' % (env,))
        analysis = subprocess.Popen(command, stdin=fnull, env=env)
        code = analysis.wait()
        del analysis
        if code != 0:
            raise WorkerRuntimeError('Non-zero analysis exit status %s!' % code)

        return self.store.put_loc(
            '%s/ROI_%s.npz' % (self.subject_path, self.row['RID']),
            'ROI_%s.npz' % self.row['RID'],
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
        filtered_filename = '%s_only.csv' % base
        filtered_file = open(filtered_filename, 'w', newline='')
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

    def compute_synspy_stats(self, csv_url, existing_row={}):
        """Process input CSV URL and return stats column value updates."""
        filename = self.get_file(csv_url)
        c, m, s, p = util.load_segment_info_from_csv(filename, (0.4,0.26,0.26), filter_status=(3,7))
        if c.shape[0] > 0:
            stats = {
                'Core Min.': float(m[:,0].min()),
                'Core Max.': float(m[:,0].max()),
                'Core Sum': float(m[:,0].sum()),
                '#Centroids': int(m.shape[0]),
                'Core Mean': float(m[:,0].mean()),
            }
        else:
            stats = {
                'Core Min.': None,
                'Core Max.': None,
                'Core Sum': None,
                '#Centroids': 0,
                'Core Mean': None,
            }
        return {
            k: v
            for k, v in stats.items()
            if k not in existing_row or existing_row[k] != v
        }

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
        n1_outfile = 'ImagePair_%s_n1_registered.csv' % self.row['RID']
        n2_outfile = 'ImagePair_%s_n2_registered.csv' % self.row['RID']
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
        s1_outfile = 'SynapticPair_%s_s1_registered.csv' % self.row.get('RID')
        s2_outfile = 'SynapticPair_%s_s2_registered.csv' % self.row.get('RID')
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

    def put_row_update(self, update_row):
        self.catalog.put(
            '%s;%s' % (
                self.unit.put_update_baseurl,
                ','.join([
                    urlquote(col, safe='')
                    for col in list(update_row.keys())
                    if col not in ['ID', 'RID']
                ])
            ),
            json=[update_row]
        )
        sys.stderr.write('\nupdated in ERMrest: %s' % json.dumps(update_row, indent=2))
    
    work_units = _work_units # these are defined above w/ their funcs and URLs...

    @classmethod
    def look_for_work(cls):
        """Find, claim, and process work for each work unit.

        Do find/claim with HTTP opportunistic concurrency control and
        caching for efficient polling and quiescencs.

        On error, set Status="failed: reason"

        Result:
         true: there might be more work to claim
         false: we failed to find any work
        """
        found_work = False

        for unit in cls.work_units:
            # this handled concurrent update for us to safely and efficiently claim a record
            unit.idle_etag, batch = cls.catalog.state_change_once(
                unit.get_claimable_url,
                unit.put_claim_url,
                unit.claim_input_data,
                unit.idle_etag
            )
            # batch may be empty if no work was found...
            for row, claim in batch:
                found_work = True
                handler = None
                try:
                    handler = cls(row, unit)
                    unit.run_row_job(handler)
                except WorkerBadDataError as e:
                    sys.stderr.write("Aborting task %s on data error: %s\n" % (row["RID"], e))
                    cls.catalog.put(unit.put_claim_url, json=[unit.failure_input_data(row, e)])
                    # continue with next task...?
                except Exception as e:
                    # TODO: eat some exceptions and return True to continue?
                    if unit.failure_input_data is not None:
                        cls.catalog.put(unit.put_claim_url, json=[unit.failure_input_data(row, e)])
                    raise
                finally:
                    if handler is not None:
                        handler.cleanup()

        return found_work

    @classmethod
    def blocking_poll(cls):
        return cls.catalog.blocking_poll(cls.look_for_work, polling_seconds=cls.poll_seconds)

@atexit.register
def _atexit_cleanup():
    os.chdir(Worker.startup_working_dir)
    for dirname in list(Worker.working_dirs):
        Worker.cleanup_working_dir(dirname)
        del Worker.working_dirs[dirname]

