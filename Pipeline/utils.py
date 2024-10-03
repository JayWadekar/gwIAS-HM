import numpy as np
import scipy.signal as signal
import scipy.special as special
import params
import os
import glob
import json
import sys
from collections import defaultdict
import warnings
import getpass
import platform
import importlib
import dill
from numba import vectorize
from scipy.interpolate import interp1d
import datetime
import time
from astropy.time import Time as astrotime

from python_utils import abs2 as abs_sq, abbar, next_power
from python_utils import load_module, import_matplotlib, argmaxnd, \
    npy_append_rows, npy_append_cols, is_numpy_int, is_numpy_float, \
    store_symmetrix_matrix, load_symmetrix_matrix
import copy
import pathlib
import h5py
import gc
import re


# %% Commands and constants
FFT = np.fft.fft
IFFT = np.fft.ifft
RFFT = np.fft.rfft
IRFFT = np.fft.irfft
FFTIN = np.zeros

try:
    import pyfftw
    FFTIN = pyfftw.zeros_aligned
except ImportError:
    pyfftw = None
    FFTIN = np.zeros

# Conversion factor between MAD and sigma
MAD2SIGMA = 1./(np.sqrt(2.) * special.erfinv(0.5))
# Seconds in a year
YEAR = 3.154e7

# Minimum GPS time in O1, everything is 4096 x n + this
# For future runs, keep adding to this and BOUNDS_RUNS below
TMIN = TMIN_O1 = 1126068224
TMAX_O1 = 1137254400
TMIN_O2 = 1164558336
TMAX_O2 = 1187737600
# TMIN_O3 = 1238166018
TMIN_O3 = 1238163456  # This is the minimum GPS time of the starting file
TMIN_O3a = 1238166018
TMAX_O3a = 1253977218
TMIN_O3b = 1256655618
TMAX_O3b = 1269363618


class CustomDefaultdict(dict):
    # Custom defaultdict that passes the key to the factory function
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and callable(args[0]):
            self.factory = args[0]
            args = args[1:]
        else:
            self.factory = None
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.factory is not None:
            self[key] = self.factory(key)
            return self[key]
        else:
            close_hdf5()
            raise KeyError(key)


BOUNDS_RUNS = CustomDefaultdict(
    lambda x: handle_missing_run(x),
    o1=(TMIN_O1, TMAX_O1),
    o2=(TMIN_O2, TMAX_O2),
    o3a=(TMIN_O3, TMAX_O3a),
    o3b=(TMIN_O3b, TMAX_O3b))

# If you have a non-standard run name (HM doesn't count), add it here
BASE_RUNS = CustomDefaultdict(lambda x: standardize_run_name(x), o1new='O1')

# %% Directories and files
# Directories
# Directory with code
from pathlib import Path
CODE_DIR = Path(__file__).parent

# DATA_ROOT is where the data and template banks will be stored
# Add a PERSONALIZED path to the DATA_ROOT in the if-else getpass.getuser() block below
if 'sns.ias.edu' in platform.uname().node.lower():
    DATA_ROOT = '/data/bzackay/GW' # (default directory on the IAS cluster)
else:
    DATA_ROOT = 'Create a personalized DATA_ROOT path in utils.py'

if 'sns.ias.edu' in platform.uname().node.lower():
    # Data directories on the IAS cluster used for past publications
    O3a_DATA_ROOT = "/scratch/lustre/tejaswi/GW"
    O3b_DATA_ROOT = "/scratch/lustre/srolsen/GW"
    HM_O3a_DATA_ROOT = "/data/jayw/IAS/GW/Data/HM_O3a_search"
    HM_O3b_DATA_ROOT = "/data/jayw/IAS/GW/Data/HM_O3b_search"
else:
    O3a_DATA_ROOT = ''
    O3b_DATA_ROOT = ''
    HM_O3a_DATA_ROOT = ''
    HM_O3b_DATA_ROOT = ''

# Make a new entry below for PERSONALIZED paths
if getpass.getuser() == 'bzackay':
    CODE_DIR = '/home/bzackay/temp/gw_detection_ias'
    if platform.uname().node.split('.')[0] == 'mbp99':
        CODE_DIR = '/Users/bzackay/Dropbox (IAS)/python/GW/gw_detection_ias'
        DATA_ROOT = '/Users/bzackay/Data/GW'
        O3a_DATA_ROOT = DATA_ROOT

elif getpass.getuser() == 'barakzackay':
    CODE_DIR = '/Users/barakzackay/python/gw_detection_ias'
    DATA_ROOT = '/Users/barakzackay/data/GW/'
    O3a_DATA_ROOT = DATA_ROOT

elif getpass.getuser() in ['tejaswi', 'Anisha']:
    pname = platform.uname().node.lower()
    if ('mbp77' in pname) or ('mbp104' in pname) or \
            ('wireless.ucsb.edu' in pname):
        # Laptop
        CODE_DIR = '/Users/tejaswi/Work/gw_detection_ias'
        DATA_ROOT = '/Users/tejaswi/Work/gw_detection_ias/largedata'
        O3a_DATA_ROOT = DATA_ROOT
    elif 'anishas-imac' in pname:
        # Home desktop
        CODE_DIR = '/Users/Anisha/Teja/gw_detection_ias'
        DATA_ROOT = '/Users/Anisha/Teja/gw_detection_ias/largedata'
        O3a_DATA_ROOT = DATA_ROOT
    else:
        # Server
        CODE_DIR = '/home/tejaswi/Work/gw_detection_ias'

elif getpass.getuser() == 'jroulet':
    CODE_DIR = '/home/jroulet/GW/gw_detection_ias'

elif getpass.getuser() == 'matiasz':
    CODE_DIR = '/home/matiasz/gw_detection_ias'

elif getpass.getuser() == 'srolsen':
    CODE_DIR = '/home/srolsen/research/gw_detection_ias'
elif getpass.getuser() == 'seth':
    CODE_DIR = '/home/seth/research/gw_detection_ias'

elif getpass.getuser() == 'hschia':
    CODE_DIR = '/home/hschia/PE/gw_detection_ias'

elif getpass.getuser() == 'jayw':
    if platform.uname().node.split('.')[0] == 'mbp179':
        DATA_ROOT = '/Users/jayw/Documents/Acad/GW/Pipeline_data'

elif getpass.getuser() == 'javier':
    CODE_DIR = '/home/javier/work/GW_search/gw_detection_ias'
    DATA_ROOT = '/home/javier/work/GW_search/gw_detection_ias/largedata'
    O3a_DATA_ROOT = DATA_ROOT

elif getpass.getuser() == 'amehta':
    # UCSB knot cluster
    CODE_DIR = '/home/amehta/gw_detection_ias'
    # DATA_ROOT = '/home/amehta/gw_detection_ias/largedata'

elif getpass.getuser() == 'teja':
    # UCSB knot cluster
    CODE_DIR = '/home/teja/Work/gw_detection_ias'
    DATA_ROOT = '/home/teja/Work/gw_detection_ias/largedata'
    
elif getpass.getuser() == 'isha':
    # UCSB knot cluster
    CODE_DIR = '/home/isha/gw_detection_ias'
    DATA_ROOT = '/home/isha/O3a_data'
    O3a_DATA_ROOT = '/home/isha/O3a_data'
    O3b_DATA_ROOT = '/home/isha/O3b_data'

if platform.uname().node == 'dendro.weizmann.ac.il':
    CODE_DIR = '/Users/barakzackay/python/gw_detection_ias'
    DATA_ROOT = '/Users/barakzackay/data/GW'

if (('wexac.weizmann.ac.il' in platform.uname().node) and
        (getpass.getuser() == 'barakz' or getpass.getuser() == 'jonatahm')):
    if getpass.getuser() == 'barakz':
        CODE_DIR = '/home/labs/barakz/Collaboration-gw/barakz/gw_detection_ias'
    if getpass.getuser() == 'jonatahm':
        CODE_DIR = '/home/labs/barakz/Collaboration-gw/mushkin/gw_detection_ias'
    DATA_ROOT = '/home/labs/barakz/Collaboration-gw'
    O3a_DATA_ROOT = '/home/labs/barakz/Collaboration-gw'


if (('marvin' in platform.uname().node) and
        (getpass.getuser() == 'barakz' or getpass.getuser() == 'jonatahm')):
    if getpass.getuser() == 'barakz':
        CODE_DIR = '/Collaboration-gw/barakz/gw_detection_ias'
    if getpass.getuser() == 'jonatahm':
        CODE_DIR = '/Collaboration-gw/mushkin/gw_detection_ias'
    DATA_ROOT = '/Collaboration-gw'
    O3a_DATA_ROOT = '/Collaboration-gw'
# Directory with ASDs and other small pieces of shared data
SMALL_DATA_DIR = os.path.join(CODE_DIR, 'small_data')

# Strain directories
STRAIN_ROOT = {"o1": os.path.join(DATA_ROOT,'LVK_strain_data'),
               "o1new": os.path.join(DATA_ROOT,'LVK_strain_data'),
               "o2": os.path.join(DATA_ROOT, 'LVK_strain_data',"O2"),
               "o2new": os.path.join(DATA_ROOT,'LVK_strain_data', "O2"),
               "o3a": os.path.join(DATA_ROOT, 'LVK_strain_data', "O3a"),
               "O3a": os.path.join(DATA_ROOT, 'LVK_strain_data', "O3a"),
               "o3b": os.path.join(DATA_ROOT, 'LVK_strain_data', "O3b"),
               "O3b": os.path.join(DATA_ROOT, 'LVK_strain_data', "O3b")}

if O3a_DATA_ROOT=='':
    O3a_DATA_ROOT = os.path.join(DATA_ROOT, "O3a_search")

if O3b_DATA_ROOT=='':
    O3b_DATA_ROOT = os.path.join(DATA_ROOT, "O3b_search")

if HM_O3a_DATA_ROOT=='':
    HM_O3a_DATA_ROOT = os.path.join(DATA_ROOT, "HM_O3a_search")

if HM_O3b_DATA_ROOT=='':
    HM_O3b_DATA_ROOT = os.path.join(DATA_ROOT, "HM_O3b_search")


# Directories with output from the pipeline

SEARCH_OUTPUT_DIR = defaultdict(
    lambda: DATA_ROOT,
    o3a=O3a_DATA_ROOT,
    O3a=O3a_DATA_ROOT,
    o3b=O3b_DATA_ROOT,
    O3b=O3b_DATA_ROOT,
    hm_o3a=HM_O3a_DATA_ROOT,
    hm_o3b=HM_O3b_DATA_ROOT)

CAND_DIR = {key: os.path.join(
    SEARCH_OUTPUT_DIR[key],'Candidates') for key in SEARCH_OUTPUT_DIR}

STATS_DIR = {key: os.path.join(
    SEARCH_OUTPUT_DIR[key],'Stats_new') for key in SEARCH_OUTPUT_DIR}

TRIG_DIR = defaultdict(
    lambda: os.path.join(DATA_ROOT, 'OutputDir'),
    o3a=os.path.join(O3a_DATA_ROOT, 'OutputDir'),
    O3a=os.path.join(O3a_DATA_ROOT, 'OutputDir'),
    o3b=os.path.join(O3b_DATA_ROOT, 'OutputDir'),
    O3b=os.path.join(O3b_DATA_ROOT, 'OutputDir'),
    hm_o3a=os.path.join(HM_O3a_DATA_ROOT, 'Triggers_single_det'),
    hm_o3b=os.path.join(HM_O3b_DATA_ROOT, 'Triggers_single_det'))
# TRIG_DIR is different for backward compatibility

TEMPLATE_DIR = os.path.join(DATA_ROOT, 'templates')

# Prefixes for directory names for different runs and types of sources, this is
# to match the part of the directory name before the chirp mass bank id, and
# can contain wildcards (recommended to not end with an *)
# The part after the chirp mass id is supposed to end with the subbank id
# Warning: Don't try to be clever and have characters like *, ., etc. in the
# directory names!
BBH_PREFIXES = {
    'o1': "*Oct_22*Mcbin",
    'o1new': "O1*BBH_",
    'o2': "O2*BBH_",
    'o3a': "O3a*BBH_",
    'o3b': "O3b*BBH_",
    'hm_o3a': "BBH_",
    'hm_o3b': "BBH_"}

NSBH_PREFIXES = {
    'o1': None,       # No NSBH in O1 - NSBH_banks/* directory missing?
    'o1new': None,    # No "O1*NSBH_"
    'o2': "O2*NSBH_"}

BNS_PREFIXES = {
    'o1': "*Dec_20*BNS",
    'o2': "O2*BNS_"}

SOURCE_TO_PREFIX = {
    'bbh': BBH_PREFIXES,
    'nsbh': NSBH_PREFIXES,
    'bns': BNS_PREFIXES}

# LSC PE samples directories
LSC_PE_DIR = os.path.join(DATA_ROOT, 'LSC_PE_samples')

# Files
# -----
INJ_PATH = os.path.join(SMALL_DATA_DIR, 'injections_list.txt')
# DEFAULT_ASDFILE = os.path.join(DATA_DIR, "LIGO-P1200087-v18-aLIGO_MID_LOW.txt")
DEFAULT_ASDFILE = os.path.join(SMALL_DATA_DIR, "asd_o2.npy")
DEFAULT_ASDFILE_O3 = os.path.join(SMALL_DATA_DIR, "asd_o3a.npy")

if platform.uname().node == 'dendro.weizmann.ac.il':
    DEFAULT_ASDFILE_O3 = '/Users/barakzackay/data/GW/templates/O3a/asd_o3a.npy'
if 'wexac' in platform.uname().node:
    DEFAULT_ASDFILE_O3 = '/home/labs/barakz/Collaboration-gw/templates/O3a/asd_o3a.npy'
if 'marvin' in platform.uname().node:
    DEFAULT_ASDFILE_O3 = '/Collaboration-gw/templates/O3a/asd_o3a.npy'


def before_write_save_old(fname):
    if os.path.exists(fname):
        before_write_save_old(fname+'.old')
        os.system(f'mv {fname} {fname}.old')
    return


def remove_old_versions(fname):
    if os.path.exists(fname+'.old'):
        remove_old_versions(fname+'.old')
        os.system(f'rm -f {fname}.old')
    return


def env_init_lines(
        env_command=None, module_name=None, env_name=None, python_name=None,
        cluster="helios"):
    """
    Creates text snippet to be added before submitting jobs to clusters
    :param env_command:
        Override the combination of module loading and sourcing the conda
        environment
    :param module_name: Override module to be initialized
    :param env_name: Override user-dependent conda environment name
    :param python_name: Override user- and cluster-dependent python command name
    :param cluster: Cluster being used, add options below for new clusters
    :return:
        Text snippet that is to be added in the commands to the cluster
        (no ending space)
    """
    text = ""
    read_env = True

    if env_command is not None:
        text = env_command.strip("\n") + "\n"
        read_env = False

    # Infer from the username/whatever information was provided
    if getpass.getuser() == 'tejaswi':
        if env_name is None:
            env_name = "gwias"
    elif getpass.getuser() == 'bzackay':
        if module_name is None:
            module_name = "anaconda3-user/2020.11"
        if env_name is None:
            env_name = "gwias"
        if python_name is None:
            python_name = "/opt/user/anaconda3/2020.11/envs/gwias/bin/python"
    elif getpass.getuser() == 'barakz':
        python_name = '/home/labs/barakz/barakz/anaconda3/envs/gwias/bin/python'
    elif getpass.getuser() == 'jroulet':
        if env_name is None:
            env_name = "sourcelal"
    elif getpass.getuser() == "srolsen":
        if env_name is None:
            env_name = "gwxphm"
    elif getpass.getuser() == "hschia":
        if env_name is None:
            env_name = "gwxphm_new"
    else:
        if env_name is None and read_env:
            print("Warning: not activating any conda environment!")

    if (module_name is not None) and read_env:
        text += f"module load {module_name}\n"
    if (env_name is not None) and read_env:
        text += f"source activate {env_name}\n"

    cluster_string = ""
    if (cluster.lower() == 'helios') or (cluster.lower() == 'typhon'):
        cluster_string = "srun "

    if python_name is None:
        python_name = "python"

    text += f"{cluster_string}{python_name}"

    return text


# %% IO
    
def rm_suffix(filepath, suffix='.json', new_suffix=None):
    """
    Utility to change the extension of a path
    :param filepath: String or an instance of pathlib.PosixPath
    :param suffix: Suffix to remove. Pass '.*' to remove any existing extension
    :param new_suffix: Suffix to add, if desired
    :return:
    """
    if new_suffix is None:
        new_suffix = ""

    path = pathlib.Path(filepath)

    # Treat the case where we want to remove any existing extension
    if suffix == '.*':
        path = path.with_suffix('')
    elif suffix and path.name.endswith(suffix):
        path = path.with_name(path.name[:-len(suffix)])

    # Add the new string
    path = path.with_name(path.name + new_suffix)

    # Return the result in the same type as the input
    if isinstance(filepath, pathlib.PosixPath):
        return path
    else:
        return str(path)


# Treat numpy objects gracefully during I/O
class NumpyEncoder(json.JSONEncoder):
    @staticmethod
    def np_out_hook(item):
        npf = NumpyEncoder.np_out_hook
        if isinstance(item, tuple):
            # Fix for tuple of numpy arrays
            return {'__tuple__': True, 'items': npf(list(item))}
        elif isinstance(item, list):
            # Fix for list of numpy arrays
            return [npf(e) for e in item]
        elif isinstance(item, dict):
            # Fix for dict with numpy arrays
            return {key: npf(value) for key, value in item.items()}
        elif isinstance(item, np.ndarray):
            return item.tolist()
        elif is_numpy_int(item):
            return int(item)
        elif is_numpy_float(item):
            return float(item)
        elif isinstance(item, (np.complex_, np.complex64, np.complex128, complex)):
            return {'real': item.real, 'imag': item.imag}
        else:
            return item

    @staticmethod
    def np_in_hook(obj):
        npf = NumpyEncoder.np_in_hook
        try:
            if isinstance(obj, dict):
                if '__tuple__' in obj:
                    return tuple(npf(obj['items']))
                else:
                    return {key: npf(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return np.asarray(obj, dtype=object)
            else:
                return obj
        except:
            return obj

    # def encode(self, obj):
    #     return super().encode(self.np_out_hook(obj))

    def default(self, obj):
        return self.np_out_hook(obj)

    # def default(self, obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     return json.JSONEncoder.default(self, obj)


# Gracefully treat tuples in JSON files
class TupleEncoder(json.JSONEncoder):
    @staticmethod
    def tuple_out_hook(item):
        tupf = TupleEncoder.tuple_out_hook
        if isinstance(item, tuple):
            return {'__tuple__': True, 'items': list(item)}
        elif isinstance(item, list):
            return [tupf(e) for e in item]
        elif isinstance(item, dict):
            return {key: tupf(value) for key, value in item.items()}
        else:
            return item

    @staticmethod
    def tuple_in_hook(obj):
        if '__tuple__' in obj:
            return tuple(obj['items'])
        else:
            return obj

    def encode(self, obj):
        return super().encode(self.tuple_out_hook(obj))


# get gwf channel name
def get_gwf_channel_names(fname):
    import gwpy.io.gwf
    return gwpy.io.gwf.get_channel_names(fname)


def handle_missing_run(run_name):
    close_hdf5()
    raise ValueError(f"Run {run_name} not recognized")


def standardize_run_name(run_name):
    """Maintain consistency in run names, mainly used to confirm with the GWOSC
    naming conventions
    :param run_name: Run name to be standardized
    :return Standardized run name, with leading hm_ removed and capitalized
    """
    lower_case_run_name = run_name.lower()
    if lower_case_run_name.startswith("hm_"):
        # Remove the HM prefix
        lower_case_run_name = lower_case_run_name[3:]
    return lower_case_run_name.capitalize()


def get_strain_fnames(t, run=None, check_exists=True):
    """
    Returns the strain file names given a time
    :param t: GPS time
    :param run: String identifying the run, if None, infers from the GPS time
    :param check_exists: If True, returns None for missing files
    :return: List of length n_detectors with strain file names
    """
    if run is None:
        run = get_run(t)
    # Get the standardized run name for strain files
    run = BASE_RUNS[run]

    t0 = TMIN + int(np.floor((t - TMIN) / 4096)) * 4096
    strain_root = STRAIN_ROOT[run.lower()]
    strain_files = []
    if run.lower() == "o1":
        for det in ["H", "L"]:
            IFO = det + "1"
            strain_dir = os.path.join(strain_root, IFO)
            strain_files.append(
                os.path.join(
                    strain_dir, f"{det}-{det}1_LOSC_4_V1-{t0}-4096.hdf5"))
    else:
        for det in ["H", "L", "V"]:
            strain_dir = os.path.join(strain_root, det + "1")
            strain_files.append(
                os.path.join(
                    strain_dir,
                    f"{det}-{det}1_GWOSC_{run}_4KHZ_R1-{t0}-4096.hdf5"))

    # Return something only if it exists
    if check_exists:
        strain_files = [f if os.path.isfile(f) else None for f in strain_files]

    return strain_files


def get_left_right_fnames(fname, return_only_existing=True):
    disjoint_name = fname.split('-')
    left_fname = '-'.join(
        disjoint_name[:-2] +
        [str(int(disjoint_name[-2]) - 4096)] +
        disjoint_name[-1:])
    right_fname = '-'.join(
        disjoint_name[:-2] +
        [str(int(disjoint_name[-2]) + 4096)] +
        disjoint_name[-1:])
    if return_only_existing:
        # Return files only if they exist
        if not os.path.isfile(left_fname):
            left_fname = None
        if not os.path.isfile(right_fname):
            right_fname = None
    return left_fname, right_fname


def get_coincident_json_filelist(
        dir_name, enumerated_epochs=None, n_epochs=None, run=None,
        det1="H1", det2="L1"):
    """
    Assumes that the file names go like "Det-Detn-...." where Detn is the key
    :param dir_name: Directory with json and trig files, created per subbank
    :param enumerated_epochs: List of epochs, if needed
    :param n_epochs: Number of epochs, if needed
    :param run: String identifying the run
    :param det1: Key identifying the first detector (e.g., "H1", "L1", "V1")
    :param det2: Key identifying the second detector (e.g., "H1", "L1", "V1")
    :return: 1. List of json files in the first detector
                (includes those with no coincidence with the second)
             2. List of coincident json files in the second detector
             3. Integer list of epochs corrresponding to 1
             4. Integer list of epochs corrresponding to 2
    """
    # Get files to look in
    # -------------------------------------------------------------
    if enumerated_epochs is None:
        files_1 = glob.glob(os.path.join(dir_name, f"*{det1}*.json"))
        files_2 = glob.glob(os.path.join(dir_name, f"*{det2}*.json"))
    else:
        # Ensure the right format
        enumerated_epochs = [int(x) for x in enumerated_epochs]

        if run is None:
            # Warning: Ensure that we don't pass multiple runs together
            # TODO: Make it safe in the super future
            run = get_run(enumerated_epochs[0])

        # List of H files
        files_1 = [get_json_fname(dir_name, ep, det1, run)
                   for ep in enumerated_epochs]

        # Add in left and right files for L
        files_2 = []
        for epoch in enumerated_epochs:
            eplist = [epoch,
                      int(epoch + params.DEF_FILELENGTH),
                      int(epoch - params.DEF_FILELENGTH)]
            files_2 += [get_json_fname(dir_name, ep, det2, run)
                        for ep in eplist]

    # Remove repetitions that occur if consecutive files were passed
    files_1 = list(set(files_1))
    files_2 = list(set(files_2))

    # Keep only files that exist
    files_1 = [f for f in files_1 if os.path.isfile(f)]
    files_2 = [f for f in files_2 if os.path.isfile(f)]

    # Sort by t0 to make the order predictable
    files_1 = sorted(files_1, key=lambda f: int(f.split("-")[-2]))
    files_2 = sorted(files_2, key=lambda f: int(f.split("-")[-2]))

    # List of epochs
    epochs_1 = np.array([int(f.split("-")[-2]) for f in files_1])
    epochs_2 = np.array([int(f.split("-")[-2]) for f in files_2])
    joint_epochs = [ep for ep in epochs_1 if ep in epochs_2]
    print("Number of joint epochs:", len(joint_epochs), flush=True)

    if n_epochs is not None:
        files_1 = files_1[:n_epochs]
        epochs_1 = epochs_1[:n_epochs]

    return files_1, files_2, epochs_1, epochs_2


# Functions to locate files of the required kind
def get_dirs(dtype='trigs', vers_suffix='', source='BBH', runs=('O2',)):
    """
    Gives a list of length n_runs with each entry being a dictionary of
    dictionaries with the outer dictionary having chirp mass ids as keys and
    the inner dictionary having subbank ids as keys and subdirectory names as
    values. Figures out the number of subbanks and chirp mass ids by itself
    :param dtype: Type of directory (trigs, cand, stats)
    :param vers_suffix:
        Suffix after the directory name if needed
        (e.g., for candidates version 4, we used 'cand4' in the past)
    :param source: Source type (BBH, NSBH, BNS)
    :param runs: List of run names
    :return:
        list of n_runs dictionaries with chirp mass ids as keys and list of
        subbank directories as values
    """
    root_dirs = get_root_dirs(dtype=dtype, runs=runs)

    run_dicts = []
    for run, root_dir in zip(runs, root_dirs):
        prefix_dict = SOURCE_TO_PREFIX.get(source.lower(), None)
        if prefix_dict is None:
            close_hdf5()
            raise RuntimeError(f"Could not recognize source {source}!")

        prefix = prefix_dict.get(run.lower(), None)
        if prefix is None:
            close_hdf5()
            raise RuntimeError(f"Could not find files for {source} in {run}!")

        run_dicts.append(
            create_chirp_mass_directory_dict(root_dir, prefix, vers_suffix))

    return run_dicts


def get_root_dirs(dtype='trigs', runs=('O2',)):
    """Gets root directories by run, used since O3a is on scratch"""
    if dtype.lower() in ['trigs', 'output']:
        root_dirs = [TRIG_DIR[run.lower()] for run in runs]
    elif dtype.lower() == 'cand':
        root_dirs = [CAND_DIR[run.lower()] for run in runs]
    elif dtype.lower() == 'stats':
        root_dirs = [STATS_DIR[run.lower()]for run in runs]
    else:
        close_hdf5()
        raise RuntimeError(f"Could not recognize directory type {dtype}")
    return root_dirs


def preprocess_wildcards(input_string, remove_trailing_star=False):
    # Replace '*' with '.*' only if it is not already followed by '.'
    processed_string = re.sub(r'(?<!\.)\*(?!\.)', '.*', input_string)

    # Remove trailing '*' if needed
    if remove_trailing_star and processed_string.endswith(".*"):
        processed_string = processed_string[:-2]

    return processed_string


def extract_parts(directory_name, prefix, suffix):
    """
    Extracts the parts of the directory name based on the prefix and suffix
    :param directory_name: Subdirectory name to extract parts from
    :param prefix:
        Prefix to match the subdirectories for the given run and source type,
        can have wildcards (see BBH_PREFIXES etc., for examples). The prefix
        should match the part of the subdirectory name before the chirp mass id
        if the chirp mass id is absent, it makes it zero (like O1 BNS)
        The part after the chirp mass id is supposed to end with the subbank id
    :param suffix:
        Any desired suffix to demand at the end of the subdirectory's name
        (cand type is the use case). Can be the empty string
    :return:
    """
    # Preprocess the prefix and suffix to handle '*' and '.*'
    if not suffix:
        # Runaway problem if the end of prefix is * and suffix is ""
        prefix = preprocess_wildcards(prefix, remove_trailing_star=True)
    else:
        prefix = preprocess_wildcards(prefix)
    suffix = preprocess_wildcards(suffix)

    # Construct the regex pattern with the processed prefix and suffix
    # \d* is used to match zero or more digits (zero digits for BNS in O1)
    # (.*?) is a non-greedy match for characters between the digits and the
    # suffix
    # \d+ is used to match one or more digits for the subbank id
    pattern = re.compile(rf'^{prefix}(\d*)(.*?)(\d+){suffix}$')

    matches = pattern.match(directory_name)
    if matches:
        # Assign 0 if no digits are present
        chirp_mass_id = matches.group(1) if matches.group(1) else '0'
        subbank_id = matches.group(3)
        return chirp_mass_id, subbank_id
    return None


def create_chirp_mass_directory_dict(base_path, prefix, suffix):
    """
    Reads the names of all subdirectories, and creates a dictionary of
    dictionaries, with the outer dictionary having chirp mass ids as keys and
    the inner dictionary having subbank ids as keys and subdirectory names as
    values
    :param base_path: Base path to look for subdirectories
    :param prefix:
        Prefix to match the subdirectories for the given run and source type,
        can have wildcards (see BBH_PREFIXES etc for examples). The prefix
        should match the part of the subdirectory name before the chirp mass id
        if the chirp mass id is absent, it makes it zero (like O1 BNS)
        The part after the chirp mass id is supposed to end with the subbank id
    :param suffix:
        Any desired suffix to demand at the end of the subdirectories' names
        (cand type is the use case). Can be the empty string
    """
    base_path = pathlib.Path(base_path)
    prefix_path = pathlib.Path(prefix)

    # Edge case where the prefix has a subdirectory name
    subdirectory_path = base_path / prefix_path.parent
    prefix_expression = prefix_path.stem

    # Structure to hold the subdirectories grouped by chirp mass id
    grouped_directories = defaultdict(list)

    for subdir in subdirectory_path.iterdir():
        if subdir.is_dir():
            parts = extract_parts(subdir.name, prefix_expression, suffix)
            if parts:
                chirp_mass_id, subbank_id = parts
                grouped_directories[int(chirp_mass_id)].append(
                    (int(subbank_id), subdir))

    sorted_dict = {}
    for number in sorted(grouped_directories):
        # Sort the subdirectories by subbank id
        sorted_directories = sorted(
            grouped_directories[number], key=lambda x: x[0])
        sorted_dict[number] = {item[0]: item[1] for item in sorted_directories}

    return sorted_dict


def is_in_run(tgps, run):
    """Check if a GPS time is in a given run, works with HM/non-standard runs"""
    base_run = BASE_RUNS[run].lower()
    bounds = BOUNDS_RUNS[base_run]
    return bounds[0] <= tgps <= bounds[1]


# get run name
def get_run(tgps, use_HM=False):
    """Returns the run name for a given GPS time"""
    run_to_use = None
    for run, bounds in BOUNDS_RUNS.items():
        if bounds[0] <= tgps <= bounds[1]:
            run_to_use = run
            break

    if run_to_use is None:
        close_hdf5()
        raise ValueError(f"GPS time {tgps} isn't in available runs")

    if use_HM:
        run_to_use = "hm_" + run_to_use

    return run_to_use


def get_detector_fnames(
        t, chirp_mass_id=0, subbank=None, run=None, source='BBH', use_HM=False,
        dname=None, detectors=('H1', 'L1', 'V1'), return_only_existing=True):
    """
    :param t: GPS time
    :param chirp_mass_id: Chirp mass bank id
    :param subbank: Subbank id, if None, returns all subbanks
    :param run: Run name, if None, figures it out from t and use_HM
    :param source: Source type (BBH, NSBH, BNS)
    :param use_HM: If True, uses HM run names
    :param dname: If not None, returns the json files in this directory
    :param detectors: List of detectors to return files for
    :param return_only_existing:
        If True, returns only existing files and None for missing ones
    :return:
        An n_detector array of json file names for the TriggerLists at time t
    """
    if run is None:
        run = get_run(t, use_HM=use_HM)

    if dname is not None:
        outputdirs = [dname]
    else:
        # n_subbanks
        outputdirs = get_dirs(
            dtype='trigs', source=source, runs=[run])[0][chirp_mass_id]

    t0 = TMIN + int(np.floor((t - TMIN) / 4096)) * 4096
    # n_subbanks x n_detectors
    fnames = [[
        get_json_fname(outputdirs[i], t0, d, run) for d in detectors]
        for i in range(len(outputdirs))]

    if return_only_existing:
        # Return files only if they exist
        fnames = [[f if os.path.isfile(f) else None for f in fnamelist]
                  for fnamelist in fnames]

    if dname is not None:
        # n_detectors
        return fnames[0]

    elif subbank is not None:
        return fnames[subbank]

    else:
        return fnames


def get_json_fname(dir_name, epoch, detector, run="O3a"):
    """
    TODO: What if we choose a different fmax?
    Assumes that the file names go like "Det-Detn-...." where Detn is the
    detector key
    :param dir_name: Directory with json and trig files, created per subbank
    :param epoch: Integer with epoch (t0)
    :param detector: Key identifying the detector (e.g., "H1", "L1", "V1")
    :param run: String identifying the run
    :return: Absolute json fname
    """
    epoch = int(epoch)
    if run is None:
        run = get_run(epoch)
    # Get the standardized run name for strain files, which the json inherits
    run = BASE_RUNS[run]

    det = detector.rstrip("0123456789")
    if run.lower() in ["o1", "o1new"]:
        return os.path.join(
            dir_name,
            f"{det}-{detector}_LOSC_4_V1-{epoch}-4096_config.json")
    else:
        return os.path.join(
            dir_name,
            f"{det}-{detector}_GWOSC_{run}_4KHZ_R1-{epoch}-4096_config.json")


def get_dtype(data):
    try:
        return data.dtype
    except AttributeError:
        if hasattr(data, '__iter__') and (len(data) > 0) and \
                not isinstance(data, str):
            return get_dtype(data[0])
        else:
            return type(data)


# %% Functions to deal with hdf5 files
# List of operations to support for numpy-like behavior
_SUPPORTED_OPS = [
    ('add', np.add, '+'),
    ('sub', np.subtract, '-'),
    ('mul', np.multiply, '*'),
    ('truediv', np.divide, '/'),
    ('floordiv', np.floor_divide, '//'),
    ('mod', np.mod, '%'),
    ('pow', np.power, '**'),
    ('and', np.bitwise_and, '&'),
    ('or', np.bitwise_or, '|'),
    ('xor', np.bitwise_xor, '^'),
    ('invert', np.bitwise_not, '~'),
    ('lshift', np.left_shift, '<<'),
    ('rshift', np.right_shift, '>>'),
    ('lt', np.less, '<'),
    ('le', np.less_equal, '<='),
    ('eq', np.equal, '=='),
    ('ne', np.not_equal, '!='),
    ('gt', np.greater, '>'),
    ('ge', np.greater_equal, '>='),
    ('pos', np.positive, '+'),
    ('neg', np.negative, '-'),
    ('abs', np.absolute, 'abs'),
    ('divmod', np.divmod, 'divmod'),
]

_SUPPORTED_REVERSE_OPS = [
    ('radd', np.add, '+'),
    ('rsub', np.subtract, '-'),
    ('rmul', np.multiply, '*'),
    ('rtruediv', np.divide, '/'),
    ('rfloordiv', np.floor_divide, '//'),
    ('rmod', np.mod, '%'),
    ('rpow', np.power, '**'),
    ('rand', np.bitwise_and, '&'),
    ('ror', np.bitwise_or, '|'),
    ('rxor', np.bitwise_xor, '^'),
    ('rlshift', np.left_shift, '<<'),
    ('rrshift', np.right_shift, '>>'),
    ('rdivmod', np.divmod, 'divmod'),
]

_SUPPORTED_INPLACE_OPS = [
    ('iadd', np.add, '+='),
    ('isub', np.subtract, '-='),
    ('imul', np.multiply, '*='),
    ('itruediv', np.divide, '/='),
    ('ifloordiv', np.floor_divide, '//='),
    ('imod', np.mod, '%='),
    ('ipow', np.power, '**='),
    ('iand', np.bitwise_and, '&='),
    ('ior', np.bitwise_or, '|='),
    ('ixor', np.bitwise_xor, '^='),
    ('ilshift', np.left_shift, '<<='),
    ('irshift', np.right_shift, '>>='),
]


def populate_magic_methods(cls):
    """Decorator to populate magic methods for HDF5DatasetSubset, they enable
    us to interact with it like it's a numpy array when needed"""
    # First define the magic methods for the operations
    for op_name, op_func, _ in _SUPPORTED_OPS:
        def method(self, other, op_func=op_func):
            result = op_func(self.dataset[self.base_index], other)
            return result
        setattr(cls, f'__{op_name}__', method)

    # Then define the magic methods for the reverse operations
    for op_name, op_func, _ in _SUPPORTED_REVERSE_OPS:
        def method(self, other, op_func=op_func):
            result = op_func(other, self.dataset[self.base_index])
            return result
        setattr(cls, f'__{op_name}__', method)

    # Finally define the magic methods for the inplace operations
    for op_name, op_func, _ in _SUPPORTED_INPLACE_OPS:
        def method(self, other, op_func=op_func):
            self.dataset[self.base_index] = op_func(
                self.dataset[self.base_index], other)
            return self  # Return the modified object
        setattr(cls, f'__{op_name}__', method)

    return cls


@populate_magic_methods
class HDF5DatasetSubset:
    """Simulates a copy-on-write reference to a subset of a multidimensional or
    variable-length HDF5 dataset. This is needed because by default, edits to a
    HDF5 dataset's elements don't propagate back to the underlying dataset"""
    __slots__ = ['dataset', 'base_index']
    
    def __init__(self, dataset, base_index):
        """
        Initializes dataset[base_index] without making an intermediate copy as
        would happen with fancyindexing. Copies are made when
        dataset[base_index][key] is called
        :param dataset:
            Instance of h5py.Dataset (can't be EditableHDF5Dataset as that would
            cause infinite recursion)
        :param base_index: Index to the subset in the dataset
        """
        self.dataset = dataset
        self.base_index = base_index

    def __getitem__(self, key):
        # Get item(s) from the dataset
        # Can be inefficient as we're creating a copy...
        return self.dataset[self.base_index][key]

    def __setitem__(self, key, value):
        # Set the stored data in the dataset
        row_data = self.dataset[self.base_index]
        row_data[key] = value
        self.dataset[self.base_index] = row_data

    def __iter__(self):
        # Load the dataset only once and iterate over it
        for entry in self.dataset[self.base_index]:
            yield entry

    def __len__(self):
        # Python doesn't route len through __getattr__
        # Inefficient as we're creating a copy...
        return len(self.dataset[self.base_index])

    def __repr__(self):
        # Python doesn't route repr through __getattr__
        s = repr(self.dataset[self.base_index])
        return s.replace("array", "HDF5DatasetSubset").replace(
            '\n       ', '\n                   ')

    def __getattr__(self, name):
        # Forward attribute access to anything else
        return getattr(self.dataset[self.base_index], name)


class EditableHDF5Dataset:
    """Wrapper of hdf5 dataset that returns references instead of copies when
    indexed if it is multidimensional or variable-length"""
    __slots__ = ['_dataset', '_is_vlen', '_dshape', '_dlen']
    
    def __init__(self, dataset):
        # Underscores as datasets can have attributes exposed to the user
        self._dataset = dataset
        self._is_vlen = h5py.check_vlen_dtype(dataset.dtype)
        self._dshape = self._dataset.shape
        self._dlen = len(self._dataset)

    def __getattr__(self, name):
        # Forward attribute access to the underlying dataset
        return getattr(self._dataset, name)

    def __len__(self):
        # Python doesn't route len through __getattr__
        return self._dlen

    def return_scalar(self, key):
        """
        Checks whether the return type is a scalar
        :param key: Key used to index into the dataset
        :return: True if the return type is a scalar
        """
        key = key if isinstance(key, tuple) else (key,)
        for item in key:
            if isinstance(item, bool) or not isinstance(item, int):
                # Fancyindexing or slicing is in play
                return False

        if len(key) >= len(self._dshape):
            # Scalar or error that we don't need to handle
            return True
        else:
            return False

    def __getitem__(self, key):
        # Recall that we have to be consistent with fancyindexing
        # A[(0, 1)] is different from A[(0, 1),] - the former is the element at
        # row index = 0, column index = 1, and the latter picks the first two
        # rows
        # Get item(s) from the dataset
        if self._dshape is None or len(self._dshape) == 0:
            # It's not our responsibility to handle errors here
            return self._dataset[key]

        if not self._is_vlen and self.return_scalar(key):
            return self._dataset[key]
        else:
            return HDF5DatasetSubset(self._dataset, key)

    def __setitem__(self, key, value):
        # Delegate item assignment to the underlying dataset as it doesn't go
        # through __getattr__
        self._dataset[key] = value

    def __iter__(self):
        # Python doesn't route __iter__ through __getattr__
        if len(self._dshape) == 0:
            close_hdf5()
            raise TypeError("Can't iterate over a scalar dataset")
        elif len(self._dshape) == 1:
            # Dereference once and keep serving from the numpy array
            for entry in self._dataset[()]:
                yield entry
        else:
            for i in range(self._dshape[0]):
                yield self[i]

    def __repr__(self):
        # Python doesn't route repr through __getattr__
        # Build off the representation of the dataset
        s = repr(self._dataset)
        return s.replace("<HDF5 dataset", "<EditableHDF5Dataset")


class CustomHDF5Group(h5py.Group):
    """Custom h5py group object that wraps datasets with EditableHDF5Dataset
    on access"""
    @classmethod
    def create_custom_group(cls, group):
        """Factory method to create an CustomHDF5Group instance"""
        instance = cls.__new__(cls)
        # instance._id = group.id
        super(CustomHDF5Group, instance).__init__(group.id)

        # Copy attributes from the parent group
        for key, value in group.attrs.items():
            instance.attrs[key] = value

        return instance

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if (isinstance(item, h5py.Dataset) and
                (item.shape is not None and len(item.shape) > 0)):
            # Wrap the dataset object with EditableHDF5Dataset
            return EditableHDF5Dataset(item)
        elif isinstance(item, h5py.Group):
            # Recursively wrap datasets within groups
            return CustomHDF5Group.create_custom_group(item)
        else:
            # Return other objects as-is
            return item

    def __repr__(self):
        # Build off the representation of the group
        s = super().__repr__()
        return s.replace("<HDF5 group", "<CustomHDF5Group")


class CustomHDF5File(h5py.File):
    """Custom h5py file object that wraps datasets with EditableHDF5Dataset
    on access"""
    def __getitem__(self, key):
        item = super().__getitem__(key)
        if (isinstance(item, h5py.Dataset) and
                (item.shape is not None and len(item.shape) > 0)):
            # Wrap the dataset object with EditableHDF5Dataset
            return EditableHDF5Dataset(item)
        elif isinstance(item, h5py.Group):
            # Recursively wrap datasets within groups
            try:
                return CustomHDF5Group.create_custom_group(item)
            except (KeyError, OSError, RuntimeError):
                # Weird error when accessing attributes of read only files
                print(f"Couldn't access {key} properly, falling back " + 
                      f"to base group")
                return item
        else:
            # Return other objects as-is
            return item

    def __repr__(self):
        # Build off the representation of the File
        s = super().__repr__()
        return s.replace("<HDF5 file", "<CustomHDF5File")


def extract_filename(fname):
    """
    Extracts the filename from a string representation of a buffer object
    :param fname: String
    :return: Filename if found, or the original string
    """
    mtch = re.search(r"<_io\.BufferedReader name='(.+?)'>", fname)
    if mtch:
        return mtch.group(1)
    else:
        return fname


HDF5_MODE_DICT = {"r": ['r', 'r+', 'a'],
                  "r+": ['r+', 'a'],
                  "w": ['w'],
                  "w-": ['w-', 'x'],
                  "x": ['x', 'w-'],
                  "a": ['r+', 'a']}


def get_hdf5_file(source, mode="a", raise_error=False, **creation_kwargs):
    """
    Returns a hdf5 file object with the given mode, given a source. Warning, if
    the file is already open in read-only mode, we can't return a writeable
    version.
    :param source: The path to the hdf5 object, or the File object itself
    :param mode:
        Mode to open the file in (recommended "r" for read only, and "a" for
        read/write)
    :param raise_error: If True, raise an error if the file is not found
    :param creation_kwargs:
        Keyword arguments to pass to h5py.File if creating it for the first time
    :return:
        1. File object that satisfies the mode conditions. Returns None if
           source is None, and reuses existing object if it satisfies
           the conditions
        2. Boolean indicating if the file was opened in the function
    """
    if source is not None:
        if isinstance(source, h5py.File):
            try:
                if source.mode in HDF5_MODE_DICT[mode]:
                    # The file is already in the required mode
                    f = source
                    toclose = False
                else:
                    # Check if it was created from a buffer instead of a string
                    hdf5_fname = extract_filename(source.filename)
                    track_order = bool(
                        source.id.get_create_plist().get_link_creation_order())
                    f = CustomHDF5File(
                        hdf5_fname, mode, track_order=track_order)
                    toclose = True
            except ValueError:
                if raise_error:
                    close_hdf5()
                    raise
                else:
                    f = None
                    toclose = False
        else:
            hdf5_fname = source
            try:
                creation_kwargs["track_order"] = creation_kwargs.get(
                    "track_order", True)
                f = CustomHDF5File(hdf5_fname, mode, **creation_kwargs)
                toclose = True
            except FileNotFoundError:
                if raise_error:
                    close_hdf5()
                    raise
                else:
                    f = None
                    toclose = False
    else:
        f = None
        toclose = False

    return f, toclose


def close_hdf5(fnames=None):
    """
    :param fnames: If known, close only these filenames (single or list)
    :return: Closes open hdf5 files (all, or selected)
    """
    if fnames is not None:
        if isinstance(fnames, str):
            fnames = [fnames]
    for obj in gc.get_objects():
        try:
            flag = isinstance(obj, h5py.File)
        except ReferenceError:
            continue
        if flag:
            if fnames is not None:
                try:
                    flag = obj.filename not in fnames
                    if flag:
                        continue
                except ValueError:
                    continue
            try:
                obj.close()
            except AttributeError:
                pass
    return


def read_hdf5_node(fobj, h5path, create=True, raise_error=False):
    """
    Convenience function to read/create a group from a hdf5 file
    :param fobj: hdf5 file or group
    :param h5path: String, or an iterable of strings to reach the leaf
    :param create: If True, we create what doesn't exist
    :param raise_error: If True, raise an error if the path doesn't exist
    :return:
        group object, or None if it doesn't exist and create and raise_error
        are False
    """
    if isinstance(h5path, str):
        h5path = [h5path]

    g = fobj
    for sub in h5path:
        if sub in g.keys():
            g = g[sub]
        elif create:
            _ = g.create_group(sub)
            g = g[sub]  # Do it this way to ensure CustomHDF5Group is used
        elif raise_error:
            close_hdf5()
            raise RuntimeError(f"Path {h5path} not in {fobj.filename}")
        else:
            return None

    return g


def mmap_h5(path, h5path, mode="r", dtype=None):
    """
    Memory mapping is much faster for single row access than non-chunked hdf5,
    as long as we have a 64 bit architecture
    It is slower for random access, as the kernel version on the IAS systems
    is older and doesn't activate the madvise system call
    WARNING: If the hdf5 file is already open (e.g., in a writeable manner), the
    values in the mmap object might not be updated until the changes to the hdf5
    file are flushed (despite the description of the dedault UNIX driver, it has
    a small buffer, set by rdcc_nbytes)!
    :param path: Path to hdf5 file, or the file object itself
    :param h5path: Name of dataset within the file, or iterable with path to it
    :param mode: {‘r+’, ‘r’, ‘w+’, ‘c’}, see documentation of mmap
    :param dtype:
        Interpret the data using this datatype (defaults to the saved type,
        different from type conversion! use only for complex/real)
    :return: mmap object to query as needed
    """
    # First open the file to read offsets
    f, toclose = get_hdf5_file(path, "r", raise_error=True)

    if f is None:
        raise RuntimeError(f"File {path} not found")

    ds = read_hdf5_node(f, h5path, create=False)
    # We get the dataset address in the HDF5 field
    offset = ds.id.get_offset()
    # We ensure we have a non-compressed contiguous array
    assert ds.chunks is None
    assert ds.compression is None
    assert offset > 0
    if dtype is None:
        dtype = ds.dtype
        shape = ds.shape
    else:
        shape = list(ds.shape)
        shape[-1] *= np.dtype(ds.dtype).itemsize
        shape[-1] = shape[-1] // np.dtype(dtype).itemsize
        shape = tuple(shape)

    source = f.filename

    if toclose:
        f.close()

    arr = np.memmap(source, mode=mode, shape=shape, offset=offset, dtype=dtype)

    return arr


def delete_hdf5_datasets(source, h5path, leaves):
    """
    Convenience function to prune some leaves from a hdf5 file
    :param source: Path to the hdf5 file, or the file object itself
    :param h5path:
        String, or an iterable of strings to reach the parent of the leaves
    :param leaves: String, or iterable of strings with names of leaves to delete
    :return:
    """
    f, toclose = get_hdf5_file(source, "r+")

    if f is None:
        return

    if isinstance(h5path, str):
        h5path = [h5path]

    if isinstance(leaves, str):
        leaves = [leaves]

    g = f
    for sub in h5path:
        if sub in g.keys():
            g = g[sub]
        else:
            # The leaf cannot be reached
            if toclose:
                f.close()
            return

    for leaf in leaves:
        if leaf in g.keys():
            del g[leaf]

    if toclose:
        f.close()

    return


def write_hdf5_node(
        fobj, h5path, data, dtype=None, overwrite=True, outformat=h5py.Dataset,
        outdtype=None, outmode="r+"):
    """
    Convenience function to create/overwrite a leaf in a hdf5 file
    :param fobj: hdf5 file object
    :param h5path:
        String, or an iterable of strings to reach the leaf. The last entry is
        the leaf's name
    :param data: Data to write in the leaf
    :param dtype: If known, type of data. The default of None is to infer it
    :param overwrite:
        Flag indicating whether to overwrite the leaf if it exists
        0: Returns the existing key without doing anything
        1: Deletes the existing dataset and creates a new one
        2. Uses the existing and allocated space if it is possible
        If boolean, False = 0, True = 2
    :param outformat:
        Format of object to create (default is dataset, can also be mmap).
        It can be a string or a type.
    :param outdtype: If known, how to read the data (only used for mmap)
    :param outmode: Mode to open the file in (only used for mmap)
    :return: The leaf object in the desired format
    """
    if checkempty(h5path):
        # Nothing to do
        return None

    if isinstance(h5path, str):
        h5path = [h5path]

    overwrite = bool2int(overwrite)

    root = h5path[:-1]
    leaf = h5path[-1]

    g = read_hdf5_node(fobj, root, create=True)
    rval = None
    dtype_to_check = dtype if dtype is not None else get_dtype(data)

    if leaf in g.keys():
        if overwrite == 0:
            print(f"Leaf {leaf} already exists in {fobj.filename}")
            rval = g[leaf]
        elif overwrite == 1 or h5py.check_vlen_dtype(dtype_to_check):
            # Start a fresh leaf
            del g[leaf]
            fobj.flush()
        else:
            # Try to use the existing space if possible
            try:
                dataset = g[leaf]
                if (dataset.shape == data.shape and
                        dataset.dtype == dtype_to_check):
                    # We can reuse the space
                    dataset[...] = data
                    rval = dataset
                else:
                    # Start a fresh leaf
                    del g[leaf]
                    fobj.flush()
            except AttributeError:
                # Start a fresh leaf
                del g[leaf]
                fobj.flush()

    if rval is None:
        if dtype is not None:
            # Do it this way to ensure EditableHDF5Dataset is returned
            _ = g.create_dataset(leaf, data=data, dtype=dtype)
        else:
            _ = g.create_dataset(leaf, data=data)
        rval = g[leaf]

    if ((isinstance(outformat, str) and
         outformat.lower() in ("mmap", "memmap")) or
            (isinstance(outformat, type) and
             issubclass(outformat, np.memmap))):
        # Ensure the hdf5 object is flushed to disk as we just wrote data
        fobj.flush()
        return mmap_h5(fobj, h5path, mode=outmode, dtype=outdtype)
    else:
        return rval


def save_dict_to_hdf5_attrs(fobj, dic, keys_to_skip=None, overwrite=False):
    """
    Saves a dictionary to the attributes of a hdf5 file
    :param fobj: hdf5 file or group object
    :param dic: Dictionary to save
    :param keys_to_skip: List of keys to skip
    :param overwrite: If True, overwrite existing attributes
    :return: None
    """
    overwrite = bool2int(overwrite)

    if checkempty(keys_to_skip):
        keys_to_skip = []

    for key, value in dic.items():
        if key in keys_to_skip:
            continue

        if overwrite == 0 and key in fobj.attrs.keys():
            print(f"Key {key} already exists in {fobj}!")
            continue

        try:
            fobj.attrs[key] = value
        except TypeError:
            # Save as a binary blob
            bytes_out = dill.dumps(value)
            fobj.attrs[key] = np.void(bytes_out)

    return


def load_dict_from_hdf5_attrs(fobj, keys=None, outdict=None):
    """
    Loads a dictionary from the attributes of a hdf5 file
    :param fobj: hdf5 file or group object
    :param keys: List of keys to load. If None, load all keys
    :param outdict: Dictionary to load into, if None, create a new one
    :return:
        Dictionary with the keys and values if outdict is None, else modifies
        what was passed
    """
    # Fix weird error in sequentially reading the attributes for CustomHDF5File
    dict_to_read = dict(fobj.attrs)
    if keys is None:
        keys = dict_to_read.keys()

    return_dict = False
    if outdict is None:
        outdict = {}
        return_dict = True

    for key in keys:
        val = dict_to_read.get(key, None)
        if isinstance(val, np.void):
            val = dill.loads(val.tobytes())
        outdict[key] = val

    if return_dict:
        return outdict
    else:
        return


# %% ASD and waveform interpolation
def asdf_fromfile(asdfile):
    """
    :param asdfile: File with frequencies (in Hz) and ASD (in 1/sqrt(Hz))
    :return:
        Function that takes frequencies (in Hz), returns ASD (in 1/sqrt(Hz))
    """
    try:
        data = np.loadtxt(asdfile)
    except UnicodeDecodeError:
        data = np.load(asdfile)

    if data.ndim == 1:
        # Bug where frequencies and asd were concatenated
        data = data.reshape(2, -1).T

    elif data.shape[1] == 2 and data.shape[0] > 2:
        # Were stored as column vectors, turn into rows
        data = data.T

    freq, asd = data

    def asdf(f_axis):
        """
        Amplitude spectral density in 1/sqrt(Hz) obtained by linear
        interpolation in log(f in Hz), log(ASD in 1/sqrt(Hz)).
        """
        log_f = np.zeros(len(f_axis))
        mask = f_axis > 0
        log_f[mask] = np.log(f_axis[mask])
        log_f[~mask] = -np.inf
        return np.exp(np.interp(
            log_f, np.log(freq), np.log(asd), left=np.inf, right=np.inf))

    return asdf


def interpolate_asd(old_f, old_asd, log_in=False):
    """
        get log-log interpolant of ASD with old_f[0] >= 0,
        :param old_f: (ordered) array of rfft frequencies >= 0 (in Hz) where ASD is saved
        :param old_asd: len(old_f) array with ASD = sqrt(PSD)
        :param log_in: bool indicating if old_asd is already a log (note old_f is never log)
        :return: Function that takes frequencies (Hz) & returns ASD at those frequencies
    """
    i0 = (0 if old_f[0] > 0 else 1)
    f0 = old_f[i0]
    if log_in:
        old_logasd = old_asd[i0:]
        asd0 = np.exp(old_asd[i0])
    else:
        old_logasd = np.log(old_asd[i0:])
        asd0 = old_asd[i0]
    log_asdf = interp1d(
        np.log(old_f[i0:]), old_logasd, kind='slinear', bounds_error=True,
        assume_sorted=True)

    def asd_func(new_f):
        nbelow = np.count_nonzero(new_f < f0)
        return (np.concatenate([asd0 * np.ones(nbelow), np.exp(log_asdf(np.log(new_f[nbelow:])))])
                if nbelow > 0 else np.exp(log_asdf(np.log(new_f))))
    return asd_func


def interpolate_wf_fd(old_f, old_wf, log_in=False):
    """
    get log-log interpolant of amp & lin of phase with old_f[0] >= 0,
    :param old_f: (ordered) array of rfft frequencies >= 0 (in Hz) where waveform is saved
    :param old_wf: len(old_f) array with frequency domain waveform h(f)
        OR log waveform log(h) = np.log(np.abs(h)) + 1j*np.unwrap(np.angle(h)) => set log_in=True
        --> BE SURE TO UNWRAP PHASE if log_in !!!
    :param log_in: bool indicating if old_wf is log(h) => real, imag = amp, unwrapped_phase
    :return: Function that takes frequencies (Hz) & returns waveform at those frequencies
    """
    i0 = (0 if old_f[0] > 0 else 1)
    f0 = old_f[i0]
    if log_in:
        old_logamp, old_phase = old_wf.real, old_wf.imag
        wf0 = np.exp(old_logamp[i0] + 1j*old_phase[i0])
    else:
        old_logamp, old_phase = np.log(np.abs(old_wf)), np.unwrap(np.angle(old_wf))
        wf0 = old_wf[i0]
    log_ampf = interp1d(
        np.log(old_f[i0:]), old_logamp[i0:], kind='slinear', bounds_error=True,
        assume_sorted=True)
    phasef = interp1d(
        old_f[i0:], old_phase[i0:], kind='slinear', bounds_error=True,
        assume_sorted=True)

    def wf_func(new_f):
        nbelow = np.count_nonzero(new_f < f0)
        return (np.concatenate([wf0 * np.ones(nbelow), np.exp(log_ampf(np.log(new_f[nbelow:]))
                                                              + 1j*phasef(new_f[nbelow:]))])
                if nbelow > 0 else np.exp(log_ampf(np.log(new_f)) + 1j*phasef(new_f)))
    return wf_func


def change_wf_fd_grid(wf_fd, fs_in, fs_out, pad_mode='center'):
    """
    :param wf_fd: FD waveform at frequencies in fs_in
    :param fs_in: np.fft.rfftfreq(n=nfft_in, d=dt_in)[-keeplen:]
        where keeplen represents a possible low-frequency cutoff,
        and dt_in = 1 / (2*fs_in[-1]), nfft_in = T_in / dt_in,
        with T_in = 1 / df_in = 1 / (fs_in[1] - fs_in[0])
    :param fs_out: np.fft.rfftfreq(n=nfft_out, d=dt_out)[-getlen:]
        where getlen represents a possible low-frequency cutoff
        and nfft_out, dt_out are defined from fs_out the same
        way as nfft_in, dt_in
    :param pad_mode: str (`center`, `left`, `right`) specifying
        where in the TD grid is safe to pad, i.e., location of
        the interval of zeros connecting the end of the merger
        signal back to the start of the inspiral
    """
    if len(fs_in) == len(fs_out):
        # if fs_in = fs_out, do nothing
        if np.allclose(fs_in, fs_out):
            return wf_fd
    # get df and check that grids are uniform
    df_in, df_out = fs_in[1] - fs_in[0], fs_out[1] - fs_out[0]
    assert np.allclose(np.diff(fs_in), df_in), 'fs_in is not uniform!'
    assert np.allclose(np.diff(fs_out), df_out), 'fs_out is not uniform!'
    # allow for possibility that fs_in and/or fs_out have a low-frequency cutoff
    fnyq_in, fnyq_out = fs_in[-1], fs_out[-1]
    dt_in, dt_out = 1 / (2 * fnyq_in), 1 / (2 * fnyq_out)
    T_in, T_out = 1 / df_in, 1 / df_out
    # now we can get nfft even if fs_in/out doesn't go all the way down to 0
    nfft_in, nfft_out = int(T_in / dt_in), int(T_out / dt_out)
    # also need intermediate grid for increasing nyquist frequency at old df
    nfft_inter = int(T_in / dt_out)
    # now make corresponding grids (maybe remove this)
    fs_out_rfft = np.fft.rfftfreq(nfft_out, d=dt_out)
    assert np.allclose(fs_out_rfft[-len(fs_out):], fs_out), \
        'fs_out is not an RFFT grid or highpassed RFFT grid'
    fs_in_rfft = np.fft.rfftfreq(nfft_in, d=dt_in)
    assert np.allclose(fs_in_rfft[-len(fs_in):], fs_in), \
        'fs_in is not an RFFT grid or highpassed RFFT grid'
    fs_inter = np.fft.rfftfreq(nfft_inter, d=dt_out)
    # now account for possibility that input wf had low-freqs cut out
    wf_fd_in = np.zeros(len(fs_in_rfft), dtype=np.complex128)
    wf_fd_in[-len(wf_fd):] = wf_fd.copy()
    # now populate intermediate grid, which is either a
    # truncation or (right) zero-padding of wf_fd_in
    wf_fd_inter = np.zeros(len(fs_inter), dtype=np.complex128)
    keeplen_inter_fd = min(len(wf_fd_in), len(wf_fd_inter))
    wf_fd_inter[:keeplen_inter_fd] = wf_fd_in[:keeplen_inter_fd]
    # now change the times with pad_mode specifying where
    # in the time domain waveform it is ok to pad with zeros
    wf_fd_out = change_filter_times_fd(wf_fd_inter, nfft_inter, nfft_out, pad_mode)
    return wf_fd_out[-len(fs_out):]


# %% Algebraic operations and general functions
def threshold_rv(dist, nsamp, *dist_args, nfire=params.NPERFILE, onesided=True):
    """
    :param dist: Distribution in scipy.stats
    :param nsamp: Number of samples
    :param dist_args: Extra arguments for distribution
    :param nfire: Number of times noise should cross the threshold in samples
    :param onesided: False if we want lower threshold too
    :return: Threshold for random variable that is exceeded nfire times in
             nsamp samples on average (upper and lower thresholds if
             onesided=False)
    """
    upper_threshold = dist.isf(nfire / nsamp, *dist_args)
    if onesided:
        return upper_threshold
    else:
        lower_threshold = dist.ppf(nfire / nsamp, *dist_args)
        return lower_threshold, upper_threshold


def bincent(bins):
    return 0.5 * (bins[1:] + bins[:-1])


def is_within(val, bounds):
    return (bounds[0] < val) & (val < bounds[1])


# %% Array operations
def sigma_from_median(arr, axis=-1):
    """Computes sigma from median for an array with Gaussian samps + outliers
    Silently returns 1 if we passed in an empty array
    :param arr: Array with samples
    :param axis: Axis to compute the sigma along
    """
    if checkempty(arr):
        return 1
    median = np.median(arr, axis=axis)
    if isinstance(median, np.ndarray):
        median = np.expand_dims(median, axis=axis)
    mad = np.median(np.abs(arr - median), axis=axis)
    return MAD2SIGMA * mad


def unbias_split(scores, vsq):
    """
    Function to compute split scores - expectations from total score
    :param scores: nchunk x nscore matrix with split scores
                   (can be vector for nscores=1)
    :param vsq:
        Array of length nchunk with fractional norm^2 of cosine waveforms,
        sums to one
    :return nchunk x nscore matrix with unbiased split scores
            (can be vector for nscores=1)
    """
    scores_tot = np.sum(scores, axis=0)
    if len(scores.shape) > 1:
        # Many scores
        scores_unbiased = scores - np.dot(vsq[:, np.newaxis],
                                          scores_tot[np.newaxis, :])
    else:
        # One score
        scores_unbiased = scores - vsq * scores_tot
    return scores_unbiased


def orthogonalize_split(scores, proj_l, submask_l, submask_h=None):
    """
    Function to compute orthogonalized scores to those in the L subset
    :param scores: nchunk x nscore matrix with split scores
                   (can be vector for nscores=1)
    :param proj_l:
        Matrix to multiply scores in the L subset before subtracting from those
        in H to orthogonalize the latter
    :param submask_l: Boolean mask into nchunk to select splits in L
    :param submask_h:
        Boolean mask into nchunk to select splits in H (if not given, assume
        complement of submask_l)
    :return:
        np.count_nonzero(submask_h) x nscore matrix with orthogonalized split
        scores (can be vector for nscores=1)
    """
    if submask_h is None:
        submask_h = np.logical_not(submask_l)

    if len(scores.shape) > 1:
        # Many scores
        scores_orthogonalized = scores[submask_h, :] - \
            np.dot(proj_l, scores[submask_l, :])
    else:
        # One score
        scores_orthogonalized = scores[submask_h] - \
            np.dot(proj_l, scores[submask_l])
    return scores_orthogonalized


def submask(bigmask, *indarrays):
    """
    :param bigmask:
        Boolean mask that picks out a subset of `valid' elements within a set
    :param indarrays:
        Each indarray is either a list of indices or a Boolean mask, into the
        parent set of bigmask, to be retained, subject to validity
    :return: Boolean mask (of length np.count_nonzero(bigmask)) into the set of
    `valid' elements that picks out elements indexed by the union of the
    indarrays
    """
    indmask_orig = np.zeros_like(bigmask, dtype=bool)
    for indarray in indarrays:
        indmask_orig[indarray] = True
    return indmask_orig[np.nonzero(bigmask)[0]]


def splitarray(
        dat, splitvec, interval, axis=0, return_split_keys=False, origin=0):
    """
    Splits a single or multi-dimensional array into sub-arrays based on a
    coordinate
    :param dat:
        List/array to split (can be multidimensional, in which case it has to
        be a numpy array)
    :param splitvec: Numpy vector to split array dimension by
    :param interval: Length of buckets for values in splitvec
    :param axis:
        (>0) Axis to split dat along, dimension should match len(splitvec)
    :param return_split_keys:
        Flag indicating whether to return keys that we split according to
    :param origin: Split according to offsets from this origin
    :return:
        If return_split_keys is True,
        1. Keys split according to
        2. List of subarrays of dat split along axis according to values in
           splitvec every interval
        else, only the second one
    """
    isnp = type(dat) is np.ndarray
    if (axis > 0) and (not isnp):
        raise RuntimeError("Multidimensional arrays need to be numpy arrays")
    if (dat.shape[axis] if isnp else len(dat)) != len(splitvec):
        raise RuntimeError("Dimensions of array and splitvec do not match")
    if len(splitvec) == 0:
        raise RuntimeError("Cannot split according to an empty array!")
    splitvec = np.asarray(splitvec)

    # Sort according to splitvec
    if isnp:
        # Numpy array, create possibly multidimensional slice
        multislice = tuple(
            [slice(None)] * axis + [np.argsort(splitvec)] +
            [slice(None)] * (dat.ndim - axis - 1))
        dat = dat[multislice]
    else:
        # List
        dat = [dat[idx] for idx in np.argsort(splitvec)]

    # Find indices that split dimension into buckets every time_tol seconds
    splitvec = splitvec[np.argsort(splitvec)]

    # Check if we need to use floats for weird edge case that doesn't return
    # originals
    dtype_split = splitvec.dtype
    dtype_interval = type(interval)
    dtype_origin = type(origin)
    if (((dtype_split == np.dtype(int)) and (dtype_interval == np.dtype(int)))
            and (dtype_origin == np.dtype(int))):
        indarray = (splitvec - origin) // interval
    else:
        indarray = np.floor(((splitvec - origin) / interval)).astype(int)

    # Split data according to indices
    split_keys, split_indices = np.unique(indarray, return_index=True)
    dat_split = np.split(dat, split_indices[1:], axis=axis)

    if return_split_keys:
        return split_keys, dat_split
    else:
        return dat_split


def index_limits(window_index, extra_req, jump, window_size, valid_mask):
    """
    Function that decides which indices to include in the average
    such that we always average window_size indices to avoid
    `regression-to-mean' artifacts due to fewer samples near holes
    :param window_index: Index of window to decide limits for
    :param extra_req: Number of extra indices to pull in
    :param jump: Jump in indices between window starts
    :param window_size: Size of window desired, after respecting the valid mask
    :param valid_mask: Valid mask deciding which entries are to kept
    :return: left and right limits of window with required number of entries
    """
    # Define cumulative number of valid inds
    nvalid_upto = np.cumsum(valid_mask)

    # Look at how much room we have to pull in from each side
    left_avail = 0
    # In case i * jump + window_size is larger than length of data
    right_avail = nvalid_upto[-1] - nvalid_upto[
        min(window_index * jump + window_size, len(nvalid_upto) - 1)]
    if window_index > 0:
        left_avail = nvalid_upto[window_index * jump - 1]
    if (left_avail + right_avail) < extra_req:
        raise RuntimeError(
            "Not enough data available to compute a single " +
            "PSD drift correction!")

    # Try to pull in symmetrically
    left_in = min(extra_req // 2, left_avail)
    # Pull in more from the right if we couldn't pull in enough from the left
    right_in = min(extra_req - left_in, right_avail)
    # If there wasn't enough on the right, increase from the left
    left_in += extra_req - (left_in + right_in)
    # Now find indices to pull in
    left_target = nvalid_upto[window_index * jump] - left_in
    left_ind, right_ind = np.searchsorted(
        nvalid_upto, [left_target, left_target + window_size + 1])

    return left_ind, right_ind


def checkempty(array, verbose=True):
    # First deal with irritating case when we can't make a numpy array
    if hasattr(array, "__len__"):
        # Deal with even more irritating edge case when the attribute exists,
        # but throws an error when queried
        try:
            if len(array) > 0:
                return False
        except TypeError:
            # if verbose:
            #    print("Object has `len' attribute that can't be queried")
            return True
    nparray = np.asarray(array)
    if (((nparray is None) or
         (nparray.dtype == np.dtype('O'))) or (nparray.size == 0)):
        return True
    else:
        return False


def safe_concatenate(existing, newlist):
    if checkempty(existing):
        outlist = newlist
    else:
        if checkempty(newlist):
            outlist = existing
        else:
            outlist = np.concatenate([existing, newlist], axis=0)
    return outlist


def bool2int(flag):
    if isinstance(flag, bool):
        if flag:
            flag = 2
        else:
            flag = 0
    return flag


def safelen(array):
    if checkempty(array):
        return 0
    else:
        return len(array)


def hole_edges(qmask):
    # List of edges of holes (holes extend from left_edges:right_edges)
    # Append ones at left and right to catch end holes if present
    # Warning: saving as unsigned 32 bit int
    edges = np.diff(np.r_[1, qmask, 1].astype(np.int32))
    left_edges = np.where(edges == -1)[0].astype(np.uint32)
    right_edges = np.where(edges == 1)[0].astype(np.uint32)
    return np.c_[left_edges, right_edges]


def hole_edges_to_mask(mask_edges, ninds):
    mask = FFTIN(ninds, dtype=bool)
    mask[:] = True
    for l_ind, r_ind in mask_edges:
        mask[l_ind:r_ind] = False
    return mask


def remove_bad_times(bad_time_list, time_list, time_shift_tol, *dat):
    """
    Removes elements of dat that are in the same bucket as in bad_time_list
    :param bad_time_list: List of bad times (s)
    :param time_list: List of times that elements in dat correspond to (s)
    :param time_shift_tol: Tolerance for buckets (s)
    :param dat: Any number of lists that need to be cleaned of bad times
    :return:
        dat, with bad times cleaned. If dat is a single list, returns list
        instead of list of lists
    """
    if checkempty(dat):
        return

    if checkempty(bad_time_list):
        if len(dat) == 1:
            return dat[0]
        else:
            return dat

    bucket_ids_bad = (np.asarray(bad_time_list) / time_shift_tol).astype(int)
    bucket_ids_time = (np.asarray(time_list) / time_shift_tol).astype(int)
    mask_good = np.logical_not(np.in1d(bucket_ids_time, bucket_ids_bad))
    outdat = []
    for d in dat:
        if isinstance(d, np.ndarray):
            outdat.append(d[mask_good])
        elif isinstance(d, list):
            outdat.append([d[x] for x in np.where(mask_good)[0]])
        else:
            raise RuntimeError("I don't know how to pick elements")

    if len(outdat) == 1:
        return outdat[0]
    else:
        return outdat


def find_closest_coarse_calphas(coarse_axes, fine_calphas):
    """
    Finds the closest associated coarse calphas to given fine calphas
    :param coarse_axes:
        List of length n_axes, with the i^th entry being the array of allowed
        coarse calphas in dimension i
    :param fine_calphas:
        Array with set of fine calphas whose associated coarse calphas we want,
        can be 1D for singleton
    :return:
        n_set x n_calpha array with coarse calphas assocated with each
        fine_calpha (always 2D)
    """
    if fine_calphas.size == 0:
        if fine_calphas.ndim == 0:
            return np.zeros((0, len(coarse_axes)))
        elif fine_calphas.ndim == 1:
            return fine_calphas[:, None]
        else:
            return fine_calphas

    fine_calphas = np.atleast_2d(fine_calphas)

    # Build up the coarse calphas dimension by dimension
    # calpha dimension in fine_calphas != dimension in coarse_axes
    ndim = min(len(coarse_axes), fine_calphas.shape[1])
    calphas_arrays = []
    for idim, coarse_axis in enumerate(coarse_axes[:ndim]):
        inds_right = np.searchsorted(coarse_axis, fine_calphas[:, idim])
        inds_right[inds_right > len(coarse_axis) - 1] = len(coarse_axis) - 1
        inds_left = inds_right - 1
        inds_left[inds_left < 0] = 0
        coarse_arr = np.c_[coarse_axis[inds_left], coarse_axis[inds_right]]
        darr = np.abs(coarse_arr - fine_calphas[:, idim][:, None])
        closest_coarse_arr = coarse_arr[
            np.arange(len(coarse_arr)), np.argmin(darr, axis=-1)]
        calphas_arrays.append(closest_coarse_arr)
        
    calphas_arrays = np.array(calphas_arrays).T

    return calphas_arrays


def scalar(x):
    if np.isscalar(x):
        return x
    else:
        return x[0]


def index_after_removal(lst, target, to_remove=None):
    found_remove = False
    for i, x in enumerate(lst):
        if x == target:
            return i - 1 if found_remove else i
        if to_remove is not None and x == to_remove:
            found_remove = True
    return None


# %% Parameter conversion functions
def q_and_eta(q=None, eta=None):
    if q is not None:
        eta = q / (1. + q)**2
    elif eta is not None:
        q = (1. - 2. * eta - np.sqrt(1. - 4. * eta)) / (2. * eta)
    else:
        raise RuntimeError("I need parameters to convert")
    return {'q': q, 'eta': eta}


def mass_conversion(**dic):
    """
    Computes all conversions given a subset of parameters describing the masses
    :param dic:
        Dictionary with known subset of mc, mt, m1, m2, q, eta
        (all can be scalars or numpy arrays)
    :return: dic with all values solved for
    """
    mc = dic.get('mc', dic.get('mchirp'))
    mt = dic.get('mt', dic.get('mtot'))
    m1 = dic.get('m1')
    m2 = dic.get('m2')
    q = dic.get('q')
    eta = dic.get('eta')
    if (q is None) and (dic.get('q1') is not None):
        q = 1 / dic['q1']
        dic['q'] = q

    # Check if we have to do any work
    if not np.any(map(lambda x: x is None, [mc, mt, m1, m2, q, eta])):
        return dic

    # First ensure that q is on the right domain
    if (q is not None) and np.any(q > 1):
        if hasattr(q, "__len__"):
            q[q > 1] = 1/q[q > 1]
        else:
            q = 1/q
        dic['q'] = q
        return mass_conversion(**dic)

    # Second ensure that q and eta are always defined together
    if (q is not None) != (eta is not None):
        dic_q = q_and_eta(q=q, eta=eta)
        dic.update(dic_q)
        return mass_conversion(**dic)

    # If q is not defined, do what is needed to get it
    if q is None:
        if mt is not None:
            if mc is not None:
                dic['eta'] = (mc / mt)**(5. / 3.)
                return mass_conversion(**dic)
            elif m1 is not None:
                dic['q'] = mt / m1 - 1
                return mass_conversion(**dic)
            elif m2 is not None:
                dic['q'] = m2 / (mt - m2)
                return mass_conversion(**dic)
            else:
                raise RuntimeError("Enough parameters weren't defined")
        elif m1 is not None:
            if mc is not None:
                r = (mc / m1)**5
                if np.any(np.logical_or(0 > r, r > 1/2)):
                    raise RuntimeError("I couldn't find a physical solution")
                if hasattr(r, "__len__"):
                    qdic = []
                    for rval in r:
                        qvals = np.roots([1, 0, -rval, -rval])
                        qdic.append(qvals[np.isreal(qvals)][0].real)
                    dic['q'] = np.asarray(qdic)
                else:
                    qvals = np.roots([1, 0, -r, -r])
                    dic['q'] = qvals[np.isreal(qvals)][0].real
                return mass_conversion(**dic)
            elif m2 is not None:
                dic['q'] = m2 / m1
                return mass_conversion(**dic)
            else:
                raise RuntimeError("Enough parameters weren't defined")
        elif m2 is not None:
            if mc is not None:
                r = (m2 / mc)**5
                if np.any(np.logical_or(0 > r, r > 2)):
                    raise RuntimeError("I couldn't find a physical solution")
                if hasattr(r, "__len__"):
                    qdic = []
                    for rval in r:
                        qvals = np.roots([1, 1, 0, -rval])
                        qdic.append(
                            qvals[np.logical_and(
                                np.isreal(qvals), qvals > 0)][0].real)
                    dic['q'] = np.asarray(qdic)
                else:
                    qvals = np.roots([1, 1, 0, -r])
                    dic['q'] = qvals[
                        np.logical_and(np.isreal(qvals), qvals > 0)][0].real
                return mass_conversion(**dic)
            else:
                raise RuntimeError("Enough parameters weren't defined")
        else:
            raise RuntimeError("Enough parameters weren't defined")
    else:
        if m1 is not None:
            dic['m2'] = m1 * q
            dic['mt'] = m1 * (1 + q)
            dic['mc'] = m1 * q**0.6 / (1 + q)**0.2
            return dic
        if mt is not None:
            dic['m1'] = mt / (1 + q)
            return mass_conversion(**dic)
        if mc is not None:
            dic['mt'] = mc / dic['eta']**0.6
            return mass_conversion(**dic)
        raise RuntimeError("Enough parameters weren't defined")


def m1_m2_s1_s2_to_chieff_chia(m1, m2, s1, s2, pe=True):
    chieff = (m1 * s1 + m2 * s2) / (m1 + m2)
    if pe:
        chia = (s1 - s2) / 2
    else:
        chia = (m1 * s1 - m2 * s2) / (m1 + m2)
    return chieff, chia


def m1_m2_mt(m1=None, m2=None, mt=None):
    """complete m1, m2, mt or check consistency if all given --> return m1, m2, mt"""
    if mt is None:
        mt = m1 + m2
    elif m2 is None:
        m2 = mt - m1
    elif m1 is None:
        m1 = mt - m2
    else:
        assert np.allclose(m1 + m2, mt), 'parameter error: mt = m1 + m2 must hold!'
    assert np.any(m1 < m2) == False, 'cannot have m1 < m2'
    return m1, m2, mt


def s1z_s2z_chieff_chia_from_pars(pe=True, **pars):
    """
    *NOTE* this uses chia = (m1*s1z - m2*s2z) / (m1 + m2)
    :param **pars: pars must have at least two of mt, m1, m2,
                  and at least two of s1z (or chi1), s2z (or chi2), chia, chieff
    :return: s1z, s2z, chieff, chia
    """
    s1z, s2z, chieff, chia = [pars.get(k, None) for k in ['s1z', 's2z', 'chieff', 'chia']]
    m1, m2, mt = m1_m2_mt(m1=pars.get('m1', None), m2=pars.get('m2', None), mt=pars.get('mt', pars.get('mtot', None)))
    # NOTE: pathology is possible here if you are using chi1 = [s1x, s1y, s1z] and each s1j has length 3
    if (s1z is None) and hasattr(pars.get('chi1', None), '__len__'):
        s1z = (pars['chi1'][2] if np.shape(pars['chi1'])[0] == 3 else pars['chi1'][:, 2])
    if (s2z is None) and hasattr(pars.get('chi2', None), '__len__'):
        s2z = (pars['chi2'][2] if np.shape(pars['chi2'])[0] == 3 else pars['chi2'][:, 2])

    # insufficient set if missing more than 2 of these
    if np.count_nonzero([(s1z is None), (s2z is None), (chieff is None), (chia is None)]) > 2:
        raise RuntimeError(f'missing necessary spin parameter(s):\n\tpars =\n{pars}')
    else:
        # otherwise find highest priority pair available to define the other two
        if (s1z is not None) and (s2z is not None):
            # if have s1z and s2z, use them
            chieff = (m1*s1z + m2*s2z) / mt
            chia = (m1*s1z - m2*s2z) / mt
        elif chieff is not None:
            # elif have chieff, use that with (priority order) s1z or s2z or chia
            if s1z is not None:
                s2z = mt * (chieff - m1*s1z/mt) / m2
                chia = 2*m1*s1z/mt - chieff
            elif s2z is not None:
                s1z = mt * (chieff - m2*s2z/mt) / m1
                chia = chieff - 2*m2*s2z/mt
            else:
                s1z, s2z = mt * (chieff + chia) / (2 * m1), mt * (chieff - chia) / (2 * m2)
        else:
            # otherwise must have (s1z, chia) or (s2z, chia)
            if s1z is not None:
                s2z, chieff = mt * (m1*s1z/mt - chia) / m2, 2*m1*s1z/mt - chia
            else:
                s1z, chieff = mt * (chia + m2*s2z/mt) / m1, chia + 2*m2*s2z/mt

    return s1z, s2z, chieff, chia


def pars_from_pars(**pars):
    """
    WARNING: this sees keys 'chi1' and 'chi2' as being the vector spins [sjx, sjy, sjz]
     *UNLIKE* in other places where chij = sqrt(sjx^2 + sjy^2 + sjz^2)
    complete intrinsic parameter dictionary from sufficient parts
    """
    # mass ratio alternative
    if pars.get('q1', None) is not None:
        pars['q'] = (pars['q1'] if np.any(pars['q1'] < 1) else 1. / pars['q1'])
    # mass completion
    pars.update(mass_conversion(**pars))
    pars['q1'] = 1. / pars['q']
    if np.any(pars['q1'] < 1):
        raise RuntimeError('pars_from_pars malfunction: mass conversion failed')

    # spin completion --> individual components take precedent
    s1x, s1y, s2x, s2y, chi1, chi2 = [
        pars.get(k, None) for k in ['s1x', 's1y', 's2x', 's2y', 'chi1', 'chi2']]
    # first determine in-plane spins OR set to 0 if not given
    zero = (0 if not hasattr(pars['q'], '__len__') else np.zeros(len(pars['q'])))
    # NOTE: pathology is possible here if you are using chi1 = [s1x, s1y, s1z] and each s1j has length 3
    if s1x is None:
        s1x = (zero if chi1 is None else (chi1[0] if np.shape(chi1)[0] == 3 else chi1[:, 0]))
    if s1y is None:
        s1y = (zero if chi1 is None else (chi1[1] if np.shape(chi1)[0] == 3 else chi1[:, 1]))
    if s2x is None:
        s2x = (zero if chi2 is None else (chi2[0] if np.shape(chi2)[0] == 3 else chi2[:, 0]))
    if s2y is None:
        s2y = (zero if chi2 is None else (chi2[1] if np.shape(chi2)[0] == 3 else chi2[:, 1]))
    # then complete z spins (with error if insufficient info)
    s1z, s2z, chieff, chia = s1z_s2z_chieff_chia_from_pars(**pars)
    chi1, chi2 = [s1x, s1y, s1z], [s2x, s2y, s2z]
    pars.update({'s1x': s1x, 's1y': s1y, 's1z': s1z, 'chi1': [s1x, s1y, s1z],
                 's2x': s2x, 's2y': s2y, 's2z': s2z, 'chi2': [s2x, s2y, s2z],
                 'chieff': chieff, 'chia': chia})

    # tidal deformabilities
    pars['l1'] = pars.get('l1', 0)
    pars['l2'] = pars.get('l2', 0)
    return pars


# %% Signal processing functions
def define_coarser_mask(freqs_in, mask_freqs_in, freqs_out):
    """
    :param freqs_in: Array with fine frequency grid
    :param mask_freqs_in: Boolean mask on fine frequency grid
    :param freqs_out: Array with coarse frequency grid
    :return:
        Mask on freqs_out with product of entries of mask_freqs_in in the
        relevant range
    """
    mask_freqs_out = np.zeros_like(freqs_out, dtype=bool)

    df_freqs_in = freqs_in[1] - freqs_in[0]
    df_freqs_out = freqs_out[1] - freqs_out[0]
    # assert df_freqs_in <= df_freqs_out, \
    #     "input frequency grid must be finer than the output one"

    if df_freqs_in > df_freqs_out:
        warnings.warn("Input frequency grid is coarser than the output one!")

    for i in range(len(mask_freqs_out)):
        start_ind_in = int(df_freqs_out * i / df_freqs_in)
        end_ind_in = int(np.ceil(df_freqs_out * (i + 1) / df_freqs_in))
        mask_freqs_out[i] = np.prod(mask_freqs_in[start_ind_in:end_ind_in])

    return mask_freqs_out


def band_filter(
        dt, fmin=None, fmax=None, wfac=None, btype='bandpass',
        order=params.ORDER, filter_type='butter'):
    """Creates desired filter, and computes its impulse response length
    :param dt: Sampling interval (s)
    :param fmin: Lower critical frequency (-3dB point). Pass None for lowpass
    :param fmax: Higher critical frequency (-3dB point). Pass None for highpass
    :param wfac: Fraction of impulse response to capture. If None, we use
                 defaults from params
    :param btype: Kind of filter, should be either 'bandpass', 'bandstop',
                  'high', or 'low'
    :param order: Order of filter
    :param filter_type: Kind of filter to create
    :return: sos representation of filter coefficients, impulse response length
    """
    trivial = False
    if fmin is None:
        if fmax is None:
            wn = 1
            trivial = True
        else:
            # Overwrite btype
            btype = 'low'
            wn = fmax * 2. * dt
            if wn >= 1:
                trivial = True
            if wfac is None:
                wfac = params.IRL_EPS_HIGH
    else:
        wmin = fmin * 2. * dt
        wmax = 1
        if fmax is not None:
            wmax = fmax * 2. * dt
            if wmax >= 1:
                # No point going above Nyquist
                wmax = 1
        if wmax < 1:
            wn = (wmin, wmax)
            if wfac is None:
                wfac = params.IRL_EPS_BAND
        else:
            # Max is above Nyquist
            wn = wmin
            # Overwrite btype
            if btype == 'bandpass':
                btype = 'high'
            elif btype == 'bandstop':
                btype = 'low'
            if wfac is None:
                wfac = params.IRL_EPS_HIGH
    if trivial:
        # return np.array([1, 0]), np.array([1, 0]), 0
        return np.array([[1, 0, 0, 1, 0, 0]]), 0
    else:
        # b, a = signal.butter(N=order, Wn=wn, btype=btype)
        # b, a = signal.iirfilter(N=order, Wn=wn, btype=btype, ftype=filter_type)
        # # Compute impulse response length
        # z, p, k = signal.tf2zpk(b, a)
        sos = signal.iirfilter(
            N=order, Wn=wn, btype=btype, ftype=filter_type, output='sos')
        # Compute impulse response length
        z, p, k = signal.sos2zpk(sos)
        r = np.max(np.abs(p))
        approx_impulse_len = int(np.ceil(np.log(wfac) / np.log(r)))
        if approx_impulse_len < 0:
            raise RuntimeError("Filter coefficients were unstable!")
        return sos, approx_impulse_len


def notch_filter(dt, fc, df):
    """
    Creates notch filter
    :param dt: Sampling interval (s)
    :param fc: Central frequency of notch (Hz)
    :param df: -3dB bandwidth in frequency, high - low (Hz)
    :return: Numerator and denominator coeffs of notch filter
    """
    b, a = signal.iirnotch(fc * 2. * dt, fc / df)
    return b, a


def notch_filter_sos(
        dt, freqs, mask_freqs, flow=None, fhigh=None, dfmin_notch=None,
        notch_pars_in=None):
    """
    Defines set of sos filters to apply to notch out lines
    :param dt: Time interval between successive elements of data (s)
    :param freqs: Regular array with frequencies for line identification
    :param mask_freqs: Mask on freqs with zeros at lines
    :param flow: Apply notch filters only on frequencies f > flow
    :param fhigh: Apply notch filters only on frequencies f < fhigh
    :param dfmin_notch: Minimum frequency width (Hz) that each notch should have
    :param notch_pars_in:
        If we already have a list of notch parameters, pass it in to append to
        and return
    :return:
        List of 2-tuples with sos coefficients, and impulse response lengths,
        to apply to data
    """
    if not np.any(np.logical_not(mask_freqs)):
        # No lines to notch
        return [(np.array([[1, 0, 0, 1, 0, 0]]), 0)]

    deltaf = freqs[1] - freqs[0]
    mask_freqs_edges = hole_edges(mask_freqs)
    l_line_inds, r_line_inds = \
        mask_freqs_edges[:, 0], mask_freqs_edges[:, 1]
    l_line_freqs = freqs[l_line_inds]
    if len(r_line_inds) > 0:
        # Treat boundary case in which the right edge is outside data, which
        # can happen if mask_freqs[-1] is zero
        r_line_freqs = freqs[r_line_inds - 1] + deltaf
    else:
        r_line_freqs = np.array([])

    # Mask identifying which blocks of frequencies are to be notched
    if flow is None:
        flow = 0
    if fhigh is None:
        fhigh = freqs[-1]
    # Stay below Nyquist
    fhigh = min(fhigh, 1./2/dt)
    notch_mask = np.logical_and(l_line_freqs <= fhigh, r_line_freqs >= flow)

    if notch_pars_in is not None:
        notch_pars = notch_pars_in
    else:
        notch_pars = []

    for fmin_ind, fmax_ind in zip(
            l_line_inds[notch_mask], r_line_inds[notch_mask]):
        fc_notch = freqs[fmin_ind] + (fmax_ind - fmin_ind - 1) * deltaf / 2
        df_notch = (fmax_ind - fmin_ind) * deltaf
        if dfmin_notch is not None:
            df_notch = max(df_notch, dfmin_notch)
        # b, a = utils.notch_filter(dt, fc_notch, df_notch)
        sos, irl = band_filter(
            dt, fmin=fc_notch - df_notch / 2, fmax=fc_notch + df_notch / 2,
            btype='bandstop', filter_type='bessel')
        # print("fc_notch: ", fc_notch)
        # print("df_notch: ", df_notch)
        # print("irl: ", irl)
        # strain_wt = signal.filtfilt(b, a, strain_wt)
        # strain_wt = signal.sosfiltfilt(sos, strain_wt, padlen=irl)
        notch_pars.append((sos, irl))

    return notch_pars


def sine_gaussian(dt, fc, df):
    """
    Function to return time-domain sine-gaussian pulses (ready for FFT)
    :param dt: Sampling interval (s)
    :param fc: Central frequency of transient (Hz)
    :param df: Spread in frequency, high - low (Hz)
    :return: 1. Time-domain cosine pulse
             2. Time-domain sine pulse, both satisfying sum_t pulse^2 = 1
             3. Support of pulse (has 2 * support - 1 nonzero entries)
    """
    half_support = signal.gausspulse('cutoff', fc=fc, bw=df/fc)
    nhalfinds = int(np.ceil(half_support / dt))
    times = FFTIN(2 * nhalfinds + 1)
    out_cos = FFTIN(2 * nhalfinds + 1)
    out_sin = FFTIN(2 * nhalfinds + 1)
    times[:] = np.arange(-nhalfinds, nhalfinds + 1) * dt
    cos_pulse, sin_pulse, *_ = signal.gausspulse(
        times, fc=fc, bw=df/fc, retquad=True)
    out_cos[:] = np.roll(cos_pulse[::-1], -nhalfinds)[:] / \
        np.linalg.norm(cos_pulse)
    out_sin[:] = np.roll(sin_pulse[::-1], -nhalfinds)[:] / \
        np.linalg.norm(sin_pulse)
    return out_cos, out_sin, nhalfinds + 1


def condition_filter(
        filt, support=None, truncate=False, shorten=False, taper=False,
        notch_pars=None, flen=None, wfac=params.WFAC_FILT, in_domain='fd',
        out_domain='fd', def_fftsize=params.DEF_FFTSIZE,
        taper_fraction=0.2, min_trunc_len=2):
    """
    Compute support, weight, and truncate input frequency domain filter
    :param filt:
        If in_domain is 'fd':
            Complex array of length nrfft(len(data)) with RFFT of
            {whitening filter with time domain weight at the edges}
        else:
            Real array of length len(data) with whitening filter with time
            domain weight at the edges
    :param support:
        Support to truncate at (in indices). If None, it is calculated from the
        filter itself
    :param truncate:
        Flag whether to truncate the time-domain support of the filter
    :param shorten:
        Flag whether to shorten the time-domain length of the filter
    :param taper:
        Flag whether to taper the time domain response of the filter with a
        Tukey window. Applied after truncating if truncate==True, else on the
        entire length
    :param notch_pars:
        Optional parameters of notches to apply, output of notch_filter_sos
    :param flen:
        Total length of filter in time domain (zero-padded in frequency domain
        if needed)
    :param wfac: (1 - Weight of the filter to capture)/2
    :param in_domain: Domain of input ('fd' or 'td')
    :param out_domain: Domain of output ('fd' or 'td')
    :param def_fftsize:
        Default fftsize, one of the limits for the time domain length of the
        output
    :param taper_fraction:
        Fraction of response to taper with a Tukey window, if applicable
        (0 is boxcar, 1 is Hann)
    :param min_trunc_len: minimum N_samples//2 after truncation
    :return: 1. If out_domain is 'fd':
                    Complex array of the same size as filt with RFFT of
                    conditioned filter
                else:
                    Real array of the same size as filt with conditioned filter
             2. Support of filter (TD filter has 2 * support - 1 nonzero coeffs)
             3. Total weight of filter (sum w(t)^2)
    """
    if in_domain.lower() == 'fd':
        # Transform to time domain
        filt = IRFFT(filt, n=flen)

    # Condition the time domain filter
    support_in = support
    filt, support, tot_cum_wt = condition_filter_td(
        filt, support=support_in, truncate=truncate, taper=taper, wfac=wfac,
        taper_fraction=taper_fraction, min_trunc_len=min_trunc_len)

    # Notch it and recondition if needed
    if notch_pars is not None:
        # Bring weight into the center
        filt = np.fft.fftshift(filt)

        # Apply notches
        for sos_notch, irl_notch in notch_pars:
            filt = signal.sosfiltfilt(sos_notch, filt, padlen=irl_notch)

        # Bring weight back to the edges, and redo the conditioning
        filt = np.fft.ifftshift(filt)
        filt, support, tot_cum_wt = condition_filter_td(
            filt, support=support_in, truncate=truncate, wfac=wfac)

    # Shorten the filter, if needed
    if shorten:
        fftsize = max(
            min(def_fftsize, next_power(len(filt)) // 4),
            next_power(2 * support))
        filt = change_filter_times_td(filt, len(filt), fftsize)

    if out_domain.lower() == 'fd':
        # Transform to frequency domain
        filt = RFFT(filt)

    return filt, support, tot_cum_wt


def condition_filter_td(
        filt_td, support=None, truncate=False, taper=False,
        wfac=params.WFAC_FILT, taper_fraction=0.2, min_trunc_len=2):
    """
    Compute support, weight, and truncate input time domain filter
    :param filt_td: Array with time domain filter with weight at the edges
    :param support: Support to truncate at (in indices). If None, it is
                    calculated from the filter itself
    :param truncate: Flag indicating whether to truncate time-domain filter
    :param taper:
        Flag whether to taper the time domain response of the filter with a
        Tukey window. Applied ater truncating if truncate==True, else on the
        entire length
    :param wfac: (1 - Weight of the filter to capture)/2
    :param taper_fraction:
        Fraction of response to taper with a Tukey window, if applicable
        (0 is boxcar, 1 is Hann)
    :param min_trunc_len: minimum N_samples//2 after truncation
    :return: 1. Array of the same size as filt_td with conditioned filter
             2. Support of filter (TD filter has 2 * support - 1 nonzero coeffs)
             3. Total weight of filter (sum w(t)^2)
    """
    cum_wt = np.cumsum(np.fft.fftshift(filt_td) ** 2)
    tot_cum_wt = cum_wt[-1]

    # Find support
    if support is None:
        inds_wt = np.searchsorted(
            cum_wt, [wfac * tot_cum_wt, (1. - wfac) * tot_cum_wt])
        support = inds_wt[1] - inds_wt[0]

    if support < min_trunc_len:
        print(f'WARNING: requested support={support}, below',
              f'min_trunc_len={min_trunc_len}',
              '\n--> setting support=min_trunc_len')
        support = min_trunc_len

    # Truncate at large lags
    if truncate is True:
        # Keep a factor of two for safety
        if 2 * support >= len(filt_td):
            print("Warning: Data chunk may be too small!")
            support = support // 2
        filt_td_out = FFTIN(len(filt_td))
        filt_td_out[:] = filt_td[:]
        filt_td_out *= np.r_[np.ones(support),
                             np.zeros(len(filt_td) - 2 * support + 1),
                             np.ones(support - 1)]
        window_len = 2 * support
    else:
        filt_td_out = filt_td
        window_len = len(filt_td)

    if taper:
        # Tukey window to taper the whitening filter with
        # The sym factor ensures that the Tukey has real-valued Fourier
        # coefficients after we apply an ifftshift
        window = np.fft.ifftshift(signal.tukey(
            window_len, alpha=taper_fraction, sym=bool(window_len % 2)))
        window = change_filter_times_td(
            window, window_len, len(filt_td_out), pad_mode='center')
        filt_td_out *= window

    tot_cum_wt = np.sum(filt_td_out ** 2)

    return filt_td_out, support, tot_cum_wt


def change_filter_times_td(filter_td, orig_len, chop_len, pad_mode='center'):
    """Converts a conditioned filter to a different time array by
    zero-padding in time-domain
    :param filter_td:
        n_filter x orig_len array with time-domain filters
        (can be vector for n_filter = 1)
    :param orig_len: Length of original time series
    :param chop_len: Length of different time series
    :param pad_mode: Where to pad, can be 'center', 'left' or 'right'
    :return Time domain filter(s) living on chop_len
    """
    if orig_len == chop_len:
        return filter_td

    out_dims = list(filter_td.shape)
    out_dims[-1] = chop_len
    filter_chop_td = FFTIN(out_dims, dtype=filter_td.dtype)

    n_copy = min(chop_len, orig_len)
    if pad_mode.lower() == 'center':
        n_copy_l = n_copy - n_copy // 2
        n_copy_r = n_copy // 2
    elif pad_mode.lower() == 'left':
        n_copy_l = 0
        n_copy_r = n_copy
    elif pad_mode.lower() == 'right':
        n_copy_l = n_copy
        n_copy_r = 0
    else:
        raise RuntimeError(f"Pad mode {pad_mode} not recognized!")

    filter_chop_td[..., :n_copy_l] = filter_td[..., :n_copy_l]
    if n_copy_r > 0:
        filter_chop_td[..., -n_copy_r:] = filter_td[..., -n_copy_r:]

    return filter_chop_td


def change_filter_times_fd(filter_fd, orig_len, chop_len, pad_mode='center'):
    """Converts a conditioned filter to a different time array by zero-padding
    in time-domain
    :param filter_fd:
        n_filter x len(rfftfreq(orig_len)) array with frequency-domain filters
        (can be vector for n_filter = 1)
    :param orig_len: Length of original time series
    :param chop_len: Length of different time series
    :param pad_mode: Where to pad, can be 'center', 'left' or 'right'
    :return Frequency domain filter(s) living on rfftfreq(chop_len)
    """
    if orig_len == chop_len:
        return filter_fd

    filter_td = IRFFT(filter_fd, n=orig_len, axis=-1)
    filter_chop_td = change_filter_times_td(
        filter_td, orig_len, chop_len, pad_mode=pad_mode)
    filter_chop_fd = RFFT(filter_chop_td, axis=-1)

    return filter_chop_fd


def hilbert_transform(wfs_cos, axis=-1):
    return IRFFT(RFFT(wfs_cos, axis=axis) * 1j, n=wfs_cos.shape[axis])


def match(
        wfs_cos_1, wfs_cos_2, allow_shift=True, allow_phase=True,
        return_cov=False):
    """
    Computes match, or cosine, between waveforms
    :param wfs_cos_1:
        n_wf x len(wf) array with whitened waveforms (can be vector if n_wf = 1)
    :param wfs_cos_2:
        n_wf x len(wf) array with whitened waveforms (can be vector if n_wf = 1)
    :param allow_shift: Flag to allow shifts when computing the match
    :param allow_phase: Flag to allow phases when computing the match
    :param return_cov: Flag to return timeseries of complex match
    :return:
        If return cov, timeseries of complex match between the waveforms, else
        the match or cosine (defined as |complex match|)
    """
    # Ensure that the waveforms are normalized to SNR=1
    wfs_cos_1 = wfs_cos_1 / np.linalg.norm(wfs_cos_1, axis=-1)
    wfs_cos_2 = wfs_cos_2 / np.linalg.norm(wfs_cos_2, axis=-1)

    if allow_phase:
        # Complexify one of the waveforms
        wfs_comp_1 = FFTIN(wfs_cos_1.shape, dtype=np.complex128)
        wfs_comp_1[:] = wfs_cos_1 + 1j * hilbert_transform(wfs_cos_1)
    else:
        wfs_comp_1 = wfs_cos_1

    if allow_shift:
        # Compute the match using FFT
        cov = IFFT(
            FFT(wfs_comp_1, axis=-1) *
            FFT(wfs_cos_2, axis=-1).conj(), axis=-1)
    else:
        cov = np.atleast_1d(np.sum(wfs_comp_1 * wfs_cos_2, axis=-1))

    if return_cov:
        return cov
    else:
        return np.max(abs_sq(cov), axis=-1)**0.5


# %% Post-processing of triggers
def sinc_interp_by_factor_of_2(
        t, x, left_ind=None, right_ind=None,
        support=params.SUPPORT_SINC_FILTER, n_interp=1):
    """
    :param t: Time axis of the original x array.
    :param x: Samples to be interpolated. The last axis of the 
              array will be interpolated.
    :param left_ind: If known, left index of region of interest
    :param right_ind: If known, right index of region of interest
    :param support: Support of the interpolation filter
    :param n_interp: Number of times to sinc-interpolate by factor of 2
    :return: t_interp, x_interp
    # Apologies to grad students
    Length of x_interp is 2*len(x) - 4*(support+1) if n_interp = 1
    """
    if len(t) < 1:
        return t, x

    # Amount of data we lose from each side in the limit of an infinite
    # number of sinc interpolations
    support_edge_data = 2 * (support + 1)

    t_sinc = np.arange(-support, support) + 0.5
    kernel = np.sinc(t_sinc)
    kernel = kernel / np.sum(kernel)

    # If desired, use subset of data
    # Take a bit of extra from the right and left so that the support of the
    # sinc filter is valid, we also want to see if the max is a little bit to
    # the left/right of the region of interest
    left_lim = 0
    right_lim = len(t)
    if left_ind is not None:
        left_lim = max(left_ind - support_edge_data, 0)
    if right_ind is not None:
        right_lim = right_ind + support_edge_data
    t = t[left_lim:right_lim]
    x = x[..., left_lim:right_lim]

    for ind in range(n_interp):
        if len(t) < 1:
            return t, x

        dt = t[1] - t[0]

        x_half = x.reshape(-1, x.shape[-1])
        x_half = np.array([np.convolve(_, kernel)
                           for _ in x_half])
        x_half = x_half.reshape(x.shape[:-1]+(-1,))
        x_half = x_half[..., 2*support:-2*support-1]
        t_half = (np.arange(support, support + x_half.shape[-1]) + 0.5) * dt + t[0]

        t_merge = np.zeros(len(t_half) * 2)
        t_merge[::2] = t[support:-support-2]
        t_merge[1::2] = t_half
        x_merge = np.zeros(x.shape[:-1]+(len(t_merge),), dtype=x.dtype)
        x_merge[..., ::2] = x[..., support:-support-2]
        x_merge[..., 1::2] = x_half

        t = t_merge
        x = x_merge

    return t, x


def sinc_interp_x2D(
        t, x2D, left_ind=None, right_ind=None,
        support=params.SUPPORT_SINC_FILTER, n_interp=1):
    """
    exact same as sinc_interp_by_factor_of_2() except along last axis of
    two dimensional array x2D
    """
    if len(t) < 1:
        return t, x2D

    # Amount of data we lose from each side in the limit of an infinite
    # number of sinc interpolations
    support_edge_data = 2 * (support + 1)

    t_sinc = np.arange(-support, support) + 0.5
    kernel = np.sinc(t_sinc)
    kernel = kernel / np.sum(kernel)

    # If desired, use subset of data
    # Take a bit of extra from the right and left so that the support of the
    # sinc filter is valid, we also want to see if the max is a little bit to
    # the left/right of the region of interest
    left_lim = 0
    right_lim = len(t)
    if left_ind is not None:
        left_lim = max(left_ind - support_edge_data, 0)
    if right_ind is not None:
        right_lim = right_ind + support_edge_data
    t = t[left_lim:right_lim]
    x2D = x2D[:, left_lim:right_lim]

    for ind in range(n_interp):
        if len(t) < 1:
            return t, x2D

        dt = t[1] - t[0]

        x_half = np.array([np.convolve(x, kernel)
                           for x in x2D])[:, 2*support:-2*support-1]
        t_half = (np.arange(support, support + x_half.shape[-1]) + 0.5) * dt + t[0]

        t_merge = np.zeros(len(t_half) * 2)
        t_merge[::2] = t[support:-support-2]
        t_merge[1::2] = t_half
        x_merge = np.zeros((x2D.shape[0], len(t_merge)), dtype=x2D.dtype)
        x_merge[:, ::2] = x2D[:, support:-support-2]
        x_merge[:, 1::2] = x_half

        t = t_merge
        x2D = x_merge

    return t, x2D


def amend_indices(left_indices, right_indices, n_tol=params.SUPPORT_EDGE_DATA):
    """Joins blocks together if they are separated by less than n_tol"""

    if len(left_indices) == 0:
        # Not a single candidate with this waveform
        return left_indices, right_indices

    # Add leftmost left index
    left_inds_output = [left_indices[0]]
    right_inds_output = []
    for i in range(1, len(left_indices)):
        if (left_indices[i]-right_indices[i-1]) > n_tol:
            left_inds_output.append(left_indices[i])
            right_inds_output.append(right_indices[i-1])

    # Add rightmost right index
    right_inds_output.append(right_indices[-1])

    return np.array(left_inds_output), np.array(right_inds_output)


def make_template_ids(calphas, eps_c_alpha=0.01):
    # Caution: Assumes that there is more than one dimension!
    # TODO: Make it robust to this ?
    # Returns a list unless a calphas is a single vector
    if len(calphas) == 0:
        # No calphas passed
        return []

    if hasattr(calphas[0], "__len__"):
        # Multiple sets of calphas passed
        calphas = np.round(np.asarray(calphas) / eps_c_alpha).astype(np.int64)
        return [hash(tuple(x)) for x in calphas]
    else:
        # One set of calphas passed
        return hash(tuple(
            (np.round(np.asarray(calphas) / eps_c_alpha)).astype(np.int64)))


def make_trigger_ids(triggers, c0_pos, eps_time=1/4096, eps_c_alpha=0.01):
    # Caution: Assumes that there is more than one dimension!
    # TODO: Make it robust to this? Currently we already store zero,
    #  so it should be fixed already
    # Returns a list unless triggers is a single vector
    if len(triggers) == 0:
        # No triggers passed
        return np.array([], dtype=np.int64)

    if hasattr(triggers[0], "__len__"):
        # Multiple triggers passed
        triggers = np.asarray(triggers)
        times = np.round(triggers[:, 0] / eps_time).astype(np.int64)
        calphas = np.round(triggers[:, c0_pos:] / eps_c_alpha).astype(np.int64)
        keys = np.c_[times, calphas]
        return np.asarray([hash(tuple(x)) for x in keys], dtype=np.int64)
    else:
        # One trigger passed
        time = np.round(triggers[0] / eps_time).astype(np.int64)
        calphas = np.round(triggers[c0_pos:] / eps_c_alpha).astype(np.int64)
        key = np.r_[time, calphas]
        return hash(tuple(key))


def get_injection_details():
    try:
        with open(INJ_PATH, 'r') as f:
            injdata = f.readlines()
    except FileNotFoundError:
        warnings.warn("Couldn't find injection file!", Warning)
        return [], [], [], []

    lines = [x for x in injdata if x.split()[-1] == 'cbc']
    injection_starts = [float(x.split('-')[0]) for x in lines]
    injection_ends = [float(x.split('-')[1].split()[0]) for x in lines]
    injection_snrs = [float(x.split()[-3]) for x in lines]

    m1 = [float(x.split()[1]) for x in lines]
    m2 = [float(x.split()[2]) for x in lines]
    injection_chirp_masses = [
        (m1[i]*m2[i])**(3/5)/(m1[i]+m2[i])**(1/5) for i in range(len(m1))]

    return injection_starts, injection_ends, injection_snrs, \
        injection_chirp_masses


def is_close_to(t_1, t_2, t_1_start=None, eps=4, return_close=False):
    """
    :param t_1: List of reference times (ends of waveforms)
    :param t_2:
        Time, or list of times that we are checking against the reference times
    :param t_1_start: if available, list of start times of waveforms
    :param eps: Tolerance to mark associated triggers
    :param return_close:
        Flag to return len(t_2) x len(t_1) boolean matrix indicating closeness
    :return:
        Boolean flag / array of flags listing whether time / list of times in
        t_2 is close to any of the t_1
        if return_close, returns adjacency matrix too
    """
    if checkempty(t_1) or checkempty(t_2):
        if hasattr(t_2, '__len__'):
            if return_close:
                if hasattr(t_1, '__len__'):
                    return np.zeros(len(t_2), dtype=bool), \
                        np.zeros((len(t_2), len(t_1)), dtype=bool)
                else:
                    return np.zeros(len(t_2), dtype=bool), \
                        np.zeros(len(t_2), dtype=bool)
            else:
                return np.zeros(len(t_2), dtype=bool)
        else:
            if return_close:
                if hasattr(t_1, '__len__'):
                    return False, np.zeros(len(t_1), dtype=bool)
                else:
                    return False, False
            else:
                return False

    # Convert scalars into vectors
    if not hasattr(t_1, '__len__'):
        t_1 = np.array([t_1])
    else:
        t_1 = np.asarray(t_1)

    if t_1_start is not None:
        if not hasattr(t_1_start, '__len__'):
            t_1_start = np.array([t_1_start])
        else:
            t_1_start = np.asarray(t_1_start)

    # Make t2 a len(t_2) x 1 matrix
    if not hasattr(t_2, '__len__'):
        t_2 = np.array([[t_2]])
    else:
        t_2 = np.asarray(t_2)[:, None]

    # Create len(t_2) x len(t_1) matrix
    if t_1_start is None:
        closeflags = np.abs(t_1 - t_2) <= eps
    else:
        endclose = np.abs(t_1 - t_2) <= eps
        startclose = np.abs(t_1_start - t_2) <= eps
        middle = ((t_1_start - t_2) <= 0) * ((t_1 - t_2) >= 0)
        closeflags = np.logical_or(endclose, startclose)
        closeflags = np.logical_or(closeflags, middle)

    isclose = np.any(closeflags, axis=1)
    if len(t_2) == 1:
        if return_close:
            return isclose[0], closeflags[0]
        else:
            return isclose[0]
    else:
        if return_close:
            return isclose, closeflags
        else:
            return isclose


def incoherent_score(triggers, no_sum=False, **kwargs):
    """
    :param triggers: 
        n_cand x n_detector x row of processedclists
        (can be a 2d array if n_cand = 1)
    :param no_sum: Flag to return individual scores
    :return: Vector of incoherent scores (scalar if n_cand = 1)
    """
    if no_sum:
        return triggers[..., 1]
    else:
        return np.sum(triggers[..., 1], axis=-1)


def coherent_score(x):
    """Combine H1 terms from scores_vetoed_max, returns
    scalar for single element"""
    return np.sum(x, axis=-1)


@vectorize(nopython=True)
def offset_background(dt, time_slide_jump, dt_shift):
    """
    Finds the amount to shift the detectors' data streams by.
    It returns an integer multiple of dt_shift
    :param dt: Time delays w.r.t reference trigger (s) (scalar or array)
    :param time_slide_jump: The least count of time slides (s)
    :param dt_shift: Time resolution (s)
    :return:
        Amount to add to detector to bring the timeseries to zero lag
        wrt the reference detector (opposite convention to coherent_score_mz.py)
    """
    # dts = t_det - t_h1, where each is evaluated at the peak of the respective
    # SNR^2 timeseries
    dt0 = dt % time_slide_jump
    dt1 = dt % time_slide_jump - time_slide_jump
    if abs(dt0) < abs(dt1):
        shift = dt0 - dt
    else:
        shift = dt1 - dt
    return round(shift / dt_shift) * dt_shift


# %% Multiprocessing functions
def multiprocessing(func, input_list, num_processors):
    """
    A shorthand function to apply multiprocessing in a one liner
    :param func: Function to apply. Callable with single objects in input list
    :param input_list: inputs for the function.
    :param num_processors: number of processors to use. If 1, will apply locally in the shell.
    :return: the list [func(inp) for inp in input_list]
    """
    if num_processors == 1:
        return [func(inp) for inp in input_list]

    from multiprocessing import Pool
    p = Pool(processes=num_processors)
    res = list(p.map(func, input_list))
    p.close()
    return res


def track_job(
        job, jobname, n_tasks, n_tasks_prev=0, n_tasks_tot=None,
        update_interval=10):
    """Track a multiprocessing job that was submitted in chunks"""
    if n_tasks_tot is None:
        n_tasks_tot = n_tasks

    old_tasks_remaining = n_tasks + job._chunksize
    while job._number_left > 0:
        tasks_remaining = job._number_left * job._chunksize
        if tasks_remaining < old_tasks_remaining:
            old_tasks_remaining = tasks_remaining
            print(f"Fraction of {jobname} done: " +
                  f"{(n_tasks_prev + n_tasks - tasks_remaining)/n_tasks_tot}",
                  flush=True)
        time.sleep(update_interval)


# %% Plotting routines
def plot_veto_details(
        tobj, candidate, signal_enhancement_factor=1, dt_l=10, dt_r=6, nfft=128,
        noverlap=None, use_HM=False, **kwargs):
    """
    Make four-panel diagnostic plot for vetoes
    :param tobj: Trigger object
    :param candidate: Array with row of processed clist
    :param kwargs: Extra keyword arguments to pass to vetoes
    :param use_HM: use triggers_single_detector_HM.py instead of 
                    triggers_single_detector.py
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    if use_HM:
        import triggers_single_detector_HM as trig
    else:
        import triggers_single_detector as trig

    if noverlap is None:
        noverlap = int(3/4*nfft)
    # Apply all vetoes
    zz = tobj.veto_trigger_all(candidate, lazy=False, verbose=True, **kwargs)
    if not zz[0]:
        print("Failed veto, what is this candidate doing here?")

    # Identify right-index of subtracted waveform
    ind_sub = np.searchsorted(tobj.time_sub, candidate[0])

    # Collect split test statistics
    # Read off parameters used for the split tests
    split_chunks = kwargs.get("split_chunks", params.SPLIT_CHUNKS)
    chi2vars_split = []
    for ind in range(len(split_chunks)):
        residuals = trig.split_test_details[ind][0](trig.scores_split)
        chi2vars_split.append(
            abs_sq(residuals) * 2 / trig.split_test_details[ind][1])

    # Make plots
    fig, axes = plt.subplots(
        ncols=2, nrows=2, constrained_layout=True, figsize=(12, 8))

    # Plot subtraction residuals
    fs = int(np.round(1 / tobj.dt))
    fmax = kwargs.get("fmax", fs/2)
    axes[0, 0].specgram(
        tobj.strain_sub[ind_sub - int(dt_l * fs):
                        ind_sub + int(dt_r * fs)],
        NFFT=nfft, Fs=fs, noverlap=noverlap, vmin=0, vmax=25 / fs,
        scale='linear')
    axes[0, 0].set_ylim(top=fmax)
    axes[0, 0].set_title('Whitened data')
    axes[1, 0].specgram(
        signal_enhancement_factor * trig.strain_sub[ind_sub - int(dt_l * fs):
                                                    ind_sub + int(dt_r * fs)] -
        (signal_enhancement_factor - 1) * tobj.strain_sub[ind_sub - int(dt_l * fs):
                                                          ind_sub + int(dt_r * fs)],
        NFFT=nfft, Fs=fs, noverlap=noverlap, vmin=0, vmax=25 / fs,
        scale='linear')
    axes[1, 0].set_ylim(top=fmax)
    axes[1, 0].set_title('Whitened data - bestfit waveform')

    # Plot chi2 test residuals
    # Read off threshold used for the chi2 test
    pthresh_chi2 = kwargs.get("pthresh_chi2", params.THRESHOLD_CHI2)
    chi2_min_eigval = kwargs.get("chi2_min_eigval", params.CHI2_MIN_EIGVAL)
    if trig.continue_to_chi2:
        # Collect chi2 test statistics
        chi2var_chi2 = trig.scores_to_chi2(trig.scores_split)
        axes[0, 1].hist(
            chi2var_chi2[trig.valid_inds], histtype='step', bins=100, density=True,
            log=True, label='_nolegend_')
        x = np.linspace(0, 30, num=100)
        axes[0, 1].semilogy(x, stats.chi2.pdf(x, trig.ndof), label='_nolegend_')
        axes[0, 1].axvline(
            chi2var_chi2[trig.rel_index_sub], c='k', label='Achieved')
        axes[0, 1].axvline(
            stats.chi2.isf(pthresh_chi2, trig.ndof), c='k', ls='--', label='Threshold')
        threshold_plus_safety = stats.ncx2.isf(pthresh_chi2, trig.ndof, trig.ncpar_chi2)
        axes[0, 1].axvline(
            stats.ncx2.isf(pthresh_chi2, trig.ndof, trig.ncpar_chi2), c='k',
            ls=':', label='Threshold + Safety')
        axes[0, 1].legend(frameon=False)
        axes[0, 1].set_title('Chi-Squared test')
        xlim = axes[0, 1].get_xlim()
        axes[0, 1].set_xlim(right=max([xlim[-1], threshold_plus_safety + 1]))

    # Plot split test residuals
    pthresh_split = kwargs.get("pthresh_split", params.THRESHOLD_SPLIT)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    x = np.linspace(0, 15, num=100)
    axes[1, 1].semilogy(x, stats.chi2.pdf(x, 2), c='k', label='_nolegend_')
    axes[1, 1].axvline(stats.chi2.isf(pthresh_split, 2), c='k', ls='--',
                       label='_nolegend_')
    for ind in range(len(split_chunks)):
        axes[1, 1].hist(
            chi2vars_split[ind][trig.valid_inds], histtype='step', bins=100,
            density=True, log=True, color=colors[ind],
            label=','.join([str(x) for x in split_chunks[ind]]))
        axes[1, 1].axvline(
            chi2vars_split[ind][trig.rel_index_sub], c=colors[ind],
            label='_nolegend_')
        axes[1, 1].axvline(
            stats.ncx2.isf(pthresh_split, 2, trig.split_test_results[ind][1]),
            c=colors[ind], ls=':', label='_nolegend_')
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title('Split tests')

    return fig


def gen_step_fd(f_grid, f_power=1):
    twenty_hz_ind = np.searchsorted(f_grid, 20)
    glitch_wf = 1j/f_grid**f_power
    glitch_wf[:twenty_hz_ind] = 0
    return glitch_wf


# TODO: Remove arbitrary numbers from dechirp function and make them controllable from outside
def get_spectrum(trigger, det_ind=1, N_freqs=64, bns=True, use_HM=False):
    """
        Function used to dechirp
        Gets a trigger in the format of trigger, bank_id

        Returns:
            fs_spec,
            spectrum,              - power spectrum of the candidate
            power_spectrum_noise,  - average power spectrum of noise around the candidate
            asd,                   - The ASD at the time,
                                     important if you want to compare to expectation
            dechirped_data         - Data around the event, dechirped such that the
                                     candidate should contain all the power
                                     in a single spectrum
    """
    if use_HM:
        import triggers_single_detector_HM as trig
    else:
        import triggers_single_detector as trig
    gps_time = trigger[0][0]

    fname = get_detector_fnames(
        gps_time, trigger[-1][0], trigger[-1][1],
        source=['BBH', 'BNS'][bns])[det_ind]
    if fname is None:
        return None
    print(fname)
    T = trig.TriggerList.from_json(fname)
    trig_calpha, relevant_index, relevant_index_sub = \
        T.prepare_subset_for_optimization(
            trigger=trigger[0], location=None, dt=32,
            subset_defined=False)

    # Define the global GPS time, safe to the edge case when the score is
    # achieved at the end
    score_time = T.time[0] + relevant_index * T.dt

    # Frequency domain conjugate to time_sub
    fs_sub = np.fft.rfftfreq(len(T.time_sub), d=T.dt)

    # Compute the overlaps, hole corrections, and valid inds for the
    # given calphas without sinc interpolation. Avoid zeroing invalid
    # scores because it is nasty in the Fourier domain

    wf_no_amp_fd = np.exp(1j * T.templatebank.gen_phases_from_calpha(trig_calpha, fs_sub))

    overlaps, hole_corrections, valid_inds = T.gen_scores(
        wf_no_amp_fd, subset=True, zero_invalid=False)

    # The complex overlaps have only positive frequencies
    overlaps_rfft = FFTIN(len(fs_sub), dtype=np.complex128)
    overlaps_rfft[:] = RFFT(overlaps.real) + \
        1j * RFFT(overlaps.imag)

    # Shift the zero-lag overlap to the zeroth index
    overlaps_rfft[:] = overlaps_rfft * np.exp(
        2. * np.pi * 1j * fs_sub * relevant_index_sub * T.dt)

    if use_HM:
        amp_over_asd = np.interp(fs_sub, T.templatebank.amp[:, 0], T.templatebank.amp[:, 1]) / T.asdfunc(fs_sub)
    else:
        amp_over_asd = np.interp(fs_sub, T.templatebank.fs_basis, T.templatebank.amp) / T.asdfunc(fs_sub)
    amp_over_asd *= (np.sqrt(len(fs_sub) / np.sum(amp_over_asd**2)))

    psd_drift_corr = trigger[0][T.psd_drift_pos]

    _, frac_time = T.get_time_index(trigger[0])
    check_overlaps_fft = np.concatenate(
        [amp_over_asd*overlaps_rfft * np.exp(
            2*np.pi*1j*fs_sub * (
                    frac_time + T.templatebank.shift_whitened_wf*T.dt)),
         np.zeros(len(fs_sub)-2)])
    check_overlaps = np.fft.ifft(check_overlaps_fft)/psd_drift_corr

    check_overlaps_truncated = np.roll(np.roll(check_overlaps, N_freqs//2)[:N_freqs], -N_freqs//2)
    phase_to_apply = np.conj(check_overlaps_truncated[0]/np.abs(check_overlaps_truncated[0]))
    spectrum_matched_filtered = np.fft.fft(check_overlaps_truncated)
    fs_spec = np.fft.fftfreq(N_freqs, 1 / 1024)[:N_freqs // 2]

    dechirp_fft = np.concatenate(
        [overlaps_rfft * np.exp(
            2*np.pi*1j*fs_sub * (
                    frac_time + T.templatebank.shift_whitened_wf*T.dt)),
         np.zeros(len(fs_sub) - 2)])
    dechirp = np.fft.ifft(dechirp_fft)/psd_drift_corr
    print(len(dechirp))
    dechirp_truncated = np.roll(np.roll(dechirp, N_freqs//2)[:N_freqs], -N_freqs//2)
    spectrum = np.fft.fft(dechirp_truncated*phase_to_apply)[:N_freqs//2 + 1]/(2*N_freqs)**0.5

    power_spectrum_noise = np.mean(
        [np.abs(np.fft.fft(dechirp[k:k+N_freqs])[:N_freqs//2 + 1]/(2*N_freqs)**0.5)**2
         for k in range(N_freqs//2, N_freqs//2 + 256*N_freqs, N_freqs)], axis=0)

    asd = T.asdfunc(fs_spec)

    return fs_spec, spectrum, power_spectrum_noise, asd, np.roll(dechirp, len(dechirp)//2)


def colorbar(mappable, **kwargs):
    # Creates colorbar of the right size
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return fig.colorbar(mappable, cax=cax, **kwargs)


# %% Spherical harmonics
# TESTED AGAINST LAL's ylm
def A_lm_inclin(l, m, iota):
    if l == 2:
        if m == 2:
            # (2, 2)
            return 0.5*np.sqrt(5/np.pi)*(np.cos(iota*0.5)**4)
        elif m == -2:
            # (2, -2)
            return 0.5*np.sqrt(5/np.pi)*(np.sin(iota*0.5)**4)
        elif m == 1:
            # (2, 1)
            return np.sqrt(5/np.pi)*(np.cos(iota*0.5)**3)*np.sin(iota*0.5)
        elif m == -1:
            # (2, -1)
            return 0.5*np.sqrt(5/np.pi)*(np.sin(iota*0.5)**2)*np.sin(iota)
        elif m == 0:
            # (2, 0)
            return np.sqrt(15/(32*np.pi))*(np.sin(iota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 3:
        if m == 3:
            # (3, 3)
            return -np.sqrt(21/(2*np.pi))*(np.cos(iota*0.5)**5)*np.sin(iota*0.5)
        elif m == -3:
            # (3, -3)
            return np.sqrt(21/(8*np.pi))*(np.sin(iota*0.5)**4)*np.sin(iota)
        elif m == 2:
            # (3, 2)
            return np.sqrt(7/np.pi)*(np.cos(iota*0.5)**4)*(-1 + 1.5*np.cos(iota))
        elif m == -2:
            # (3, -2)
            return np.sqrt(7/np.pi)*(1 + 1.5*np.cos(iota))*(np.sin(iota*0.5)**4)
        elif m == 1:
            # (3, 1)
            return np.sqrt(35/(8*np.pi))*(np.cos(iota*0.5)**3)*(-1 + 3*np.cos(iota))*np.sin(iota*0.5)
        elif m == -1:
            # (3, -1)
            return np.sqrt(35/(8*np.pi))*np.cos(iota*0.5)*(1 + 3*np.cos(iota))*(np.sin(iota*0.5)**3)
        elif m == 0:
            # (3, 0)
            return np.sqrt(105/(32*np.pi))*np.cos(iota)*(np.sin(iota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 4:
        if m == 4:
            # (4, 4)
            return 3*np.sqrt(7/np.pi)*(np.cos(iota*0.5)**6)*(np.sin(iota*0.5)**2)
        elif m == -4:
            # (4, -4)
            return 0.75*np.sqrt(7/np.pi)*(np.sin(iota*0.5)**4)*(np.sin(iota)**2)
        elif m == 3:
            # (4, 3)
            return -3*np.sqrt(7/(2.*np.pi))*(np.cos(iota*0.5)**5)*(-1 + 2*np.cos(iota))*np.sin(iota*0.5)
        elif m == -3:
            # (4, -3)
            return 3*np.sqrt(7/(2.*np.pi))*np.cos(iota*0.5)*(1 + 2*np.cos(iota))*(np.sin(iota*0.5)**5)
        elif m == 2:
            # (4, 2)
            return 0.75*(np.cos(iota*0.5)**4)*(9 - 14*np.cos(iota) + 7*np.cos(2*iota))/np.sqrt(np.pi)
        elif m == -2:
            # (4, -2)
            return 0.75*(9 + 14*np.cos(iota) + 7*np.cos(2*iota))*(np.sin(iota*0.5)**4)/np.sqrt(np.pi)
        elif m == 1:
            # (4, 1)
            return 3*(np.cos(iota*0.5)**3)*(6 - 7*np.cos(iota) + 7*np.cos(2*iota))*np.sin(iota*0.5)/np.sqrt(8*np.pi)
        elif m == -1:
            # (4, -1)
            return 3*np.cos(iota*0.5)*(6 + 7*np.cos(iota) + 7*np.cos(2*iota))*(np.sin(iota*0.5)**3)/np.sqrt(8*np.pi)
        elif m == 0:
            # (4, 0)
            return np.sqrt(45/(512*np.pi))*(5 + 7*np.cos(2*iota))*(np.sin(iota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 5:
        if m == 5:
            # (5, 5)
            return -np.sqrt(330/np.pi)*(np.cos(iota*0.5)**7)*(np.sin(iota*0.5)**3)
        elif m == -5:
            # (5, -5)
            return np.sqrt(330/np.pi)*(np.cos(iota*0.5)**3)*(np.sin(iota*0.5)**7)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    else:
        raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')


# TESTED AGAINST LAL's ylm
def Y_lm(l, m, iota, azim):
    """NOTE: azim = pi/2 - vphi"""
    return np.exp(1j * m * azim) * A_lm_inclin(l, m, iota)


# TESTED AGAINST LAL's ylm
def A_lm_halfinclin(l, m, halfiota):
    if l == 2:
        if m == 2:
            # (2, 2)
            return 0.5*np.sqrt(5/np.pi)*(np.cos(halfiota)**4)
        elif m == -2:
            # (2, -2)
            return 0.5*np.sqrt(5/np.pi)*(np.sin(halfiota)**4)
        elif m == 1:
            # (2, 1)
            return np.sqrt(5/np.pi)*(np.cos(halfiota)**3)*np.sin(halfiota)
        elif m == -1:
            # (2, -1)
            return np.sqrt(5/np.pi)*(np.sin(halfiota)**3)*np.cos(halfiota)
        elif m == 0:
            # (2, 0)
            return np.sqrt(15/(32*np.pi))*(np.sin(2*halfiota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 3:
        if m == 3:
            # (3, 3)
            return -np.sqrt(10.5/np.pi)*(np.cos(halfiota)**5)*np.sin(halfiota)
        elif m == -3:
            # (3, -3)
            return np.sqrt(10.5/np.pi)*(np.sin(halfiota)**5)*np.cos(halfiota)
        elif m == 2:
            # (3, 2)
            return np.sqrt(7/np.pi)*(np.cos(halfiota)**4)*(-1 + 1.5*np.cos(2*halfiota))
        elif m == -2:
            # (3, -2)
            return np.sqrt(7/np.pi)*(np.sin(halfiota)**4)*(1 + 1.5*np.cos(2*halfiota))
        elif m == 1:
            # (3, 1)
            return np.sqrt(35/(8*np.pi))*(np.cos(halfiota)**3)*np.sin(halfiota)*(-1 + 3*np.cos(2*halfiota))
        elif m == -1:
            # (3, -1)
            return np.sqrt(35/(8*np.pi))*np.cos(halfiota)*(np.sin(halfiota)**3)*(1 + 3*np.cos(2*halfiota))
        elif m == 0:
            # (3, 0)
            return np.sqrt(105/(32*np.pi))*np.cos(2*halfiota)*(np.sin(2*halfiota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 4:
        if m == 4:
            # (4, 4)
            return 3*np.sqrt(7/np.pi)*(np.cos(halfiota)**6)*(np.sin(halfiota)**2)
        elif m == -4:
            # (4, -4)
            return 3*np.sqrt(7/np.pi)*(np.sin(halfiota)**6)*(np.cos(halfiota)**2)
        elif m == 3:
            # (4, 3)
            return 3*np.sqrt(7/(2.*np.pi))*(np.cos(halfiota)**5)*np.sin(halfiota)*(1 - 2*np.cos(2*halfiota))
        elif m == -3:
            # (4, -3)
            return 3*np.sqrt(7/(2.*np.pi))*np.cos(halfiota)*(np.sin(halfiota)**5)*(1 + 2*np.cos(2*halfiota))
        elif m == 2:
            # (4, 2)
            return 0.75*(np.cos(halfiota)**4)*(9 - 14*np.cos(2*halfiota) + 7*np.cos(4*halfiota))/np.sqrt(np.pi)
        elif m == -2:
            # (4, -2)
            return 0.75*(np.sin(halfiota)**4)*(9 + 14*np.cos(2*halfiota) + 7*np.cos(4*halfiota))/np.sqrt(np.pi)
        elif m == 1:
            # (4, 1)
            return 3*(np.cos(halfiota)**3)*np.sin(halfiota)*(
                    6 - 7*np.cos(2*halfiota) + 7*np.cos(4*halfiota))/np.sqrt(8 * np.pi)
        elif m == -1:
            # (4, -1)
            return 3*np.cos(halfiota)*(np.sin(halfiota)**3)*(
                    6 + 7*np.cos(2*halfiota) + 7*np.cos(4*halfiota))/np.sqrt(8 * np.pi)
        elif m == 0:
            # (4, 0)
            return np.sqrt(45/(512*np.pi))*(5 + 7*np.cos(4*halfiota))*(np.sin(2*halfiota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 5:
        if m == 5:
            # (5, 5)
            return -np.sqrt(330/np.pi)*(np.cos(halfiota)**7)*(np.sin(halfiota)**3)
        elif m == -5:
            # (5, -5)
            return np.sqrt(330/np.pi)*(np.cos(halfiota)**3)*(np.sin(halfiota)**7)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    else:
        raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')


# TESTED AGAINST LAL's ylm
def Y_lm_halfinclin(l, m, halfiota, azim):
    """NOTE: azim = pi/2 - vphi"""
    return np.exp(1j * m * azim) * A_lm_halfinclin(l, m, halfiota)


# %% LIGO-Virgo Collaboration (LVC) run summary files
# Functions to know if there is a LIGO-made hole at a given time
LIGO_mask_times, LIGO_mask = np.load(
    os.path.join(SMALL_DATA_DIR, 'LIGO_holes.npy'))
LIGO_mask = np.append(LIGO_mask, False)


def is_LIGO_valid(t):
    return LIGO_mask[np.searchsorted(LIGO_mask_times, t)]


@np.vectorize
def is_LIGO_valid_between(t1, t2):
    """
    Take two gps times t1 and t2 and return:
        0 if the hole chunk from t1 to t2 is invalid
        1 if it is partially valid
        2 if it is fully valid
    where valid means that both H and L have all the
    ['DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3']
    flags True.
    (e.g. for assessing whether a waveform between t1 and t2
    had the right to be found or not)
    """
    i1 = np.searchsorted(LIGO_mask_times, t1)
    i2 = np.searchsorted(LIGO_mask_times, t2)
    if i1 == i2:
        return 2 * LIGO_mask[i1]
    else:
        return 1


# Functions used to make the file '/data/bzackay/GW/LIGO_holes.npy' loaded above:
def get_HL_filenames(t0):
    try:
        run = get_run(t0)
        if run == 'O1':
            fns = [f'/data/bzackay/GW/H1/H-H1_LOSC_4_V1-{t0}-4096.hdf5',
                   f'/data/bzackay/GW/L1/L-L1_LOSC_4_V1-{t0}-4096.hdf5']
        elif run == 'O2':
            fns = [f'/data/bzackay/GW/O2/H1/H-H1_GWOSC_O2_4KHZ_R1-{t0}-4096.hdf5',
                   f'/data/bzackay/GW/O2/L1/L-L1_GWOSC_O2_4KHZ_R1-{t0}-4096.hdf5']
        else:
            # TODO: Handle other runs
            fns = ['']
    except ValueError:
        fns = ['']
    return fns


def make_LIGO_mask(
        mask_keys=('DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3'), tmin=TMIN_O1,
        tmax=TMAX_O2, save=False):
    """
    Run this only once to make the file /data/bzackay/GW/LIGO_holes.npy
    """
    import readligo
    t_changes = []
    values = []
    for t0 in np.arange(int(tmin // 4096 * 4096), tmax, 4096).astype(int):
        fns = get_HL_filenames(t0)
        if not all([os.path.isfile(fn) for fn in fns]):
            t_changes.append(t0 + 4096)
            values.append(False)
            continue
        data = [readligo.loaddata(fn, readstrain=False) for fn in fns]
        masks = []
        for d in data:
            assert d[1][0] == t0, t0
            for k in mask_keys:
                m = np.zeros(4096, dtype=bool)
                # LVC files have a bug: if the masks start
                # True and then are False, they are not
                # 4096 s long and they are all True
                m[:len(d[2][k])] = d[2][k]
                masks.append(m)
        mask = np.logical_and.reduce(masks)

        i_chunk_end = np.where(np.diff(mask) != 0)[0]
        vals = mask[i_chunk_end]

        t_changes += list(t0 + np.append(i_chunk_end+1, 4096))
        values += list(vals) + [mask[-1]]

    t_changes = np.array(t_changes)
    values = np.array(values)

    # Remove redundant
    keep = np.append(np.diff(values) != 0, True)
    values = values[keep]
    t_changes = t_changes[keep]
    if save:
        np.save('/data/bzackay/GW/LIGO_holes.npy',
                np.stack([t_changes, values]))
    else:
        return t_changes, values


LVC_RUN_PIPELINE_EVENT_TIME_FILES = \
    {'O1': {'gwtc1': os.path.join(SMALL_DATA_DIR, 'O1_gwtc1_event_times.npy')},
     'O2': {'gwtc1': os.path.join(SMALL_DATA_DIR, 'O2_gwtc1_event_times.npy')},
     'O3a': {'pycbc': os.path.join(SMALL_DATA_DIR, 'O3a_pycbc_L1_event_times.npy'),
             'gstlal': os.path.join(SMALL_DATA_DIR, 'O3a_gstlal_L1_event_times.npy'),
             'cwb': None,
             'gwtc2': os.path.join(SMALL_DATA_DIR, 'O3a_gwtc2_event_times.npy')},
     'O3b': {'gwtc3': os.path.join(SMALL_DATA_DIR, 'O3b_gwtc3_event_times.npy')},
     'O1-O3': {'gwosc': os.path.join(SMALL_DATA_DIR, 'O1-O3_event_times.npy')}}


def get_lsc_event_times(runs='all', pipelines='all'):
    if isinstance(runs, str):
        if runs.lower() == 'all':
            runs = ['O1', 'O2', 'O3a', 'O3b']
        else:
            runs = [runs]
    if isinstance(pipelines, str):
        if pipelines.lower() == 'all':
            pipelines = ['pycbc', 'gstlal', 'cwb', 'gwtc1', 'gwtc2', 'gwtc3', 'gwosc']
        else:
            pipelines = [pipelines]
    times = []
    for r in runs:
        for p in pipelines:
            if LVC_RUN_PIPELINE_EVENT_TIME_FILES[r].get(p.lower(), None) is not None:
                fname = LVC_RUN_PIPELINE_EVENT_TIME_FILES[r][p.lower()]
                if isinstance(fname, str) and os.path.exists(fname):
                    if fname[-3:] == 'npy':
                        times += list(np.load(fname))
                    elif fname[-3:] == 'pkl':
                        times += list(dill.load(fname))
                    else:
                        print(f'Unrecognized file type for {fname}')
                else:
                    print(f'No file for run = {r}, pipeline = {p}')
    return times


# Function to read LSC PE samples from O3
def get_O3_lsc_pe_samples(evname, root=LSC_PE_DIR, comoving=True):
    """
    Returns a PESummary object to interact with the LSC PE samples
    :param evname: Name of the event
    :param root: Path to directory with the samples of all O3 events
    :param comoving: Flag whether to load samples labeled by 'comoving'
    :return:
        1. PEsummary object with posterior samples
        2. Numpy recarray with prior samples
        The list of approximants is in object.samples_dict.keys(), and for each
        approximant, the samples are in object.samples_dict['approximant']
        The PSD is in object.psd['approximant']
        See https://dcc.ligo.org/public/0169/P2000223/005/PEDataReleaseExample.html
    """
    pe_io = load_pesummary_io()

    # Find events even if we give only the first part of their name
    fnames = glob.glob(os.path.join(root, evname + '*'))
    if len(fnames) == 0:
        print(f"Event {evname} not found!")
        return

    fnames_prior = [f for f in fnames if 'prior' in f]
    if len(fnames_prior) > 1:
        print(f"{len(fnames_prior)} events found matching {evname}!")
        print(f"Picking the one with the lowest length")
        fnames_prior = sorted(fnames_prior, key=lambda x: len(x))

    fname_prior = fnames_prior[0]
    if (evname[0] == 'S') or (not comoving):
        fname_posterior = rm_suffix(fname_prior, '_prior.npy', '.h5')
    else:
        fname_posterior = rm_suffix(fname_prior, '_prior.npy', '_comoving.h5')

    posterior = pe_io.read(fname_posterior)
    prior = np.load(fname_prior)

    return posterior, prior


def load_pesummary_io():
    """I need to do some monkey patching to make it work on my machine"""
    sp = load_module('seaborn.palettes')
    sd = load_module('seaborn.distributions')
    smnp = load_module('statsmodels.nonparametric.api')
    s_core = load_module('seaborn._core')
    s_utils = load_module('seaborn.utils')
    plt = load_module('matplotlib.pyplot')

    from six import string_types

    def _bivariate_kdeplot(x, y, filled, fill_lowest,
                           kernel, bw, gridsize, cut, clip,
                           axlabel, cbar, cbar_ax, cbar_kws, ax, **kwargs):
        """Plot a joint KDE estimate as a bivariate contour plot."""
        # Determine the clipping
        if clip is None:
            clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        elif np.ndim(clip) == 1:
            clip = [clip, clip]

        # Calculate the KDE
        if sd._has_statsmodels:
            xx, yy, z = sd._statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)
        else:
            xx, yy, z = sd._scipy_bivariate_kde(x, y, bw, gridsize, cut, clip)

        # Plot the contours
        n_levels = kwargs.pop("n_levels", 10)
        cmap = kwargs.get("cmap", "BuGn" if filled else "BuGn_d")
        if isinstance(cmap, string_types):
            if cmap.endswith("_d"):
                pal = ["#333333"]
                pal.extend(sp.color_palette(cmap.replace("_d", "_r"), 2))
                cmap = sp.blend_palette(pal, as_cmap=True)
            else:
                cmap = plt.cm.get_cmap(cmap)

        kwargs["cmap"] = cmap
        contour_func = ax.contourf if filled else ax.contour
        cset = contour_func(xx, yy, z, n_levels, **kwargs)
        if filled and not fill_lowest:
            cset.collections[0].set_alpha(0)
        kwargs["n_levels"] = n_levels

        if cbar:
            cbar_kws = {} if cbar_kws is None else cbar_kws
            ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

        # Label the axes
        if hasattr(x, "name") and axlabel:
            ax.set_xlabel(x.name)
        if hasattr(y, "name") and axlabel:
            ax.set_ylabel(y.name)

        return ax, cset

    def _statsmodels_univariate_kde(data, kernel, bw, gridsize, cut, clip,
                                    cumulative=False):
        """Compute a univariate kernel density estimate using statsmodels."""
        fft = kernel == "gau"
        kde = smnp.KDEUnivariate(data)
        kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
        if cumulative:
            grid, y = kde.support, kde.cdf
        else:
            grid, y = kde.support, kde.density
        return grid, y

    # monkey patching to make the update work
    sd._bivariate_kdeplot = _bivariate_kdeplot
    sd._statsmodels_univariate_kde = _statsmodels_univariate_kde
    s_utils.categorical_order = s_core.categorical_order

    pe_io = load_module('pesummary.io')
    return pe_io


# back when evnames were 8 chars
SPECIAL_EVNAMES = {'GW150914': {'tgps': 1126259462.4},
                   'GW151012': {'tgps': 1128678900.4},
                   'GW151216': {'tgps': 1134293073.165},
                   'GW151226': {'tgps': 1135136350.6},
                   'GW170104': {'tgps': 1167559936.6},
                   'GW170121': {'tgps': 1169069154.565},
                   'GW170202': {'tgps': 1170079035.715},
                   'GW170304': {'tgps': 1172680691.356},
                   'GW170403': {'tgps': 1175295989.221},
                   'GW170425': {'tgps': 1177134832.178},
                   'GW170608': {'tgps': 1180922494.5},
                   'GW170727': {'tgps': 1185152688.019},
                   'GW170729': {'tgps': 1185389807.3},
                   'GW170809': {'tgps': 1186302519.8},
                   'GW170814': {'tgps': 1186741861.5},
                   'GW170817': {'tgps': 1187008882.4},
                   'GW170818': {'tgps': 1187058327.1},
                   'GW170823': {'tgps': 1187529256.5},
                   # 'GW190412': {'tgps': 1239082262.2},
                   'GW190521': {'tgps': 1242442967.4},
                   'GW190814': {'tgps': 1249852257.0}}


# converting between tgps and evname
def get_tgps_from_evname(evn, prefix='GW'):
    if evn in SPECIAL_EVNAMES:
        return SPECIAL_EVNAMES[evn]['tgps']
    npre = len(prefix)
    assert (len(evn) == npre+13) and (evn[:npre]+evn[npre+6:npre+7] == f'{prefix}_'), \
        f'evn must have form {prefix}yymmdd_hhmmss if not in SPECIAL_EVNAMES'
    return astrotime(datetime.datetime(2000+int(evn[npre:npre+2]), int(evn[npre+2:npre+4]),
                                       int(evn[npre+4:npre+6]), int(evn[npre+7:npre+9]),
                                       int(evn[npre+9:npre+11]), int(evn[npre+11:]))).gps


def get_evname_from_tgps(tgps_sec, prefix='GW', force_full=False):
    s = str(astrotime(astrotime(tgps_sec, format='gps'),
                      format='iso', scale='utc'))
    evn = f'{prefix}{s[2:4]}{s[5:7]}{s[8:10]}_{s[11:13]}{s[14:16]}{s[17:19]}'
    if ((not force_full) and (evn[:8] in SPECIAL_EVNAMES) and
        (abs(SPECIAL_EVNAMES[evn[:8]]['tgps'] -
             get_tgps_from_evname(evn, prefix=prefix)) < 1)):
        return evn[:8]
    return evn


# def get_lsc_event_times():
#    pycbc_O3a = dill.load(open('/home/hschia/PE/gw_detection_ias/data/O3a_pycbc_event_times.pkl', 'rb'))
#    gstlal_O3a = dill.load(open('/home/hschia/PE/gw_detection_ias/data/O3a_gstlal_event_times.pkl', 'rb'))
#
#    return [1128678900, 1126259462, 1135136350,
#            1167559936.6, 1180922494.5, 1185389807.3, 1186302519.8,
#            1186741861.5, 1187008882.4, 1187058327.1, 1187529256.5] + pycbc_O3a + gstlal_O3a

pass
