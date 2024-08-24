import numpy as np
import sys
import os
import importlib
from numba import vectorize
#from numba import jit
from scipy import signal as spsig


########################################################################
#### MODULE I/O
def load_module(mname):
    """load python module by name (string)"""
    if mname not in sys.modules:
        mkey = importlib.import_module(mname)
    else:
        mkey = sys.modules[mname]
    return mkey


def import_matplotlib(figsize=None, **subplots_kwargs):
    """
    Avoid annoying errors due to loading matplotlib on the server
    :return: Figure and axis to plot into
    """
    plt = load_module('matplotlib.pyplot')
    fig, ax = plt.subplots(figsize=figsize, **subplots_kwargs)
    return fig, ax


########################################################################
#### BASIC FILE I/O
def archive_paths(paths):
    if isinstance(paths, str) or (not hasattr(paths, '__len__')):
        paths = [paths]
    for p in [str(p) for p in paths]:
        new_p = p + '.old'
        if os.path.exists(new_p):
            archive_paths(new_p)
        os.system(f'mv {p} {new_p}')

########################################################################
#### DICTIONARY USAGE

def invert_dict(dict_in, iter_val=False):
    """return dictionary with inverted key-value pairs"""
    if iter_val is True:
        dict_out = {}
        for key, val in dict_in.items():
            dict_out.update({v: key for v in val})
        return dict_out
    else:
        return {val: key for key, val in dict_in.items()}

def merge_dicts_safely(dics):
    """
    merge multiple dictionaries into one, accepting repeated keys if
    values are consistent, otherwise raise ValueError
    """
    merged = {}
    for dic in dics:
        for key in merged.keys() & dic.keys():
            if merged[key] != dic[key]:
                raise ValueError(f'Found incompatible values for {key}')
        merged |= dic
    return merged

def check_equal(a, b):
    """recursive x == y that handles ragged mixed-type containers"""
    if isinstance(a, dict):
        return [check_equal(a.get(k, None), v) for k, v in b.items()]
    if hasattr(a, '__len__') and (not isinstance(a, str)):
        return [check_equal(aa, bb) for aa, bb in zip(a, b)]
    return a == b

########################################################################
#### ARITHMETIC

@vectorize(nopython=True)
def abs2(x):
    """x.real^2 + x.imag^2"""
    return (x.real ** 2) + (x.imag ** 2)

#@jit(nopython=True)
def abs2sum(x):
    """sum x.real^2 + x.imag^2 along last axis"""
    return np.sum((x.real ** 2) + (x.imag ** 2), axis=-1)

@vectorize(nopython=True)
def abbar(c1, c2):
    """return c1 * np.conjugate(c2) with numba vectorization"""
    return c1 * np.conjugate(c2)

def next_power(n):
    """get next power of 2 following n"""
    return 2 ** int(np.ceil(np.log2(n)))

########################################################################
#### numpy object handling

def argmaxnd(arr):
    """get unraveled (tuple) index of maximal element in arr"""
    return np.unravel_index(np.argmax(arr), arr.shape)

def argmax_lastax(arr):
    """get axis=-1 (innermost `column') index of max in arr"""
    return np.unravel_index(np.argmax(arr), arr.shape)[-1]

def zeropad_end(arr, pad_to_N):
    """
    pad arr axis=-1 with zeros at (right/back) end to length pad_to_N
      if arr.shape[-1] < pad_to_N, raise ValueError
    """
    if pad_to_N == arr.shape[-1]:
        return arr
    if pad_to_N > arr.shape[-1]:
        new_arr = np.zeros((*arr.shape[:-1], pad_to_N), dtype=arr.dtype)
        new_arr[..., :arr.shape[-1]] = arr
        return new_arr
    raise ValueError(f'wfarr of shape {arr.shape} cannot be padded to {pad_to_N}')

def is_numpy_int(x):
    """check if type(x) is one of numpy's integer types"""
    return isinstance(x, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64))
def is_numpy_float(x):
    """check if type(x) is one of numpy's float types"""
    return isinstance(x, (np.float_, np.float16, np.float32, np.float64))

def npy_append_rows(infile_npy, add_rows, outfile_npy=None):
    """
    append rows to array from .npy file at path infile
      if outfile_new given, write the result there;
      otherwise rewrite infile_npy with appended rows
    """
    if outfile_npy is None:
        outfile_npy = infile_npy
    np.save(outfile_npy, np.concatenate([np.atleast_2d(np.load(infile_npy)),
                                         np.atleast_2d(add_rows)], axis=0))
    return

def npy_append_cols(infile_npy, add_cols, outfile_npy=None):
    """
    append columns to array from .npy file at path infile
      if outfile_new given, write the result there;
      otherwise rewrite infile_npy with appended columns
    """
    if outfile_npy is None:
        outfile_npy = infile_npy
    old_cols = np.load(infile_npy)
    if old_cols.ndim == 1:
        old_cols = old_cols[:, np.newaxis]
    cols_2d = (add_cols[:, np.newaxis] if np.ndim(add_cols) == 1 else add_cols)
    np.save(outfile_npy, np.concatenate([old_cols, cols_2d], axis=1))
    return

def zip_to_array(*args):
    """
    O(10^3)x faster np.array([row for row in np.broadcast(*args)])
    usable when args are all at most 1d.
    """
    lengths = [(len(a) if hasattr(a, '__len__') else 1) for a in args]
    imaxlen = np.argmax(lengths)
    maxlen = lengths[imaxlen]
    if maxlen == 1: # if all scalars or length 1
        return np.transpose([np.atleast_1d(a) for a in args])
    if np.allclose(lengths, maxlen): # if all same length > 1
        return np.transpose(args)
    # otherwise combine with broadcasting
    assert all([(l == maxlen) for l in lengths if (l != 1)]), \
        'All non-scalar arguments must have the same length!'
    outrows = np.zeros((maxlen, len(args)), dtype=type(args[imaxlen][0]))
    for jcol, a in enumerate(args):
        outrows[:, jcol] = a
    return outrows

########################################################################
#### STORING and LOADING SYMMETRIC MATRICES (efficiently)

def store_symmetrix_matrix(mat):
    """store n x n symmetric matrix as 1d array of length n*(n+1)/2"""
    n = len(mat)
    flatmat = np.zeros(int(n * (n + 1) / 2))
    iflat = 0
    for j in range(n):
        uplen = n - j
        flatmat[iflat:iflat + uplen] = mat[j][j:]
        iflat += uplen
    return flatmat

def load_symmetrix_matrix(flatmat, dimlen):
    """load n x n symmetric matrix from 1d array formatted as in store_symmetric_matrix()"""
    nn = len(flatmat)
    assert dimlen*(dimlen+1)/2 == nn, 'flatmat must have length dimlen*(dimlen+1)/2'
    outmat = np.zeros((dimlen, dimlen))
    iflat = 0
    for j in range(dimlen):
        uplen = dimlen - j
        outmat[j, j:] = outmat[j:, j] = flatmat[iflat:iflat + uplen]
        iflat += uplen
    return outmat

########################################################################
#### PROBABILITY DISTRIBUTIONS

def flat_in_log(vmin, vmax, n):
    """get n samples from log-uniform distribution between vmin and vmax"""
    logmin = np.log(vmin)
    logmax = np.log(vmax)
    return np.exp(np.random.uniform(logmin, logmax, n))

def rand_azim(nangles=1):
    """get nangles (int or tuple) samples ~ U(0, 2*np.pi)"""
    if nangles == 1:
        return np.random.uniform(0, 2*np.pi)
    return np.random.uniform(0, 2*np.pi, nangles)

def rand_cos(nangles=1):
    """get nangles (int or tuple) samples ~ U(-1, 1)"""
    if nangles == 1:
        return np.random.uniform(-1, 1)
    return np.random.uniform(-1, 1, nangles)

def rand_polar(nangles=1):
    """get nangles (int or tuple) samples ~ np.arccos(U(-1, 1))"""
    return np.arccos(rand_cos(nangles=nangles))

########################################################################
#### FFT and RFFT UTILITIES

def nfft_of_rfft(rfft_arr):
    """ASSUMES nfft even <=> len(rfft_arr) odd"""
    return 2 * (rfft_arr.shape[-1] - 1)

def nfft_of_rfftlen(rfftlen):
    """ASSUMES nfft even <=> rfftlen odd"""
    return 2 * (rfftlen - 1)

def rfftlen_of_nfft(nfft):
    """ASSUMES nfft even <=> len(rfft_arr) odd"""
    return (nfft // 2) + 1

########################################################################
#### SIGNAL WINDOWING UTILITIES

def tukwin_front(nfront):
    """front taper of tukey window (first half of hann)"""
    return spsig.hann(2*nfront, sym=True)[:nfront]

def tukwin_back(nback):
    """back taper of tukey window (last half of hann)"""
    return spsig.hann(2*nback, sym=True)[-nback:]

def tukwin_npts(ntot, nwin):
    """get tukey window based on numbers of total and windowed points"""
    tukwin = np.ones(ntot)
    hannwin = spsig.hann(2*nwin, sym=True)
    tukwin[:nwin] = hannwin[:nwin]
    tukwin[-nwin:] = hannwin[-nwin:]
    return tukwin

def tukwin_bandpass(taper_width, f_nyq=None, rfft_len=None, f_rfft=None):
    """
    taper_width is frequency interval length in Hz to be tapered at each end
    f_nyq & rfft_len are the nyquist frequency and length of the rfftfreq array
    """
    if f_rfft is None:
        return spsig.tukey(rfft_len, alpha=(2 * taper_width / f_nyq), sym=True)
    else:
        return spsig.tukey(len(f_rfft), alpha=(2 * taper_width / f_rfft[-1]), sym=True)

########################################################################
#### PRINTING

def printarr(arr, prec=4, pre='', post='', sep='  ', form='f'):
    """wrapper for np.array2string printing"""
    print(pre + np.array2string(np.asarray(arr), separator=sep,
                                max_line_width=np.inf, threshold=np.inf,
                                formatter={'float_kind':lambda x: f"%.{prec}{form}" % x}) + post)
    return

def fmt(num, prec=4, form='f'):
    """formatting number as string"""
    formstr = '{:.' + str(prec) + form + '}'
    return formstr.format(num)
