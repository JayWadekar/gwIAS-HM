# This should be the main file
# Should receive a file name and configuration parameters, and output all
# candidates that pass glitch_analysis.
# It should also record:
# 0) high quality fraction
# 1) glitch blocked fraction
# 2) glitch
import numpy as np
import os
import glob
import readligo as rl
import scipy.signal as signal
import scipy.linalg as scilin
import scipy.stats as stats
import params
import utils
import warnings
import copy


# Some parameters for data analysis
# ---------------------------------
# Specgrams are not independent due to overlaps, oversubscription factor
OVERSUB = 1 / (1 - params.OVERLAP_FAC)
# Moving average length for long mode removal (in units of intervals)
AVG_LEN = int(params.N_INDEP_MOVING_AVG_EXCESS_POWER * OVERSUB)


# Functions to work with PSDs
# ------------------------------------------------------------------------
def scipy_12_welch(
        x, y=None, fs=1.0, window='hann', nperseg=None, noverlap=None,
        nfft=None, detrend='constant', return_onesided=True,
        scaling='density', axis=-1, average='mean', mask=None,
        line_id_ver="old"):
    """Cheating to copy Scipy 1.2's Welch method, to access the average
    attribute
    Note: Assumes fs is an integer in converting times to mask indices
    :param line_id_ver:
        Flag to use old or new version of line-identification code
    :return: 1. Array with frequency axis (Hz)
             2. Array with one-sided PSD (Hz^-1)
             3. Crude line mask with lines zeroed
             4. Loud line mask with lines zeroed
    """
    if y is None:
        # Default to the auto-PSD
        y = x

    try:
        from scipy.signal._spectral_py import _spectral_helper as spectral_helper
    except ModuleNotFoundError:
        from scipy.signal.spectral import _spectral_helper as spectral_helper

    freqs, times, pxy = spectral_helper(
        x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided,
        scaling, axis, mode='psd')

    # Default PSD overlap is 1/2
    nindpsd = len(times) / 2

    # FFTs become unreliable around preexisting holes. Cut out these segments
    if mask is not None:
        if nperseg is None:
            raise RuntimeError("I have no way of converting times to indices!")
        segduration = nperseg / fs

        # Create boolean mask indexing good times
        specmask = np.zeros_like(times, dtype=bool)
        for ind, wcent in enumerate(times):
            specmask[ind] = np.prod(
                mask[int(np.round(wcent - segduration / 2)):
                     int(np.round(wcent + segduration / 2))])

        # Default PSD overlap is 1/2
        nindpsd = np.count_nonzero(specmask)/2
        if nindpsd < params.MINSAMP_PSD:
            raise RuntimeError("File is too intermittent, cannot measure PSD!")

        pxy = pxy[:, specmask]

    # Average over windows
    if pxy.ndim >= 2 and pxy.size > 0:
        if pxy.shape[-1] > 1:
            if average.lower() == 'none':
                # Pass pxy through
                return freqs, pxy.real
            elif average.lower() == 'median':
                pxy = np.nanmedian(pxy, axis=-1) / median_bias(pxy.shape[-1])
            elif average.lower() == 'mean':
                pxy = pxy.mean(axis=-1)
            else:
                raise ValueError(
                    'average must be "median" or "mean", got %s' % (average,))
        else:
            pxy = np.reshape(pxy, pxy.shape[:-1])

    # Find lines
    if line_id_ver.lower() == "old":
        crude_line_mask, loud_line_mask = define_crude_line_mask(
            freqs, pxy.real, nindpsd)
    else:
        crude_line_mask, loud_line_mask = define_crude_line_mask_v2(
            freqs, pxy.real, nindpsd)

    return freqs, pxy.real, crude_line_mask, loud_line_mask


def define_crude_line_mask(
        freqs, psds, nindpsd, sm_time=1, fmin=params.FMIN_ANALYSIS):
    """Marks lines in the PSD with zeros, errs on the side of marking lines"""
    # Define lowpass filter (in the frequency domain) to remove noise in
    # the PSD
    df = freqs[1] - freqs[0]
    sos, irl = utils.band_filter(df, fmax=sm_time, btype='low')
    asds = np.sqrt(psds)
    # Smooth the asds with this filter
    smoothed_asds = signal.sosfiltfilt(sos, asds, padlen=irl)
    # Mark lines as more than 4 sigma fluctuations in the ASD
    fractional_deviation = np.abs(smoothed_asds / asds - 1)
    # globals()["asds"] = asds.copy()
    # globals()["smoothed_asds"] = smoothed_asds.copy()
    # globals()["nindpsd"] = nindpsd
    # globals()["fractional_deviation"] = fractional_deviation
    line_freqs = np.logical_not(np.logical_and(
        fractional_deviation > params.LINE_SIGMA / (2. * np.sqrt(nindpsd)),
        freqs > fmin))
    loud_line_freqs = np.logical_not(np.logical_and(
        fractional_deviation > params.LOUD_LINE_SIGMA / (2. * np.sqrt(nindpsd)),
        freqs > fmin))
    return line_freqs, loud_line_freqs


def define_crude_line_mask_v2(
        freqs, psds, nindpsd, lw_check=0.5, fmin=params.FMIN_ANALYSIS, niter=3):
    """
    Function to flag lines in the PSD, errs on the side of marking lines
    :param freqs: Array with frequencies (in Hz)
    :param psds: Array with PSDs (in 1/Hz)
    :param nindpsd: Number of independent windows in Welch estimate for the PSD
    :param lw_check:
        Parameter for smoothing window to check for lines (in Hz), equal to the
        assumed width of the lines, as well as roughly 1/3 the width of the
        frequency interval we smooth over
    :param fmin: Look for lines only at f >= fmin
    :param niter: Number of passes to find lines
    :return: 1. Boolean array of length freqs with zeros marking lines
             2. Boolean array of length freqs with zeros marking loud lines
    """
    # ASDs
    asds = np.sqrt(psds)

    # Define moving average filter (in the frequency domain) to remove noise in
    # the PSD, and exclude the line itself
    df = freqs[1] - freqs[0]
    # Guard against bad choice of resolution
    nw = max(2*(int(lw_check/df) // 2), 2)
    sm_filter = np.r_[np.ones(nw), np.zeros(nw + 1), np.ones(nw)]
    sm_filter /= np.sum(sm_filter)
    lfilt = len(sm_filter)

    line_freqs = np.ones_like(freqs, dtype=bool)
    loud_line_freqs = np.ones_like(freqs, dtype=bool)
    for ind in range(niter):
        # Smooth the asds with the filter
        asd_padded = np.pad(
            asds, (lfilt // 2, lfilt - lfilt // 2 - 1), mode="edge")
        smoothed_asds = signal.convolve(asd_padded, sm_filter, mode="valid")

        # Mark lines (loud lines) as more than 4 (500) sigma fluctuations
        # in the ASD
        fractional_deviation = asds / smoothed_asds - 1

        line_mask = np.logical_and(
            fractional_deviation > params.LINE_SIGMA / (2. * np.sqrt(nindpsd)),
            freqs > fmin)
        loud_line_mask = np.logical_and(
            fractional_deviation > (params.LOUD_LINE_SIGMA /
                                    (2. * np.sqrt(nindpsd))), freqs > fmin)

        line_freqs[line_mask] = False
        loud_line_freqs[loud_line_mask] = False

        # Remove lines for the next iteration
        asds[line_mask] = smoothed_asds[line_mask]

    return line_freqs, loud_line_freqs


def median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.
    See arXiv:gr-qc/0509116 Appendix B for details.
    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.
    Returns
    -------
    bias : float
        Calculated bias.
    """
    ii_2 = 2 * np.arange(1., (n-1) // 2 + 1)
    return 1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2)


def asd_func(fint, psdint, fmin=None, fmax=None):
    """
    Returns function that takes frequencies in Hz and gives the interpolated
    ASD in units of 1/Hz^0.5
    :param fint: Array with list of sampled frequencies
    :param psdint: Array with list of computed PSDs
    :param fmin: Blow up PSDs below fmin due to lack of calibration (None if no
                 fmin)
    :param fmax: Blow up PSDs above fmax to avoid features in the measured PSD
                 (Pass None if no fmax)
    :return: Function that takes frequencies and returns ASDs by linearly
             interpolating the PSDs
    """
    if fmin is not None:
        imin = np.searchsorted(fint, fmin)
    else:
        imin = 0
    if fmax is not None:
        imax = np.searchsorted(fint, fmax)
    else:
        imax = len(fint)

    def fn(f):
        return np.sqrt(np.interp(
            f, fint[imin:imax], psdint[imin:imax], left=np.inf, right=np.inf))

    return fn


def makeasdplots(asddir, asd_function=None, show=True, fname="ASDS.pdf"):
    """
    Makes comparison plots of ASDs with the data-derived one
    :param asddir: Directory with ASD files
    :param asd_function: Function that takes frequencies (in Hz) and returns
                     data-derived asd (in 1/Hz**0.5)
    :param show: Flag indicated whether to show or save
    :param fname: Filename
    :return:
    """
    import matplotlib.pyplot as plt
    import re

    asdfiles = sorted(glob.glob(os.path.join(asddir, "*.txt")))
    reg = re.compile(r"aLIGO_(.*)\.txt")
    labels = [re.search(reg, fname).group(1) for fname in asdfiles]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    fs = np.logspace(1., np.log10(3000.))  # default frequencies

    for fname, label in zip(asdfiles, labels):
        asd = np.loadtxt(fname)
        fs = asd[:, 0]
        ax.loglog(asd[:, 0], asd[:, 1], label=label)

    if asd_function is not None:
        ax.loglog(fs, asd_function(fs), label='DATA')

    ax.legend(frameon=False)

    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(asddir, fname))


# Functions to perform matched filtering
# ------------------------------------------------------------------------
def chunkedfft(data, fftsize, wl, padmode='left', wraparound=True):
    """
    Returns FFTs of chunks of data for overlap-save. Coordinates assumed
    dimensionless, as in DFT
    :param data: Array with the input data (time domain)
    :param fftsize: Size of FFT for each sub-chunk
    :param wl: Length of window (time domain)
    :param padmode:
        How to pad the data (useful for various choices of window weights and
        time conventions)
    :param wraparound:
        Flag indicating whether to copy data into the padded part to mimic a
        full FFT
    :return: n_chunk x nrfft(fftsize) array of rFFTs of chunks
    """
    # Size of chunk
    chunksize = fftsize - wl + 1

    if padmode == 'left':
        # The first entry is when the right-edge of the filter touches the
        # left edge of the data
        # Captures the starting transient, but not the ending one
        leftpad = wl - 1
    elif padmode == 'center':
        # The first entry is when the filter center touches the left edge
        # of the data
        # Captures half of the starting and ending transients
        # Reproduces the 'same' mode of scipy fftconvolve if wraparound = False
        leftpad = wl // 2
    else:
        # The first entry is when the left-edge of the filter touches the
        # left edge of the data
        # Captures the ending transient but not the starting one
        leftpad = 0
    rightpad = wl - 1 - leftpad

    # Number of chunks (note that >= 2)
    nchunk = int(np.ceil(len(data) / chunksize))
    paddeddata = utils.FFTIN(wl - 1 + chunksize * nchunk)

    # Left and right-pad as needed
    paddeddata[leftpad:leftpad + len(data)] = data

    if wraparound:
        if leftpad > 0:
            # Add right data to left for wrapping around to mimic an fft
            paddeddata[:leftpad] = data[-leftpad:]
        # Add left data to right for wrapping around to mimic fft
        paddeddata[leftpad + len(data):leftpad + len(data) + rightpad] = \
            data[:rightpad]

    # Structure to hold rFFT of data
    rfft_vec = utils.FFTIN((nchunk, fftsize // 2 + 1), dtype=np.complex128)
    for chid in range(nchunk):
        rfft_vec[chid] = utils.RFFT(
            paddeddata[chid * chunksize:chid * chunksize + fftsize])

    return rfft_vec


def overlap_save(chunked_data_f, window_f, fftsize, wl):
    """
    Convolution with the overlap-save method. Look at the modes in chunkedfft
    to understand what transients are captured at the edges
    Coordinates assumed dimensionless, as in DFT
    Note: Overlap-save is only justified when the window is strictly compact
    and contained within wl
    :param chunked_data_f: n_chunk x nrfft(fftsize) array of rFFTs of data chunks
    :param window_f: Complex array of length nrfft(fftsize) RFFT of {window with
                     support within wl, padded to fftsize in the time domain}
    :param fftsize: Size of FFT for each sub-chunk
    :param wl: Length of window (time domain)
    :return: Array of length nchunk x chunksize with convolved data
    """
    # IRFFT along the last axis by default
    fft_in = utils.FFTIN(chunked_data_f.shape, dtype=np.complex128)
    fft_in[:] = chunked_data_f * window_f
    conv = utils.IRFFT(fft_in, n=fftsize)

    # Discard transients and return contiguous output
    chunksize = fftsize - wl + 1
    if conv.ndim == 1:
        return conv[wl - 1:fftsize]
    else:
        return conv[:, wl - 1:fftsize].reshape(len(conv) * chunksize)


def norm_matched_filter_overlap(
        chunked_data_f, wf_whitened_fd, fftsize, support_wf):
    """
    Computes matched filter overlap for a sliding template. Coordinates
    assumed dimensionless, as in DFT. Returns complex number whose real and
    imaginary parts are N(0,1)
    :param chunked_data_f: n_chunk x rfftsize(fftsize) array with data in
                           frequency domain
    :param wf_whitened_fd: Frequency domain template (length rfft(fftsize))
                           Convention: power is towards the right side
    :param fftsize: Size of FFT for each sub-chunk
    :param support_wf: Time domain support for whitened waveform template
    :return: Array of length n_chunk x chunksize with cosine and sine components
             of matched filtering overlap
    """
    fft_in = utils.FFTIN(wf_whitened_fd.shape, dtype=np.complex128)
    fft_in[:] = wf_whitened_fd.conj()
    # mf_cos = IRFFT(fft_in) / normalization
    # mf_sin = IRFFT(fft_in) / normalization
    mf_cos = overlap_save(chunked_data_f, fft_in, fftsize, support_wf)
    fft_in *= -1j
    mf_sin = overlap_save(chunked_data_f, fft_in, fftsize, support_wf)

    mf_out = utils.FFTIN(mf_cos.shape, dtype=np.complex128)
    mf_out[:] = mf_cos + 1j * mf_sin

    return mf_out


# Functions to load data for matched filtering
# ------------------------------------------------------------------------
def loaddata(
        fname=None, left_fname=None, right_fname=None,
        quality_flags=('DEFAULT', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3'),
        chunktime_psd=params.DEF_CHUNKTIME_PSD,
        wf_duration_est=params.MIN_WFDURATION, edgepadding=True,
        fmax=params.FMAX_OVERLAP, **load_kwargs):
    """
    Load data
    Warning:
    1. Assumes that the quality flags are sampled at 1 Hz
    2. Assumes that the sampling rate of data is an integer
    # TODO: If mask is too intermittent, make it flat since it is unreliable...
    :param fname: Filename
    :param left_fname: File with strain data on the left, if known
    :param right_fname: File with strain data on the right, if known
    :param quality_flags:
        Tuple of LIGO data quality flags to apply cuts on
        Check out https://losc.ligo.org/techdetails/
    :param chunktime_psd: Length of time chunk for measuring PSD (s)
    :param wf_duration_est:
        Estimated waveform duration (used to figure out how much to pull in)
    :param edgepadding:
        Flag indicating whether to pad data with zeros that we will fill later
    :param fmax:
        Maximum frequency involved in the analysis, used to estimate
        buffer indices
    :param load_kwargs:
        Dictionary with details to load from GPS times instead of files.
        Should have the following keys:
        'run': O1/O2/O3a
        'IFO': H1/L1/V1
        'tstart': Starting time
        'tstop': Stopping time
        No loading extra seconds
    :return: 1. Times
             2. Strains
             3. Boolean mask with quality flags applied
             4. Loaded channel_dict with all flags (at 1 Hz resolution)
             5. Time at the left edge of central file
    """
    # Calculate some derived loading parameters
    # -----------------------------------------
    # Estimate of how many indices will be unusable at edges of full files
    # Impulse response length of high-pass filter that we will use
    # (invariant to whether the frequency is a power of two)
    dt_analysis = 1 / (2 * fmax)
    _, irl = utils.band_filter(dt_analysis, fmin=params.FMIN_PSD)
    irl_sec = int(np.round(irl * dt_analysis))

    # Estimate of safety length at edge of file
    support_blue_max = int(chunktime_psd)
    edgesafety_est = support_blue_max + irl_sec

    # Limit of hole-size at the edge of the file where we force a break
    max_endbreak = int(max(wf_duration_est, chunktime_psd))

    # Default length to pull in from the left and right
    print("Estimated waveform duration: ", int(wf_duration_est))
    invalid_left_sec_est = int(wf_duration_est) + edgesafety_est
    print("Estimated invalid seconds to the left:", invalid_left_sec_est)
    invalid_right_sec_est = edgesafety_est
    print("Estimated invalid seconds to the right:", invalid_right_sec_est)

    # Minimum file duration (s)
    min_fileduration = chunktime_psd * params.MIN_FILELENGTH_FAC

    if invalid_left_sec_est > min_fileduration:
        warnings.warn(
            "Waveforms are very long, check the loading code!", Warning)

    # LIGO doesn't properly flag left edges of big holes (possibly due to
    # latency in their flagging system). If we terminate a file here, most
    # likely we will create an unfillable hole and kill the file. Safety
    # time to expand the hole's left edge by if we aren't filling there
    bad_hole_safety = int(params.IMPROPER_FLAGGING_SAFETY_DURATION)

    # Load data in primary file or interval
    # -------------------------------------
    if fname is not None:
        strain, time, channel_dict = rl.loaddata(fname)
    else:
        # Read segment
        tstart = load_kwargs['tstart']
        tstop = load_kwargs['tstop']
        IFO = load_kwargs['IFO']
        run = load_kwargs['run']
        flist = rl.FileList(os.path.join(utils.STRAIN_ROOT[run.lower()], IFO))
        strain, tdict, channel_dict = rl.getstrain(
            tstart, tstop, IFO, filelist=flist)
        time = np.arange(tdict['start'], tdict['stop'], tdict['dt'])

    mask = np.prod([channel_dict[x] for x in quality_flags], axis=0)

    # Some parameters of the data
    t0 = time[0]            # Leftmost time
    dt = time[1] - time[0]  # Sampling interval
    fs = int(1 / dt)

    # Add or subtract elements to/from the left edge, if reading from a file
    # ----------------------------------------------------------------------
    # Find indices of holes in primary file
    mask_edges = utils.hole_edges(mask)
    left_edges_hole, right_edges_hole = mask_edges[:, 0], mask_edges[:, 1]
    hole_lengths = right_edges_hole - left_edges_hole

    # Load left file if given
    if left_fname is not None:
        strain_l, time_l, channel_dict_l = rl.loaddata(left_fname)
        mask_l = np.prod([channel_dict_l[x] for x in quality_flags], axis=0)
        # Find indices of holes in left file
        mask_edges_l = utils.hole_edges(mask_l)
        left_edges_hole_l, right_edges_hole_l = \
            mask_edges_l[:, 0], mask_edges_l[:, 1]
        hole_lengths_l = right_edges_hole_l - left_edges_hole_l

    # Check if the left edge of the primary file is inside a hole
    left_hole_len = 0
    if (len(left_edges_hole) > 0) and (left_edges_hole[0] == 0):
        left_hole_len += hole_lengths[0]
        # Add length by which hole extends into the left file, if available
        if (left_fname is not None) and \
                (len(right_edges_hole_l) > 0) and \
                (right_edges_hole_l[-1] == len(mask_l)):
            left_hole_len += hole_lengths_l[-1]

    # Default behaviour is to catch enough to not lose indices at the
    # beginning of the file due to signal processing
    left_len = invalid_left_sec_est
    if left_fname is None:
        left_len = 0

    # Clip the primary file at its left edge if it has too big a hole at the
    # beginning
    # Note: This will make us lose events partially overlapping with large
    # holes at the ends of files, but that's life
    if left_hole_len > 0:
        if (left_hole_len >= max_endbreak) or (left_fname is None):
            first_good_point = right_edges_hole[0]
            time = time[first_good_point * fs:]
            strain = strain[first_good_point * fs:]
            mask = mask[first_good_point:]
            for key in channel_dict.keys():
                channel_dict[key] = channel_dict[key][first_good_point:]
            # Don't pull in data from left
            left_len = 0

    if left_len > 0:
        # We're definitely adding elements from the left file
        # Expand/contract default length of segment we pull in, if needed
        # Look for big holes
        bad_hole_inds_l = np.where(hole_lengths_l >= chunktime_psd)[0]

        # If there is a big hole within the last min_fileduration seconds
        # TODO: If there is a large hole to the left + a small hole to the
        #   right, we don't load the left file now
        # 1. If the hole is entirely within invalid_left_sec_est, don't change
        #    left_len
        # 2. If the left edge is farther than invalid_left_sec_est, but the
        #    right one isn't, restrict to right of hole to avoid zeros at edge
        # 3. If the left and right edges are farther than invalid_left_sec_est,
        #    but the right one is closer than min_fileduration, expand since
        #    the data is better off in this file rather than by itself as the
        #    estimated PSD would be too noisy
        if len(bad_hole_inds_l) > 0:
            left_to_edge = len(mask_l) - left_edges_hole_l[bad_hole_inds_l[-1]]
            right_to_edge = len(mask_l) - right_edges_hole_l[bad_hole_inds_l[-1]]
            # Check ensures that we expand/restrict only if needed
            if ((right_to_edge < min_fileduration) and
                    (left_to_edge >= invalid_left_sec_est)):
                left_len = right_to_edge

    # Add or subtract elements to/from the right edge of file
    # -------------------------------------------------------
    # Flag indicating whether we ended at a big hole
    big_hole_termination = False

    # Primary file might have been clipped, so recompute hole indices
    mask_edges = utils.hole_edges(mask)
    left_edges_hole, right_edges_hole = mask_edges[:, 0], mask_edges[:, 1]
    hole_lengths = right_edges_hole - left_edges_hole

    if right_fname is not None:
        strain_r, time_r, channel_dict_r = rl.loaddata(right_fname)
        mask_r = np.prod([channel_dict_r[x] for x in quality_flags], axis=0)
        # Find indices of holes in right file
        mask_edges_r = utils.hole_edges(mask_r)
        left_edges_hole_r, right_edges_hole_r = \
            mask_edges_r[:, 0], mask_edges_r[:, 1]
        hole_lengths_r = right_edges_hole_r - left_edges_hole_r

    # Check if the right edge of the primary file is inside a hole
    right_hole_len = 0
    if (len(right_edges_hole) > 0) and (right_edges_hole[-1] == len(mask)):
        right_hole_len += hole_lengths[-1]
        # Add length by which hole extends into the right file, if available
        if (right_fname is not None) and \
                (len(left_edges_hole_r) > 0) and \
                (left_edges_hole_r[0] == 0):
            right_hole_len += hole_lengths_r[0]

    # Default behaviour is to catch enough to not lose indices at the
    # end of the file due to signal processing
    right_len = invalid_right_sec_est
    if right_fname is None:
        right_len = 0

    # Clip the primary file at its right edge if it has too big a hole at
    # its end
    if right_hole_len > 0:
        if (right_hole_len >= max_endbreak) or (right_fname is None):
            last_good_point = left_edges_hole[-1]
            time = time[:last_good_point * fs]
            strain = strain[:last_good_point * fs]
            mask = mask[:last_good_point]
            for key in channel_dict.keys():
                channel_dict[key] = channel_dict[key][:last_good_point]
            # Don't pull in data from right
            right_len = 0
            if right_hole_len >= max_endbreak:
                # Mark termination at a big hole
                big_hole_termination = True

    if right_len > 0:
        # We're definitely adding elements from the right file
        # Expand/contract default length of segment we pull in, if needed
        # Look for big holes
        bad_hole_inds_r = np.where(hole_lengths_r >= chunktime_psd)[0]

        # If there is a big hole in the first min_fileduration seconds
        # TODO: If there is a large hole to the right + a small hole to the
        # left, we don't load the right file now
        # 1. If the hole is entirely within invalid_right_sec_est, don't change
        #    right_len
        # 2. If the right edge is farther than invalid_right_sec_est, but the
        #    left one isn't, restrict to left of hole to avoid zeros at edge
        # 3. If the left and right edges are farther than invalid_right_sec_est,
        #    but the left one is closer than min_fileduration, expand since the
        #    data is better off in this file rather than by itself as the
        #    estimated PSD would be too noisy
        if len(bad_hole_inds_r) > 0:
            left_to_edge = left_edges_hole_r[bad_hole_inds_r[0]]
            right_to_edge = right_edges_hole_r[bad_hole_inds_r[0]]
            # Check ensures that we expand/restrict only if needed
            if ((left_to_edge < min_fileduration) and
                    (right_to_edge >= invalid_right_sec_est)):
                right_len = left_to_edge
                # Mark termination at a big hole
                big_hole_termination = True

    print(f"Length of data pulled in from left file: {left_len}")
    print(f"Length of data pulled in from right file: {right_len}")

    # Collect data together
    # ---------------------
    # First deal with LIGO's bad masking
    central_sec = len(mask)
    if big_hole_termination:
        print(f"Chopping off {bad_hole_safety} seconds to mitigate " +
              f"bad masking")
        # How much we chop off from what we pull in from the right hand file
        right_cut = min(right_len, bad_hole_safety)
        right_len -= right_cut
        # If we pulled in lesser than the amount to chop off, take the rest
        # from the central file
        central_sec -= (bad_hole_safety - right_cut)

    # Define concatenated data arrays with aligned memory
    left_sec = left_len
    right_sec = right_len

    if edgepadding:
        # Always add a hole of this length to both sides that we will fill in
        left_sec += support_blue_max
        right_sec += support_blue_max

    # Populate times, we do it this way because we might be adding some zeros
    # due to edge padding where we have no data anyway
    time_total = utils.FFTIN((left_sec + central_sec + right_sec) * fs)
    time_total[:] = time[0] - left_sec + dt * np.arange(len(time_total))

    # Define strain
    strain_total = utils.FFTIN((left_sec + central_sec + right_sec) * fs)
    # Masks are false by default
    mask_total = utils.FFTIN(left_sec + central_sec + right_sec, dtype=bool)
    channel_dict_total = {}
    for flag, quality_mask in channel_dict.items():
        # Channel dicts are true by default
        cdict_ones = utils.FFTIN(left_sec + central_sec + right_sec, dtype=bool)
        cdict_ones[:] = True
        channel_dict_total.update({flag: cdict_ones})
    # Except for DEFAULT
    cdict_zeros = utils.FFTIN(left_sec + central_sec + right_sec, dtype=bool)
    channel_dict_total.update({'DEFAULT': cdict_zeros})

    # Fill center data
    strain_total[left_sec * fs:(left_sec + central_sec) * fs] = \
        strain[:central_sec * fs]
    mask_total[left_sec:left_sec + central_sec] = mask[:central_sec]
    for flag in channel_dict_total.keys():
        channel_dict_total[flag][left_sec:left_sec + central_sec] = \
            channel_dict[flag][:central_sec]

    # Fill left data
    if left_len > 0:
        # Fill only amount pulled in, i.e., if we edgepadded, let those entries
        # remain zero
        strain_total[(left_sec - left_len) * fs:left_sec * fs] = \
            strain_l[-(left_len * fs):]
        mask_total[left_sec - left_len:left_sec] = mask_l[-left_len:]
        for flag in channel_dict_total.keys():
            channel_dict_total[flag][left_sec - left_len:left_sec] = \
                channel_dict_l[flag][-left_len:]

    # Fill right data
    if right_len > 0:
        # Fill only amount pulled in, i.e., if we edgepadded, let those entries
        # remain zero
        strain_total[(left_sec + central_sec) * fs:
                     (left_sec + central_sec + right_len) * fs] = \
            strain_r[:right_len * fs]
        mask_total[left_sec + central_sec:left_sec + central_sec + right_len] = \
            mask_r[:right_len]
        for flag in channel_dict_total.keys():
            channel_dict_total[flag][left_sec + central_sec:
                                     left_sec + central_sec + right_len] = \
                channel_dict_r[flag][:right_len]

    # LIGO returns nan when data is undefined, set it to zero so that we do not
    # kill the FFTs operating on the data. Also update the mask during the
    # seconds we set nan to zero, to be safe
    badinds = np.isnan(strain_total)
    mask_total[(np.where(badinds)[0] * dt).astype(int)] = 0
    strain_total = np.nan_to_num(strain_total)

    return time_total, strain_total, mask_total, channel_dict_total, t0


def data_to_asdfunc(
        strains, mask, fs, chunktime_psd=params.DEF_CHUNKTIME_PSD,
        fmax=params.FMAX_PSD, average='median', line_id_ver="old"):
    """
    Passes the data to welch and returns a function that maps frequencies to ASD
    :param strains: Array with strains
    :param mask: Boolean quality mask on strain data at 1 Hz
    :param fs: Sampling frequency (Hz)
    :param chunktime_psd: Chunktime for PSD estimation
    :param fmax: Maximum frequency to estimate the PSD at
    :param average: Averaging method to use
    :param line_id_ver:
        Flag to use old or new version of line-identification code
    :return: 1. Array with frequencies (Hz)
             2. Array with PSD (Hz^-1)
             3. Function mapping frequencies to ASD
             4. Crude mask on frequencies with lines marked with zeros
             5. Mask on frequencies with loud lines marked with zeros
    """
    # Compute median PSD (safe to localized glitches, and reasonable numbers of
    # isolated blocks of flagged bad data)
    freq_axis, psd, crude_line_mask, loud_line_mask = scipy_12_welch(
         strains, fs=fs, nperseg=int(chunktime_psd * fs),
         average=average, mask=mask, line_id_ver=line_id_ver)

    # Define function returning the ASD
    asd_func_data = asd_func(freq_axis, psd, fmin=params.FMIN_PSD, fmax=fmax)

    return freq_axis, psd, asd_func_data, crude_line_mask, loud_line_mask


# Functions to prepare data for matched filtering
# ------------------------------------------------------------------------
def process_data(
        time, strain, mask, asd_func_data, do_signal_processing=True,
        sigma_clipping_threshold=None, sine_gaussian_intervals=None,
        sine_gaussian_thresholds=None, bandlim_transient_intervals=None,
        bandlim_power_thresholds=None, excess_power_intervals=None,
        excess_power_thresholds=None, erase_bands=False, freqs_in=None,
        crude_line_mask=None,loud_line_mask=None, freqs_to_notch=None,
        notch_wt_filter=False,
        renorm_wt=True, times_to_fill=None, times_to_save=None,
        fmax=params.FMAX_OVERLAP, notch_format="old", taper_wt_filter=False,
        taper_fraction=0.5, min_filt_trunc_time=1):
    """
    Preferably send in 2**N samples
    :param time: Array of times (s)
    :param strain: Array of strain data
    :param mask: Quality mask on strain data at 1 Hz
    :param asd_func_data:
        Function returning ASDs (1/sqrt{Hz}) given frequencies (Hz)
    :param do_signal_processing:
        Flag indicating whether to do signal processing to identify glitches
        and fill holes
    :param sigma_clipping_threshold:
        Threshold for sigma clipping calculated from waveform. If None, clipping
        depends on params.NPERFILE
    :param sine_gaussian_intervals:
        Frequency bands within which we look for Sine-Gaussian noise
        [central frequency, df = (upper - lower frequency)] Hz
    :param sine_gaussian_thresholds:
        Amplitude thresholds for sine gaussian transients computed from
        waveforns. If None, detection depends on params.NPERFILE
    :param bandlim_transient_intervals:
        Array with set of time-interval-frequency intervals (s, Hz)
        [[dt_i, [f_i_min, f_i_max]],...]
    :param bandlim_power_thresholds:
        Thresholds for bandlimited transient detection computed from waveforms.
        If None, detection depends on params.NPERFILE
    :param excess_power_intervals:
        Array with timescales to look for excess power on (s)
    :param excess_power_thresholds:
        Thresholds for excess power detection computed from waveforms. If None,
        detection depends on params.NPERFILE
    :param erase_bands: Flag to erase bands in time-frequency space
    :param freqs_in:
        Array with frequencies on which we looked for lines (optional)
    :param crude_line_mask:
        Boolean array with zeros marking crudely identified lines (optional)
    :param loud_line_mask:
        Boolean array with zeros marking identified loud lines, that will be
        notched (optional)
    :param freqs_to_notch:
        If desired, list of frequency ranges to notch [[f_min_i, f_max_i]]
    :param notch_wt_filter:
        Flag to apply notches to the whitening filter as well, setting it to
        True avoids biasing the SNR, but creates artifacts at low frequencies
        (safe to use False as well, since we have PSD drift downstream)
    :param renorm_wt:
        Flag whether to scale the whitened data to have unit variance after
        highpass (if True, we scale by some amount to account for the fraction
        of the band lost)
    :param times_to_fill:
        If desired, list of time ranges to create holes in [[t_min, t_max], ...]
    :param times_to_save:
        If desired, list of time ranges not to create holes in
        [[t_min, t_max], ...]
    :param fmax: Maximum frequency involved in the analysis
    :param notch_format:
        Flag to pick the format for notching. "new" tries to center notches on
        the finer grid if available
    :param taper_wt_filter:
        Flag whether to taper the time domain response of the whitening filter
        with a Tukey window
    :param taper_fraction:
        Fraction of response to taper with a Tukey window, if applicable
        (0 is boxcar, 1 is Hann)
    :return: 1. Array of downsampled times
             2. Downsampled and whitened strain data
             3. Boolean mask with zeros where data has been zeroed/inpainted
             4. Boolean mask with zeros where we cannot trust whitened data
             5. Fine grequency grid on which we identify lines
             6. Mask on the fine frequency grid with zeros at varying lines
             7. n_glitch_tests x ntimes boolean array with 0 where test fired
             8. Whitening filter in time domain (array of length fftsize, with
                weight at the ends)
             9. Support of whitening filter (TD filter has
                2 * support - 1 nonzero coeffs)
             10. Normalization factor for whitening filter (to whiten the data,
                convolve with norm * whitening_filter_fd)
             11. Factor to multiply the whitened strain with to get each entry
                 to be N(0,1) (differs from 1 due to bandpass, the value = 1 =
                 incorrect if notch_wt_filter is True)
             12. Freqs x times boolean mask with zeros at turbulent bands
    """
    # Preprocessing that does not depend on data
    # ------------------------------------------
    # Compute ASDs at full resolution
    dt_full = time[1] - time[0]
    fs_full = np.fft.rfftfreq(len(strain), dt_full)
    asds_full = asd_func_data(fs_full)

    # Use frequencies up to fmax_overlap
    ind_sub = fs_full.searchsorted(fmax) + 1
    assert ind_sub > 1, "Too low fmax_overlap passed"
    fs_down = fs_full[:ind_sub]
    asds_down = asds_full[:ind_sub]

    # Define downsampled time grid (always even-numbered in length)
    time_down = time[0] + (1. / (2. * fs_down[-1])) * np.arange(2 * ind_sub - 2)
    dt_down = time_down[1] - time_down[0]
    sub_fac = fs_full[-1] / fs_down[-1]

    # Highpass filter that we apply to fix leakage/bad data at low frequencies
    sos, irl = utils.band_filter(dt_down, fmin=params.FMIN_PSD)

    # Initialize masks for lines and notches, if any
    # Set frequency resolution, according to longest duration over
    # which we look for transients, default = 1 s
    tmax = 1.
    if utils.safelen(excess_power_intervals) > 0:
        tmax = max(tmax, max(excess_power_intervals))
    if utils.safelen(bandlim_transient_intervals) > 0:
        tmax = max(
            tmax, max([t for t, df in bandlim_transient_intervals]))

    # Define masks
    freqs_lines = np.fft.rfftfreq(int(tmax / dt_down), d=dt_down)
    mask_freqs = np.ones_like(freqs_lines, dtype=bool)

    if notch_format.lower() == "old":
        freq_axis_notch = freqs_lines
    else:
        if freqs_in is None:
            freq_axis_notch = freqs_lines
        else:
            # Pick the finer grid
            freq_axis_notch = freqs_in

    mask_notch = np.ones_like(freq_axis_notch, dtype=bool)
    # Add anything the user desired to the mask to be notched
    if freqs_to_notch is not None:
        for fmin_notch, fmax_notch in freqs_to_notch:
            mask_notch *= ((freq_axis_notch - fmin_notch) *
                           (freq_axis_notch - fmax_notch)) >= 0

    # Add loud lines to the mask to be notched
    if (freqs_in is not None) and (loud_line_mask is not None):
        mask_notch *= utils.define_coarser_mask(
            freqs_in, loud_line_mask, freq_axis_notch)

    # Compute filter coefficients for notches
    if notch_format.lower() == "old":
        dfmin_notch = None
    else:
        dfmin_notch = 1./tmax
    notch_pars = utils.notch_filter_sos(
        dt_down, freq_axis_notch, mask_notch, dfmin_notch=dfmin_notch)

    # Preprocess the data
    # --------------------------------------------------------------------------
    # Downsample strain to lower time resolution, leakage due to brick wall
    # The loading function makes holes on both sides
    # Warning: We're not going to widen the untrusted region due to this!
    # We assume that irl catches the required length
    strain_down_rfft = utils.RFFT(strain)[:ind_sub]

    # Fix the Nyquist frequency to be real
    if len(time_down) % 2 == 0:
        strain_down_rfft[-1] = 2. * strain_down_rfft[-1].real

    # Ensure aligned memory
    strain_down = utils.FFTIN(len(time_down))
    strain_down[:] = utils.IRFFT(strain_down_rfft / sub_fac, n=len(time_down))

    # Fix leakage of Nyquist frequency into zero frequency
    strain_down = strain_down - np.mean(strain_down)

    # Define whitening filter and apply bandpasses/filters
    # --------------------------------------------------------------------------
    # Define time domain whitening filter with suppressed ringing
    # Raw frequency domain whitening filter
    wt_filter_fd_unconditioned = 1. / asds_down

    if notch_wt_filter:
        # Add highpass to notches, and eventually the whitening filter
        notch_pars.append((sos, irl))

        # By linearity, we can also notch the whitening filter, and `whiten'
        # the raw data with this notched filter. If asked, do it this way to
        # avoid biasing the SNR, and to maintain the finite support of the
        # whitening filter
        wt_filter_td, support_wt, tot_wt = utils.condition_filter(
            wt_filter_fd_unconditioned, truncate=True, shorten=True,
            taper=taper_wt_filter, notch_pars=notch_pars, flen=len(time_down),
            taper_fraction=taper_fraction, in_domain='fd', out_domain='td',
            min_trunc_len=int(min_filt_trunc_time / dt_down))

        # Normalization factor for whitening is (1 / fmax_psd = 2 * dt)** 0.5
        norm_wt = (2. * dt_down)**0.5

        # Reproduce my memory of O2, which was the only case when
        # notch_wt_filter might have been true. Looking at some files, it seems
        # norm_wt was actually not (2. * dt_down)**0.5, so maybe it was not True
        # Note that this isn't actually the required value of sc_n01...
        sc_n01 = 1.
    else:
        # Highpass filter the data to remove low frequency components where we
        # cannot trust the strain, since we are not zeroing them in the
        # whitening filter
        strain_down = signal.sosfiltfilt(sos, strain_down, padlen=irl)

        wt_filter_td, support_wt, tot_wt = utils.condition_filter(
            wt_filter_fd_unconditioned, truncate=True, shorten=True,
            taper=taper_wt_filter, flen=len(time_down),
            taper_fraction=taper_fraction, in_domain='fd', out_domain='td',
            min_trunc_len=int(min_filt_trunc_time / dt_down))

        # Normalization factor for each entry to be N(0,1) is
        # 1. / (fmax_psd - fmin_psd) ** 0.5, but we can account for boundary
        # effects. We can also compute from IRFFT in time domain, but this
        # way is robust to transients at the boundaries
        # psd_drift_correction accounts for error due to bandpassing
        nf = np.count_nonzero(np.isfinite(asds_down))
        norm_n01 = ((time_down[-1] - time_down[0]) / nf) ** 0.5
        # Should have been norm_wt = np.sqrt(dt_down * len(time_down) / nf)

        if renorm_wt:
            norm_wt = norm_n01
            # Extra factor to multiply the strain to make each entry N(0, 1)
            sc_n01 = 1.
        else:
            norm_wt = (2. * dt_down)**0.5
            # Extra factor to multiply the strain to make each entry N(0, 1)
            sc_n01 = norm_n01 / norm_wt

    # This flag is used in glitch tests
    renorm_wt = False if notch_wt_filter else renorm_wt

    # Move weight to the left to prepare for overlap-save
    fftsize = len(wt_filter_td)
    wt_filter_fd_fft = utils.RFFT(np.roll(wt_filter_td, support_wt - 1))

    # Safety margin at edges if we have no holes at the edges, accounts for
    # corruption by highpass filter
    edgesafety = support_wt + irl

    # Things that signal-processing will return
    # Whitened strain data
    strain_wt_down = utils.FFTIN(len(time_down))
    # Boolean quality mask sampled at 1/dt_down, can also use repeat,
    # but this is safer if the sampling rate is not an integer
    # Note: Assumes qmask is sampled at 1 Hz
    inds_mask = (time_down - time_down[0]).astype(int)
    qmask_down = mask[inds_mask].astype(bool)

    # Add any holes we want to force
    if times_to_fill is not None:
        for t1, t2 in times_to_fill:
            ind1 = max(0, int(np.ceil(((t1 - time_down[0]) / dt_down))))
            ind2 = max(0, int(np.ceil(((t2 - time_down[0]) / dt_down))))
            qmask_down[ind1:ind2] = 0

    # Record where we don't want to create holes
    #TODO: Make magic eraser use this
    if times_to_save is not None:
        mask_save = utils.FFTIN(len(time_down), dtype=bool)
        mask_save[:] = True
        for interval in times_to_save:
            i1, i2 = np.searchsorted(time_down, interval)
            mask_save[i1:i2] = False
    else:
        mask_save = None

    # Mask before holes were added
    # Initialize to ones to fill holes present in the beginning
    qmask_prev = utils.FFTIN(len(qmask_down), dtype=bool)
    qmask_prev[:] = True

    # Where we cannot trust the overlaps
    valid_mask = utils.FFTIN(len(time_down), dtype=bool)
    valid_mask[:] = True

    # Mask stating why we made the outliers
    # 2 = LIGO mask + sigma_clipping_threshold
    nglitchtests = 2 + utils.safelen(sine_gaussian_intervals) + \
        utils.safelen(bandlim_transient_intervals) + \
        utils.safelen(excess_power_intervals)
    outlier_mask = utils.FFTIN((nglitchtests, len(time_down)), dtype=bool)
    outlier_mask[:] = True

    # Widen bad intervals because of Butterworth. We assume that this fixes
    # the Gibbs phenomenon due to downsampling
    bad_inds = np.where(np.logical_not(qmask_down))[0]
    for ind in bad_inds:
        qmask_down[max(ind - irl, 0): ind + irl] = 0
        outlier_mask[0, max(ind - irl, 0): ind + irl] = 0

    # Treat whether we're filling holes at the edges
    edgefilling = False
    qmask_down_edges = utils.hole_edges(qmask_down)
    left_hole_inds, right_hole_inds = \
        qmask_down_edges[:, 0], qmask_down_edges[:, 1]
    if len(left_hole_inds) > 0 and left_hole_inds[0] == 0:
        if len(right_hole_inds) > 0 and right_hole_inds[-1] == len(qmask_down):
            # We're filling edge holes
            edgefilling = True
            edgesafety = 1
            
    band_erased = False

    if not do_signal_processing:
        # Whiten data with the previously computed whitening filter
        chunked_strain_f = chunkedfft(
            strain_down, fftsize, 2 * support_wt - 1, padmode='center')
        strain_wt_down[:] = norm_wt * overlap_save(
            chunked_strain_f, wt_filter_fd_fft,
            fftsize, 2 * support_wt - 1)[:len(time_down)]
            
        mask_stft = np.ones((int(params.LINE_TRACKING_DT/2/dt_down)+1,
                int(np.ceil(len(strain_wt_down) * dt_down
                /(params.LINE_TRACKING_DT/2)))+1),dtype=bool)

        # Notch filter the whitened data, if we didn't notch the whitening
        # filter (the highpass filter has either been applied already, or is
        # included in the notches)
        if not notch_wt_filter:
            print(f"Notch filtering the whitened strain at a frequency " +
                  f"resolution of {freqs_lines[1] - freqs_lines[0]} Hz")
            for sos_notch, irl_notch in notch_pars:
                strain_wt_down = signal.sosfiltfilt(
                    sos_notch, strain_wt_down, padlen=irl_notch)

        # We don't really care about marking lines, but need to return
        # something sensible
        # mask_freqs = None
        mask_freqs = np.ones_like(freqs_lines, dtype=bool)

        # Zero areas where we cannot be sure we whitened the data
        # We always have holes at the edges, we're not guaranteeing overlaps
        # that advance into these holes
        if not edgefilling and (edgesafety > 1):
            strain_wt_down[:(edgesafety - 1)] = \
                strain_wt_down[-(edgesafety - 1):] = 0

    else:
        # TODO: Define mask_stft and place it in the right place
        # Some parameters for data quality and filling
        irl_to_sec = dt_down * irl
        minclip = params.MIN_CLIPWIDTH
        wlength = max(minclip, irl_to_sec)

        # Convert power intervals into inputs to excess power function
        power_intervals_inp = [[t, [0, np.inf]] for t in excess_power_intervals]

        # List of glitch indices
        glitch_indices = []

        loudoutlier = False
        
        for n in range(params.N_GLITCHREMOVAL + 1):
            # Inpaint holes in place in the unwhitened data
            # We want to go back to the old mode of functioning
            # if loudoutlier
            fill_holes_file(
                strain_down, qmask_down, qmask_prev, valid_mask,
                wt_filter_td, support_wt, erase_bands, 
                band_erased if not loudoutlier else True)
            # Edges of the file should be mandatory zeros
            # and we fiddled with it in fill_holes_file() in an edge case
            # where a large hole appeared at the edge of the file
            if edgefilling and erase_bands:
                qmask_down[0:right_hole_inds[0]]=False
                qmask_down[left_hole_inds[-1]:]=False
            #TODO (Teja): Think about hole filling

            # Record where we filled inside the mask before making new holes,
            # to avoid re-doing work
            qmask_prev[:] = qmask_down[:]
            
            if n==0:
                # Whiten and notch, modify strain_wt_down in place, and identify 
                # varying lines in mask_freqs
                mask_freqs, amp_ratio = gen_whitened_notched_strain(
                    strain_down, wt_filter_fd_fft, valid_mask, strain_wt_down, 
                    fftsize, support_wt, norm_wt, edgefilling=edgefilling, 
                    edgesafety=edgesafety, notch_wt_filter=notch_wt_filter, 
                    notch_pars=notch_pars, detect_varying_lines=True, 
                    qmask_down=qmask_down, dt_down=dt_down, tmax=tmax, 
                    freqs_in=freqs_in, crude_line_mask=crude_line_mask, fmax=fmax)
            else:
                # Whiten and notch, modify strain_wt_down in place
                gen_whitened_notched_strain(
                    strain_down, wt_filter_fd_fft, valid_mask, strain_wt_down, 
                    fftsize, support_wt, norm_wt, edgefilling=edgefilling, 
                    edgesafety=edgesafety, notch_wt_filter=notch_wt_filter, 
                    notch_pars=notch_pars, detect_varying_lines=False)
                        
            # Applying band eraser
            if (n>0) and (not band_erased) and (not loudoutlier):
                if erase_bands:
                    
                    mask_stft = band_eraser(strain_down, strain_wt_down, 
                        qmask_down, valid_mask, dt_down, notch_pars)
                    
                    # notching already done in band_eraser so avoid doing twice
                    # so passed notch_wt_filter=True
                    gen_whitened_notched_strain(
                    strain_down, wt_filter_fd_fft, valid_mask, strain_wt_down, 
                    fftsize, support_wt, norm_wt, edgefilling=edgefilling, 
                    edgesafety=edgesafety, notch_wt_filter=True, 
                    notch_pars=notch_pars, detect_varying_lines=False)
                    
                else:
                    mask_stft = np.ones((int(params.LINE_TRACKING_DT/2/dt_down)+1,
                            int(np.ceil(len(strain_wt_down) * dt_down
                            /(params.LINE_TRACKING_DT/2)))+1),dtype=bool)
                band_erased = True

            # Find outliers and make new holes there
            if n < params.N_GLITCHREMOVAL:
                # Glitch tests in decreasing order of specificity
                # Reject outliers in whitened strain data
                # Warning: Using renorm_wt = False will not numerically
                # reproduce O2 thresholds here
                indices_outlier, whitened_outlier_mask, loudoutlier, _ = \
                    find_whitened_outliers(
                        strain_wt_down, qmask_down, valid_mask,
                        dt_down, wlength,
                        sigma_clipping_threshold=sigma_clipping_threshold,
                        outlier_fraction=params.OUTLIER_FRAC,
                        renorm_wt=renorm_wt, mask_save=mask_save)
                outlier_mask[1, :] = np.logical_and(
                    outlier_mask[1, :], whitened_outlier_mask)

                if not loudoutlier:
                    # Reject sine-Gaussian transients that are not associated
                    # with lines
                    indices_sine_gaussian, sine_gaussian_outlier_mask, _ = \
                        find_sine_gaussian_transients(
                            strain_wt_down, qmask_down, valid_mask, dt_down,
                            params.MIN_CLIPWIDTH, freqs_lines, mask_freqs,
                            sine_gaussian_intervals=sine_gaussian_intervals,
                            sine_gaussian_thresholds=sine_gaussian_thresholds,
                            edgesafety=edgesafety, fftsize=fftsize,
                            renorm_wt=renorm_wt, mask_save=mask_save)
                    outlier_mask[2:2 + len(sine_gaussian_intervals), :] = \
                        np.logical_and(
                            outlier_mask[2:2 + len(sine_gaussian_intervals), :],
                            sine_gaussian_outlier_mask)

                    # Reject intervals with excess power
                    # Find bandlimited transients that are not associated
                    # with lines
                    indices_bandlimited_transient, \
                        bandlim_transient_outlier_mask, _ = \
                        find_excess_power_transients(
                            strain_wt_down, qmask_down, valid_mask, dt_down,
                            excess_power_intervals=bandlim_transient_intervals,
                            excess_power_thresholds=bandlim_power_thresholds,
                            edgesafety=edgesafety, freqs_lines=freqs_lines,
                            mask_freqs=mask_freqs, fmax=fmax,
                            mask_save=mask_save)
                    startind = 2 + len(sine_gaussian_intervals)
                    endind = startind + len(bandlim_transient_intervals)
                    outlier_mask[startind:endind, :] = \
                        np.logical_and(outlier_mask[startind:endind, :],
                                       bandlim_transient_outlier_mask)

                    # Find times with excess power summed over all channels that
                    # are not associated with lines
                    indices_excess_power, excess_power_outlier_mask, _ = \
                        find_excess_power_transients(
                            strain_wt_down, qmask_down, valid_mask, dt_down,
                            excess_power_intervals=power_intervals_inp,
                            excess_power_thresholds=excess_power_thresholds,
                            edgesafety=edgesafety, freqs_lines=freqs_lines,
                            mask_freqs=mask_freqs, fmax=fmax,
                            mask_save=mask_save)
                    startind = 2 + len(sine_gaussian_intervals) + \
                        len(bandlim_transient_intervals)
                    endind = startind + len(excess_power_intervals)
                    outlier_mask[startind:endind, :] = \
                        np.logical_and(outlier_mask[startind:endind, :],
                                       excess_power_outlier_mask)
                else:
                    # We need another pass of sigma clipping
                    indices_sine_gaussian = []
                    indices_bandlimited_transient = []
                    indices_excess_power = []

                glitch_indices += indices_outlier
                glitch_indices += indices_sine_gaussian
                glitch_indices += indices_bandlimited_transient
                glitch_indices += indices_excess_power
                print(n, len(indices_outlier), len(indices_sine_gaussian),
                      len(indices_bandlimited_transient),
                      len(indices_excess_power), loudoutlier, wlength)

                if not loudoutlier:
                    # Increase window length
                    wlength += minclip
                    minclip *= 2

                if ((len(indices_outlier) + len(indices_sine_gaussian) +
                        len(indices_bandlimited_transient) +
                        len(indices_excess_power)) == 0):
                    break
        
        if not band_erased:
            if erase_bands:
                mask_stft = band_eraser(strain_down, strain_wt_down, qmask_down, 
                    valid_mask, dt_down, notch_pars)
                    
                
                # notching already done in band_eraser so avoid doing twice
                # so passed notch_wt_filter=True
                gen_whitened_notched_strain(
                    strain_down, wt_filter_fd_fft, valid_mask, strain_wt_down, 
                    fftsize, support_wt, norm_wt, edgefilling=edgefilling, 
                    edgesafety=edgesafety, notch_wt_filter=True, 
                    notch_pars=notch_pars, detect_varying_lines=False)
            else:
                mask_stft = np.ones((int(params.LINE_TRACKING_DT/2/dt_down)+1,
                            int(np.ceil(len(strain_wt_down) * dt_down
                            /(params.LINE_TRACKING_DT/2)))+1),dtype=bool)

    overlaps_data = valid_mask * qmask_down
    print("Length of corrupted data (s) = ",
          (len(overlaps_data) - np.count_nonzero(overlaps_data)) * dt_down)

    # Mark edges in valid mask
    if edgefilling:
        # We have holes at the edges that we filled. Overlaps that look into
        # the filling are not considered valid
        qmask_down_edges = utils.hole_edges(qmask_down)
        left_hole_inds, right_hole_inds = \
            qmask_down_edges[:, 0], qmask_down_edges[:, 1]
        valid_mask[:right_hole_inds[0]] = 0
        valid_mask[left_hole_inds[-1]:] = 0
    else:
        if edgesafety > 1:
            valid_mask[:(edgesafety - 1)] = valid_mask[-(edgesafety - 1):] = 0

    return time_down, strain_wt_down, qmask_down, valid_mask, freqs_lines, \
        mask_freqs, outlier_mask, wt_filter_td, support_wt, norm_wt, \
         sc_n01, mask_stft
         
         
def gen_whitened_notched_strain(strain_down, wt_filter_fd_fft, valid_mask, 
    strain_wt_down, fftsize, support_wt, norm_wt, edgefilling=True, edgesafety=1, 
    notch_wt_filter=False, notch_pars=None, detect_varying_lines=False,
    **varying_lines_kwargs):
    """
    Whiten and notch, modify strain_wt_down in place, and perhaps identify 
    varying lines in mask_freqs
    :param strain_down: Raw strain data
    :param wt_filter_fd_fft: FFT of wt_filter in TD
    :param valid_mask:
        Boolean mask with zeros where we cannot trust whitened data
    :param strain_wt_down: Modifies in place
    :param support_wt: Support of filter (TD filter has 2 * support - 1 nonzero coeffs)
    :param norm_wt: Normalization factor for whitening: (1 / fmax_psd = 2 * dt)** 0.5
    :param detect_varying_lines: Identify varying lines
    :param edgefilling:  Flag to fill holes at the edges
    :param edgesafety:
        Safety margin at the edge where we cannot trust the nature of the
        whitened data (we haven't applied this to the mask when we get here)
    :param notch_wt_filter:
        Flag to apply notches to the whitening filter as well, setting it to
        True avoids biasing the SNR, but creates artifacts at low frequencies
        (safe to use False as well, since we have PSD drift downstream)
    :param notch_pars: Parameters for applying the notch_wt_filter
    :return: If detecting varying lines
        1. mask_freqs: Boolean mask on freqs with zeros at varying lines
        2. amp_ratio: Array with amplitude scaling factors needed to suppress lines
                      above upper threshold
    """
    # Whiten the data with the previously computed whitening filter
    chunked_strain_f = chunkedfft(
        strain_down, fftsize, 2 * support_wt - 1, padmode='center')
    strain_wt_down[:] = norm_wt * overlap_save(
        chunked_strain_f, wt_filter_fd_fft,
        fftsize, 2 * support_wt - 1)[:len(strain_down)]

    # Zero areas where we cannot be sure of the whitened data, and
    # where hole filling failed
    if not edgefilling and (edgesafety > 1):
        strain_wt_down[:(edgesafety-1)] = \
            strain_wt_down[-(edgesafety-1):] = 0
    strain_wt_down *= valid_mask
    
    # Detect varying lines in the first iteration, with the search
    # restricted to crudely identified lines if user passed them
    if detect_varying_lines:
        qmask_down = varying_lines_kwargs['qmask_down']
        dt_down = varying_lines_kwargs['dt_down']
        tmax = varying_lines_kwargs['tmax']
        freqs_in = varying_lines_kwargs['freqs_in']
        crude_line_mask = varying_lines_kwargs['crude_line_mask']
        fmax = varying_lines_kwargs['fmax']
        freqs_lines, _, _, mask_freqs, _, amp_ratio = \
            specgram_quality(
                strain_wt_down, qmask_down, valid_mask, dt_down,
                tmax, edgesafety, freqs_lines=freqs_in,
                mask_freqs_in=crude_line_mask, fmax=fmax)

    # Notch filter the whitened data, if we didn't notch the
    # whitening filter (the highpass filter has either been applied
    # already, or is included in the notches)
    if not notch_wt_filter:
        #print(f"Notch filtering the whitened strain at a frequency " +
                #f"resolution of {freqs_lines[1] - freqs_lines[0]} Hz")
        for sos_notch, irl_notch in notch_pars:
            strain_wt_down[:] = signal.sosfiltfilt(
                sos_notch, strain_wt_down, padlen=irl_notch)
                
    if detect_varying_lines:
        return mask_freqs, amp_ratio

def fill_holes_file(
        data, qmask, qmask_prev, valid_mask, wt_filter_td, support_wt, 
        erase_bands, band_erased):
    """
    :param data: Array of len(data) with strain data
    :param qmask:
        Boolean mask of len(data) with zeros at holes in unwhitened data
    :param qmask_prev:
        Boolean mask of len(data) with mask before previous instance of hole
        filling
    :param valid_mask:
        Boolean mask with zeros marking where we cannot trust whitened data
    :param wt_filter_td: Time domain whitening filter
    :param support_wt:
        Time domain support of whitening filter (2 * support_wt - 1
        nonzero coeffs)
    :param erase_bands: Boolean flag indicating whether we are erasing bands
    :param band_erased: Boolean flag indicating whether we have erased bands
    :return: Divides hole filling task into segments and passes to appropriate
             routines
    """
    nholeinds = len(qmask) - np.count_nonzero(qmask)

    if nholeinds == 0:
        return
    elif nholeinds < params.NHOLE_BF:
        # With new code, we will never enter here
        # Brute force inversion should be fast enough
        # Check if we need to redo hole filling
        if np.allclose(qmask, qmask_prev):
            return
        wt_filter_td_full = utils.change_filter_times_td(
            wt_filter_td, len(wt_filter_td), len(data))
        wt_filter_fd = utils.RFFT(wt_filter_td_full)
        filleddata = fill_holes_bruteforce(data, qmask, wt_filter_fd)
        data[:] = filleddata[:]
    else:
        # Let's try to break the calculation into manageable chunks
        qmask_edges = utils.hole_edges(qmask)
        left_edges, right_edges = qmask_edges[:, 0], qmask_edges[:, 1]

        # Half support of blueing filter (2 * nblue - 1 nonzero coeffs)
        nblue = 2 * support_wt - 1

        # Gaps between successive holes
        gaps = left_edges[1:] - right_edges[:-1]
        # Breaks where we can split the calculation, starts after first hole
        # and ends before last hole
        breaks = np.where(gaps >= (nblue - 1))[0]

        if ((len(left_edges) > 1) and (len(breaks) == 0) and
                (nholeinds > params.NHOLE_MAX)):
            # Either too many intermittent holes to break the calculation, or
            # several large blocks that talk to each other. We'll just try our
            # luck with the whole segment. Check if we need to redo hole filling
            if np.allclose(qmask, qmask_prev):
                return
            fill_holes_segment(
                data, qmask, valid_mask, wt_filter_td, support_wt, 
                erase_bands, band_erased, qmask_prev)
            return

        # There are places where we can break the calculation
        left_index = max(0, left_edges[0] - nblue + 1)
        for i in range(len(breaks)+1):
            if i == len(breaks):
                # Treat last hole
                right_index = min(len(data), right_edges[-1] + nblue - 1)
            else:
                right_index = min(len(data), right_edges[breaks[i]] + nblue - 1)

            # Check if we need to redo hole filling
            if not np.allclose(qmask[left_index:right_index],
                               qmask_prev[left_index:right_index]):
                # Fill holes in segment
                fill_holes_segment(
                    data[left_index:right_index],
                    qmask[left_index:right_index],
                    valid_mask[left_index:right_index],
                    wt_filter_td, support_wt, erase_bands, band_erased, 
                    qmask_prev[left_index:right_index])

            # Update left index for next break
            if i < len(breaks):
                left_index = max(0, left_edges[breaks[i] + 1] - nblue + 1)

    return


def fill_holes_segment(data_seg, qmask_seg, valid_mask_seg, wt_filter_td,
                       support_wt, erase_bands, band_erased, qmask_prev_seg):
    """
    Fill holes in a segment of data
    :param data_seg: Segment of whitened strained data
    :param qmask_seg: Boolean mask with zeros at holes in unwhitened data
    :param valid_mask_seg: Boolean mask with zeros marking where we cannot
                           trust whitened data
    :param wt_filter_td: Time domain whitening filter
    :param support_wt: Time domain support of whitening filter
                       (2 * support_wt - 1 nonzero coeffs)
    :param erase_bands: Boolean flag indicating whether we are erasing bands
    :param band_erased: Boolean flag indicating whether we have erased bands
    :param qmask_prev_seg: Boolean mask with zeros at holes in unwhitened data
                            with the previous Boolean mask
    :return: Inpaints holes in data in place, expands holes and quality mask
             if needed, and marks valid_mask if we failed
    """
    # First check what we're filling
    holeinds_seg = np.where(np.logical_not(qmask_seg))[0]
    leftind_hole = holeinds_seg[0]
    rightind_hole = holeinds_seg[-1] + 1

    if not np.any(valid_mask_seg[leftind_hole:rightind_hole]):
        # Do not try where we previously failed
        return

    # Expand segment to have length = power of two
    # Number of data elements to be sent for hole filling
    ndata = len(data_seg)
    # Find power of 2 >= ndata, and use it to speed up fft
    nfft = utils.next_power(ndata)

    # Truncate whitening filter onto nfft
    wt_filt_nfft_td = utils.change_filter_times_td(
        wt_filter_td, len(wt_filter_td), nfft)
    wt_filt_nfft_fd = utils.RFFT(wt_filt_nfft_td)

    # Make data and mask arrays of size nfft
    data_nfft = utils.FFTIN(nfft)
    data_nfft[:ndata] = data_seg[:]
    qmask_nfft = np.r_[qmask_seg, np.ones(nfft - ndata, dtype=bool)]

    # First see if we can fill via brute force
    nholeinds_seg = len(holeinds_seg)

    if nholeinds_seg <= params.NHOLE_MAX:
        # We can do it by bruteforce
        filleddata = fill_holes_bruteforce(
            data_nfft, qmask_nfft, wt_filt_nfft_fd)
    else:
        # Find leftmost and rightmost indices of holes and expand
        nholeinds_consecutive = rightind_hole - leftind_hole

        if nholeinds_consecutive > nholeinds_seg:
            
            # Don't expand the holes until band_eraser has had a chance
            if erase_bands and not band_erased:
                warnings.warn("We haven't used band_eraser() yet," + \
                "so not expanding the valid_mask for now.", Warning)
                # We are not going to fill holes here so we are going to reset the mask
                qmask_seg[leftind_hole:rightind_hole] = 1
                qmask_prev_seg[leftind_hole:rightind_hole] = 1
                # TODO: fix outlier_mask to make it consistent with the above two
                return
            # We're expanding the hole
            warnings.warn(
                'Strict hole filling too demanding. Expanding the holes!',
                Warning)
        
        if nholeinds_consecutive >= params.NHOLE_MAX_CONSECUTIVE:
            # Even if we expand the holes, we cannot fill them in a
            # reasonable time. We give up on filling them, and mark
            # invalid locations where we cannot trust the whitened data
            warnings.warn(f"Hole of size {nholeinds_consecutive} too " +
                          "big to fill in a reasonable time!", Warning)

            valid_mask_seg[max(0, leftind_hole - (support_wt - 1)):
                           rightind_hole + support_wt - 1] = 0
            return

        # Fill the expanded hole
        filleddata = fill_hole_consecutive(
            data_nfft, leftind_hole, rightind_hole, wt_filt_nfft_fd)

        # Expand quality mask
        qmask_seg[leftind_hole:rightind_hole] = 0

    # Record filled data
    data_seg[:] = filleddata[:ndata]

    return


def fill_holes_bruteforce(data, qmask, wt_filter_fd):
    """
    :param data: Array with strain data
    :param qmask: Boolean mask with zeros at holes in unwhitened data
    :param wt_filter_fd: Frequency domain whitening filter. Lives in space of
                         rfft(len(data), dt)
    :return: Array of size len(data) with filled data
    """

    # Copy the data
    filleddata = utils.FFTIN(len(data))
    filleddata[:] = data[:]

    hole_inds = np.where(np.logical_not(qmask))[0]
    nholeinds = len(hole_inds)

    filleddata[hole_inds] = 0

    if nholeinds < 1:
        return

    if nholeinds > params.NHOLE_MAX:
        raise RuntimeError("Trying to fill in " + str(nholeinds) +
                           " indices, too much of the data is corrupted!")

    # First, run the square of the whitening filter on the data
    c_inv_td = utils.IRFFT(np.abs(wt_filter_fd) ** 2, n=len(data))
    c_inv_dat = utils.IRFFT(
        utils.RFFT(filleddata) * np.abs(wt_filter_fd) ** 2, n=len(data))

    x, y = np.meshgrid(hole_inds, hole_inds, indexing='ij')

    mat = utils.FFTIN((len(hole_inds), len(hole_inds)))
    mat[:] = c_inv_td[np.abs(x.flatten() - y.flatten())].reshape(
        [len(hole_inds), len(hole_inds)])[:]

    # Solve for filled values, note that at size of the matrix 10**5 x 10**5,
    # solve becomes numerically unstable
    d_fill = np.linalg.solve(mat, c_inv_dat[hole_inds])
    filleddata[hole_inds] = - d_fill

    return filleddata


def fill_hole_consecutive(data, leftind, rightind, wt_filter_fd):
    """
    :param data: Array with strain data
    :param leftind: Left index of hole
    :param rightind: Right index of hole
    :param wt_filter_fd: Frequency domain whitening filter. Lives in space of
                         rfft(len(data), dt)
    :return: Array of size len(data) with filled data
    """

    # Copy the data and zero inside the hole
    filleddata = utils.FFTIN(len(data))
    filleddata[:] = data[:]
    filleddata[leftind:rightind] = 0

    # Run the square of the whitening filter on the data
    c_inv_td = utils.IRFFT(np.abs(wt_filter_fd) ** 2, n=len(data))
    c_inv_dat = utils.IRFFT(
        utils.RFFT(filleddata) * np.abs(wt_filter_fd) ** 2, n=len(data))

    # Shuffle the data inside the hole
    d_fill = scilin.solve_toeplitz(c_inv_td[:(rightind - leftind)],
                                   c_inv_dat[leftind:rightind])
    filleddata[leftind:rightind] = - d_fill

    return filleddata


# Functions to find glitches
# ------------------------------------------------------------------------
# Sigma-clipping
# --------------
def find_whitened_outliers(
        strain_wt, qmask, valid_mask, dt, clipwidth,
        sigma_clipping_threshold=None, outlier_fraction=params.OUTLIER_FRAC,
        zero_mask=True, nfire=params.NPERFILE, renorm_wt=True, sc_n01=1.,
        mask_save=None, verbose=True):
    """
    Zeros mask in place on either side of outliers in whitened data stream
    :param strain_wt: Whitened strain data
    :param qmask: Boolean mask with zeros at holes in unwhitened data
    :param valid_mask:
        Boolean mask with zeros where we cannot trust whitened data
    :param dt: Time interval between successive elements of strain_wt (s)
    :param clipwidth: Half-width of window around outliers to zero (in s)
    :param sigma_clipping_threshold:
        Threshold for sigma clipping computed from waveform. If none, clipping
        depends on params.NPERFILE
    :param outlier_fraction:
        If outlier is more than 1/outlier_fraction x sigma clipping threshold,
        adjust threshold to avoid overcorrecting due to ringing of the outlier
        against the whitening filter
    :param zero_mask:
        Flag indicating whether to zero the mask at glitches. Turn off for
        vetoing
    :param nfire: Number of times glitch detector fires per perfect file
    :param renorm_wt:
        Flag whether we scaled the whitened data to have unit variance after
        highpass
    :param sc_n01:
        Factor to multiply the whitened data with to get each data point to be
        N(0,1), used only if renorm_wt is False (which is the default from now)
    :param mask_save:
        If desired, boolean mask on strain_wt with zeros at data to avoid
    :param verbose: If true, prints details about outlier check
    :return: Zeros mask in place, returns
             1. List of indices that jumped
             2. Mask with zeros where we should zero the data as a result
             3. Flag indicating whether we had an overly loud outlier
             4. String with details of the test
    """
    # Create a copy of the previous mask, and mark where we don't want to look
    mask_dat = qmask.copy()
    mask_dat *= valid_mask
    if mask_save is not None:
        mask_dat *= mask_save

    # Mask where we set what changed in this iteration to zero
    outlier_mask = utils.FFTIN(len(strain_wt), dtype=bool)
    outlier_mask[:] = True

    # Set triggering threshold
    # First set Gaussian noise threshold, controls when the waveform-based
    # threshold is too low
    threshold_from_dist = (utils.threshold_rv(
        stats.chi2, int(params.DEF_FILELENGTH / dt), 1, nfire=nfire)) ** 0.5

    if sigma_clipping_threshold is not None:
        # We have a threshold achieved by the waveform
        threshold_from_wf = (stats.ncx2.isf(
            params.FALSE_NEGATIVE_PROB_POWER_TESTS, 1,
            sigma_clipping_threshold)) ** 0.5
        sigma_outlier = max(threshold_from_wf, threshold_from_dist)
    else:
        sigma_outlier = threshold_from_dist

    if not renorm_wt:
        # The TD strain does not have standard deviation = unity because of
        # highpass, the standard deviation is 1/sc_n01. Scale the threshold
        sigma_outlier /= sc_n01

    # Width of window in indices
    hwclip = int(np.round(clipwidth / dt))

    # Find outliers and add to mask. Don't look inside the previous mask
    # TODO: Can we fail inside a mask?
    loudoutlier = False
    abs_strain = np.abs(strain_wt)
    indices = np.where((abs_strain > sigma_outlier) * mask_dat)[0]

    inds_out = []
    if len(indices) > 0:
        maxoutlier = np.max(abs_strain[indices])

        if (outlier_fraction * maxoutlier) > sigma_outlier:
            sigma_outlier = outlier_fraction * maxoutlier
            loudoutlier = True
            indices = np.where((abs_strain > sigma_outlier) * mask_dat)[0]

        # A temporary variable with mask to apply to indices if needed
        if mask_save is not None:
            mask_temp = utils.FFTIN(len(qmask), dtype=bool)
        else:
            mask_temp = None

        for i in indices:
            nchange = update_masks(
                max(i - hwclip, 0), i + hwclip,
                qmask=qmask if zero_mask else None,
                outlier_mask=outlier_mask, mask_save=mask_save,
                mask_temp=mask_temp)
            if nchange > 0:
                inds_out.append(i)

    emesg = ""
    emesg += f"Looking for outliers in whitened strain data\n"
    max_value = np.max(abs_strain[mask_dat])
    emesg += f"Maximum strain = {max_value:.3g}, " + \
             f"Empirical threshold = {threshold_from_dist:.3g}, " + \
             f"Applied Threshold = {sigma_outlier:.3g}\n"

    if verbose:
        print(emesg)

    return inds_out, outlier_mask, loudoutlier, emesg


# Sine-Gaussian transients in amplitude
# -------------------------------------
def find_sine_gaussian_transients(
        strain_wt, qmask, valid_mask, dt, clipwidth, freqs_lines, mask_freqs,
        sine_gaussian_intervals=None, sine_gaussian_thresholds=None,
        edgesafety=1, fftsize=params.DEF_FFTSIZE, zero_mask=True,
        nfire=params.NPERFILE, renorm_wt=True, mask_save=None, verbose=True):
    """
    Finds sine-gaussian transients in data
    :param strain_wt: Whitened strain data
    :param qmask: Boolean mask with zeros at holes in unwhitened data
    :param valid_mask:
        Boolean mask with zeros where we cannot trust whitened data
    :param dt: Time interval between successive elements of data (s)
    :param clipwidth: Half-width of window around outliers to zero (in s)
    :param freqs_lines: Array with frequencies over which we detected lines
    :param mask_freqs: Boolean mask on freqs with zeros at varying lines
    :param sine_gaussian_intervals:
        Frequency bands within which we look for Sine-Gaussian noise
        [central frequency, df = (upper - lower frequency)] Hz
    :param sine_gaussian_thresholds:
        Amplitude thresholds for sine gaussian transients computed from
        waveforms. If None, detection depends on params.NPERFILE
    :param edgesafety:
        Safety margin at the edge where we cannot trust the nature of the
        whitened data (we haven't applied this to the mask when we get here)
    :param fftsize: FFTsize for overlap-save
    :param zero_mask:
        Flag indicating whether to zero the mask at glitches. Turn off for
        vetoing
    :param nfire: Number of times glitch detector fires per perfect file
    :param renorm_wt:
        Flag whether we scaled the whitened data to have unit variance after
        highpass
    :param mask_save:
        If desired, boolean mask on strain_wt with zeros at data to avoid
    :param verbose: If true, prints details about sine-Gaussian check
    :return: Zeros qmask in place, and returns
             1. List of indices that jumped
             2. Masks with zero where we zeroed the data, for each interval
             3. String with details of the test
    """
    # Mask where we set what changed in this iteration to zero
    outlier_mask = utils.FFTIN(
        (len(sine_gaussian_intervals), len(strain_wt)), dtype=bool)
    outlier_mask[:] = True
    emesg = ""

    if sine_gaussian_intervals is None:
        return [], outlier_mask, emesg

    # Create a copy of the previous mask, and mark where we don't want to look
    mask_dat = qmask.copy()
    if edgesafety > 1:
        mask_dat[:(edgesafety - 1)] = mask_dat[-(edgesafety - 1):] = 0
    mask_dat *= valid_mask
    if mask_save is not None:
        mask_dat *= mask_save

    indices = []
    for tr_ind, (fc_sg, df_sg) in enumerate(sine_gaussian_intervals):
        # Compute scores
        scores, support_sg, scores_cos, scores_sin = \
            calculate_sine_gaussian_overlaps(
                strain_wt, dt, fc_sg, df_sg, freqs_lines, mask_freqs, fftsize)

        # Set thresholds on scores
        # First set Gaussian noise threshold, controls when the waveform-based
        # thresholds are too low
        nindsamples = int(params.DEF_FILELENGTH / ((2 * support_sg - 1) * dt))
        threshold_from_dist = (utils.threshold_rv(
            stats.chi2, nindsamples, 2, nfire=nfire)) ** 0.5

        # Inflate threshold of distribution to account for residual PSD drift
        threshold_from_dist *= (1. + params.PSD_DRIFT_SAFETY)

        if sine_gaussian_thresholds is not None:
            # We have a threshold achieved by the waveform
            non_centrality_parameter = sine_gaussian_thresholds[tr_ind]
            threshold_from_wf = (stats.ncx2.isf(
                params.FALSE_NEGATIVE_PROB_POWER_TESTS, 2,
                non_centrality_parameter)) ** 0.5
            sine_gaussian_threshold = max(
                threshold_from_dist, threshold_from_wf)
        else:
            sine_gaussian_threshold = threshold_from_dist

        if renorm_wt:
            # Note: This leads to a bug when there are many zeros in the file
            # Going forward, we are going to keep renorm_wt = False, so this is
            # a historical record to reproduce old runs
            # We rescaled the strain to have TD variance = unity even after
            # bandpassing, so the scores don't have variance = 1
            cos_sigma = utils.sigma_from_median(scores_cos)
            sin_sigma = utils.sigma_from_median(scores_sin)
            abs_std = (0.5 * (cos_sigma ** 2 + sin_sigma ** 2))**0.5
            sine_gaussian_threshold *= abs_std

        emesg_curr = ""
        emesg_curr += f"Looking for sine-Gaussians around {fc_sg} Hz, " + \
                      f"bandwidth {df_sg} Hz\n"
        max_sg_score = np.max(scores[mask_dat])
        emesg_curr += f"Maximum score = {max_sg_score:.3g}, " + \
                      f"Empirical threshold = {threshold_from_dist:.3g}, " + \
                      f"Applied Threshold = {sine_gaussian_threshold:.3g}\n"

        if verbose:
            print(emesg_curr)

        # Zero original mask without triggering on already bad seconds
        # Width of window in indices
        hwclip = int(np.round(max(clipwidth, 1/df_sg) / dt))
        local_indices = list(
            np.where((scores > sine_gaussian_threshold) * mask_dat)[0])

        # A temporary variable with mask to apply to indices if needed
        if mask_save is not None:
            mask_temp = utils.FFTIN(len(qmask), dtype=bool)
        else:
            mask_temp = None

        for ind in local_indices:
            nchange = update_masks(
                max(ind - hwclip, 0), ind + hwclip,
                qmask=qmask if zero_mask else None,
                outlier_mask=outlier_mask[tr_ind], mask_save=mask_save,
                mask_temp=mask_temp)
            if nchange > 0:
                indices.append(ind)

        # indices += local_indices
        emesg += emesg_curr

    return indices, outlier_mask, emesg


def calculate_sine_gaussian_overlaps(
        strain_wt, dt, fc_sg, df_sg, freqs_lines=None, mask_freqs=None,
        fftsize=params.DEF_FFTSIZE):
    """
    Returns sine gaussian scores
    :param strain_wt: Whitened strain data
    :param dt: Time interval between successive elements of data (s)
    :param fc_sg: Central frequency of sine-Gaussian band (Hz)
    :param df_sg: df = Upper - lower frequency of sine-Gaussian (Hz)
    :param freqs_lines: Frequencies for line identification
    :param mask_freqs: Mask on freqs with zeros at varying lines
    :param fftsize: FFTsize for overlap-save
    :return:
        1. Abs of sine-gaussian overlaps
        2. Half support of sine-gaussian filter
        3. Cosine overlaps
        4. Sine overlaps
    """
    # First notch-filter the data to remove varying lines. Filter looks inside
    # the holes, but the time-domain support is small, and if we filled the
    # holes responsibly, there are no large artifacts. We use this only to
    # detect bad regions
    strain_wt_notched = strain_wt
    if (freqs_lines is not None) and (mask_freqs is not None):
        flow, fhigh = fc_sg - df_sg / 2, fc_sg + df_sg / 2
        notch_pars = utils.notch_filter_sos(
            dt, freqs_lines, mask_freqs, flow=flow, fhigh=fhigh)
        for sos_notch, irl_notch in notch_pars:
            strain_wt_notched = signal.sosfiltfilt(
                sos_notch, strain_wt_notched, padlen=irl_notch)

    # Then compute the cosine and sine score matches
    # First compute pulses
    cos_pulse, sin_pulse, support_sg = utils.sine_gaussian(dt, fc_sg, df_sg)

    # Change them to the right fft size and put weight in the right place
    cos_pulse_td_fft = np.roll(
        utils.change_filter_times_td(
            cos_pulse, len(cos_pulse), fftsize),
        support_sg - 1)
    sin_pulse_td_fft = np.roll(
        utils.change_filter_times_td(
            sin_pulse, len(sin_pulse), fftsize),
        support_sg - 1)
    cos_pulse_fd_fft = utils.RFFT(cos_pulse_td_fft)
    sin_pulse_fd_fft = utils.RFFT(sin_pulse_td_fft)

    # Prepare data for overlap save
    chunked_strain_wt_notched = chunkedfft(
        strain_wt_notched, fftsize, 2 * support_sg - 1, padmode='center')

    # Compute overlaps and scores
    overlaps_cos = overlap_save(
        chunked_strain_wt_notched, cos_pulse_fd_fft,
        fftsize, 2 * support_sg - 1)[:len(strain_wt_notched)]
    overlaps_sin = overlap_save(
        chunked_strain_wt_notched, sin_pulse_fd_fft,
        fftsize, 2 * support_sg - 1)[:len(strain_wt_notched)]
    scores = np.sqrt(overlaps_cos ** 2 + overlaps_sin ** 2)

    return scores, support_sg, overlaps_cos, overlaps_sin


# Band-limited transients in power
# ------------------------------------
def find_excess_power_transients(
        strain_wt, qmask, valid_mask, dt, excess_power_intervals=None,
        excess_power_thresholds=None, edgesafety=1, freqs_lines=None,
        mask_freqs=None, zero_mask=True, nfire=params.NPERFILE,
        fmax=params.FMAX_OVERLAP, mask_save=None, verbose=True):
    """
    Detects bandlimited excess power transients
    :param strain_wt: Whitened strain data
    :param qmask: Boolean mask with zeros at holes in unwhitened data
    :param valid_mask:
        Boolean mask with zeros where we cannot trust whitened data
    :param dt: Time interval between successive elements of data (s)
    :param excess_power_intervals:
        Array with set of time-interval-frequency intervals (s, Hz)
        [[dt_i, [f_i_min, f_i_max]],...]. Pass f_min = 0, f_max = np.inf for
        total power
    :param excess_power_thresholds:
            Thresholds for excess power computed from waveforms. If None,
            detection depends on params.NPERFILE. The thresholds correspond
            to squared sum of bandpassed data
    :param edgesafety:
        Safety margin at the edge where we cannot trust the nature of the
        whitened data (we haven't applied this to the mask when we get here)
    :param freqs_lines:
        Array with frequencies on which we previously detected lines (optional)
    :param mask_freqs:
        Boolean mask on freqs_lines with zeros at previously detected varying
        lines (optional)
    :param zero_mask:
        Flag indicating whether to zero the mask at glitches. Turn off for
        vetoing
    :param nfire: Number of times glitch detector fires per perfect file
    :param fmax: Maximum frequency involved in the analysis
    :param mask_save:
        If desired, boolean mask on strain_wt with zeros at data to avoid
    :param verbose: If true, would print details on power detector
    :return: Zeros qmask in place, and returns
             1. List of indices that jumped
             2. Masks with zero where we zeroed the data, for each interval
             3. String with details of the test
    """
    # Mask where we set what changed in this iteration to zero
    outlier_mask = utils.FFTIN(
        (len(excess_power_intervals), len(strain_wt)), dtype=bool)
    outlier_mask[:] = True
    emesg = ""

    if excess_power_intervals is None:
        return [], outlier_mask, emesg

    indices = []
    for tr_ind, (interval, frng) in enumerate(excess_power_intervals):
        excess_power_out = calculate_excess_power(
            strain_wt, qmask, valid_mask, dt, interval, frng,
            edgesafety=edgesafety, freqs_lines=freqs_lines,
            mask_freqs_in=mask_freqs, fmax=fmax, verbose=verbose)

        if excess_power_out is None:
            continue

        times, excess_power, mask_times, times_excess_power, \
            robust_excess_power, mask_excess_power, ndof = excess_power_out

        # Set threshold for excess power over mean
        # First set Gaussian noise threshold, controls when the waveform-based
        # thresholds are too low
        nindsamples = int(params.DEF_FILELENGTH / interval)
        threshold_from_dist = utils.threshold_rv(
            stats.chi2, nindsamples, ndof, nfire=nfire) - ndof

        # Moving average helps, but not fully, hence inflate the threshold to
        # account for residual PSD drift
        # TODO: Make this consistent with amplitude rather than power
        #  Barak: Why? this looks good
        threshold_from_dist += params.PSD_DRIFT_SAFETY * ndof

        if excess_power_thresholds is not None:
            # We have thresholds achieved by the waveforms
            non_centrality_parameter = excess_power_thresholds[tr_ind]
            threshold_from_wf = stats.ncx2.isf(
                params.FALSE_NEGATIVE_PROB_POWER_TESTS, ndof,
                non_centrality_parameter) - ndof
            excess_power_threshold = max(
                threshold_from_dist, threshold_from_wf)
        else:
            excess_power_threshold = threshold_from_dist

        emesg_curr = ""
        emesg_curr += f"Looking for excess power in {interval} s, {frng} Hz\n"
        max_excess_power = np.max(robust_excess_power[mask_excess_power])
        emesg_curr += f"Maximum power = {max_excess_power:.3g}, " + \
                      f"Gaussian threshold = {threshold_from_dist:.3g}, " + \
                      f"Applied Threshold = {excess_power_threshold:.3g}\n"

        # globals()["robust_excess_power"] = robust_excess_power
        # globals()["ndof"] = ndof

        if ((type(verbose) == bool) and verbose) or (verbose > 1):
            print(emesg_curr, flush=True)

        # Zero original mask without triggering on already bad seconds
        local_indices = list(
            np.where((robust_excess_power > excess_power_threshold) *
                     mask_excess_power)[0])
        n_inds = int(interval / dt)

        # A temporary variable with mask to apply to indices if needed
        if mask_save is not None:
            mask_temp = utils.FFTIN(len(qmask), dtype=bool)
        else:
            mask_temp = None

        for obs_ind in local_indices:
            data_ind = times_excess_power[obs_ind] / dt
            left_ind = max(0, int(np.round(data_ind - n_inds / 2)))
            right_ind = int(np.round(data_ind + n_inds / 2))
            nchange = update_masks(
                left_ind, right_ind,
                qmask=qmask if zero_mask else None,
                outlier_mask=outlier_mask[tr_ind], mask_save=mask_save,
                mask_temp=mask_temp)
            if nchange > 0:
                indices.append(obs_ind)

        # We lost a bit due to the moving average at the edges, most likely
        # invalid anyway due to edgesafety. Look anyway to be safe
        half_avg_len = AVG_LEN // 2
        times_edges = times[:half_avg_len] + times[-half_avg_len:]
        mask_times_edges = mask_times[:half_avg_len] + \
            mask_times[-half_avg_len:]
        excess_power_edges = excess_power[:half_avg_len] + \
            excess_power[-half_avg_len:]
        edge_local_indices = list(
            np.where((excess_power_edges > excess_power_threshold) *
                     mask_times_edges)[0])
        for obs_ind in edge_local_indices:
            data_ind = times_edges[obs_ind] / dt
            left_ind = max(0, int(np.round(data_ind - n_inds / 2)))
            right_ind = int(np.round(data_ind + n_inds / 2))
            nchange = update_masks(
                left_ind, right_ind,
                qmask=qmask if zero_mask else None,
                outlier_mask=outlier_mask[tr_ind], mask_save=mask_save,
                mask_temp=mask_temp)
            if nchange > 0:
                indices.append(obs_ind)

        # indices += local_indices + edge_local_indices
        emesg += emesg_curr

    return indices, outlier_mask, emesg


def calculate_excess_power(
        strain_wt, qmask, valid_mask, dt, interval, frng, edgesafety=1,
        freqs_lines=None, mask_freqs_in=None, fmax=params.FMAX_OVERLAP,
        verbose=True, **spec_kwargs):
    """
    Returns samples of excess power with and without a rolling average
    :param strain_wt: Whitened strain data
    :param qmask: Boolean mask with zeros at holes in unwhitened data
    :param valid_mask:
        Boolean mask with zeros where we cannot trust whitened data
    :param dt: Time interval between successive elements of data (s)
    :param interval: Time interval to look for excess power on (s)
    :param frng: Frequency interval to look for excess power in (Hz).
                 Pass [0, np.inf] for total power
    :param edgesafety:
        Safety margin at the edge where we cannot trust the nature of the
        whitened data (we haven't applied this to the mask when we get here)
    :param freqs_lines:
        Array with frequencies on which we previously detected lines (optional)
    :param mask_freqs_in:
        Boolean mask on freqs_lines with zeros at previously detected varying
        lines (optional)
    :param fmax: Maximum frequency involved in the analysis
    :param verbose: If true, would print details on power detector
    :return: Zeros qmask in place, and returns list of indices that jumped
    """
    # Compute spectrogram with time and frequency masks
    freqs, times, spgram, mask_freqs, mask_times, _ = \
        specgram_quality(
            strain_wt, qmask, valid_mask, dt, interval, edgesafety=edgesafety,
            freqs_lines=freqs_lines, mask_freqs_in=mask_freqs_in, fmax=fmax,
            **spec_kwargs)
    n_inds = int(interval / dt)

    # Restrict to channels in analysis range
    valid_channels = (freqs > params.FMIN_ANALYSIS) * (freqs <= fmax)
    mask_freqs *= valid_channels

    if (frng[0] == 0) and (frng[1] == np.inf):
        # We're doing an excess power correction
        band_mask = mask_freqs
    else:
        # We're finding bandlimited transients
        # Check that the interval resolves the transient
        if interval < (2 / (frng[1] - frng[0])):
            raise RuntimeError("Banded transient is underresolved")

        # We cut at half frequency, so we get the power in each band
        tfft = n_inds * dt
        flow = np.round(frng[0] * tfft) / tfft
        fhigh = np.round(frng[1] * tfft) / tfft
        band_mask = (freqs >= flow) * (freqs <= fhigh)
        # Remove lines and impose frequency bounds
        band_mask *= mask_freqs

    if np.count_nonzero(band_mask) == 0:
        if verbose:
            print("Not a single valid frequency channel available for " +
                  f"dt = {interval} s, f in {frng} Hz")
        return None

    # Collapse frequency direction into observable that approximately
    # behaves like a chisquare dist with 2 * (n_freq channels) dof
    # This observable behaves like \sum_{n in interval} band_passed_strain^2(n)
    # Fixes scaling factor due to effect of highpass filter
    fs = int(1 / dt)
    obs = np.sum(spgram[band_mask, :], axis=0) * fs
    ndof = 2 * np.count_nonzero(band_mask)
    scfac = stats.chi2.median(ndof) / np.median(obs[mask_times])
    obs *= scfac

    # Excess power over mean
    # This catches outliers if the file is largely good + a little bad
    excess_power = obs - ndof

    # Moving average, slightly modifies statistics, but guards power detection
    # against long modes that are better left to the PSD drift correction
    times_excess_power, robust_excess_power, mask_excess_power = \
        robust_power_filter(
            times, excess_power, mask_times, AVG_LEN, int(OVERSUB) - 1)

    return times, excess_power, mask_times, times_excess_power, \
        robust_excess_power, mask_excess_power, ndof


def specgram_quality(
        strain_wt, qmask, valid_mask, dt, interval, edgesafety=1,
        nfire=params.NPERFILE, freqs_lines=None, mask_freqs_in=None,
        fmax=params.FMAX_OVERLAP, **spec_kwargs):
    """
    Computes specgram of whitened data, along with list of bad time and
    frequency channels in the specgram (bad time channels are windows that
    overlap with zeros in mask, and bad frequency channels are varying lines or
    those beyond the analysis range)
    :param strain_wt: Whitened strain data
    :param qmask: Boolean mask with zeros at holes in unwhitened data (we have
                  already widened holes due to the highpass filter)
    :param valid_mask:
        Boolean mask with zeros where we cannot trust whitened data
    :param dt: Time interval between successive elements of strain_wt (s)
    :param interval: Time interval for FFTs (s, sets resolution of lines)
    :param edgesafety:
        Safety margin at the edge where we cannot trust the nature of the
        whitened data (we haven't applied this to the mask when we get here)
    :param nfire: Number of times line-detector fires per perfect file
    :param freqs_lines:
        Array with frequencies on which we previously detected lines (optional)
    :param mask_freqs_in:
        Boolean mask on freqs_lines with zeros at previously detected
        lines/varying lines (optional)
    :param fmax: Maximum frequency involved in the analysis
    :return: 1. Array of length n_freqs with frequencies (Hz)
             2. Array of length n_times with times (s)
             3. n_freqs x n_times array with specgram (two-sided psd in Hz^-1)
             4. Boolean array of length n_freqs with zeros at candidate
                varying line channels
             5. Boolean array of length n_times with zeros where FFTs see
                holes/invalid data
             6. Array with amplitude scaling factors needed to suppress lines
                above upper threshold
    """
    # First define a mask with edges and failed regions marked
    mask_dat = qmask.copy()
    if edgesafety > 1:
        mask_dat[:(edgesafety - 1)] = mask_dat[-(edgesafety - 1):] = 0
    mask_dat *= valid_mask

    n_inds = int(interval / dt)
    if n_inds > len(strain_wt):
        raise RuntimeError("What are you doing giving me such a wide interval?")

    # Returns two-sided PSD. For real-valued and whitened data, each entry is
    # the sum of squares of two gaussian RVs, each of which has mean zero and
    # variance 1/(2*f_max) = 1/fs. Window functions spoil this, but the default
    # Tukey in the spectrogram is quite good with white noise, and probably ok
    # with the lines?
    fs = int(1 / dt)
    freqs, times, spgram = signal.spectrogram(
        strain_wt, fs=fs, nperseg=n_inds,
        noverlap=int(n_inds * params.OVERLAP_FAC), mode='psd', **spec_kwargs)

    # Spectrogram is unreliable around pre-existing holes. Build Boolean mask
    # indexing good times
    mask_times = np.zeros_like(times, dtype=bool)
    for mask_ind, candidate_time in enumerate(times):
        data_ind = candidate_time / dt
        left_ind = max(0, int(np.round(data_ind - n_inds / 2)))
        right_ind = int(np.round(data_ind + n_inds / 2))
        mask_times[mask_ind] = np.prod(mask_dat[left_ind: right_ind])

    if np.count_nonzero(mask_times) < OVERSUB:
        warnings.warn(
            "Not enough independent intervals available for power detector!",
            Warning)
        mask_freqs = np.ones_like(freqs, dtype=bool)
        amplitude_ratios = np.ones_like(freqs)
        return freqs, times, spgram, mask_freqs, mask_times, amplitude_ratios

    # Identify bands in which the mean varies above Gaussian level
    # First multiply by fs so that each entry is chi2 with 2 dof, sum over
    # times ideally chi2 distributed with (2 n_good times) / oversub dof
    ndof = int(2 * np.count_nonzero(mask_times) / OVERSUB)
    tot_whitened_power = np.sum(spgram[:, mask_times], axis=-1) * fs / OVERSUB

    # This will be mostly flat, and rise by a large amount near varying lines
    # These variations should not cause a problem since the PSD is high there
    # anyway. The typical value at the floor might be different from the chi2
    # expectation due to highpass filtering. Rescale to the right normalization
    scfac = stats.chi2.median(ndof) / np.median(tot_whitened_power)
    tot_whitened_power *= scfac

    # Avoid low and high frequencies
    valid_channels = (freqs > params.FMIN_ANALYSIS) * (freqs <= fmax)
    nvalid_channels = np.count_nonzero(valid_channels)

    # Find frequencies outside upper and lower limits for variability
    tot_whitened_power_low, tot_whitened_power_high = utils.threshold_rv(
        stats.chi2, nvalid_channels, ndof, onesided=False, nfire=nfire)
    upper_line_freqs = (
            (tot_whitened_power >= tot_whitened_power_high) * valid_channels)
    lower_line_freqs = (
            (tot_whitened_power <= tot_whitened_power_low) * valid_channels)
    mask_freqs = np.logical_or(upper_line_freqs, lower_line_freqs)

    # Identify as varying lines only if the frequencies have been identified as
    # lines, if done previously. Overwrites mask_freqs
    if (freqs_lines is not None) and (mask_freqs_in is not None):
        # Assume we are not applying tests on finer frequency scale than used
        # to measure the PSD. Define coarse mask with zeros at lines
        mask_lines_coarse = utils.define_coarser_mask(
            freqs_lines, mask_freqs_in, freqs)
        mask_freqs *= np.logical_not(mask_lines_coarse)

    # Make it so that lines have zeros
    mask_freqs = np.logical_not(mask_freqs)

    amplitude_ratios = np.ones_like(freqs)
    amplitude_ratios[upper_line_freqs] = np.sqrt(
        tot_whitened_power / tot_whitened_power_high)[upper_line_freqs]

    return freqs, times, spgram, mask_freqs, mask_times, amplitude_ratios


def robust_power_filter(
        times, power_sequence, mask_times, avg_len, avoid_len=1):
    """
    :param times: Array with central times of intervals in which power was
                  measured
    :param power_sequence: Array with measured power in intervals
    :param mask_times: Boolean mask indicating whether interval sees an
                       existing hole
    :param avg_len: Number of intervals that moving average test spans
    :param avoid_len: Excess power is correlated, avoid these many intervals
                      around central one
    :return: Array with central times, excess power with moving average
             subtracted, and Boolean mask into times
    """
    half_avg_len = avg_len // 2
    filt = np.ones(half_avg_len * 2 + 1)
    filt[half_avg_len - avoid_len:half_avg_len + avoid_len + 1] = 0
    nsamp = np.convolve(mask_times, filt, 'valid')
    mask = nsamp.astype(int) == 0
    avg_seq = np.zeros_like(nsamp)
    avg_seq[~mask] = np.convolve(power_sequence * mask_times, filt, 'valid')[~mask]\
                         / nsamp[~mask]
    robust_excess_power = power_sequence[half_avg_len:-half_avg_len] - avg_seq
    return times[half_avg_len:-half_avg_len], robust_excess_power, \
        mask_times[half_avg_len:-half_avg_len]


def update_masks(
        ileft, iright, qmask=None, outlier_mask=None, mask_save=None,
        mask_temp=None):
    """
    Convenience function to update a mask in a range, record where we nulled,
    and avoid some regions if needed
    :param ileft: Index of left edge of region to null
    :param iright: Index of right edge of region to null (not inclusive)
    :param qmask: Global boolean mask to update by nulling, if needed
    :param outlier_mask:
        Boolean array to null to record where we ended up nulling this round,
        if needed
    :param mask_save:
        Boolean array with zeros at indices that we want to exempt from nulling
    :param mask_temp: Boolean array for working memory
    :return: Modifies input masks in place, returns number of indices updated
    """
    if qmask is None and outlier_mask is None:
        # Nothing to do
        return 0

    if mask_save is not None:
        if mask_temp is None:
            mask_temp = utils.FFTIN(len(mask_save), dtype=bool)
        else:
            mask_temp[:] = False

        mask_temp[ileft: iright] = True
        mask_temp *= mask_save
        if qmask is not None:
            qmask[mask_temp] = False
        if outlier_mask is not None:
            outlier_mask[mask_temp] = False
        out = np.count_nonzero(mask_temp)
    else:
        if qmask is not None:
            qmask[ileft: iright] = False
        if outlier_mask is not None:
            outlier_mask[ileft: iright] = False
        out = iright - ileft

    return out
    
    
def band_eraser(
        strain_down, strain_wt_down, qmask_down, valid_mask, dt_down, notch_pars):
    """
    Detects bandlimited excess power transients
    :param strain_down: Raw strain data, modifies in place 
                        (bands with high-power cells will be removed)
    :param strain_wt_down: Modifies in place
    :param qmask_down: Boolean mask with zeros at holes in unwhitened data
    :param valid_mask:
        Boolean mask with zeros where we cannot trust whitened data
    :param dt_down: Time interval between successive elements of data (s)
    :param notch_pars: Parameters for notching loud lines from the strain
    :return: mask_stft: Mask with excess power bands removed
    """
    T = params.LINE_TRACKING_DT
    chunk_size = params.LINE_TRACKING_TIME_SCALE // params.LINE_TRACKING_DT
    hole_mask = qmask_down * valid_mask
    
    # In process
    len_data = len(strain_wt_down)
    len_cell = int(T / dt_down / 2)
    
    # if the data is not in exact multiples of cell size, append the data on the left
    if (len_data%len_cell) > 0:
        strain_wt_down = np.append(np.zeros(
                                len_cell - (len_data%len_cell)), strain_wt_down)
        strain_down_expanded = np.append(np.zeros(
                                len_cell - (len_data%len_cell)), strain_down)
    else:
        strain_down_expanded = strain_down.copy()
    
    f_stft, t_stft, strain_stft = signal.stft(strain_wt_down, nperseg=T/dt_down,
                                          noverlap=(T/dt_down)//2, fs=1/dt_down)
    
    # Calculate indices of freq bins of bands
    f_array = np.arange(16, 512, params.LINE_TRACKING_DF)
    ind_min = np.zeros_like(f_array); ind_max = np.zeros_like(f_array);
    for i in range(len(f_array)):
        f_max = f_array[i] + params.LINE_TRACKING_DF
        ind_min[i] = np.searchsorted(f_stft, f_array[i])
        ind_max[i] = np.searchsorted(f_stft, f_max)

        
    survival_scale = 7.824 # = stats.chi2.isf(0.02,2), 0.02 = prob, 2 = d.o.f
    mask_stft = np.ones_like(strain_stft, dtype=bool)

    # Finding which of the stft elements overlap with holes
    stft_overlap_hole = np.ones(len(t_stft), dtype=bool)
    for i in range(len(t_stft)-1):
        # only register if the hole is > 0.01 s
        if np.sum(~hole_mask[int(t_stft[i]/dt_down):
         int(t_stft[i+1]/dt_down)]) > 0.01/dt_down:
            stft_overlap_hole[i:i+2] = False # Flag the two bins which contain the hole
            
    for chunk in range(0, len(t_stft)-chunk_size, 1):
    
        # STFT of strain for the time chunk and remove cells which overlap with holes
        strain_stft_chunk = (strain_stft.T[chunk:chunk+chunk_size]
                                [stft_overlap_hole[chunk:chunk+chunk_size]]).T
                            
        # Calculating the median over a time chunk
        sigmaSQ = np.median(np.abs(strain_stft_chunk)**2) \
                        / median_bias(np.prod(strain_stft_chunk.shape)) / 2
                        
        poisson_lambda = 0.02 * (4 * len(strain_stft_chunk[0]))
        poisson_threshold = stats.poisson.isf(
                            0.1 / (1/dt_down * 4096), poisson_lambda)
        # rate in Gaussian noise should be 1 per 10 files
        
        for i in range(len(f_array)):
            band = strain_stft_chunk[ind_min[i]:ind_max[i]]
            bad_inds = (np.abs(band)**2) > sigmaSQ * survival_scale
            # flag the lines in mask_stft
            if np.sum(bad_inds) > poisson_threshold:
                mask_stft[ind_min[i]-2:ind_max[i]+2,
                          chunk-chunk_size//4:chunk+chunk_size+chunk_size//4] = 0
                # removing some extra data on the left and right to be conservative
                        
    # Notching the unwhitened strain before using mask_stft
    # to prevent artifacts
    for sos_notch, irl_notch in notch_pars:
        strain_down_expanded = signal.sosfiltfilt(
            sos_notch, strain_down_expanded, padlen=irl_notch)
    
    _, _, strain_stft = signal.stft(strain_down_expanded, nperseg=T/dt_down,
                                          noverlap=(T/dt_down)//2, fs=1/dt_down)
    strain_down_expanded = signal.istft(mask_stft * strain_stft, fs=1/dt_down,
                            noverlap=(T/dt_down)//2)[1]
                            
    strain_down[:] = strain_down_expanded[-len_data:]
    
    print('Used band eraser')
    return mask_stft
    
     

# if __name__ == "__main__":
    # # Check of excess power detector
    # import matplotlib.pyplot as plt
    # interval = 1
    # frng = [55, 65]
    # dt = 2**-10
    # n_inds = int(interval / dt)
    # b, a, irl = utils.band_filter(dt, frng[0], frng[1])
    # strain_wt = np.random.randn(2 ** 20)
    # transient = signal.filtfilt(b, a, 10 * np.random.randn(n_inds))
    # apod = np.exp(-0.5 * ((np.arange(n_inds) - n_inds // 2)/(n_inds * 0.2))**2)
    # strain_wt[2**19: 2**19 + n_inds] += transient * apod
    # qmask = np.ones_like(strain_wt, dtype=bool)
    # valid_mask = qmask.copy()
    # edgesafety = 1
    # freqs, times, spgram, mask_freqs, mask_times = specgram_quality(
    #         strain_wt, qmask, valid_mask, dt, interval, edgesafety)
    # if (frng[0] == 0) and (frng[1] == np.inf):
    #     band_mask = mask_freqs
    # else:
    #     if interval < (2 / (frng[1] - frng[0])):
    #         raise RuntimeError("Banded transient is underresolved")
    #     tfft = n_inds * dt
    #     flow = np.round(frng[0] * tfft) / tfft
    #     fhigh = np.round(frng[1] * tfft) / tfft
    #     band_mask = (freqs >= flow) * (freqs <= fhigh)
    #     band_mask *= mask_freqs
    # fs = int(1 / dt)
    # obs = np.sum(spgram[band_mask, :], axis=0) * fs
    # ndof = 2 * np.count_nonzero(band_mask)
    # scfac = stats.chi2.median(ndof) / np.median(obs[mask_times])
    # obs *= scfac
    # nindsamples = int(params.DEF_FILELENGTH / interval / 8)
    # excess_power_threshold = utils.threshold_rv(
    #     stats.chi2, nindsamples, ndof) - ndof
    # excess_power = obs - ndof
    # plt.hist(excess_power[mask_times], bins=100, histtype='step', density='True')
    # plt.gca().set_yscale('log')
    # plt.plot(np.arange(-10, 10), stats.chi2.pdf(np.arange(-10, 10) + ndof, ndof))
    # plt.axvline(excess_power_threshold, c='k', ls='--')

pass
