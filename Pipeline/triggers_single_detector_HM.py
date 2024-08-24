import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import params
import utils
import data_operations as d_ops
import template_bank_generator_HM as tg
import multiprocess as mp
import json
import copy
import time
import warnings
from collections import defaultdict
import scipy.signal as signal
import scipy.stats as stats
import itertools
import sys

# as in ComputeWF_array.CONSTANTS
DEFAULT_INCLINATION = 0
DEFAULT_PHIREF = 0

DEFAULT_QUALITY_FLAGS = ('DEFAULT',)
# Note: we will also catch LIGO's injections
# SUGGESTED_QUALITY_FLAGS = ('DEFAULT', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3')
# Change in O3a
SUGGESTED_QUALITY_FLAGS = ('DEFAULT', 'CBC_CAT1')
INJ_FLAGS = ('NO_CBC_HW_INJ', 'NO_BURST_HW_INJ', 'NO_DETCHAR_HW_INJ',
             'NO_STOCH_HW_INJ')
DEFAULT_TEMPLATE_CONF = utils.DATA_ROOT + \
                        "/templates/TaylorF2_20_BNS/metadata.json"
DEFAULT_BH_TEMPLATE_ROOT = utils.DATA_ROOT + \
                        "/templates/BH_fmin20_fmax512_6bins_multibanks"


def ensure_abspath(config_fname, fname):
    """
    Ensures that the path to fname is absolute, assumes it is saved
    in the same directory as config_fname!
    :param config_fname: Absolute path to config.json file
    :param fname: Absolute or relative path to file
    :return: Absolute path to fname
    """
    if not os.path.isabs(fname):
        # The path is relative to that of the config.json file
        fname = os.path.join(
            os.path.dirname(config_fname), fname)
    return fname


class TriggerList(object):
    """
    Class that takes in directories with data and template bank information,
    and generates/analyzes triggers
    # TODO: Remove notch_wt_filter options etc
    """
    # Functions to initialize and save a list of triggers and associated data
    # -------------------------------------------------------------------------
    def __init__(
            self, fname=None, left_fname=None, right_fname=None,
            quality_flags=SUGGESTED_QUALITY_FLAGS, white_data_test=False,
            fname_preprocessing=None, load_data=True, do_signal_processing=True,
            do_ffts=2, overwrite_masks=True,
            chunktime_psd=params.DEF_CHUNKTIME_PSD, line_id_ver="new",
            fmax=params.FMAX_OVERLAP,
            preserve_max_snr=params.DEF_PRESERVE_MAX_SNR,
            sine_gaussian_intervals=params.DEF_SINE_GAUSSIAN_INTERVALS,
            bandlim_transient_intervals=params.DEF_BANDLIM_TRANSIENT_INTERVALS,
            excess_power_intervals=params.DEF_EXCESS_POWER_INTERVALS,
            erase_bands=True,
            freqs_to_notch=None, notch_format="new", notch_wt_filter=False,
            renorm_wt=False, taper_wt_filter=False, taper_fraction=0.2,
            times_to_fill=None, times_to_save=None, save_preprocessing=False,
            psd_drift_interval=params.DEF_PSD_DRIFT_INTERVAL,
            average='safemean', fftsize=params.DEF_FFTSIZE,
            template_conf=DEFAULT_TEMPLATE_CONF, delta_calpha=0.7,
            template_safety=params.TEMPLATE_SAFETY, remove_nonphysical=True,
            force_zero=True, trig_fname=None, save_hole_correction=True,
            threshold_chi2=25., base_threshold_chi2=22., nbankchunks=1,
            nbankchunks_done=0, injection_args=None, override_asdfunc=None,
            fake_data=None, min_filt_trunc_time=1,
            populate_left_right_fnames=False, **load_kwargs):
        """
        Initializes data information and parameters needed to generate/analyze
        triggers
        :param fname: File with strain data
        :param left_fname: File with strain data on the left
        :param right_fname: File with strain data on the right
        :param quality_flags: Tuple of LIGO data quality flags to apply cuts on
        :param white_data_test:
            Flag indicating whether we want to do a test on white data
        :param fname_preprocessing:
            Absolute path to file to load preprocessed data from
        :param load_data:
            Set flag to false to avoid time intensive operations by not touching
            the data. If true, we load the data regardless of what is in
            fname_preprocessing
        :param do_signal_processing:
            Flag indicating whether to do signal processing to identify glitches
            and fill holes
        :param do_ffts:
            Flag indicating whether to set up templatebank and chunked FFTs
            0: No FFTs
            1: Sets up the template bank
            2. Sets up the template bank and chunked data
            If boolean, False = 0, True = 2
        :param overwrite_masks:
            Flag indicating whether to overwrite mask and valid mask, useful if
            we're redoing a file but we want to keep the old masks
        :param chunktime_psd: Length of chunk for PSD estimation (in seconds)
        :param line_id_ver:
            Flag to use old or new version of line-identification code
            ("old" in O2 and earlier)
        :param fmax: Maximum frequency involved in the analysis
        :param preserve_max_snr:
            Lower bound for SNR for the central template that is preserved by
            glitch removal. If None, removal depends on params.SIGMA_OUTLIER
        :param sine_gaussian_intervals:
            Frequency bands within which we look for Sine-Gaussian transients
            [central frequency, df = (upper - lower frequency)] Hz
        :param bandlim_transient_intervals:
            Array with set of time-interval-frequency intervals (s, Hz)
            [[dt_i, [f_i_min, f_i_max]],...]
        :param excess_power_intervals:
            Intervals to apply excess power check (s). If None, no excess power
            check performed
        :param erase_bands:
            Flag to erase bands in time-frequency space
        :param freqs_to_notch:
            If desired, list of frequency ranges to notch
            [[f_min_i, f_max_i],...]
        :param notch_format:
            Flag to pick the format for notching. "new" tries to center notches
            on the finer grid if available
        :param notch_wt_filter:
            Flag to apply notches to the whitening filter as well, set it to
            True to avoid biasing the SNR.
            (safe to use False as well, since we have PSD drift downstream)
            Warning: Setting=True creates artifacts at low frequencies, so I
            changed the default to False (My memory is that it was True in O2,
            but false in O1, and that the default in from_json might have been
            set to preserve O1. However, this was *not* borne out by the saved
            value of norm_wt in an O2 file)
        :param renorm_wt:
            Flag whether to scale the whitened data to have unit variance
            after highpass, used only if notch_wt_filter is False
            (If True, we scale to account for the fraction of the band lost.
            Warning: Setting=True biases the PE distances, but not SNRs, in
            the current implementation of gw_pe, so I changed the default)
        :param taper_wt_filter:
            Flag whether to taper the time domain response of the whitening
            filter with a Tukey window
        :param taper_fraction:
            Fraction of response to taper with a Tukey window, if applicable
            (0 is boxcar, 1 is Hann)
        :param times_to_fill:
            If desired, list of time intervals to create holes in
            [[t_min, t_max], ...]
        :param times_to_save:
            If desired, list of time intervals not to create holes in
            [[t_min, t_max], ...]
        :param save_preprocessing:
            Flag indicating whether to save preprocessing results
        :param psd_drift_interval:
            Interval to update correction for nonstationary PSD (s). Pass
            negative number to avoid doing it
        :param average:
            Method to compute the PSD drift correction
            (see gen_psd_drift_correction for options)
        :param fftsize: Size of FFT for matched filtering
        :param template_conf: Absolute path of json file with template metadata
        :param delta_calpha: Spacing of template bank (in units of basis coeffs)
        :param template_safety: Factor to inflate calpha range by
        :param remove_nonphysical:
            Flag indicating whether we want to keep only the gridpoints close to
            a physical waveform
        :param force_zero:
            Flag indicating whether we want to center the template grid to get
            as much overlap as possible with the central region
        :param trig_fname:
            Absolute path to file with triggers to load, if needed
        :param save_hole_correction:
            Flag indicating whether to save hole correction in the trig file,
            useful to operate with both old and new runs
        :param threshold_chi2: Threshold to save single-detector triggers above
        :param base_threshold_chi2:
            Threshold to keep single-detector triggers for sinc interpolation
        :param nbankchunks:
            Number of chunks the template bank was divided into for evaluation
        :param nbankchunks_done:
            Number of chunks that were completed
        :param injection_args:
            Dictionary with parameters to inject waveform(s) into strain data
            time = Scalar or list with absolute GPS time(s) (taken to be right
                   edges for pars/wf, linear-free times for calphas)
            snr =  Scalar or list with SNR(s)
            phase = Scalar or list with phase(s)
            type = Either 'par', 'pars', 'calpha', 'calphas', 'wf', or 'wfs'
            inj_pars = 1D or 2D arrays with
                m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2,
                or calphas,
                or time-domain waveforms
            approximant = Approximant
        :param override_asdfunc:
            If given, asd function to override the one measured from the data
        :param fake_data:
            If given, n_data x 2 array with times and raw/whitened and
            downsampled strain (according to white_data_test False/True) to use
        :param min_filt_trunc_time: minimum truncation length (seconds) of filter
        :param populate_left_right_fnames:
            Flag to automatically populate the left and right fnames if the
            user didn't pass them in
        :param load_kwargs:
            Dictionary with details to load from GPS times instead of files.
            Should have the following keys:
            'run': O1/O2/O3a
            'IFO': H1/L1/V1
            'tstart': Starting time
            'tstop': Stopping time,
            Also can specify edgepadding=True/False, no loading extra seconds
        :return: Instance of class
        """
        super().__init__()

        if (fname is not None) and populate_left_right_fnames:
            if (left_fname is None) or (right_fname is None):
                left_fname, right_fname = utils.get_left_right_fnames(fname)

        # Record files
        self.fname = fname
        self.left_fname = left_fname
        self.right_fname = right_fname
        self.template_conf = template_conf
        self.fname_preprocessing = fname_preprocessing

        # Record processing parameters
        self.quality_flags = quality_flags
        self.chunktime_psd = chunktime_psd
        self.line_id_ver = line_id_ver
        self.fmax = fmax
        self.preserve_max_snr = preserve_max_snr
        self.freqs_to_notch = freqs_to_notch
        self.notch_format = notch_format
        self.notch_wt_filter = notch_wt_filter
        self.renorm_wt = renorm_wt
        self.taper_wt_filter = taper_wt_filter
        self.taper_fraction = taper_fraction
        self.times_to_fill = times_to_fill
        self.times_to_save = times_to_save
        self.psd_drift_interval = psd_drift_interval
        self.average = average
        self.fftsize = fftsize
        # minimum length in seconds of truncated whitening filter
        self.min_filt_trunc_time = min_filt_trunc_time

        # Retain only frequency intervals upto Nyquist in bands
        self.sine_gaussian_intervals = \
            [x for x in sine_gaussian_intervals
             if (x[0] + (x[1] / 2)) <= self.fmax]
        self.bandlim_transient_intervals = \
            [x for x in bandlim_transient_intervals if x[1][1] <= self.fmax]
        self.excess_power_intervals = excess_power_intervals
        self.erase_bands = erase_bands

        # Record injection parameters
        self.injection_args = injection_args

        # Record triggering parameters
        self.threshold_chi2 = threshold_chi2
        self.base_threshold_chi2 = base_threshold_chi2
        self.nbankchunks = nbankchunks
        self.nbankchunks_done = nbankchunks_done

        self.save_hole_correction = save_hole_correction
        # Apology to grad student
        if self.save_hole_correction:
            self.normfac_pos = 2
            self.hole_correction_pos = 3
            self.psd_drift_pos = 6
            self.rezpos = 7
            self.imzpos = 8
            self.c0_pos = 13

            self.processedclist_keys = {
                'Description': 'processedclist stores properties '\
                        +'of collected triggers in the following format',
                0: 'time',
                1: 'incoherent_SNRsq',
                2: 'normfac_22',
                3: 'hole_correction_22',
                6: 'psd_drift_correction',
                7: 'Re(SNR_22)',
                8: 'Im(SNR_22)',
                9: 'Re(SNR_33)',
                10: 'Im(SNR_33)',
                13: 'calpha_0'}
        else:
            self.normfac_pos = 2
            self.hole_correction_pos = None
            self.psd_drift_pos = 3
            self.rezpos = 4
            self.imzpos = 5
            self.c0_pos = 10

            self.processedclist_keys = {
                'Description': 'processedclist stores properties '\
                        +'of collected triggers in the following format',
                0: 'time',
                1: 'incoherent_SNRsq',
                2: 'normfac_22',
                3: 'psd_drift_correction',
                4: 'Re(SNR_22)',
                5: 'Im(SNR_22)',
                6: 'Re(SNR_33)',
                7: 'Im(SNR_33)',
                10: 'calpha_0'}

        # Record template bank parameters
        self.delta_calpha = delta_calpha
        self.template_safety = template_safety
        self.remove_nonphysical = remove_nonphysical
        self.force_zero = force_zero

        # Read t0 from filename, if given
        if fname is not None:
            self.t0 = float(fname.split('-')[-2])

        # List of reasons we can make holes
        self.outlier_reasons = \
            ['LIGO HOLE', 'SIGMA CLIPPING'] + \
            ['SINE GAUSSIAN: ' + str(x)
             for x in self.sine_gaussian_intervals] + \
            ['BANDLIMITED EXCESS POWER: ' + str(x)
             for x in self.bandlim_transient_intervals] + \
            ['EXCESS POWER: ' + str(x)
             for x in self.excess_power_intervals]

        # Treat do_ffts properly
        do_ffts = utils.bool2int(do_ffts)
        
        # Quantities useful for generating triggers on subset of data
        self.time_sub = None
        self.strain_sub = None
        self.mask_sub = None
        self.valid_mask_sub = None
        self.chunked_data_sub = None
        self.chunked_mask_sub = None
        self.psd_drift_correction_sub = None

        # Parameters of subset
        self.relevant_index = None
        self.left_inds = None
        self.right_inds = None
        self.offset_sub = None

        # Deal with preprocessing
        # ---------------------------------------------------------------------
        if white_data_test:
            # Make white data
            if fake_data is not None:
                # Warning: make sure you pass the right format!
                self.time = fake_data[:, 0]
                self.strain = fake_data[:, 1]
                self.t0 = self.time[0]
                self.dt = self.time[1] - self.time[0]
            else:
                self.t0 = 0
                self.dt = 1 / (2 * self.fmax)
                nsamp = int(params.DEF_FILELENGTH / self.dt)
                self.time = self.t0 + self.dt * np.arange(nsamp)
                self.strain = np.random.randn(nsamp)

            # Didn't load data
            self.channel_dict = {}

            # Save PSD variables
            fs_full = int(4 / self.dt)
            fs_down = int(1 / self.dt)
            self.freq_axis = np.fft.rfftfreq(
                int(chunktime_psd * fs_full), d=self.dt / 4)
            self.psd = tg.DEFAULT_ASDF(self.freq_axis) ** 2
            self.crude_line_mask = np.ones_like(self.freq_axis, dtype=bool)
            self.loud_line_mask = np.ones_like(self.freq_axis, dtype=bool)

            # No lines here
            self.freqs_lines = np.fft.rfftfreq(int(1 * fs_down), d=self.dt)
            self.mask_lines = np.ones_like(self.freqs_lines, dtype=bool)

            # No holes here
            nglitchtests = \
                2 + utils.safelen(self.sine_gaussian_intervals) + \
                utils.safelen(self.bandlim_transient_intervals) + \
                utils.safelen(self.excess_power_intervals)
            self.outlier_mask = np.ones(
                (nglitchtests, len(self.time)), dtype=bool)
            self.sigma_clipping_threshold = None
            self.sine_gaussian_thresholds = None
            self.bandlim_power_thresholds = None
            self.excess_power_thresholds = None

            # No fancy bandpassing
            self.norm_wt = np.sqrt(2. * self.dt)
            self.sc_n01 = 1.

            # Define the template bank
            bank = tg.TemplateBank.from_json(template_conf)

            self.fftsize, wt_filter_fd, self.support_wt = \
                self.safe_set_waveform_conditioning(
                    bank, self.fftsize, self.dt,
                    taper_wt_filter=self.taper_wt_filter,
                    taper_fraction=self.taper_fraction,
                    min_filt_trunc_time=self.min_filt_trunc_time)

            # Save in class variables
            self.templatebank = bank
            # self.support_wt = support_wt
            self.normfac = bank.normfac

            # Define masks
            self.mask = np.ones_like(self.strain, dtype=bool)
            self.valid_mask = np.ones_like(self.strain, dtype=bool)
            template_size = bank.support_whitened_wf
            self.valid_mask[:template_size - 1] = 0

            # Perform injections if needed
            # -------------------------------------------------------------
            if injection_args is None:
                injected_wf = None
                injected_wf_whitened = None
            else:
                injected_wf = self.inject_wf_into_data(
                    self.time, self.strain, bank, tg.DEFAULT_ASDF,
                    whitened=True, taper_wt_filter=self.taper_wt_filter,
                    taper_fraction=self.taper_fraction,
                    min_filt_trunc_time=self.min_filt_trunc_time,
                    **injection_args)
                # time_full = np.arange(len(injected_wf)) * self.dt/4
                # mask_inj = np.ones_like(injected_wf, dtype=bool)
                # processed_noiseless = d_ops.process_data(
                #     time_full, injected_wf, mask_inj, tg.DEFAULT_ASDF,
                #     do_signal_processing=False)
                # injected_wf_whitened = processed_noiseless[1]

            # Save injection details
            self.injected_wf = injected_wf
            # self.injected_wf_whitened = injected_wf_whitened
            self.injected_wf_whitened = injected_wf
            
            if erase_bands:
                self.mask_stft = np.ones((int(params.LINE_TRACKING_DT/2/self.dt)+1,
                    int(np.ceil(
                    len(self.strain)/(params.LINE_TRACKING_DT/2/self.dt)))+1),
                    dtype=bool)

        else:
            if fname_preprocessing is not None:
                # Check whether we are loading from a file
                if not os.path.exists(fname_preprocessing):
                    fname_preprocessing_ext = fname_preprocessing + ".npz"
                    if os.path.exists(fname_preprocessing_ext):
                        fname_preprocessing = fname_preprocessing_ext

                if os.path.exists(fname_preprocessing):
                    self.load_data_from_preprocessing(
                        fname_preprocessing, do_ffts=do_ffts)

            if load_data:
                # -------------------------------------------------------------
                # First make a template bank with the fiducial ASD to get a
                # typical waveform length to decide how much to load
                bank = tg.TemplateBank.from_json(template_conf)
                dt_down = 1 / (2 * self.fmax)

                # At this point, if the fftsize is too short, we throw an error
                self.fftsize, *_ = self.safe_set_waveform_conditioning(
                    bank, self.fftsize, dt_down,
                    taper_wt_filter=self.taper_wt_filter,
                    taper_fraction=self.taper_fraction,
                    min_filt_trunc_time=self.min_filt_trunc_time)

                # The waveform duration estimate in seconds
                wf_duration_est = int(
                    np.ceil(bank.support_whitened_wf * dt_down))

                # Load data at full resolution
                if fake_data is not None:
                    # Warning: make sure you pass the right format!
                    time_full = fake_data[:, 0]
                    strain_full = fake_data[:, 1]
                    mask_full = np.ones_like(time_full, dtype=bool)
                    channel_dict = {}
                    self.t0 = time_full[0]
                else:
                    # We're loading from GPS times/files
                    time_full, strain_full, mask_full, channel_dict, t0 = \
                        d_ops.loaddata(
                            fname=fname, left_fname=left_fname,
                            right_fname=right_fname,
                            quality_flags=quality_flags,
                            chunktime_psd=self.chunktime_psd,
                            wf_duration_est=wf_duration_est, edgepadding=True,
                            fmax=self.fmax, **load_kwargs)

                    if fname is not None:
                        # Check fiducial left edge of central file, if loading
                        # from file
                        if t0 != self.t0:
                            raise RuntimeError(
                                "What have you done with the filename LIGO?")
                    else:
                        self.t0 = t0

                # If we don't have enough chunks to get an accurate PSD,
                # reduce the PSD chunksize
                if (np.sum(mask_full) <
                        (self.chunktime_psd * params.MIN_FILELENGTH_FAC)):
                    self.chunktime_psd /= 4
                    if (np.sum(mask_full) >
                            (self.chunktime_psd * params.MIN_FILELENGTH_FAC)):
                        warnings.warn(
                            "Cannot reliably estimate PSD, reducing the " +
                            f"chunktime. On time: {np.sum(mask_full)} s", Warning)
                    else:
                        raise RuntimeError("On time in file is too short! " +
                                           f"On time: {np.sum(mask_full)} s")

                # Compute the ASD
                dt_full = time_full[1] - time_full[0]
                fs_full = int(1 / dt_full)

                freq_axis, psd, asdfunc, crude_line_mask, loud_line_mask = \
                    d_ops.data_to_asdfunc(
                        strain_full, mask_full, fs_full,
                        chunktime_psd=self.chunktime_psd,
                        fmax=max(self.fmax, params.FMAX_PSD),
                        line_id_ver=self.line_id_ver)
                
                if override_asdfunc is not None:
                    asdfunc = override_asdfunc
                    print('Overrode asdfunc')
                
                # Save some results of loading
                self.freq_axis = freq_axis
                self.psd = psd
                self.asdfunc = asdfunc
                self.crude_line_mask = crude_line_mask
                self.loud_line_mask = loud_line_mask
                self.channel_dict = channel_dict

                # Perform injections if needed
                # -------------------------------------------------------------
                if injection_args is None:
                    injected_wf = None
                    injected_wf_whitened = None
                else:
                    # Warning: notching and bandpassing can change the PSD we
                    # use at a later stage if notch_wt_filter is True, so we
                    # measure the SNR later, and avoid notching at this time
                    injected_wf = self.inject_wf_into_data(
                        time_full, strain_full, bank, asdfunc,
                        taper_wt_filter=self.taper_wt_filter,
                        taper_fraction=self.taper_fraction,
                        min_filt_trunc_time=self.min_filt_trunc_time,
                        **injection_args)
                    time_inj = np.arange(len(injected_wf)) * dt_full
                    mask_inj = np.ones_like(injected_wf, dtype=bool)
                    processed_noiseless = d_ops.process_data(
                        time_inj, injected_wf, mask_inj, self.asdfunc,
                        do_signal_processing=False, notch_wt_filter=False,
                        renorm_wt=self.renorm_wt, fmax=self.fmax,
                        taper_wt_filter=self.taper_wt_filter,
                        taper_fraction=self.taper_fraction)
                    injected_wf_whitened = processed_noiseless[1]

                # Save injection details
                self.injected_wf = injected_wf
                self.injected_wf_whitened = injected_wf_whitened

                # Load data and perform signal processing
                # -------------------------------------------------------------
                # Set thresholds for glitch detectors on whitened data
                if do_signal_processing and (preserve_max_snr is not None):
                    # Define whitening filter on downsampled data and
                    # condition the bank with the filter
                    fftsize_sp, *_ = self.safe_set_waveform_conditioning(
                        bank, self.fftsize, dt_down, asdfunc=asdfunc,
                        shorten_fftsize=True,
                        taper_wt_filter=self.taper_wt_filter,
                        taper_fraction=self.taper_fraction,
                        min_filt_trunc_time=self.min_filt_trunc_time)
                    if fftsize_sp > self.fftsize:
                        self.fftsize = fftsize_sp

                    # Compute glitch detector thresholds
                    print("Computing glitch detector thresholds")
                    # Generate whitened waveforms with SNR = 1 that live at the
                    # extreme corners of the bank
                    wt_wfs_cos = bank.gen_boundary_whitened_wfs_td()[...,0,:]
                    # Note that we have only used 22 waveforms here
                    # but we will add the HM wfs with the flag 'make_HM_wfs'
                    # in the function below
                    
                    glitch_thresholds = self.get_glitch_thresholds(
                        wt_wfs_cos, bank.dt, preserve_max_snr,
                        self.sine_gaussian_intervals,
                        self.bandlim_transient_intervals,
                        self.excess_power_intervals, include_HM_wfs=True, bank=bank)
                    sigma_clipping_threshold = glitch_thresholds[0][0]
                    sine_gaussian_thresholds = glitch_thresholds[1][0]
                    bandlim_power_thresholds = glitch_thresholds[2][0]
                    excess_power_thresholds = glitch_thresholds[3][0]
                else:
                    sigma_clipping_threshold = None
                    sine_gaussian_thresholds = None
                    bandlim_power_thresholds = None
                    excess_power_thresholds = None

                # Save glitch removal parameters
                self.sigma_clipping_threshold = sigma_clipping_threshold
                self.sine_gaussian_thresholds = sine_gaussian_thresholds
                self.bandlim_power_thresholds = bandlim_power_thresholds
                self.excess_power_thresholds = excess_power_thresholds

                # Load data
                print("Starting basic data processing")
                tmp_time = time.time()
                time_sub, strain_sub, mask_sub, valid_mask, freqs_lines, \
                    mask_lines, outlier_mask, wt_filter_td, support_wt, \
                    norm_wt, sc_n01, mask_stft = d_ops.process_data(
                        time_full, strain_full, mask_full, asdfunc,
                        do_signal_processing=do_signal_processing,
                        sigma_clipping_threshold=sigma_clipping_threshold,
                        sine_gaussian_intervals=self.sine_gaussian_intervals,
                        sine_gaussian_thresholds=sine_gaussian_thresholds,
                        bandlim_transient_intervals=self.bandlim_transient_intervals,
                        bandlim_power_thresholds=bandlim_power_thresholds,
                        excess_power_intervals=self.excess_power_intervals,
                        excess_power_thresholds=excess_power_thresholds,
                        erase_bands=self.erase_bands,
                        freqs_in=freq_axis,
                        crude_line_mask=crude_line_mask,
                        loud_line_mask=loud_line_mask,
                        freqs_to_notch=self.freqs_to_notch,
                        notch_format=self.notch_format,
                        notch_wt_filter=notch_wt_filter,
                        renorm_wt=self.renorm_wt,
                        times_to_fill=times_to_fill,
                        times_to_save=times_to_save,
                        fmax=self.fmax,
                        taper_wt_filter=self.taper_wt_filter,
                        taper_fraction=self.taper_fraction)
                print("Finished basic data processing")
                print("Basic data processing time:", time.time()-tmp_time)

                # Save data
                self.time = time_sub
                self.strain = strain_sub

                # Save some results of signal processing
                self.mask = getattr(self, "mask", None)
                if (self.mask is None) or overwrite_masks:
                    self.mask = mask_sub
                self.freqs_lines = freqs_lines
                self.mask_lines = mask_lines
                self.dt = time_sub[1] - time_sub[0]
                self.outlier_mask = outlier_mask
                self.wt_filter_td = wt_filter_td
                self.support_wt = support_wt
                self.norm_wt = norm_wt
                self.sc_n01 = sc_n01
                self.mask_stft = mask_stft

                # Update waveform conditioning parameters in template bank
                # with results of signal processing
                wt_filter_td_fft = utils.change_filter_times_td(
                    wt_filter_td, len(wt_filter_td), self.fftsize)
                wt_filter_fd_fft = utils.RFFT(wt_filter_td_fft)
                bank.set_waveform_conditioning(
                    self.fftsize, self.dt, wt_filter_fd=wt_filter_fd_fft)
                self.normfac = bank.normfac
                self.templatebank = bank

                # Define grid parameters
                self.grid_axes, _, self.dcalphas = bank.define_important_grid(
                    self.delta_calpha, fudge=self.template_safety,
                    force_zero=self.force_zero)

                self.valid_mask = getattr(self, "valid_mask", None)
                if (self.valid_mask is None) or overwrite_masks:
                    # Expand invalid indices in valid mask to account for the
                    # size of the template and save to class variable
                    valid_mask_edges = utils.hole_edges(valid_mask)
                    l_valid_edges, r_valid_edges = \
                        valid_mask_edges[:, 0], valid_mask_edges[:, 1]
                    template_size = bank.support_whitened_wf
                    for r_edge in r_valid_edges:
                        valid_mask[r_edge:r_edge + template_size - 1] = 0
                    self.valid_mask = valid_mask

                # Always ensure PSD drift correction is recomputed
                self.psd_drift_correction = None

        # Define mask where we forced the glitch detection not to touch the data
        self.mask_save = None
        if self.times_to_save is not None:
            ts = getattr(self, "time", None)
            if ts is not None:
                self.mask_save = utils.FFTIN(len(ts), dtype=bool)
                self.mask_save[:] = True
                for interval in self.times_to_save:
                    i1, i2 = np.searchsorted(ts, interval)
                    self.mask_save[i1:i2] = False

        # Compute quantities needed for generating triggers
        # -------------------------------------------------
        # Quantities useful for generating triggers on full data
        self.chunked_data_f = None
        self.chunked_mask_f = None
        self.psd_drift_correction = getattr(self, "psd_drift_correction", None)

        # Check if we have data
        self.strain = getattr(self, "strain", None)
        if (self.strain is not None) and (do_ffts > 1):
            self.prepare_for_triggers(average=self.average)

        # Save preproccessing results to file if given
        # --------------------------------------------
        if save_preprocessing:
            self.save_preprocessing(self.fname_preprocessing)

        # Record trigger parameters
        # -------------------------------------------------------------
        self.trig_fname = None
        self.processedclist = None
        self.calpha_range = None

        if trig_fname is not None:
            if not os.path.exists(trig_fname):
                trig_fname = utils.rm_suffix(trig_fname, ".npy", ".npy")
            if os.path.exists(trig_fname):
                self.trig_fname = trig_fname
                with open(trig_fname, "rb") as f:
                    self.processedclist = np.load(f, allow_pickle=True)
                    #TODO: remove comment when packing is ready
                    if False:#self.packed_trig_format:
                        self.processedclist = self.unpack_trigs(
                            self.processedclist)

                # Check if there were no triggers at all
                if not utils.checkempty(self.processedclist):
                    # n_basis coefficient x 2 array with coefficient ranges
                    self.calpha_range = np.c_[
                        np.min(self.processedclist[:, self.c0_pos:], axis=0),
                        np.max(self.processedclist[:, self.c0_pos:], axis=0)]

        # Initialize filterred triggerlist to work with
        self.filters = {}
        self.rejects = defaultdict(list)
        self.filteredclist = copy.deepcopy(self.processedclist)

        return

    # Functions to load and save processed data and triggers
    # ------------------------------------------------------
    @classmethod
    def empty_init(cls):
        instance = cls(load_data=False)
        return instance

    @classmethod
    def from_json(
            cls, config_fname, load_trigs=True, load_preproc=True, do_ffts=2,
            load_data=False, do_signal_processing=True, return_instance=True,
             load_injected_waveforms=False, **kwargs):
        """
        Instantiates class from json file created by previous run/gen_triggers
        Load using load_data=True and do_signal_processing=False to access the
        data without any glitch rejection
        :param config_fname: Absolute path of json file
        :param load_trigs: Flag indicating whether to load triggers
        :param load_preproc:
            Flag indicating whether to load preprocessing (if False, doesn't
            load the npz file)
        :param do_ffts:
            Flag indicating whether to define templatebank and chunked FFTs
            (only used if load_preproc or load_data is True)
            0: No FFTs
            1: Sets up the template bank
            2. Sets up chunked data
            If boolean, False = 0, True = 2
        :param load_data:
            Set flag to False in order to avoid time intensive operations by not
            touching the data. If True, does preprocessing even if file exists
        :param do_signal_processing:
            Flag indicating whether to do signal processing to identify glitches
            and fill holes. Only relevant if load_data is True
        :param load_injected_waveforms:
            Inidcates whether injected waveforms need to be loaded from a separate
            file with filename(s) injection_args['wf_filename'].
        :param kwargs: Override any of the other input parameters to init
        :return: Instance of TriggerList
        """
        with open(config_fname, 'r') as fp:
            dic = json.load(fp, object_hook=utils.NumpyEncoder.np_in_hook)
        # Make sure the currect paths are used. Demands that the path ends with 'RUN/DETECTOR/FILENAME.hdf5'
        # e.g.: /home/labs/barakz/Collaboration-gw/O3a/H1/H-H1_GWOSC_O3a_4KHZ_R1-1248841728-4096.hdf5
        for kw in ['fname','left_fname', 'right_fname']:
            if dic[kw] is not None:
                if not(dic[kw].startswith(utils.DATA_ROOT)):
                    dic[kw] = os.path.join(utils.DATA_ROOT, *dic[kw].split('/')[-3:])


        # Pop things that might be present but not needed
        t_event = dic.pop('t_event', None)

        load_kwargs = {}
        load_kwargs.update(dic)
        # if load_injected_waveforms=True in function call and I saved the injection_args
        # without inj_pars and with just the filenames that the waveforms are stored in-
        # then repopulate 'inj_pars' and pop 'wf_filename' and 'indices'
        if load_injected_waveforms:
            if load_kwargs['injection_args'].get('wf_filename',None) is not None:
                filenames = load_kwargs['injection_args'].pop('wf_filename')
                order_indices = load_kwargs['injection_args'].pop('indices')
                wf_array = np.concatenate([np.load(f) for f in filenames])
                load_kwargs['injection_args'].update({'inj_pars': wf_array[order_indices.astype('int')]})

        # We set a few default values. Note, some defaults are different from
        # those in _init_. We set these to load old files, since they are
        # different from the new defaults
        load_kwargs['remove_nonphysical'] = dic.get('remove_nonphysical', False)
        load_kwargs['force_zero'] = dic.get('force_zero', False)
        load_kwargs['save_hole_correction'] = \
            dic.get('save_hole_correction', False)
        load_kwargs['base_threshold_chi2'] = dic.get('base_threshold_chi2', 16.)
        load_kwargs['nbankchunks'] = dic.get('nbankchunks', 1)
        load_kwargs['nbankchunks_done'] = dic.get('nbankchunks_done', 1)
        load_kwargs['average'] = dic.get('average', 'median')
        load_kwargs['notch_format'] = dic.get('notch_format', "old")
        load_kwargs['notch_wt_filter'] = dic.get('notch_wt_filter', False)
        load_kwargs['renorm_wt'] = dic.get('renorm_wt', True)
        load_kwargs['fmax'] = dic.get('fmax', 512)
        load_kwargs['line_id_ver'] = dic.get('line_id_ver', "old")
        load_kwargs['erase_bands'] = dic.get('erase_bands', False)

        if load_kwargs.get('excess_power_intervals') is None:
            # Old run
            load_kwargs['excess_power_intervals'] = \
                [dic.get('excess_power_interval')]
            _ = load_kwargs.pop('excess_power_interval')

        if not load_trigs:
            load_kwargs['trig_fname'] = None
        elif load_kwargs.get('trig_fname', None) is not None:
            # Deal with relative paths
            load_kwargs['trig_fname'] = ensure_abspath(
                config_fname, load_kwargs['trig_fname'])

        if not load_preproc:
            load_kwargs['fname_preprocessing'] = None
        else:
            # Deal with relative paths
            load_kwargs['fname_preprocessing'] = ensure_abspath(
                config_fname, load_kwargs['fname_preprocessing'])

        # Make other code work
        do_ffts = utils.bool2int(do_ffts)

        # Set extra arguments, and override others if provided
        load_kwargs['load_data'] = load_data
        load_kwargs['do_signal_processing'] = do_signal_processing
        load_kwargs['do_ffts'] = do_ffts
        load_kwargs['save_preprocessing'] = False

        # If the template configuration absolute path does not exist, try to restore sanity with a relative path.
        if load_kwargs.get('template_conf', None) is not None:
            if not os.path.exists(load_kwargs['template_conf']):
                load_kwargs['template_conf'] = utils.TEMPLATE_DIR + load_kwargs['template_conf'].split('templates')[1]

        # Override any of the kwargs
        for item in kwargs.items():
            load_kwargs[item[0]] = item[1]
        
        if return_instance:
            instance = cls(**load_kwargs)
            
            if t_event is not None:
                instance.t_event = t_event

            return instance
        
        else:
            return load_kwargs

    @classmethod
    def from_gwosc(
            cls, path=None, evname='GW150914', tgps=None, detector='H1',
            verbose=True, **load_kwargs):
        """
        Loads data from gwosc and makes a trigger file, records the event
        time as given by LIGO (different from the linear-free time) in t_event
        Make sure to pass an appropriate template_conf for the parameters of
        the event in load_kwargs
        If we are doing something custom (e.g., adding holes etc), save using
        to_json along with fname_preprocessing, and we can reload it later
        with the desired modifications passed to from_json
        :param path: Path to root directory where the file will be saved
        :param evname: Event around which which we want the strain data
        :param tgps: GPS time around which which we want the strain data
        :param detector: Name of the detector for which we want the strain data
        :param verbose: Print information
        :param load_kwargs: Extra arguments to pass to the init function
        :return: Instance of TriggerList
        """
        requests = utils.load_module('requests')
        locate = utils.load_module('gwosc.locate')
        datasets = utils.load_module('gwosc.datasets')

        # By default, gwosc has 32 or 4096
        duration = int(load_kwargs.pop('duration', 4096))

        if evname is not None:
            fnames = locate.get_event_urls(
                evname, detector=detector, duration=duration)
            t_event = datasets.event_gps(evname)
        elif tgps is not None:
            fnames = locate.get_urls(detector=detector, start=tgps, end=tgps)
            fnames = [f for f in fnames if str(duration) in f]
            t_event = tgps
        else:
            raise RuntimeError("I need information to query the data")

        if len(fnames) > 0:
            fname = fnames[0]
        else:
            print("No valid filename found!")
            return

        if path is not None:
            fname_save = os.path.join(path, fname.split("/")[-1])
        else:
            # Save in the current directory
            fname_save = fname.split("/")[-1]

        if not os.path.isfile(fname_save):
            if verbose:
                print(f"Downloading {fname} from gwosc")
            r = requests.get(fname, allow_redirects=True)

            if verbose:
                print(f"Saving file to {fname_save}")
            with open(fname_save, "wb") as f:
                f.write(r.content)

        instance = cls(fname=fname_save, **load_kwargs)
        instance.t_event = t_event

        return instance

    def load_data_from_preprocessing(self, fname_preprocessing, do_ffts=True):
        """
        Load preprocessed data from file
        :param fname_preprocessing: Filename to load preprocessed data from
        :param do_ffts:
            Flag indicating whether to define whitening filter
            If boolean, True/False
            If integer, >0 we define the whitening filter
        :return:
        """
        # We are loading from a file
        preprocfile = np.load(fname_preprocessing, allow_pickle=True)

        # ASD function
        self.psd = preprocfile["psd"]
        freq_min, df = preprocfile["freq_axis_psd"]
        self.freq_axis = freq_min + df * np.arange(len(self.psd))

        if "loud_line_mask" in preprocfile.files:
            self.loud_line_mask = preprocfile["loud_line_mask"]

        self.asdfunc = d_ops.asd_func(
            self.freq_axis, self.psd, fmin=params.FMIN_PSD,
            fmax=max(self.fmax, params.FMAX_PSD))

        # Channel dict
        self.channel_dict = preprocfile["channel_dict"].item()

        # Injected waveform
        self.injected_wf = preprocfile["injected_wf"]
        if "injected_wf_whitened" in preprocfile.files:
            self.injected_wf_whitened = preprocfile["injected_wf_whitened"]
        else:
            self.injected_wf_whitened = None

        # Glitch detection thresholds
        self.excess_power_thresholds = \
            preprocfile["excess_power_thresholds"]
        self.bandlim_power_thresholds = \
            preprocfile["bandlim_power_thresholds"]
        sigma_clipping_threshold = preprocfile["sigma_clipping_threshold"]
        if utils.checkempty(sigma_clipping_threshold):
            self.sigma_clipping_threshold = None
        else:
            self.sigma_clipping_threshold = float(sigma_clipping_threshold)
        if "sine_gaussian_thresholds" in preprocfile.files:
            self.sine_gaussian_thresholds = \
                preprocfile["sine_gaussian_thresholds"]

        # Data
        left_time, dt = preprocfile["time_axis_strain"]
        self.dt = dt
        self.strain = preprocfile["strain"]
        self.time = left_time + dt * np.arange(len(self.strain))

        # Results of signal processing
        if "mask" in preprocfile.files:
            self.mask = preprocfile["mask"]
        else:
            self.mask = utils.hole_edges_to_mask(
                preprocfile["mask_edges"], len(self.strain))

        if "valid_mask" in preprocfile.files:
            self.valid_mask = preprocfile["valid_mask"]
        else:
            self.valid_mask = utils.hole_edges_to_mask(
                preprocfile["valid_mask_edges"], len(self.strain))
                
        if "mask_stft" in preprocfile.files:
            self.mask_stft = preprocfile["mask_stft"]
        else: self.mask_stft = None

        if (("freqs_lines" in preprocfile.files) and
                ("mask_lines" in preprocfile.files)):
            self.mask_lines = preprocfile["mask_lines"]
            f0, df = preprocfile["freqs_lines"]
            self.freqs_lines = f0 + df * np.arange(len(self.mask_lines))

        if "outlier_reasons" in preprocfile.files:
            self.outlier_reasons = preprocfile["outlier_reasons"]
        else:
            self.outlier_reasons = \
                ['LIGO HOLE', 'SIGMA CLIPPING'] + \
                ['SINE GAUSSIAN: ' + str(x)
                 for x in self.sine_gaussian_intervals] + \
                ['BANDLIMITED EXCESS POWER: ' + str(x)
                 for x in self.bandlim_transient_intervals] + \
                ['EXCESS POWER: ' + str(x)
                 for x in self.excess_power_intervals]

        if "outlier_times" in preprocfile.files:
            self.outlier_mask = preprocfile["outlier_times"]
        else:
            self.outlier_mask = utils.FFTIN(
                (len(self.outlier_reasons), len(self.strain)),
                dtype=bool)
            for ind, reason in enumerate(self.outlier_reasons):
                self.outlier_mask[ind, :] = \
                    utils.hole_edges_to_mask(
                        preprocfile[reason], len(self.strain))

        # Template bank parameters
        self.normfac = None
        self.support_wt = None
        self.norm_wt = None
        self.sc_n01 = 1.
        if "normfac" in preprocfile.files:
            self.normfac = float(preprocfile["normfac"])
        if "support_wt" in preprocfile.files:
            self.support_wt = float(preprocfile["support_wt"])
        if "norm_wt" in preprocfile.files:
            self.norm_wt = float(preprocfile["norm_wt"])
        if "sc_n01" in preprocfile.files:
            self.sc_n01 = float(preprocfile["sc_n01"])

        # Define template bank
        bank = tg.TemplateBank.from_json(self.template_conf)
        self.templatebank = bank

        # Define grid parameters
        self.grid_axes, _, self.dcalphas = bank.define_important_grid(
            self.delta_calpha, fudge=self.template_safety,
            force_zero=self.force_zero)

        do_ffts = utils.bool2int(do_ffts)

        if do_ffts > 0:
            wt_filter_td_saved = preprocfile["wt_filter_td"]
            wt_filter_td_fft = utils.change_filter_times_td(
                wt_filter_td_saved, len(wt_filter_td_saved), self.fftsize)
            self.wt_filter_td = wt_filter_td_fft
            wt_filter_fd_fft = utils.RFFT(wt_filter_td_fft)
            bank.set_waveform_conditioning(
                self.fftsize, self.dt, wt_filter_fd=wt_filter_fd_fft)

            # Overwrite normfac
            self.normfac = bank.normfac

        # Load PSD drift correction if saved
        if "psd_drift_correction" in preprocfile.files:
            self.psd_drift_correction = preprocfile["psd_drift_correction"]

        return

    def to_json(self, config_fname, preprocessing_fname=None, overwrite=False):
        """
        Saves information about trigger list to json file
        :param config_fname: Absolute path of json file
        :param preprocessing_fname:
            Override preprocessing file, pass without extension
        :param overwrite: Flag to overwrite existing preprocessing file
        :return:
        """
        if preprocessing_fname is not None:
            # Override filename
            preprocessing_fname = ensure_abspath(
                config_fname, preprocessing_fname)
            self.fname_preprocessing = preprocessing_fname

        # Save preprocessing file only if it's not already present
        if self.fname_preprocessing is not None:
            fname_preprocessing = self.fname_preprocessing
            if overwrite or not os.path.exists(fname_preprocessing):
                fname_preprocessing = \
                    utils.rm_suffix(fname_preprocessing, ".npz", ".npz")
                if not os.path.exists(fname_preprocessing):
                    self.save_preprocessing(fname_preprocessing)
        preproc_fname = self.fname_preprocessing
        if preproc_fname is not None:
            preproc_fname = os.path.basename(preproc_fname)

        trig_fname = self.trig_fname
        if trig_fname is not None:
            trig_fname = os.path.basename(trig_fname)

        # Save relative paths for preprocessing files and trig files
        dic = {'fname': self.fname,
               'left_fname': self.left_fname,
               'right_fname': self.right_fname,
               'quality_flags': self.quality_flags,
               'fname_preprocessing': preproc_fname,
               'chunktime_psd': self.chunktime_psd,
               'line_id_ver': self.line_id_ver,
               'fmax': self.fmax,
               'preserve_max_snr': self.preserve_max_snr,
               'sine_gaussian_intervals': self.sine_gaussian_intervals,
               'bandlim_transient_intervals': self.bandlim_transient_intervals,
               'excess_power_intervals': self.excess_power_intervals,
               'erase_bands': self.erase_bands,
               'freqs_to_notch': self.freqs_to_notch,
               'notch_format': self.notch_format,
               'notch_wt_filter': self.notch_wt_filter,
               'renorm_wt': self.renorm_wt,
               'times_to_fill': self.times_to_fill,
               'times_to_save': self.times_to_save,
               'psd_drift_interval': self.psd_drift_interval,
               'average': self.average,
               'fftsize': self.templatebank.fftsize,
               'template_conf': self.template_conf,
               'delta_calpha': self.delta_calpha,
               'template_safety': self.template_safety,
               'remove_nonphysical': self.remove_nonphysical,
               'force_zero': self.force_zero,
               'trig_fname': trig_fname,
               'save_hole_correction': self.save_hole_correction,
               'threshold_chi2': self.threshold_chi2,
               'base_threshold_chi2': self.base_threshold_chi2,
               'nbankchunks': self.nbankchunks,
               'nbankchunks_done': self.nbankchunks_done,
               'injection_args': self.injection_args,
               'min_filt_trunc_time': self.min_filt_trunc_time
               }
        if hasattr(self, 't_event'):
            dic['t_event'] = self.t_event

        # Save json file
        with open(config_fname, 'w') as f:
            json.dump(dic, f, indent=2, cls=utils.NumpyEncoder)
        os.system("chmod 666 " + config_fname)

        return

    def save_preprocessing(self, fname_preprocessing):
        """
        Save preprocessing to file
        :param fname_preprocessing: Filename to save preprocessed data in
        :return:
        """
        
        dname = os.path.dirname(fname_preprocessing)
        if not os.path.isdir(dname):
            os.system(f'mkdir -p {dname}')

        # Whitening filter is compact in time domain, so save a small array
        wt_filter_td_fft = utils.IRFFT(
            self.templatebank.wt_filter_fd, n=self.fftsize)
        td_savesize = min(self.fftsize, utils.next_power(2 * self.support_wt))
        wt_filter_td_saved = utils.change_filter_times_td(
            wt_filter_td_fft, self.fftsize, td_savesize)

        # Compress masks by storing zero edges
        # Save mask and valid mask
        mask_edges = utils.hole_edges(self.mask)
        valid_mask_edges = utils.hole_edges(self.valid_mask)

        # Save masks where glitch detectors fired
        outlier_mask_edges = {}
        for ind, reason in enumerate(self.outlier_reasons):
            outlier_mask_edges.update(
                {reason: utils.hole_edges(self.outlier_mask[ind])})

        # Save whitened strain in single precision
        if self.injected_wf_whitened is not None:
            injected_wf_whitened = self.injected_wf_whitened.astype(np.float32)
        else:
            injected_wf_whitened = None

        np.savez(
            fname_preprocessing,
            freq_axis_psd=[
                self.freq_axis[0], self.freq_axis[1] - self.freq_axis[0]],
            psd=self.psd,
            channel_dict=self.channel_dict,
            injected_wf=self.injected_wf,
            injected_wf_whitened=injected_wf_whitened,
            excess_power_thresholds=self.excess_power_thresholds,
            bandlim_power_thresholds=self.bandlim_power_thresholds,
            sigma_clipping_threshold=self.sigma_clipping_threshold,
            sine_gaussian_thresholds=self.sine_gaussian_thresholds,
            time_axis_strain=[self.time[0], self.dt],
            strain=self.strain.astype(np.float32),
            mask_edges=mask_edges,
            valid_mask_edges=valid_mask_edges,
            mask_stft=self.mask_stft.astype(bool) if self.mask_stft is not None else None,
            freqs_lines=[
                self.freqs_lines[0], self.freqs_lines[1] - self.freqs_lines[0]],
            mask_lines=self.mask_lines,
            loud_line_mask=self.loud_line_mask,
            outlier_reasons=self.outlier_reasons,
            wt_filter_td=wt_filter_td_saved,
            normfac=self.normfac,
            support_wt=self.support_wt,
            norm_wt=self.norm_wt,
            sc_n01=self.sc_n01,
            psd_drift_correction=self.psd_drift_correction.astype(np.float32),
            **outlier_mask_edges)

        os.system("chmod 666 " + fname_preprocessing)

        return

    def save_candidatelist(self, trig_fname, append=True):
        """
        :param trig_fname: Absolute path to trigger file to save triggers to
        :param append: Flag indicating whether to append to existing file
        If append is True,
            1. If trig_fname doesn't exist, we create it and save
               self.processedclist to it
            2. If trig_fname exists, we append self.processedclist to its
               contents
        :return Updates self.trig_fname
        """
        self.trig_fname = trig_fname

        if trig_fname is None:
            return

        if utils.checkempty(self.processedclist):
            return

        # We're saving to a file
        # Treat if user passes name without .npy
        trig_fname_ext = utils.rm_suffix(trig_fname, ".npy", ".npy")
        # if not os.path.exists(trig_fname_ext):
        #     trig_fname_ext += ".npy"

        # If the preprocessing file exists, check it is in the same directory
        # The code should have ensured it is always the case, but check in case
        # there is an edge case that I forgot
        if self.fname_preprocessing is not None:
            assert (os.path.dirname(self.fname_preprocessing) ==
                    os.path.dirname(trig_fname_ext))

        # Check again (now with extension if added)
        self.packed_trig_format = False
        #TODO: remove when packing is ready
        if os.path.exists(trig_fname_ext) and append:
            self.trig_fname = trig_fname_ext

            # We're loading from an old file and appending
            with open(trig_fname_ext, "rb") as f:
                old_processedclist = np.load(f)
                if self.packed_trig_format:
                    old_processedclist = self.unpack_trigs(old_processedclist)

            # If old list wasn't empty, append existing triggers
            # Avoid overwriting or redefining self.processedclist
            # to save memory
            old_processedclist = utils.safe_concatenate(
                old_processedclist, self.processedclist)

            # Now save self.processedclist to self.trig_fname
            if self.packed_trig_format:
                old_processedclist = self.pack_trigs(old_processedclist)
            np.save(self.trig_fname, old_processedclist)

        else:
            # We're saving to a new file, avoid .npy.npy if user passes
            # name with .npy that doesn't exist
            self.trig_fname = trig_fname

            # Now save self.processedclist to self.trig_fname
            if self.packed_trig_format:
                self.processedclist = self.pack_trigs(self.processedclist)
            np.save(self.trig_fname, self.processedclist)

        # Ensure that we always have .npy else command will fail
        trig_fname_ext = utils.rm_suffix(trig_fname, ".npy", ".npy")
        # trig_fname_ext = trig_fname
        # if not os.path.exists(trig_fname_ext):
        #     trig_fname_ext += ".npy"
        os.system("chmod 666 " + trig_fname_ext)
        return

    # Functions involved in the preliminary setup of the class
    # -------------------------------------------------------------------------
    @staticmethod
    def safe_set_waveform_conditioning(
            bank, fftsize, dt, asdfunc=None, shorten_fftsize=False,
            taper_wt_filter=False, taper_fraction=0.2, min_filt_trunc_time=1):
        """
        Sets waveform conditioning given the ASD, and expands fftsize by a factor
        of two if needed to make it work. Useful when trialing small FFTsizes for
        BNS-like banks
        :param bank: Instance of TemplateBank
        :param fftsize: Integer with input candidate fftsize
        :param dt: Sampling interval of template (in seconds)
        :param asdfunc: Function returning ASDs (Hz^-0.5) given frequencies (Hz)
        :param shorten_fftsize:
            If True, we try to condition with a shorter fftsize if possible
        :param taper_wt_filter:
            Flag whether to taper the time domain response of the whitening filter
            with a Tukey window
        :param taper_fraction:
            Fraction of response to taper with a Tukey window, if applicable
            (0 is boxcar, 1 is Hann)
        :return:
            1. FFTsize for which conditioning worked (either input or 2x input)
            2. Whitening filter in the Fourier domain
            3. Half-support of the whitening filter in the time-domain
        """
        if asdfunc is None:
            asdfunc = bank.asdf

        if asdfunc is None:
            asdfunc = tg.DEFAULT_ASDF

        input_fftsize = fftsize
        for iter_id in range(2):
            try:
                # Generate the whitening filter from the PSD function
                fs_fft = np.fft.rfftfreq(fftsize, d=dt)
                wt_filter_fd_unconditioned = np.nan_to_num(1. / asdfunc(fs_fft))
                wt_filter_fd, support_wt, _ = utils.condition_filter(
                    wt_filter_fd_unconditioned, truncate=True, flen=fftsize,
                    taper=taper_wt_filter, taper_fraction=taper_fraction,
                    min_trunc_len=int(min_filt_trunc_time / dt))

                if shorten_fftsize and (iter_id == 0):
                    # Let's save ourselves some computation by making the length
                    # shorter, but not too short w.r.t the whitening filter
                    tempfftsize = utils.next_power(
                        min(max(8 * support_wt,
                                2 * bank.support_whitened_wf), fftsize))

                    wt_filter_fd = utils.change_filter_times_fd(
                        wt_filter_fd, fftsize, tempfftsize)
                    fftsize = tempfftsize

                # Condition the bank with this whitening filter
                bank.set_waveform_conditioning(
                    fftsize, dt, wt_filter_fd=wt_filter_fd)

                return fftsize, wt_filter_fd, support_wt
            except AssertionError:
                if iter_id == 1:

                    if shorten_fftsize:
                        # Try with the given fftsize
                        return TriggerList.safe_set_waveform_conditioning(
                            bank, input_fftsize, dt, asdfunc=asdfunc,
                            shorten_fftsize=False,
                            taper_wt_filter=taper_wt_filter,
                            taper_fraction=taper_fraction,
                            min_filt_trunc_time=min_filt_trunc_time)

                    raise RuntimeError(
                        "Increased fftsize by a factor of two, but the " +
                        "waveform is still too long. Check and pass a " +
                        "sensible fftsize.")

                # The FFTsize was too short, let's try to expand it once
                fftsize *= 2

    def pack_trigs(self, processed_clist):
        return processed_clist
        # TODO: Finish implementing! commented out because of accidental push
        #  in the middle of implementation
        # ncalpha = len(self.processedclist[0])-self.c0_pos
        # #TODO: rho_sq redundant?
        # #TODO: better way to find out the number of calphas?
        # #TODO: set up the array and assign the values.
        # #TODO: Expose the resolution of the array elements?
        # my_dtype = np.dtype([('t_minus_t0', np.int32),
        #                      ('rho_sq', np.uint16),
        #                      ('z_real', np.int16),
        #                      ('z_imag', np.int16),
        #                      ('psd_drift', np.int16),
        #                      ('normfac', np.int16),
        #                      ('calpha', np.int16, ncalpha)
        #                      ])
        #
        # arr_to_save = np.zeros(len(processed_clist), dtype=my_dtype)
        # self.save_deltas = [1e-4, 1e-2, 1e-5, 1e-3, 1e-3, 1e-3,
        #                     1e-3,
        #                     1e-1,
        #                     1e-1]
        #
        # self.normfac_pos = 2
        # self.hole_correction_pos = 3
        # self.psd_drift_pos = 4
        # self.rezpos = 5
        # self.imzpos = 6
        # self.c0_pos = 7
        #
        # arr_to_save['t_minus_t0'] = np.round(
        #     (processed_clist[:, 0] - self.t0)/self.save_deltas[0]).astype(
        #     'int32')
        # arr_to_save['rho_sq'] = np.round(
        #     (processed_clist[:, 1]) / self.save_deltas[1]).astype('int16')
        # arr_to_save['z_real'] = np.round(
        #     (processed_clist[:, self.rezpos]) / self.save_deltas[
        #         self.rezpos]).astype('int16')
        # arr_to_save['z_imag'] = np.round(
        #     (processed_clist[:, self.imzpos]) / self.save_deltas[
        #         self.imzpos]).astype('int16')
        # arr_to_save['psd_drift'] = np.round(
        #     (processed_clist[:, self.psd_drift_pos]) / self.save_deltas[
        #         self.psd_drift_pos]).astype('int16')
        # arr_to_save['normfac'] = np.round(
        #     (processed_clist[:, self.normfac_pos]) / self.save_deltas[
        #         self.normfac_pos]).astype('int16')
        # arr_to_save['calpha'] = np.round(
        #     (processed_clist[:, self.c0_pos:]) / self.save_deltas[
        #         self.c0_pos]).astype('int16')
    def unpack_trigs(self, packed_trigs):
        #TODO: Write some unpacking function to the packing function.
        return packed_trigs

    @staticmethod
    def inject_wf_into_data(
            times, strain, bank, asdfunc, whitened=False,
            taper_wt_filter=False, taper_fraction=0.2,
            min_filt_trunc_time=1, **injection_args):
        """
        Injects waveform into strain data, read description of injection_args
        for time convention, modifies strain in place as strain is a mutable type
        :param times: Array with times (s)
        :param strain: Array with strain data
        :param bank: Instance of TemplateBank
        :param asdfunc: Function returning ASDs (Hz^-0.5) given frequencies (Hz)
        :param whitened:
            Flag indicating whether the strain data is raw or whitened
        :param taper_wt_filter:
            Flag whether to taper the time domain response of the whitening filter
            with a Tukey window
        :param taper_fraction:
            Fraction of response to taper with a Tukey window, if applicable
            (0 is boxcar, 1 is Hann)
        :param injection_args:
            Parameters to inject waveform(s) into strain data
            time = Scalar or list with absolute GPS time(s) (taken to be right
                   edges for pars/wf, linear-free times for calphas,
                   fft overlap convention (right edge + 1))
            snr =  Scalar or list with SNR(s)
            phase = Scalar or list with phase(s)
            type = Either 'par', 'pars', 'calpha', 'calphas', 'wf', or 'wfs'
            inj_pars = 1D or 2D arrays with
                m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2,
                or calphas,
                or time-domain waveforms
            approximant = Approximant
        :return:
            Inserts waveform into strain data, returns noiseless stream with
            injection(s)
        """
        # Check if we have nothing to do
        if injection_args.get("time") is None:
            return utils.FFTIN(len(strain))

        # Time interval between samples
        dt = times[1] - times[0]

        # Save old bank conditioning parameters
        old_fftsize = bank.fftsize
        old_dt = bank.dt
        old_wt_filter_fd = bank.wt_filter_fd

        if dt != bank.dt:
            # We need to condition bank to a finer time grid
            # fftsize doesn't matter for generation, only depends on dt
            # and highpass filter
            # fs_inj = np.fft.rfftfreq(len(strain), dt)
            # Warning: Restricting to 128 s, careful if injecting BNS!
            flen = utils.next_power(int(128 / dt))
            fs_inj = np.fft.rfftfreq(flen, dt)
            asds_inj = asdfunc(fs_inj)
            wt_filter_fd_inj_unconditioned = 1. / asds_inj
            wt_filter_fd_inj, _, _ = utils.condition_filter(
                wt_filter_fd_inj_unconditioned, truncate=True, flen=flen,
                taper=taper_wt_filter, taper_fraction=taper_fraction,
                min_trunc_len=int(min_filt_trunc_time / dt))
            bank.set_waveform_conditioning(flen, dt, wt_filter_fd_inj)

        # Read off injection parameters
        inj_times = np.atleast_1d(injection_args.get("time"))
        snrs = injection_args.get("snr", [None] * len(inj_times))
        phases = injection_args.get("phase", [0] * len(inj_times))
        injtype = injection_args.get("type")
        inj_pars = injection_args.get("inj_pars")
        approximant = injection_args.get("approximant", None)
        inclination = injection_args.get(
            "inclination", DEFAULT_INCLINATION * np.ones_like(inj_times))
        phiRef = injection_args.get(
            "phiRef", DEFAULT_PHIREF * np.ones_like(inj_times))

        # Ensure that we are setup for multiple waveforms
        if not hasattr(inj_times, "__len__"):
            inj_times = [inj_times]
            snrs = [snrs]
            phases = [phases]
            inj_pars = [inj_pars]
            inclination = [inclination]
            phiRef = [phiRef]

        noiseless_stream = utils.FFTIN(len(strain))
        for ind, (inj_time, snr, phase, inj_par, iota, vphi) in enumerate(
                zip(inj_times, snrs, phases, inj_pars, inclination, phiRef)):
            if len(inj_times) <= 10:
                print(f"Injecting waveform {ind}")
            elif ind % 10 == 0:
                print(f"Injecting waveform {ind}, " +
                      f"{ind / len(inj_times) * 100} % done")

            # Do appropriate type of injection
            if injtype.lower() in ('par', 'pars'):
                # Injecting using parameters
                m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2 = inj_par

                # Compute waveform using lal
                # LAL does the conditioning, so OK to use same time
                # convention for H1 and L1
                wf_short = bank.gen_wf_td_from_pars(
                    target_snr=snr, phase=phase, approximant=approximant,
                    m1=m1, m2=m2, s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x,
                    s2y=s2y, s2z=s2z, l1=l1, l2=l2,
                    inclination=iota, phiRef=vphi)

                if whitened:
                    # Pad before whitening to avoid losing the ringdown
                    dt_extra = 4
                    n_inds_extra = int(np.ceil(dt_extra / bank.dt))
                    wf_td = utils.FFTIN(bank.fftsize)
                    wf_td[-n_inds_extra-len(wf_short):-n_inds_extra] = \
                        wf_short[:]
                    # Whiten the waveform
                    wf_short = np.sqrt(2. * bank.dt) * utils.IRFFT(
                        utils.RFFT(wf_td) * bank.wt_filter_fd)
                else:
                    dt_extra = 0

            elif injtype.lower() in ('calpha', 'calphas'):
                # Injecting using calphas
                calphas = np.asarray(inj_par)

                if whitened:
                    dt_extra = 0
                    lwf = bank.support_whitened_wf
                    wf_cos = bank.gen_whitened_wfs_td(calphas)
                    wf_sin = utils.IRFFT(1j * utils.RFFT(wf_cos))
                else:
                    dt_extra = 4
                    n_inds_extra = int(np.ceil(dt_extra / bank.dt))
                    lwf = bank.support_whitened_wf + n_inds_extra
                    wf_cos = bank.gen_wfs_td_from_calpha(
                        calphas, dt_extra=dt_extra, target_snr=1,
                        truncate=True)
                    wf_sin = utils.IRFFT(1j * utils.RFFT(wf_cos))

                # Apply the right phase
                wf_short = (snr * (np.cos(phase) * wf_cos +
                            np.sin(phase) * wf_sin))[-lwf:]

                # Fix convention so that inj_time is the linear free
                # time, note shift < 0
                dt_extra -= bank.shift_whitened_wf * bank.dt

            elif injtype.lower() in ('wf', 'wfs'):
                # Injecting waveform that the user passed to us, assumed at
                # native resolution of the data before downsampling
                # Warning: assumes shorter than 128 s!
                wf = inj_par

                if whitened:
                    # Pad before whitening to avoid losing the ringdown
                    dt_extra = 4
                    n_inds_extra = int(np.ceil(dt_extra / bank.dt))
                    wf_td = utils.FFTIN(bank.fftsize)
                    wf_td[-n_inds_extra-len(wf):-n_inds_extra] = wf[:]
                    # Whiten the waveform
                    wf_short = np.sqrt(2. * bank.dt) * utils.IRFFT(
                        utils.RFFT(wf_td) * bank.wt_filter_fd)
                else:
                    dt_extra = 0
                    wf_short = wf

                # Subject to highpass to make sure of no artifacts due to
                # generation
                wf_short = signal.sosfiltfilt(
                    bank.sos, wf_short, padlen=bank.irl, axis=-1)

            else:
                raise RuntimeError(f"Injection type {injtype} not recognized")

            # Apply subgrid shift
            # First index > right index
            inj_time += dt_extra
            inj_ind = int(np.ceil((inj_time - times[0]) / dt))
            # inj_ind = np.searchsorted(times, inj_time)
            dt_subgrid = inj_time - (times[0] + dt * inj_ind)
            wf_short_fd = utils.RFFT(wf_short)
            fs_fft = np.fft.rfftfreq(len(wf_short), d=dt)
            wf_short_fd[:] *= np.exp(- 2. * np.pi * 1j * fs_fft * dt_subgrid)
            wf_short[:] = utils.IRFFT(wf_short_fd, n=len(wf_short))

            # globals()["times"] = times.copy()
            # globals()["wf_short"] = wf_short.copy()

            # Find length of injected waveform within the file
            left_ind = inj_ind - len(wf_short)
            if (inj_ind < 0) or (left_ind > len(strain)):
                continue

            wf_left_ind = 0
            wf_right_ind = len(wf_short)
            if left_ind < 0:
                wf_left_ind = -left_ind
                left_ind = 0
            if inj_ind > len(strain):
                wf_right_ind = len(wf_short) - (inj_ind - len(strain))
                inj_ind = len(strain)
            # len_inject = min(inj_ind, len(strain)) - min(left_ind, len(strain))

            # globals()["lstrain"] = len(strain)
            # globals()["inj_ind"] = inj_ind
            # globals()["lwf"] = len(wf_short)

            # Define injected data
            injected_wf = utils.FFTIN(len(strain))
            injected_wf[left_ind:inj_ind] = wf_short[wf_left_ind:wf_right_ind]

            # Inject into data with and without noise
            strain += injected_wf
            noiseless_stream += injected_wf

        # Restore bank's old conditioning parameters if needed
        if dt != old_dt:
            bank.set_waveform_conditioning(
                old_fftsize, old_dt, old_wt_filter_fd)

        return noiseless_stream

    @staticmethod
    def get_glitch_thresholds(
            wt_wfs_cos, dt, preserve_max_snr, sine_gaussian_intervals,
            bandlim_transient_intervals, excess_power_intervals,
            include_HM_wfs=False, bank=None):
        """
        Function giving test-statistic values for various glitch tests attained
        in the presence of noiseless waveforms
        :param wt_wfs_cos: Whitened waveform(s) with SNR = 1
                           Array of size: nwfs x fftsize
                           Combine the modes first if you want to pass a waveform with HM
        :param dt: Sampling interval (s)
        :param preserve_max_snr:
            We assure that waveforms with this SNR are preserved
        :param sine_gaussian_intervals:
        :param bandlim_transient_intervals:
        :param excess_power_intervals:
        :param include_HM_wfs: Boolean flag whether to add to wt_wfs_cos specific 
                               waveforms simulated using parameters from
                               template bank metadata
        :param bank: Instance of TemplateBank
        :return: 1. Threshold, and index for sigma clipping
                 2. Thresholds, and indices for sine gaussian transients
                 3. Thresholds, and indices for bandlimited power transients
                 4. Thresholds, and indices for excess power
        """
        
        if include_HM_wfs:
            if bank.glitch_threshold_wf_params is None: include_HM_wfs=False
        
        if include_HM_wfs:
            wt_wfs_cos_original = wt_wfs_cos.copy()
            
            intervals_bank = bank.glitch_threshold_intervals_saved
            _ = np.array(bank.glitch_threshold_wf_params)
            calphas = _[:,-2:]; [r33, r44, phiRef] = _[:,:-2].T
            
            h = bank.gen_whitened_wfs_td(calpha=calphas, orthogonalize=True)
            h_hilbert = utils.hilbert_transform(h).T
            h = h.T
            extra_wfs = (np.cos(2*phiRef)*h[:,0,:] - np.sin(2*phiRef)*h_hilbert[:,0,:])+\
                    r33*(np.cos(3*phiRef)*h[:,1,:] - np.sin(3*phiRef)*h_hilbert[:,1,:])+\
                    r44*(np.cos(4*phiRef)*h[:,2,:] - np.sin(4*phiRef)*h_hilbert[:,2,:])
                    
            extra_wfs /= np.linalg.norm(extra_wfs, axis=0)
            extra_wfs = extra_wfs.T
            
            # Appending the wf for the sigma clipping case below
            wt_wfs_cos = np.r_[wt_wfs_cos_original, [extra_wfs[0]]]
            
        # 1. Compute sigma clipping threshold (units of strain^2)
        # ----------------------------------------------------
        wt_wfs_fd = utils.RFFT(wt_wfs_cos, axis=-1)
        wt_wfs_sin = utils.IRFFT(wt_wfs_fd * 1j, n=wt_wfs_cos.shape[-1], axis=-1)
        envelope = np.sqrt(wt_wfs_cos ** 2 + wt_wfs_sin ** 2)
        maxind_sc = np.argmax(envelope)

        if envelope.ndim > 1:
            sigma_clipping_threshold = (preserve_max_snr * envelope[
                maxind_sc // envelope.shape[-1],
                maxind_sc % envelope.shape[-1]])**2
        else:
            sigma_clipping_threshold = \
                (preserve_max_snr * envelope[maxind_sc])**2

        maxind_sc = maxind_sc % envelope.shape[-1]

        # 2. Compute sine gaussian thresholds (units of SNR^2)
        # -------------------------------------------------
        sine_gaussian_thresholds = []
        sine_gaussian_maxinds = []
        for sg_interval in sine_gaussian_intervals:
            # First define cosine and sine pulses
            fc, df = sg_interval
            if include_HM_wfs:
                if list(sg_interval) in intervals_bank:
                    wt_wfs_cos = np.r_[wt_wfs_cos_original,
                                [extra_wfs[intervals_bank.index(list(sg_interval))]]]
            cos_pulse, sin_pulse, _ = utils.sine_gaussian(dt, fc, df)
            # Change them to fftsize
            cos_pulse_fft = utils.change_filter_times_td(
                cos_pulse, len(cos_pulse), wt_wfs_cos.shape[-1])
            sin_pulse_fft = utils.change_filter_times_td(
                sin_pulse, len(sin_pulse), wt_wfs_cos.shape[-1])
            # Run these over the waveform
            overlaps_cos = utils.IRFFT(
                utils.RFFT(cos_pulse_fft) * wt_wfs_fd, axis=-1)
            overlaps_sin = utils.IRFFT(
                utils.RFFT(sin_pulse_fft) * wt_wfs_fd, axis=-1)
            overlaps = np.sqrt(overlaps_cos ** 2 + overlaps_sin ** 2)
            maxind_sg = np.argmax(overlaps)

            if overlaps.ndim > 1:
                threshold = (preserve_max_snr * overlaps[
                    maxind_sg // overlaps.shape[-1],
                    maxind_sg % overlaps.shape[-1]])**2
            else:
                threshold = (preserve_max_snr * overlaps[maxind_sg])**2

            sine_gaussian_thresholds.append(threshold)
            sine_gaussian_maxinds.append(maxind_sg % overlaps.shape[-1])

        # 3. Compute power thresholds for band-limited transients
        # ----------------------------------------------------
        bandlim_power_thresholds = []
        bandlim_power_maxinds = []
        
        for bandlim_interval in bandlim_transient_intervals:
            interval, frange = bandlim_interval
            # First check that the interval resolves the transient
            if interval < (2 / (frange[1] - frange[0])):
                raise RuntimeError(
                    "Banded transient is underresolved!")
            if include_HM_wfs:
                if list(bandlim_interval) in intervals_bank:
                    wt_wfs_cos = np.r_[wt_wfs_cos_original,
                                [extra_wfs[intervals_bank.index(list(bandlim_interval))]]]
            
            # Define the bandpass filter cutting at half interval
            ninds = int(interval / dt)
            tfft = ninds * dt
            fmin = (np.round(frange[0] * tfft) - 0.5) / tfft
            fmax = (np.round(frange[1] * tfft) + 0.5) / tfft
            # b, a, irl = utils.band_filter(
            #     dt, fmin=fmin, fmax=fmax, btype='bandpass')
            sos, irl = utils.band_filter(
                dt, fmin=fmin, fmax=fmax, btype='bandpass')

            # Pad waveforms on the right to avoid wraparound before bandpassing
            # Assumes irl + support_whitened_wf < fftsize
            if wt_wfs_cos.ndim > 1:
                wt_wfs_padded = utils.FFTIN(
                    (len(wt_wfs_cos), wt_wfs_cos.shape[-1] + irl))
                wt_wfs_padded[:, :wt_wfs_cos.shape[-1]] = wt_wfs_cos[:]
            else:
                wt_wfs_padded = utils.FFTIN(wt_wfs_cos.shape[-1] + irl)
                wt_wfs_padded[:wt_wfs_cos.shape[-1]] = wt_wfs_cos[:]

            # Filter the waveforms
            # wt_wfs_filt = signal.filtfilt(b, a, wt_wfs_padded, axis=-1)
            wt_wfs_filt = signal.sosfiltfilt(sos, wt_wfs_padded, axis=-1)

            # Pad by zero to get the first entry into the cumsums
            if wt_wfs_cos.ndim > 1:
                wt_wfs_filt = np.c_[np.zeros(len(wt_wfs_filt)), wt_wfs_filt]
            else:
                wt_wfs_filt = np.r_[0, wt_wfs_filt]

            # Compute the thresholds
            cmsm_wfs = np.cumsum(wt_wfs_filt ** 2, axis=-1)
            if wt_wfs_cos.ndim > 1:
                bandlim_power = cmsm_wfs[:, ninds:] - cmsm_wfs[:, :-ninds]
            else:
                bandlim_power = cmsm_wfs[ninds:] - cmsm_wfs[:-ninds]
            maxind_bandlim = np.argmax(bandlim_power)
            if wt_wfs_cos.ndim > 1:
                threshold = preserve_max_snr ** 2 * bandlim_power[
                        maxind_bandlim // bandlim_power.shape[-1],
                        maxind_bandlim % bandlim_power.shape[-1]]
            else:
                threshold = \
                    preserve_max_snr ** 2 * bandlim_power[maxind_bandlim]

            bandlim_power_thresholds.append(threshold)
            bandlim_power_maxinds.append(
                (maxind_bandlim % bandlim_power.shape[-1]) + (ninds // 2))

        # 4. Compute power detection thresholds
        # ----------------------------------
        # Barak: We would use these as the non-centrality parameters, for the
        # actual determination of the threshold we need to know the number of
        # deg of freedom, which we can know only much later, in real time ...
        # Pad by zero to get the first entry into the cumsums
        
        if include_HM_wfs:
            wt_wfs_cos = wt_wfs_cos_original.copy()
            for excess_power_interval in excess_power_intervals:
                if excess_power_interval in intervals_bank:
                    wt_wfs_cos = np.r_[wt_wfs_cos,[extra_wfs[
                                    intervals_bank.index(excess_power_interval)]] ]
        
        if wt_wfs_cos.ndim > 1:
            wt_wfs_cos_padded = np.c_[np.zeros(len(wt_wfs_cos)), wt_wfs_cos]
        else:
            wt_wfs_cos_padded = np.r_[0, wt_wfs_cos]

        # Compute the thresholds
        cmsm_wfs = np.cumsum(wt_wfs_cos_padded ** 2, axis=-1)

        excess_power_thresholds = []
        excess_power_maxinds = []
        for excess_power_interval in excess_power_intervals:
            ninds = int(excess_power_interval / dt)
            if wt_wfs_cos.ndim > 1:
                cmsm_diff = cmsm_wfs[:, ninds:] - cmsm_wfs[:, :-ninds]
            else:
                cmsm_diff = cmsm_wfs[ninds:] - cmsm_wfs[:-ninds]
            maxind_power = np.argmax(cmsm_diff)
            if wt_wfs_cos.ndim > 1:
                threshold = preserve_max_snr ** 2 * cmsm_diff[
                        maxind_power // cmsm_diff.shape[-1],
                        maxind_power % cmsm_diff.shape[-1]]
            else:
                threshold = preserve_max_snr ** 2 * cmsm_diff[maxind_power]

            excess_power_thresholds.append(threshold)
            excess_power_maxinds.append(
                (maxind_power % cmsm_diff.shape[-1]) + (ninds // 2))

        return (sigma_clipping_threshold, maxind_sc), \
               (sine_gaussian_thresholds, sine_gaussian_maxinds), \
               (bandlim_power_thresholds, bandlim_power_maxinds), \
               (excess_power_thresholds, excess_power_maxinds)

    # Functions to compute SNR corrections due to imperfections in the data
    # -------------------------------------------------------------------------
    # Static to use on subsets of data
    @staticmethod
    def hole_snr_correction(
            wfs_whitened_fd, chunked_mask_f, fftsize, support_whitened_wf,
            cheat_overlap_save=False, only_22=False, erase_bands=False,
            **erase_bands_kwargs):
        """
        Computes SNR loss due to the mask (with appropriately inpainted data)
        Uses the stationary phase approximation, so valid only when the
        frequencies >> 1/hole size. We do not assume waveforms are normalized.
        Note: extended by Jay to include corrections due to bands in mask_stft
        :param wfs_whitened_fd:
            nwf x 3 x length rfft(fftsize) complex array with frequency domain
            waveforms (can be vector for nwf = 1)
            Convention: power is towards the right side
        :param chunked_mask_f: Chunked FFT of mask
        :param fftsize: FFTsize for chunked mask
        :param support_whitened_wf: Support of whitened waveform
        :param cheat_overlap_save:
            Flag indicating to not use support_whitened_wf in overlap-save
        :param only_22: Use only 22 mode instead of the 3 modes everywhere
        :param erase_bands: Flag to erase bands in time-frequency space
        :param erase_bands_kwargs:
            If erase_bands, pass corresponding kwargs: mask_stft, noverlaps, dt
        :return: nwf x 3 x (nchunk x chunksize) array with hole corrections
                 (use first len(data) values in each row)
                 Can be vector if wf_whitened_fd is a vector
        """
        # Whitened time domain sine and cosine waveforms
        wfs_whitened_td_cos = utils.IRFFT(wfs_whitened_fd, n=fftsize, axis=-1)
        wfs_whitened_td_sin = utils.IRFFT(
            1j * wfs_whitened_fd, n=fftsize, axis=-1)

        # Envelope of waveforms
        envelopes = wfs_whitened_td_cos**2 + wfs_whitened_td_sin**2
        env_norms = np.sum(envelopes, axis=-1)

        # Truncate for overlap save
        envelopes[..., :-support_whitened_wf] = 0
        envelopes_fft = utils.RFFT(envelopes, axis=-1).conj()

        # Compute SNR correction
        if cheat_overlap_save:
            support_whitened_wf = 1
            
        chunksize = fftsize - support_whitened_wf + 1
        if chunked_mask_f.ndim == 1:
            nchunk = 1
        else:
            nchunk = len(chunked_mask_f)
            
        if only_22:
            if envelopes.ndim == 1:
                # Just one waveform
                corrections = (np.abs(d_ops.overlap_save(chunked_mask_f, envelopes_fft,
                                fftsize, support_whitened_wf) / env_norms)) ** 0.5
            else:
                # Multiple waveforms
                nwf = len(envelopes)
                corrections = utils.FFTIN((nwf, nchunk * chunksize))
                for i in range(nwf):
                    corrections[i, :] = (np.abs(d_ops.overlap_save(
                        chunked_mask_f, envelopes_fft[i], fftsize,
                        support_whitened_wf) / env_norms[i])) ** 0.5
        
        else:
            if envelopes.ndim == 2:
                # Just one waveform
                corrections = utils.FFTIN((3, nchunk * chunksize))
                for j in range(3): # corresponding to modes
                    corrections[j] = (np.abs(d_ops.overlap_save(
                        chunked_mask_f, envelopes_fft[j], fftsize,
                        support_whitened_wf) / env_norms[j])) ** 0.5
            else:
                # Multiple waveforms
                nwf = len(envelopes)
                corrections = utils.FFTIN((nwf, 3, nchunk * chunksize))
                for i in range(nwf):
                    for j in range(3):
                        corrections[i, j, :] = (np.abs(d_ops.overlap_save(
                            chunked_mask_f, envelopes_fft[i,j], fftsize,
                            support_whitened_wf) / env_norms[i,j])) ** 0.5
        
        # For the case when band eraser is applied
        if erase_bands:
            mask_stft = erase_bands_kwargs.get('mask_stft')
            dt = erase_bands_kwargs.get('dt')
            subset = erase_bands_kwargs.get('subset', False)
            if subset:
                left_ind = erase_bands_kwargs.get('relevant_index') - \
                                 erase_bands_kwargs.get('left_inds')
                right_ind = erase_bands_kwargs.get('relevant_index') + \
                                 erase_bands_kwargs.get('right_inds')
                start = int(left_ind*dt) * int(1/dt)
                mask_stft = mask_stft[:,int(left_ind*dt):int(right_ind*dt)+2]
                left_ind -= start; right_ind -= start
                noverlaps = right_ind - left_ind
            else:
                noverlaps = erase_bands_kwargs.get('noverlaps')
                
            (fs_len, t_len) = mask_stft.shape
                
            mask_symmetric = np.ones( ((fs_len-1)*2, mask_stft.shape[1]), dtype=bool)
            mask_symmetric[:fs_len,:] = mask_stft
            mask_symmetric[fs_len:,:] = np.delete(mask_stft[::-1,:],[0,-1], axis=0)
            noisy_freq_bands = np.sum(1-mask_symmetric, axis=1)>0
            # If none of the freq bands are noisy, skip the hole correction due to mask_stft
            if np.any(noisy_freq_bands)==False:
                return corrections
            mask_symmetric = mask_symmetric[noisy_freq_bands]
            
            wfs_whitened_td_cos = (wfs_whitened_td_cos.T/np.linalg.norm(
                                                        wfs_whitened_td_cos, axis=-1)).T
            wfs_whitened_td_cos = np.fft.fftshift(wfs_whitened_td_cos, axes=-1)
            nperseg = int(params.LINE_TRACKING_DT / dt)
            t_ind = np.arange(int((t_len-1)/dt))
            sampling_rate = 2 # in Hz, this is an arb. choice but keep in powers of two
                        
            def corrections_template (template_original):
            
                corrections_sampled = np.zeros(int((t_len-1)*sampling_rate))
                
                for i_shift in range(sampling_rate): 
                # corresponding to shifting the template in time within the STFT elements
                # (each STFT element is 1 sec, which equals 1/dt time bins)
                # currently template shifted in units of 0.5 sec (arb. choice)
                    
                    template = np.roll(template_original, int(i_shift/sampling_rate/dt))
                    
                    power_temp = (
                    signal.stft(template, nperseg=nperseg, noverlap=nperseg//2,
                     window='hann', return_onesided=False, fs=1/dt)[2] *\
                    signal.stft(template, nperseg=nperseg, noverlap=nperseg//2,
                     window='boxcar', return_onesided=False, fs=1/dt)[2].conj() *\
                    (nperseg//2) ).real
                    
                    corr_sq = np.sum(signal.fftconvolve(
                                            mask_symmetric,
                                            power_temp[:,::-1][noisy_freq_bands],
                                            axes=-1, mode='same')[:,:-1].real
                                        , axis=0)
                    corr_sq += np.sum(power_temp[~noisy_freq_bands])
                    # correcting errors caused by numerical precision
                    corr_sq[corr_sq<0]=0.; corr_sq[corr_sq>1]=1.
                    corrections_sampled[i_shift::sampling_rate] = np.sqrt(corr_sq)
                
                corrections_sampled = np.interp(t_ind, t_ind[::int(1/dt/sampling_rate)],
                                     corrections_sampled)
                if subset:
                    return (corrections_sampled[left_ind:right_ind])
                else:
                    return(corrections_sampled[-noverlaps:])
                
        
            if only_22:
                if wfs_whitened_td_cos.ndim == 1:
                    # Just one waveform
                    corrections_BE = corrections_template(wfs_whitened_td_cos)
                else:
                    # Multiple waveforms
                    nwf = len(wfs_whitened_td_cos)
                    corrections_BE = utils.FFTIN((nwf, noverlaps))
                    for i in range(nwf):
                        corrections_BE[i, :] = corrections_template(
                                                            wfs_whitened_td_cos[i])
                                
            else:
                if wfs_whitened_td_cos.ndim == 2:
                    # Just one waveform
                    corrections_BE = utils.FFTIN((3, noverlaps))
                    for j in range(3): # corresponding to modes
                        corrections_BE[j] = corrections_template(wfs_whitened_td_cos[j])
                else:
                    # Multiple waveforms
                    nwf = len(wfs_whitened_td_cos)
                    corrections_BE = utils.FFTIN((nwf, 3, noverlaps))
                    for i in range(nwf):
                        for j in range(3):
                            corrections_BE[i, j, :] = corrections_template(
                                                        wfs_whitened_td_cos[i,j])

            corrections[..., :noverlaps] *= corrections_BE # BE = band eraser
            
            return corrections
            
        return corrections

    def gen_psd_drift_correction(
            self, wf_whitened_fd=None, calphas=None, jump=None,
            tol=params.PSD_DRIFT_TOL, avg='median', override_window_size=None,
            verbose=True, indices=None, return_only_at_indices=False):
        """Computes the proper normalization for varying PSD, optionally
        returns correction, scores, and score-indices used at a particular index
        # Warning: Currently only the 22 wf is used
        # Warning: Doesn't apply a linear-free correction to the indices,
        ensure its use is consistent
        :param wf_whitened_fd:
            Array of size nmode x len(rfftfreq(fftsize)) with
             frequency domain waveform
        :param calphas:
            Set of calphas for waveform, used if waveform itself isn't provided
        :param jump:
            In unit of index (a reasonable choice is 1/dt) Ensure that jump is
            smaller than support_whitened_wf. If None, uses default in class
            (used this order to ensure that Matias's code works)
        :param tol: Tolerance on correction factor^2 (stdev of this qty/mean)
        :param avg:
            Flag indicating the average to use. Can be one of
            mean, median, trimmedmean, and safemean
        :param override_window_size: Override the window size (in indices)
        :param verbose: Flag indicating whether to print details of computation
        :param indices:
            Return window used to compute correction at indices within this list
        :param return_only_at_indices: Flag to restrict computation to indices
        :return: if indices is not None
                    if return_only_at_indices:
                        PSD drift corrections at indices,
                        array with [leftind, rightind] used for score at indices
                    else:
                        Array of length self.time with PSD drift correction
                        array with [leftind, rightind] used for score at indices
                 else:
                    Array of length self.time with PSD drift correction
        """
        # Define jump in units of indices, if not given
        if jump is None:
            jump = int(np.round(self.psd_drift_interval / self.dt))

        bank = self.templatebank
        if wf_whitened_fd is None:
            # Generate from the bank
            wf_whitened_td = bank.gen_whitened_wfs_td(calphas)
            # Change to the full domain
            wf_whitened_td = utils.change_filter_times_td(
                wf_whitened_td, bank.fftsize, self.fftsize)
            wf_whitened_fd = utils.RFFT(wf_whitened_td, axis=-1)
        else:
            # Change if needed
            wf_whitened_fd = utils.change_filter_times_fd(
                wf_whitened_fd, 2 * (wf_whitened_fd.shape[-1] - 1),
                self.fftsize, pad_mode='left')

        # Ensure that the window is longer than the jump
        if override_window_size is None:
            # Estimate number of samples needed (from FD argument)
            # This applies for the calculation using the mean
            # If we use the median, worsens the tolerance
            # (by a factor ~1.6 in the uncorrelated case, measured to 1.2 for
            # Mcbin3 bank0)
            nsamp = (2. * 2. * np.sum(np.abs(wf_whitened_fd[0,:]) ** 4) /
                     (tol ** 2 * self.fftsize))
            window_size = int(np.max([np.ceil(nsamp), 4 * jump]))
        else:
            window_size = int(override_window_size)

        if verbose:
            print("PSD window size: ", window_size)
            if window_size < bank.support_whitened_wf:
                warnings.warn("PSD drift correction scale might be shorter " +
                              "than waveform!", Warning)

        # Number of windows
        n_var_corrections = len(self.time) // jump - (window_size // jump - 1)
        std_pr = np.zeros(n_var_corrections)

        # First compute hole-corrected scores for the given template
        overlaps, hole_corrections, valid_inds = \
            self.gen_scores(wfs_whitened_fd=wf_whitened_fd[0,:],
                            calphas=calphas, only_22=True)
        # We've only used the 22 wf as of now

        h_corr_overlaps = overlaps / (hole_corrections + params.HOLE_EPS)

        # Record the indices of windows that encompass indices of interest
        # that the user passed
        if indices is not None:
            indices = np.asarray(indices)
            indices_window = \
                np.round((indices - window_size // 2) / jump).astype(int)
            indices_window[indices_window < 0] = 0
            indices_window[indices_window > (n_var_corrections - 1)] = \
                n_var_corrections - 1
        else:
            indices_window = []

        # Decide if we want to compute the correction for all windows or
        # only those of interest
        if return_only_at_indices and (indices is not None):
            windows_to_compute = indices_window
        else:
            windows_to_compute = range(n_var_corrections)

        data_indices_window = []
        for i in windows_to_compute:
            valid_sub = valid_inds[i * jump:i * jump + window_size]
            # Always average window_size indices to avoid `regression-to-mean'
            # artifacts due to fewer samples near holes
            nvalid_inds = np.count_nonzero(valid_sub)
            extra_req = window_size - nvalid_inds
            if extra_req > 0:
                # Define index limits
                left_ind, right_ind = utils.index_limits(
                    i, extra_req, jump, window_size, valid_inds)
            else:
                left_ind = i * jump
                right_ind = i * jump + window_size

            # Apply the index range
            valid_sub = valid_inds[left_ind:right_ind]
            over_sub = h_corr_overlaps[left_ind:right_ind]
            rel_scores = over_sub[valid_sub]

            if avg.lower() == 'median':
                cos_sigma = utils.sigma_from_median(rel_scores.real)
                sin_sigma = utils.sigma_from_median(rel_scores.imag)
                std_pr[i] = np.sqrt(0.5 * (cos_sigma**2 + sin_sigma**2))

            elif avg.lower() == 'mean':
                # Warning, not safe to injections at relevant SNRs...
                std_pr[i] = np.sqrt(np.var(rel_scores)/2)

            elif avg.lower() in ['trimmedmean', 'safemean']:
                # First estimate the standard-deviation using the median
                cos_sigma = utils.sigma_from_median(rel_scores.real)
                sin_sigma = utils.sigma_from_median(rel_scores.imag)
                std_median = np.sqrt(0.5 * (cos_sigma**2 + sin_sigma**2))

                # To make the mean robust, clip points that should barely
                # be achieved over this interval
                thresh = std_median**2 * stats.chi2.isf(
                    params.PSD_DRIFT_SAFEMEAN_THRESH/len(rel_scores), 2)
                good_inds = utils.abs_sq(rel_scores) < thresh

                if avg.lower() == 'safemean':
                    # If the outlier is due to a signal from the bank, it
                    # comes with friends that affect the PSD drift correction
                    # at the percent level if neglected, so we throw out a
                    # bit more from the part used to compute the average
                    safety = int(params.PSD_DRIFT_SAFETY_LEN / self.dt)
                    bad_limits = utils.hole_edges(good_inds)
                    for left_edge, right_edge in bad_limits:
                        good_inds[
                            max(0, left_edge - safety):
                            right_edge + safety] = False

                    # TODO: In an edge case, we can lose precision if we throw
                    #  too much out, so maybe we should increase the window
                    #  size to compensate if needed?
                rel_scores = rel_scores[good_inds]
                std_pr[i] = np.sqrt(np.var(rel_scores) / 2)
            else:
                raise RuntimeError(f"Averaging method {avg} not defined!")

            # Record left and right indices of window for index of interest
            if i in indices_window:
                data_indices_window.append([left_ind, right_ind].copy())

        # Define return value
        if return_only_at_indices and (indices is not None):
            return std_pr[indices_window], np.array(data_indices_window)

        # Returning full array
        # Pad upto midpoint of first window to center events within windows
        final_pad_length = len(self.strain) - \
                           (n_var_corrections * jump + window_size//2)
        padded_stdpr = np.pad(np.repeat(std_pr, jump),
                              (window_size//2, final_pad_length), 'edge')

        # Don't correct if there were no valid overlaps due to a large hole
        invalid = np.isnan(padded_stdpr)
        padded_stdpr[invalid] = 1.0

        if indices is not None:
            return padded_stdpr, np.array(data_indices_window)
        else:
            return padded_stdpr

    # Functions to generate triggers
    # -------------------------------------------------------------------------
    def prepare_for_triggers(self, average=None):
        """
        Set things up to generate triggers
        :param average:
            Method to compute the PSD drift correction
            (see gen_psd_drift_correction for options, defaults to class setup)
        :return:
        """
        if average is None:
            average = self.average

        # Compute and save the chunked ffts of downsampled data and mask
        self.chunked_data_f = d_ops.chunkedfft(
            self.strain, self.fftsize, self.templatebank.support_whitened_wf)
        mask_float = self.mask.astype(float)
        self.chunked_mask_f = d_ops.chunkedfft(
            mask_float, self.fftsize, self.templatebank.support_whitened_wf)

        # Compute correction for SNR normalization for varying PSD
        if self.psd_drift_correction is None:
            if self.psd_drift_interval > 0:
                # Generate central frequency domain waveform
                wf_whitened_td = self.templatebank.gen_whitened_wfs_td()
                wf_whitened_fd = utils.RFFT(wf_whitened_td, axis=-1)
                # Compute PSD drift correction
                self.psd_drift_correction = \
                    self.gen_psd_drift_correction(wf_whitened_fd, avg=average)

        return

    def prepare_subset_for_triggers(
            self, relevant_index, left_inds_scores, right_inds_scores,
            zero_pad=True):
        """
        Prepare to compute everything on a reduced set of data to save time
        for BH-like waveforms. Defines subset of data, performs relevant FFTs,
        and stores in class variables
        Note that if relevant_index + right_inds_scores goes past the edge of
        the data, then we will have zeros in all the elements for which we
        don't have data
        :param relevant_index:
            Index into self.time for trigger score (right edge of waveform)
            If 0, ensure that len(self.time) is passed instead
        :param left_inds_scores:
            Left indices in scores to guarantee w.r.t relevant index,
            excluding relevant index
        :param right_inds_scores:
            Right indices in scores to guarantee w.r.t relevant index,
            excluding relevant index
        :param zero_pad:
            Flag indicating whether to zero pad, or pad with existing data
        """
        lwf = self.templatebank.support_whitened_wf

        # Amount of data to take on each side for overlaps to be valid
        # We do it this way in case the trigger is too close to the edge
        # Can happen because of ninds and support
        more_left_data = True
        more_right_data = True
        left_inds = lwf + left_inds_scores
        if left_inds > relevant_index:
            left_inds = relevant_index
            more_left_data = False
        # Add 1 since the conventions of where the data lies, and the
        # PSD drift correction are shifted by 1
        right_inds = max(1, right_inds_scores + 1)
        if right_inds > (len(self.time) - relevant_index):
            right_inds = len(self.time) - relevant_index
            more_right_data = False
        fftsize_sub = left_inds + right_inds
        if fftsize_sub < lwf:
            # Pull in data from the side in which it is available
            if more_left_data:
                left_inds += lwf - fftsize_sub
            elif more_right_data:
                right_inds += lwf - fftsize_sub
            else:
                raise RuntimeError("Data length isn't enough!")
        ldat = left_inds + right_inds
        # Ensure fftsize is a power of 2
        fftsize_sub = utils.next_power(ldat)

        # Define subsets of data
        self.strain_sub = utils.FFTIN(fftsize_sub)
        self.mask_sub = utils.FFTIN(fftsize_sub, dtype=bool)
        self.valid_mask_sub = utils.FFTIN(fftsize_sub, dtype=bool)
        self.psd_drift_correction_sub = utils.FFTIN(fftsize_sub)

        # If right_inds == 0, our matched filtering indexing convention
        # makes it okay to not include strain[relevant_index], but we need the
        # PSD drift correction to be included
        self.time_sub = self.time[relevant_index - left_inds] + \
            self.dt * np.arange(fftsize_sub)
        self.strain_sub[:ldat] = self.strain[relevant_index - left_inds:
                                             relevant_index + right_inds]
        self.mask_sub[:ldat] = self.mask[relevant_index - left_inds:
                                         relevant_index + right_inds]
        self.valid_mask_sub[:ldat] = \
            self.valid_mask[relevant_index - left_inds:
                            relevant_index + right_inds]
        self.psd_drift_correction_sub[:ldat] = \
            self.psd_drift_correction[relevant_index - left_inds:
                                      relevant_index + right_inds]

        # Record what was used to make the subset
        self.relevant_index = relevant_index
        self.left_inds = left_inds
        self.right_inds = right_inds

        if not zero_pad:
            num_zeros = fftsize_sub - ldat
            max_extra_inds = len(self.strain) - (relevant_index + right_inds)
            pad_length = min(num_zeros, max_extra_inds)

            # Update right inds
            self.right_inds += pad_length

            self.strain_sub[ldat:ldat + pad_length] = \
                self.strain[relevant_index + right_inds:
                            relevant_index + right_inds + pad_length]
            self.mask_sub[ldat:ldat + pad_length] = \
                self.mask[relevant_index + right_inds:
                          relevant_index + right_inds + pad_length]
            self.valid_mask_sub[ldat:ldat + pad_length] = \
                self.valid_mask[relevant_index + right_inds:
                                relevant_index + right_inds + pad_length]
            self.psd_drift_correction_sub[ldat:ldat + pad_length] = \
                self.psd_drift_correction[
                    relevant_index + right_inds:
                    relevant_index + right_inds + pad_length]

        # Use window length = 1 to cheat overlap-save
        self.chunked_data_sub = d_ops.chunkedfft(
            self.strain_sub, fftsize_sub, 1)
        self.chunked_mask_sub = d_ops.chunkedfft(
            self.mask_sub.astype(float), fftsize_sub, 1)

        # Define offset between index in subset and the global file, negative
        # number to be added to global index to get subset index. Convention
        # is that if left_inds == 0, it turns up as fftsize_sub instead
        self.offset_sub = left_inds - relevant_index

        return

    @staticmethod
    def scores_wf(
            wfs_whitened_fd, chunked_data_f, chunked_mask_f, valid_mask,
            fftsize, support_whitened_wf, cheat_overlap_save=False,
            zero_invalid=True, only_22=False, erase_bands=False, **erase_bands_kwargs):
        """
        Computes overlaps and hole corrections for multiple waveforms
        We do not assume waveforms are normalized
        :param wfs_whitened_fd:
            nwf x 3 x length rfft(fftsize) complex array with frequency domain
            waveforms
            Convention: power is towards the right side
        :param chunked_data_f: Chunked FFT of data
        :param chunked_mask_f: Chunked FFT of mask
        :param valid_mask: Valid mask
        :param fftsize: FFTsize for chunked data and mask
        :param support_whitened_wf: Support of whitened waveform
        :param cheat_overlap_save:
            Flag indicating to not use support_whitened_wf in overlap-save
        :param zero_invalid: Flag indicating whether to zero invalid scores
        :param only_22: Only use 22 instead of the 3 modes everywhere
        :param erase_bands: Flag to erase bands in time-frequency space
        :param erase_bands_kwargs:
            If erase_bands, pass corresponding kwargs (mask_stft, noverlaps, dt)
        :return:
            1. nwf x 3 x (nchunk x chunksize) array with overlaps
               (zero where the hole correction cannot be trusted)
            2. nwf x 3 x (nchunk x chunksize) array with hole corrections
            3. nwf x 3 x (nchunk x chunksize) Boolean array indicating validity of
               overlaps
            Use first len(data) values of both, can be vectors if wf_whitened_fd
            is a vector
        """
        # First compute hole correction for the given template
        hole_corrections = TriggerList.hole_snr_correction(
            wfs_whitened_fd, chunked_mask_f, fftsize, support_whitened_wf,
            cheat_overlap_save=cheat_overlap_save, only_22=only_22, 
            erase_bands=erase_bands, **erase_bands_kwargs)
                
        # Then compute the scores for the given templates
        if cheat_overlap_save:
            support_whitened_wf = 1
            
        chunksize = fftsize - support_whitened_wf + 1
        if chunked_data_f.ndim == 1:
            nchunk = 1
        else:
            nchunk = len(chunked_data_f)
            
        if only_22:
            
            if len(wfs_whitened_fd.shape) == 1:
                # Just one waveform
                overlaps = d_ops.norm_matched_filter_overlap(
                    chunked_data_f, wfs_whitened_fd, fftsize, support_whitened_wf)
            else:
                # Multiple waveforms
                nwf = len(wfs_whitened_fd)
                overlaps = utils.FFTIN(
                    (nwf, nchunk * chunksize), dtype=np.complex128)
                for i in range(nwf):
                        overlaps[i, :] = d_ops.norm_matched_filter_overlap(
                            chunked_data_f, wfs_whitened_fd[i], fftsize,
                            support_whitened_wf)
            
        else:
        
            if len(wfs_whitened_fd.shape) == 2:
                # Just one waveform
                overlaps = utils.FFTIN((3, nchunk * chunksize), dtype=np.complex128)
                for j in range(3): # num. of modes
                    overlaps[j] = d_ops.norm_matched_filter_overlap(
                        chunked_data_f, wfs_whitened_fd[j], fftsize, support_whitened_wf)
            else:
                # Multiple waveforms
                nwf = len(wfs_whitened_fd)
                overlaps = utils.FFTIN(
                    (nwf, 3, nchunk * chunksize), dtype=np.complex128)
                for i in range(nwf):
                    for j in range(3): # num. of modes
                        overlaps[i, j, :] = d_ops.norm_matched_filter_overlap(
                            chunked_data_f, wfs_whitened_fd[i, j], fftsize,
                            support_whitened_wf)

        # Find valid inds
        # h_corr_overlaps = overlaps / (hole_corrections + params.HOLE_EPS)
        valid_inds = (hole_corrections >= params.HOLE_CORRECTION_MIN)

        valid_inds[..., :len(valid_mask)] *= valid_mask
            
        if not only_22:
            # Final valid inds are combination of those for individual modes
            # I have included the mode dimension so np array multiplication is possible
            valid_inds[...,0,:] = valid_inds[...,0,:] * valid_inds[...,1,:] \
                                    * valid_inds[...,2,:]
            valid_inds[...,1,:] = valid_inds[...,0,:]
            valid_inds[...,2,:] = valid_inds[...,0,:]
        
        if zero_invalid:
            # h_corr_overlaps *= valid_inds
            overlaps *= valid_inds

        return overlaps, hole_corrections, valid_inds

    def gen_scores(
            self, wfs_whitened_fd=None, calphas=None, subset=False,
            zero_invalid=True, only_22=False, orthogonalize_modes=True):
        """
        Convenience function to generate overlaps and hole corrections, for
        multiple waveforms or calphas, on full data or a subset of it
        :param wfs_whitened_fd:
            nwf x 3 x length rfft(fftsize) complex array with frequency domain
            waveforms
            Convention: power is towards the right side
        :param calphas:
            Optional nwf x n_pars array with coefficients of basis functions,
            overrides waveforms if given, (can be vector for nwf = 1)
        :param subset:
            Flag indicating whether to generate triggers on full data or subset
        :param zero_invalid: Flag indicating whether to zero invalid scores
        :param only_22: Use only 22 instead of the 3 modes everywhere
        :param orthogonalize_modes: Orthogonalizes the different mode wfs
            If false, returns the covariance matrix between modes 
            (upper triangular elements)
        :return:
            1. nwf x 3 x len(data/data_sub) array with overlaps
               (zero where the hole correction cannot be trusted)
            2. nwf x 3 x len(data/data_sub) array with hole corrections
            3. nwf x 3 x len(data/data_sub) Boolean array indicating validity of
               overlaps
            Can be vectors if wfs_whitened_fd or calphas is a vector
        """
        bank = self.templatebank

        if subset:
            chunked_data_f = self.chunked_data_sub
            chunked_mask_f = self.chunked_mask_sub
            valid_mask = self.valid_mask_sub
            fftsize = len(self.time_sub)
            support_whitened_wf = fftsize
            cheat_overlap_save = True
            noverlaps = fftsize
        else:
            chunked_data_f = self.chunked_data_f
            chunked_mask_f = self.chunked_mask_f
            valid_mask = self.valid_mask
            fftsize = self.fftsize
            support_whitened_wf = bank.support_whitened_wf
            cheat_overlap_save = False
            noverlaps = len(self.time)

        # Use calphas only if waveform(s) was/were not given
        if wfs_whitened_fd is not None:
            # Ensure waveforms live on the correct domain
            wfs_whitened_fd = utils.change_filter_times_fd(
                wfs_whitened_fd, 2 * (wfs_whitened_fd.shape[-1] - 1),
                fftsize, pad_mode='left')
        elif calphas is not None:
            # Ensure waveforms live on the correct domain
            wfs_whitened_td = bank.gen_whitened_wfs_td(calphas,
                                orthogonalize=orthogonalize_modes)
            wfs_whitened_td = utils.change_filter_times_td(
                wfs_whitened_td, bank.fftsize, fftsize, pad_mode='left')
            wfs_whitened_fd = utils.RFFT(wfs_whitened_td, axis=-1)
        else:
            raise RuntimeError("I need to know which waveform to use!")

        # h_corr_overlaps, valid_inds = self.scores_wf(
        #     wfs_whitened_fd, chunked_data_f, chunked_mask_f,
        #     valid_mask, fftsize, support_whitened_wf,
        #     cheat_overlap_save=cheat_overlap_save)
        
        overlaps, hole_corrections, valid_inds = self.scores_wf(
            wfs_whitened_fd, chunked_data_f, chunked_mask_f,
            valid_mask, fftsize, support_whitened_wf,
            cheat_overlap_save=cheat_overlap_save,
            zero_invalid=zero_invalid, only_22=only_22, 
            erase_bands=self.erase_bands, mask_stft=self.mask_stft,
            dt=self.dt, subset=subset, noverlaps=noverlaps, 
            relevant_index=self.relevant_index, left_inds=self.left_inds,
            right_inds=self.right_inds)

        # if len(wfs_whitened_fd.shape) > 1:
        #     # Multiple waveforms
        #     return h_corr_overlaps[:, :noverlaps], valid_inds[:, :noverlaps]
        # else:
        #     # One waveform
        #     return h_corr_overlaps[:noverlaps], valid_inds[:noverlaps]

        return overlaps[..., :noverlaps], hole_corrections[..., :noverlaps], \
                valid_inds[..., :noverlaps]

    def gen_overlaps(
            self, wfs_whitened_fd=None, calphas=None, random_calpha=True):
        """
        Convenience function to generate hole-corrected overlaps on the
        whole file
        The first three parameters are prefered in order of appearance
        :param wfs_whitened_fd:
            Array of n_wf x 3 x len(rfftfreq(fftsize)) with FFT of waveform
            (can be a vector if n_wf = 1)
        :param calphas:
            Array of n_wf x calphas with coefficients of basis for template
            (can be a vector if n_wf = 1)
        :param random_calpha: Flag to query the generator for a random template
        :return:
        """
        if wfs_whitened_fd is not None:
            wf_params = None
        elif calphas is not None:
            wfs_whitened_fd = utils.RFFT(
                self.templatebank.gen_whitened_wfs_td(calphas), axis=-1)
            wf_params = calphas
        else:
            wg = self.templatebank.wt_waveform_generator(
                self.delta_calpha, orthogonalize=True, random_order=random_calpha)
            wfs_whitened_fd, wf_params = wg.__next__()

        overlaps, hole_corrections, valid_inds = \
            self.gen_scores(wfs_whitened_fd=wfs_whitened_fd, calphas=calphas)
        h_corr_overlaps = overlaps / (hole_corrections + params.HOLE_EPS)

        return wfs_whitened_fd, wf_params, h_corr_overlaps, valid_inds

    def process_clist(self, clist, adjust_times=True):
        """
        Applies correction factor to times and SNRs and makes real array
        clist is assumed to have the following columns:
        if self.save_hole_correction is True:
        time, (overlap.real, overlap.imag) x 3, hole_correction x 3,
        psd_drift_correction, basis_coeff1, basis_coeff2, ...
        else:
        time, (overlap.real, overlap.imag) x 3, psd_drift_correction,
        basis_coeff1, basis_coeff2, ...
        :param clist: n_trigger x (4/5 + n_basis coeff) array with
                      non-shift-corrected time, non-renormalized cos + i sin
                      overlaps, psd drift correction, basis coefficients
                      (can be vector for n_trigger = 1)
        :param adjust_times:
            If True, we adjust times to assign triggers to the linear-free time
            of the calpha waveforms
        :return: n_triggers x (6/7 + basis coeffs) array with time, total SNR^2,
                 (normfac/constant bringing score to amplitude units),
                 (optional hole correction), psd_drift_correction, cosine SNR,
                 sine SNR, and basis coefficients
                 (can be vector for n_trigger = 1)
        """
        if len(clist) == 0:
            return np.array([])
        else:
            # Takes us to linear-free point + 1 index, the latter due to the
            # conj in overlap-save causing a reverse + shift of 1
            bank = self.templatebank

            # Shift to linear-free times
            if adjust_times:
                deltat = bank.shift_whitened_wf * bank.dt
            else:
                deltat = 0

            # Read off some parameters
            if self.save_hole_correction:
                hole_correction = clist[..., 7:10]
                psd_drift_correction = clist[..., 10]
                calphas = clist[..., 11:]
            else:
                hole_correction = None
                psd_drift_correction = clist[..., 7]
                calphas = clist[..., 8:]

            # Array to return
            out_shape = np.array(clist.shape)
            out_shape[-1] += 2
            adjusted_clist = np.zeros(out_shape)

            # Times
            adjusted_clist[..., 0] = clist[..., 0] + deltat

            # Absolute SNR^2
            adjusted_clist[..., 1] = (clist[...,1]** 2 + clist[...,2]** 2
                                    + clist[...,3]** 2 + clist[...,4]** 2
                                    + clist[...,5]** 2 + clist[...,6]** 2) / \
                                    (psd_drift_correction ** 2)

            # Madness to work with old runs
            if self.save_hole_correction:
                # Normfac
                adjusted_clist[..., self.normfac_pos] = bank.normfac

                # Hole correction
                adjusted_clist[...,
                               self.hole_correction_pos:
                               self.hole_correction_pos + 3] = hole_correction
            else:
                # This is the constant that brings the score back to amplitude
                # units (multiply by this). To get distance you still need to
                # correct by a template dependent factor
                adjusted_clist[..., self.normfac_pos] = \
                    psd_drift_correction * bank.normfac

            # Record PSD drift correction
            adjusted_clist[..., self.psd_drift_pos] = psd_drift_correction

            # Cos and Sin SNR
            for i in np.arange(0,6,2):
                adjusted_clist[..., i+self.rezpos] = \
                    clist[..., i+1] / psd_drift_correction
                adjusted_clist[..., i+self.imzpos] = \
                    clist[..., i+2] / psd_drift_correction

            # Basis coefficients
            adjusted_clist[..., self.c0_pos:] = calphas

            return adjusted_clist

    def filter_with_wf(
            self, wf_whitened_fd=None, calphas=None, subset=False,
            apply_threshold=True, zero_invalid_before_sinc_interp=True,
            support=params.SUPPORT_SINC_FILTER, ensure_sinc_support=False,
            recompute_psd_drift_correction=False, marginalized_score_HM=True,
            interpolate_psd_drift_correction=False, return_format='clist',
            adjust_times=1, **psd_drift_kwargs):
        """
        Generate triggers with sinc-interpolated scores with a single template
        for entire data or a subset of it
        :param wf_whitened_fd:
            Array of 3 x len(rfftfreq(fftsize)) with FFT of waveform
        :param calphas:
            Optional array of calphas representing waveform. If wf_whitened_fd
            is None, we use the calphas to generate a waveform
        :param subset:
            Flag indicating whether to filter the waveform with the entire
            data or subset of it
        :param apply_threshold:
            Flag to apply threshold on chi^2 when collecting triggers
        :param zero_invalid_before_sinc_interp:
            Flag indicating whether to zero invalid scores before
            sinc-interpolating. Default set to preserve search, exposed for
            debugging purposes
        :param support: Support of sinc-interpolating filter
        :param ensure_sinc_support:
            When sinc-interpolating inside an interval, ensure that we only
            retain scores within that interval. Set to False when collecting
            triggers to preserve older runs, use True when optimizing
        :param recompute_psd_drift_correction:
            Flag indicating whether to redefine PSD drift correction, exposed
            for debugging with astrophysical waveforms
        :param interpolate_psd_drift_correction:
            Flag whether to use an interpolated PSD drift correction
            Default set to preserve search, exposed for debugging purposes
        :param marginalized_score_HM: Use a semi-marginalized score instead of 
            |Z22|**2+|Z33|**2+|Z44|**2 as a lower threshold for storing triggers
            (basically, de-weights triggers with unphysical ratios of Amp33/Amp22
             and Amp44/Amp22)
        :param return_format: Flag whether to return clist or processedclist
        :param adjust_times:
            Flag to adjust times to assign triggers to the linear-free time
            of their implied waveforms, used only if
            return_format==processedclist. The possible values are
            0: Adjust the times
            1. Adjust the times with the shift of the calpha waveforms,
               regardless of the kind of waveforms we're using
            2: Adjust the times only if using calpha waveforms
        :param psd_drift_kwargs: Any extra arguments for recomputing PSD drift
        :return:
            1. clist/processedclist with triggers for the waveform, always a
               numpy array. Use process_clist to convert to processedclist
               Note: It adjusts the times to the linear-free ones if calphas
               is not None, unless adjust_times==False
            2. Valid mask on the scores
        """
        if (wf_whitened_fd is None) and (calphas is None):
            raise RuntimeError("I need a waveform to use!")

        # Generate overlaps, zeroed where they cannot be trusted if needed
        # ----------------------------------------------------------------------
        # Note that this preferentially uses the waveform, redefined on the
        # subset if needed
        overlaps, hole_corrections, valid_inds = self.gen_scores(
            wfs_whitened_fd=wf_whitened_fd, calphas=calphas, subset=subset,
            zero_invalid=zero_invalid_before_sinc_interp)

        # Apply hole correction before interpolation, since hole correction
        # can vary within the domain of a single sinc interpolation
        h_corr_overlaps = overlaps / (hole_corrections + params.HOLE_EPS)

        # PSD drift correction to use
        if recompute_psd_drift_correction:
            psd_drift_correction = self.gen_psd_drift_correction(
                wf_whitened_fd=wf_whitened_fd, calphas=calphas,
                **psd_drift_kwargs)
        else:
            psd_drift_correction = self.psd_drift_correction

        # Find triggers satisfying the conditions, and sinc-interpolate them
        # ----------------------------------------------------------------------
        # Define some parameters
        # Amount of data we lose from each side in the limit of an infinite
        # number of sinc interpolations
        support_edge_data = 2 * (support + 1)

        # Details of relevant data
        if subset:
            # Length of domain over which we want scores
            ldata = self.left_inds + self.right_inds

            # Restrict to the domain
            tdata = self.time_sub[:ldata]
            h_corr_overlaps = h_corr_overlaps[...,:ldata]
            hole_corrections = hole_corrections[...,:ldata]
            valid_inds = valid_inds[...,:ldata]

            # We might have recomputed the PSD drift correction, so use the
            # right subset of length ldata
            psd_drift_correction = psd_drift_correction[
                self.relevant_index - self.left_inds:
                self.relevant_index + self.right_inds]
        else:
            # Time axis
            tdata = self.time
            # Length of domain over which we want scores
            ldata = len(self.time)

        if not apply_threshold:
            # We're computing all overlaps
            left_inds = [support_edge_data]
            right_inds = [ldata - support_edge_data]
        else:
            # Find contiguous blocks above the base threshold at each index
            overlaps_sq_base_threshold = \
                (self.base_threshold_chi2 * psd_drift_correction ** 2)
            h_corr_overlaps_sq = utils.abs_sq(h_corr_overlaps)

            if marginalized_score_HM:
                Rij_samples = self.templatebank.HM_amp_ratio_samples[0]
                # Pick a sample containing one of the largest content of HM
                Rij_99_percentile = Rij_samples[np.argmin(np.abs(
                    Rij_samples[:,0]- np.percentile(Rij_samples[:,0],99)))][:-1]
        
                # We will first have a mask for the 22 SNRsq
                # by rescaling overlaps_sq_base_threshold
                # according to HM sample picked above.
                # -3 arbitrarily picked as a safety factor
                noncandidates = h_corr_overlaps_sq[0] <= (
                    overlaps_sq_base_threshold/(1+ np.sum(Rij_99_percentile**2))-3)
            else:
                noncandidates = np.zeros_like(h_corr_overlaps_sq[0]).astype(bool)
            
            # summing over modes and then including the threshold
            # over the incoherent sum to further reject triggers
            h_corr_overlaps_sq = np.sum(h_corr_overlaps_sq, axis=0)
            noncandidates = np.logical_or(noncandidates,
                h_corr_overlaps_sq <= overlaps_sq_base_threshold)
            block_edges = utils.hole_edges(noncandidates)
            left_inds, right_inds = block_edges[:, 0], block_edges[:, 1]

            # Join nearby blocks together to save on sinc-interpolations
            left_inds, right_inds = utils.amend_indices(
                left_inds, right_inds, n_tol=support_edge_data)

        if len(left_inds) == 0:
            # Not a single candidate with this waveform
            return np.array([]), np.array([])
        
        # template_prior = self.templatebank.Template_Prior_NF.log_prior(
        #    [calphas[:self.templatebank.ndims]])
        # TODO_LessImp: maybe think of implementing the template prior in the
        # incoherent score in the future

        # In the future, maybe we could also include a NF for
        # getting Rij samples for single detector trigger threshold
        # if self.templatebank.Norm_Flow_Rij_Triggering is None:
            # Rij_samples = self.templatebank.Norm_Flow_Rij_Triggering.generate_samples(
            #     calphas[..., :self.templatebank.ndims], num_samples=2048)

        # We have candidates that might rise to be useful, let's
        # sinc-interpolate them and save any triggers
        clist = []
        for left_ind, right_ind in zip(left_inds, right_inds):
            # Ignore triggers that we cannot reliably sinc interpolate
            if ((left_ind < support_edge_data) or
                    ((right_ind + support_edge_data) > ldata)):
                continue

            # Sinc-interpolate the block
            # Find the number of interpolations dynamically
            n_interp = int(np.round(np.log2(self.dt/params.DT_FINAL)))
            
            t4, x4 = utils.sinc_interp_by_factor_of_2(
                tdata, h_corr_overlaps, left_ind=left_ind,
                right_ind=right_ind, support=support, n_interp=n_interp)

            # If we didn't zero invalid scores before sinc-interpolating,
            # do it now
            if not zero_invalid_before_sinc_interp:
                # t4 cannot spread outside tdata, so safe
                orig_inds_mask = ((t4 - tdata[0]) / self.dt).astype(int)
                valid_inds_up = (valid_inds.T[orig_inds_mask].astype(bool)).T
                x4 *= valid_inds_up

            # Fix whether we use the interpolated/central value of the PSD
            # drift correction
            if interpolate_psd_drift_correction or not apply_threshold:
                current_psd_drift_corr = np.interp(
                    t4, tdata, psd_drift_correction)
            else:
                current_psd_drift_corr = psd_drift_correction[
                    (left_ind + right_ind) // 2]

            # Define which scores we will be saving
            if not apply_threshold:
                # We are going to record all the scores
                trigger_inds = np.ones_like(t4, dtype=bool)
            else:
                # Find indices above the true threshold
                overlaps_sq_threshold = \
                    (self.threshold_chi2 * current_psd_drift_corr ** 2)
                x4_sq = utils.abs_sq(x4)
                
                # Summing over modes
                x4_sq_sum = np.sum(x4_sq, axis=0)
                trigger_inds = x4_sq_sum > overlaps_sq_threshold
                
                if marginalized_score_HM:
                    # Further rejecting triggers whose HM marginalized score
                    # is below the threshold
                    x4_triggers = x4[:, trigger_inds]
                    
                    x4_sq_sum = self.templatebank.marginalized_HM_scores(
                                x4_triggers.T, Rij_samples=Rij_samples, input_Z=True)
                                    
                    trigger_inds[trigger_inds] = x4_sq_sum > overlaps_sq_threshold

            # If demanded, we retain only scores that have seen the full
            # sinc-interpolating filter
            if ensure_sinc_support:
                tmin = tdata[left_ind]
                tmax = tdata[right_ind]
                trigger_inds *= np.logical_and(t4 >= tmin, t4 <= tmax)

            # Times and scores to save
            trigger_times = t4[trigger_inds]
            trigger_scores = (x4.T[trigger_inds]).T

            # Save triggers above the threshold in the right format
            if interpolate_psd_drift_correction or not apply_threshold:
                current_psd_drift_corr = current_psd_drift_corr[trigger_inds]
            else:
                current_psd_drift_corr = np.repeat(
                    current_psd_drift_corr, len(trigger_times))

            if self.save_hole_correction:
                # Compute current hole correction to save it
                current_hole_correction = np.zeros((3,len(trigger_times)))
                for i in range(3):
                    if interpolate_psd_drift_correction:
                        current_hole_correction[i] = np.interp(
                            trigger_times, tdata, hole_corrections[i])
                    else:
                        current_hole_correction[i] = np.repeat(
                            hole_corrections[i,(left_ind + right_ind) // 2],
                            len(trigger_times))

                trigger_block = np.c_[
                    trigger_times,
                    trigger_scores[0].real,
                    trigger_scores[0].imag,
                    trigger_scores[1].real,
                    trigger_scores[1].imag,
                    trigger_scores[2].real,
                    trigger_scores[2].imag,
                    current_hole_correction[0],
                    current_hole_correction[1],
                    current_hole_correction[2],
                    current_psd_drift_corr]
            else:
                trigger_block = np.c_[
                    trigger_times,
                    trigger_scores[0].real,
                    trigger_scores[0].imag,
                    trigger_scores[1].real,
                    trigger_scores[1].imag,
                    trigger_scores[2].real,
                    trigger_scores[2].imag,
                    current_psd_drift_corr]

            if calphas is not None:
                # TODO: Sometimes (e.g. bank (3,0)), due to an edge case in force_zero,
                #  the number of grid points is not monotonic with dimension index, but
                #  this was also missed during the generator, so we never report scores
                #  for dimensions we don't enumerate. Not all calphas that are reported
                #  are important
                ndim = len(self.__dict__.get('dcalphas', np.arange(3)))
                trigger_block = np.c_[
                    trigger_block, np.repeat(
                        np.array(calphas)[np.newaxis, :ndim],
                        len(trigger_times), axis=0)]

            clist.append(trigger_block)

        if len(clist) > 0:
            clist = np.concatenate(clist)

        if return_format.lower() == 'processedclist':
            if ((int(adjust_times) == 1) or
                    ((int(adjust_times) == 2) and (calphas is not None))):
                adjust_times_flag = True
            else:
                adjust_times_flag = False
            clist = self.process_clist(
                np.asarray(clist), adjust_times=adjust_times_flag)

        clist = np.asarray(clist)
        
        return clist, valid_inds

    def gen_triggers(
            self, delta_calpha=None, template_safety=None,
            remove_nonphysical=None, force_zero=None, threshold_chi2=None,
            base_threshold_chi2=None, trig_fname=None, config_fname=None,
            marginalized_score_HM=True,
            ncores=1, n_wf_to_digest=None, subset=False, njobchunks=1):
        """
        Defines a template bank with given spacing, performs matched filtering
        of data against this bank, and saves triggers above threshold
        :param delta_calpha:
            Spacing of template bank. Defaults to self.deltac_alpha if None
        :param template_safety:
            Factor to inflate calpha range by. Defaults to self.template_safety
            if None
        :param remove_nonphysical:
            Flag indicating whether to keep only the gridpoints that are close
            to a physical waveform. Defaults to self.remove_nonphysical if None
        :param force_zero:
            Flag indicating whether to center the template grid to get as much
            overlap as possible with the central region. Defaults to
            self.force_zero if None
        :param threshold_chi2:
            Threshold to save single-detector triggers above. Defaults to
            self.threshold_chi2 if None
        :param base_threshold_chi2:
            Threshold to keep single-detector triggers for sinc interpolation.
            Defaults to self.base_threshold_chi2 if None
        :param trig_fname: Absolute path to trigger file to save new triggers
            to, if desired
            1. If trig_fname doesn't exist, we create it and save
               old processedclist + new processedclist to it
            2. If trig_fname exists, we load it, and append new processedclist
               to it
        :param config_fname: Name of configuration file to save trigger metadata
        :param marginalized_score_HM: Use a semi-marginalized score instead of 
            |Z22|**2+|Z33|**2+|Z44|**2 as a lower threshold for storing triggers
            (basically, vetoes triggers with unphysical ratios of Amp33/Amp22
             and Amp44/Amp22 close to the SNR threshold)
        :param ncores: Number of cores to parallelize over
        :param n_wf_to_digest: Run only n_wf_to_digest waveforms per core
        :param subset:
            Flag indicating whether to generate scores on a subset of data
        :param njobchunks:
            Number of chunks to divide job into, useful for restarting from an
            intermediate point
        :return: Defines self.processedclist
        """
        # Overwrite existing bank parameters if given
        if delta_calpha is not None:
            self.delta_calpha = delta_calpha

        if template_safety is not None:
            self.template_safety = template_safety

        if remove_nonphysical is not None:
            self.remove_nonphysical = remove_nonphysical

        if force_zero is not None:
            self.force_zero = force_zero

        if threshold_chi2 is not None:
            self.threshold_chi2 = threshold_chi2

        if base_threshold_chi2 is not None:
            self.base_threshold_chi2 = base_threshold_chi2

        # (Re)define grid parameters
        self.grid_axes, _, self.dcalphas = \
            self.templatebank.define_important_grid(
                self.delta_calpha, fudge=self.template_safety,
                force_zero=self.force_zero)

        # Ensure that self.trig_fname is defined (might not have been if
        # we used load_trigs = False)
        if trig_fname is not None:
            self.trig_fname = trig_fname

        # Define number of chunks that bank is divided into for evaluation
        nbankchunks = ncores * njobchunks
        if nbankchunks != self.nbankchunks:
            print(f"Number of bank chunks demanded now: {nbankchunks}")
            print("Number of bank chunks demanded " +
                  f"previously: {self.nbankchunks}")
            print("Number of bank chunks do not match, restarting job to avoid "
                  "repeated or missing waveforms")
            self.nbankchunks = nbankchunks
            self.nbankchunks_done = 0
            self.processedclist = None

        def filter_with_subbank(chunk_idx):
            # Always returns a list (not a numpy object)
            wg = self.templatebank.wt_waveform_generator(
                delta_calpha=self.delta_calpha, fudge=self.template_safety,
                remove_nonphysical=self.remove_nonphysical,
                orthogonalize=True, return_cov=False,
                force_zero=self.force_zero, ncores=nbankchunks, coreidx=chunk_idx)

            candidate_list = []
            for i, (wf, wf_params) in enumerate(wg):
                # # if i % 10 == 0:
                # #     print(f"Matched filtering with waveform {i}")
                # print(f"Matched filtering with waveform {i}")
                if n_wf_to_digest is not None and (i > n_wf_to_digest): break
                
                trigger_list, _ = self.filter_with_wf(
                    wf_whitened_fd=wf, calphas=wf_params, subset=subset,
                    marginalized_score_HM=marginalized_score_HM,
                    return_format='processedclist')
                if len(trigger_list) > 0:
                    candidate_list.append(trigger_list)
            
            return candidate_list

        def save_processedclist_and_json(new_processed_clist):
            # Check if there were any triggers
            new_processed_clist = np.asarray(new_processed_clist)
            if not utils.checkempty(new_processed_clist):
                print(f"Saving {len(new_processed_clist)} new candidates")

                # Append to self.processedclist if it has something
                self.processedclist = utils.safe_concatenate(
                    self.processedclist, new_processed_clist)

                # Append self.processedclist to old data if it exists, else
                # creates a new file and saves to it
                if trig_fname is not None:
                    self.save_candidatelist(trig_fname)

            # Update number of bankchunks finished (do even if we found no
            # triggers, so as to record the number of chunks we finished)
            if config_fname is not None:
                self.to_json(config_fname)

            return

        if ncores > 1:
            p = mp.Pool(ncores)

            # Do job in njobchunks steps, saving in between to make it easy to
            # restart a job that was killed
            njobchunks_left = int(
                np.ceil((self.nbankchunks - self.nbankchunks_done) / ncores))

            for jobind in range(njobchunks_left):
                # OK even if empty, skip the chunks that were done
                pclist = p.map_async(
                    filter_with_subbank,
                    np.arange(self.nbankchunks_done,
                              self.nbankchunks_done + ncores))
                # pclist = p.map(filter_with_subbank, np.arange(ncores))
                processedclist = sum(pclist.get(), [])
                if len(processedclist) > 0:
                    processedclist = np.vstack(processedclist)

                # Update number of chunks that were finished and save the
                # processedclist
                self.nbankchunks_done += ncores
                save_processedclist_and_json(processedclist)

                # If we saved the triggers, clear them out to avoid repetition
                if trig_fname is not None:
                    self.processedclist = None

            p.close()
            p.join()
        else:
            # Do job in njobchunks steps, saving in between to make it easy to
            # restart a job that was killed
            njobchunks_left = int(
                np.ceil((self.nbankchunks - self.nbankchunks_done) / ncores))
                
            for jobind in range(njobchunks_left):
                processedclist = filter_with_subbank(self.nbankchunks_done)
                if len(processedclist) > 0:
                    processedclist = np.vstack(processedclist)

                # Update number of chunks that were finished and save the
                # processedclist
                self.nbankchunks_done += min(
                    ncores, self.nbankchunks - self.nbankchunks_done)
                save_processedclist_and_json(processedclist)

                # If we saved the triggers, clear them out to avoid repetition
                if trig_fname is not None:
                    self.processedclist = None

        if trig_fname is not None:
            # We might have been clearing out memory, so load from saved file
            if not os.path.exists(trig_fname):
                trig_fname += ".npy"
            if os.path.exists(trig_fname):
                with open(trig_fname, "rb") as f:
                    self.processedclist = np.load(f)
                    #TODO: remove when packing is ready.
                    #if self.packed_trig_format:
                    #    self.processedclist = self.unpack_trigs(self.processedclist)
            else:
                self.processedclist = np.empty(0)

        # Refresh filters
        self.filters = {}
        self.rejects = defaultdict(list)
        self.filteredclist = copy.deepcopy(self.processedclist)

        return

    # Functions used to veto triggers
    # -------------------------------
    def get_bad_times(
            self, oversubscription=-1, maximum_interval=0.1,
            rejection_interval=25, snr2_thresh=30, nperrun=1):
        """
        Finds bad times for the file, where it overproduces triggers due to
        widespread problems
        TODO: Fix this using preserve_max_snr?
        :param oversubscription:
            Oversubscription factor / number of templates, pass -ve to skip
        :param maximum_interval:
            Interval (s) within which to count only the maximum SNR^2 trigger,
            needed to avoid penalizing events. Ensure that it is wide enough
            to catch all the friends
        :param rejection_interval:
            Interval (s) to reject based on total number of triggers within
            above SNR cut
        :param snr2_thresh:
            Threshold for maximum SNR2 to check overproduction above
        :param nperrun: Number of times this should fire per run
        :return:
            Numpy array with list of times (s), invalid times are
            (t, t + rejection_interval) for t in result
        """
        if oversubscription < 0:
            return np.array([])

        # Number of independent trials for maximum
        ntrials = int(maximum_interval / (self.dt * oversubscription))

        # Probability that max SNR^2 is greater than snr2_threshold
        pmax = 1. - stats.chi2.cdf(snr2_thresh, 2)**ntrials

        # Number of maxima to check for friends
        nmaxima = int(rejection_interval / maximum_interval)

        # Set chance that this fires on a single set of maxima
        pfire = nperrun * rejection_interval / \
            (params.DEF_FILELENGTH * params.NFILES)

        # Set threshold above which we deem that triggers have clustered due to
        # uncaught bad stuff, +1 for safety and +1 to allow an event +1 for the
        # edge-case in which the event spills over into the next 10ms bin edge
        nmax_thresh = stats.binom.isf(pfire, nmaxima, pmax) + 3

        # Compute variables to threshold
        # Chunk the triggers into groups every maximum_interval
        sorted_triggers = self.processedclist[self.processedclist[:, 0].argsort()]
        _, indices = np.unique(
            np.floor((sorted_triggers[:, 0] - self.t0) / maximum_interval),
            return_index=True)
        chunks = np.split(sorted_triggers, indices[1:])
        # Array of min_t, max(snr^2) within each group
        maxsnr2_arr = np.array([[np.min(chunk[:, 0]),
                                 np.max(chunk[:, 1])] for chunk in chunks])
        # Find intervals of maximum_interval that have a maximum above the
        # threshold
        triggers_above_snr_thresh = maxsnr2_arr[maxsnr2_arr[:, 1] > snr2_thresh]

        # These should be rare, find rejection intervals with signifiantly
        # larger number of subintervals of maximum_interval above threshold, bad
        # things are happening there
        secondary_rounded_times, secondary_buckets = np.unique(
            np.floor((triggers_above_snr_thresh[:, 0] - self.t0) /
                     rejection_interval), return_index=True)
        # Must include the length of the array so that we know how many things
        # are in the last bucket
        secondary_buckets = np.append(
            secondary_buckets, len(triggers_above_snr_thresh))
        number_of_glitches_per_big_bucket = \
            secondary_buckets[1:] - secondary_buckets[:-1]

        return rejection_interval * secondary_rounded_times[
            number_of_glitches_per_big_bucket > nmax_thresh] + self.t0

    def get_time_index(self, triggers):
        """"
        :param triggers:
            Processedclist with triggers that jumped
            (can be a vector for n_triggers = 1)
        :return: 1. Indices closest to right-edge of template + 1 = score
                 2. Subgrid shifts to the nearest time on the grid (s)
                 (scalars if n_triggers = 1)
        """
        bank = self.templatebank

        # Absolute GPS time of linear free part of waveform + 1 index, due to
        # the fact that overlap-save uses a conjugate, and conjugate is reverse
        # plus roll by one
        if triggers.ndim == 1:
            trig_times_lf = triggers[0]
        else:
            trig_times_lf = triggers[:, 0]

        # Right edge of waveform + 1 index, indexes scores with the default bank
        trig_times_right = trig_times_lf - bank.shift_whitened_wf * bank.dt

        # First index >= right-edge of template (subgrid shift will put the
        # right edge at the previous index)
        relevant_indices_right = np.searchsorted(self.time, trig_times_right)

        # Extra subgrid shift between right edge and nearest grid time (<=0)
        dts_subgrid = trig_times_right - self.time[relevant_indices_right]

        return relevant_indices_right, dts_subgrid

    def prepare_subset_for_vetoes(
            self, triggers=None, locations=None, dt_opt=params.DT_OPT,
            support=params.SUPPORT_SINC_FILTER_OPT, zero_pad=True):
        """Prepare subset that is large enough for vetos/optimization of any
        trigger in triggers
        :param triggers:
            n_trigger x len(processedclist[0]) array with triggers
            (can be a vector for n_trigger = 1)
        :param locations:
            List of length n_locations with 2-tuples with locations
            (can be a single location for n_locations = 1)
        :param dt_opt: Length of buffer (s) to allow in calpha optimization
        :param support: Support of sinc-interpolating filter
        :param zero_pad:
            Flag indicating whether to zero pad, or pad with existing data
        :return: 1. Reference index for score in full data
                 2. Number of left scores guaranteed, not inclusive of ref
                 3. Number of right scores guaranteed, not inclusive of ref
        TODO: Fix support to be in units of seconds?
        """
        # Define safety indices and scores
        # --------------------------------
        # First find safety in indices for subtraction test
        max_bandlim_interval = np.max(
            [0] + [x[0] for x in self.bandlim_transient_intervals])
        max_excess_power_interval = np.max([0] + self.excess_power_intervals)
        max_interval = np.max([max_bandlim_interval, max_excess_power_interval])

        # (params.N_INDEP_MOVING_AVG_EXCESS_POWER + 1) is the total number of
        # intervals used for excess power measurements, take enough for average
        # to be valid in the middle, wherever it may occur
        nintervals_safety = params.N_INDEP_MOVING_AVG_EXCESS_POWER / 2 + 1
        inds_safety = int(np.ceil(nintervals_safety * (max_interval / self.dt)))

        # Allow leftmost part of waveform to be vetoed by excess power
        inds_safety_left = inds_safety + self.templatebank.support_whitened_wf
        inds_safety_right = inds_safety

        # Find safety in scores for optimization
        nscores_opt = int(np.ceil(dt_opt / self.dt))
        # Add safety margin for sinc interpolation
        scores_safety = nscores_opt + 2 * (support + 1)
        scores_safety_left = scores_safety_right = scores_safety

        # Find limits that work for all triggers
        # --------------------------------------
        if triggers is not None:
            # Check if we have more than one trigger, in which case we pick the
            # first and last one. Also define the calphas of the `best' trigger
            if triggers.ndim > 1:
                # Sort by trigger time
                triggers = np.asarray(triggers)
                triggers = triggers[np.argsort(triggers[:, 0]), :]
                limiting_triggers = np.array([triggers[0], triggers[-1]])
                best_calpha = triggers[np.argmax(triggers[:, 1]), self.c0_pos:]
            else:
                limiting_triggers = np.array([triggers])
                best_calpha = triggers[self.c0_pos:]
            relevant_indices_right, _ = self.get_time_index(limiting_triggers)
        elif locations is not None:
            if type(locations) == list:
                # Number of locations, find limiting locations
                times = [x[0] for x in locations]
                limiting_times = np.array([np.min(times), np.max(times)])
                # There is no notion of `best' calpha here, pick the first one
                best_calpha = locations[0][1]
            elif type(locations) == tuple:
                limiting_times = np.array([locations[0]])
                best_calpha = locations[1]
            else:
                raise RuntimeError("Cannot interpret location")
            # Right edge of waveform + 1 index, indexes scores with the
            # default bank
            times_right = limiting_times - \
                self.templatebank.shift_whitened_wf * self.dt
            # First index >= right-edge of template
            relevant_indices_right = np.searchsorted(self.time, times_right)
        else:
            raise RuntimeError("No triggers or locations passed")

        # Find the limiting indices to compute the PSD drift corrections
        # for each trigger (used in the vetoes)
        # We just need the index limits, do it the simplest way by evaluating
        # the PSD drift correction
        # Condition the bank to self.fftsize if needed
        # (to make the PSD drift computation work)
        bank = self.templatebank
        if bank.fftsize != self.fftsize:
            bank.set_waveform_conditioning(self.fftsize, self.dt)
        _, ind_limits = self.gen_psd_drift_correction(
            calphas=best_calpha, avg='mean', verbose=False,
            indices=relevant_indices_right, return_only_at_indices=True)

        # Find the number of scores to pull in on either side for each trigger
        lwf = bank.support_whitened_wf
        # On the left
        nscores_left_list = relevant_indices_right - ind_limits[:, 0]
        nscores_left_list[nscores_left_list < 0] = 0
        nscores_left_list[nscores_left_list < (inds_safety_left - lwf)] = \
            inds_safety_left - lwf
        nscores_left_list[nscores_left_list < scores_safety_left] = \
            scores_safety_left
        # On the right
        # -1 since right_ind is not included in psd_drift_correction,
        # and prepare_subset_for_triggers is exclusive of relevant_index
        nscores_right_list = ind_limits[:, 1] - 1 - relevant_indices_right
        nscores_right_list[nscores_right_list < 0] = 0
        nscores_right_list[nscores_right_list < inds_safety_right] = \
            inds_safety_right
        nscores_right_list[nscores_right_list < scores_safety_right] = \
            scores_safety_right

        # Refer everything to the first trigger
        relevant_index_ref = relevant_indices_right[0]
        nscores_left_ref = nscores_left_list[0]
        nscores_right_ref = nscores_right_list[0]
        for relevant_index_right, nscores_left, nscores_right in zip(
                relevant_indices_right, nscores_left_list, nscores_right_list):
            # Positive if trigger is to the right of the first trigger
            dscores = relevant_index_right - relevant_index_ref
            nscores_left_ref = max(nscores_left_ref, nscores_left - dscores)
            nscores_right_ref = max(nscores_right_ref, nscores_right + dscores)

        # Define subset
        self.prepare_subset_for_triggers(
            relevant_index_ref, nscores_left_ref, nscores_right_ref,
            zero_pad=zero_pad)

        return relevant_index_ref, nscores_left_ref, nscores_right_ref

    def get_bestfit_wf(self, trigger, wf_wt_cos_td=None, individual_modes=False,
                        physical_mode_ratio=True):
        """
        :param trigger: Row of processedclist
        :param wf_wt_cos_td:
            If known, the cosine whitened waveform
            (will infer from the template bank otherwise)
        :param individual_modes: Return [22,33,44] separately instead of 
            their sum
        :param physical_mode_ratio: require A_33/A_22 and A_44/A_22 to be in
                                    the physically allowed region
        :return: Array of length support_whitened_wf with bestfit waveform
        """
        bank = self.templatebank
        _, dt_subgrid = self.get_time_index(trigger)
        
        z = np.zeros(3, dtype=np.complex128)
        # Cosine and sine overlaps, convert back from probability units
        _ = trigger[self.psd_drift_pos]
        z[0] = _ * (trigger[self.rezpos] + 1j * trigger[self.imzpos])
        z[1] = _ * (trigger[self.rezpos+2] + 1j * trigger[self.imzpos+2])
        z[2] = _ * (trigger[self.rezpos+4] + 1j * trigger[self.imzpos+4])
        
        if physical_mode_ratio:
            z = bank.marginalized_HM_scores(
                np.array([z]), marginalized=False, input_Z=True)[0]

        # Define bestfit waveform
        # Cosine waveform on original grid
        if wf_wt_cos_td is None:
            trig_calpha = trigger[self.c0_pos:]
            wf_wt_cos_td = bank.gen_whitened_wfs_td(trig_calpha,orthogonalize=True)
        wf_wt_cos_fd = utils.RFFT(wf_wt_cos_td, axis=-1)

        # Apply subgrid shift, with numpy fft convention
        # Trigger occured earlier than triglist.time[relevant_index_right] =>
        # keeping the GPS time of the `linear free point' of the waveform fixed,
        # we move the template's right edge farther, i.e., we shift the waveform
        # inside the template to the left, so that the score will jump at
        # triglist.time[relevant_index_right] instead
        wf_wt_cos_fd[:] *= np.exp(- 2. * np.pi * 1j * bank.fs_fft * dt_subgrid)

        # Define bestfit waveform
        bestfit_wf_fd = wf_wt_cos_fd * z[:,np.newaxis]
        if not individual_modes:
            bestfit_wf_fd = np.sum(bestfit_wf_fd,axis=0)
        bestfit_wf_td = utils.IRFFT(bestfit_wf_fd, n=bank.fftsize, axis=-1)
        
        return bestfit_wf_td[..., -bank.support_whitened_wf:]

    def finer_psd_drift(self, trigger, wf_whitened_fd=None, average=None):
        """
        :param trigger: Row of processedclist
        :param wf_whitened_fd:
            Frequency domain waveform to use for PSD drift. If None,
            we use the waveform corresponding to the calphas in trigger
        :param average:
            Method to compute the PSD drift correction
            (see gen_psd_drift_correction for options, defaults to class setup)
        :return: Correction factor to multiply SNR^2, equals 1 if finer
                 PSD drift correction is consistent with coarser one
        """
        # Coarse PSD drift correction
        trig_psd_drift_corr = trigger[self.psd_drift_pos]
        # Location in full data
        relevant_index, _ = self.get_time_index(trigger)
        # calphas for trigger, only used if no waveform was passed
        trig_calpha = trigger[self.c0_pos:]

        if average is None:
            average = self.average

        # Choice of tol halves the evaluation length
        finer_psd_drift_corr, _ = self.gen_psd_drift_correction(
            wf_whitened_fd=wf_whitened_fd, calphas=trig_calpha,
            tol=np.sqrt(2.) * params.PSD_DRIFT_TOL, avg=average,
            verbose=False, indices=[relevant_index],
            return_only_at_indices=True)
        finer_psd_drift_corr = finer_psd_drift_corr[0]

        # Check if significant
        # The std of the (difference between the square of the new correction
        # and the square of the old correction) equals the std of (the square of
        # the old correction) = params.PSD_DRIFT_TOL (in the Gaussian case,
        # using mean averaging). The distribution is not really a Gaussian,
        # and the std is inflated due to the median (by 1.2 for bank 3,0) but
        # it's safe enough if we set the threshold high and cut only if the
        # candidate would be killed
        if ((finer_psd_drift_corr ** 2 - trig_psd_drift_corr ** 2) >
                (params.PSD_DRIFT_VETO_THRESH * params.PSD_DRIFT_TOL)):
            return (trig_psd_drift_corr / finer_psd_drift_corr) ** 2
        else:
            return 1.0
        
    
    def veto_trigger_power(
            self, trigger,
            template_max_mismatch=params.DEF_TEMPLATE_MAX_MISMATCH,
            nfire=params.NFIRE, subset_defined=False, bestfit_wf=None,
            test_injection=False, lazy=True, physical_mode_ratio=False, verbose=0):
        """
        Checks trigger by subtracting best fit waveform and looking for excess
        power commensurate with waveform
        :param trigger: Row in TriggerList.processedclist corresp. to trigger
        :param template_max_mismatch: Reduce the preserve max snr by this factor
        :param nfire: Number of times glitch detectors fire per perfect file
        :param subset_defined:
            Flag indicating we have already defined appropriate subset of data
        :param bestfit_wf:
            If known, pass the bestfit waveform (we can use any other waveform,
            as long as we take care of the indices)
        :param test_injection: Use the injected waveform as the bestfit waveform
        :param lazy: Flag indicating whether to stop after a test fails
        :param physical_mode_ratio: require A_33/A_22 and A_44/A_22 of the best-fit wf
                                    to be in physically allowed space
        :param verbose:
            Integer flag indicating level of details to print
            0. Do not print anything
            1. Print which test failed
            2. Print which test failed, and details of the test
        :return: 0. True/False according to whether the trigger passed/failed
                 1. Correction factor to multiply SNR^2, different from 1 if
                    finer PSD drift correction is significantly different
                 2. Boolean array of size len(self.outlier_reasons) with zeros
                    marking glitch tests that fired
                    The indices correspond to:
                    0: len(self.outlier_reasons)-1:
                        (index into outlier reasons)-1 for excess-power-like
                        tests (outlier reasons[0] doesn't happen here)
                    len(self.outlier_reasons)-1: Finer PSD drift killed it
        """
        # Set up templatebank and some properties of triggers
        # ---------------------------------------------------
        bank = self.templatebank
        # SNR^2
        trig_snr2 = trigger[1]
        # PSD drift correction at location
        trig_psd_drift_correction = trigger[self.psd_drift_pos]
        # Mask into tests that failed
        glitch_mask = np.ones(len(self.outlier_reasons), dtype=bool)

        # Get bestfit waveform, and define subset of data to subtract it from
        # -------------------------------------------------------------------
        if not subset_defined:
            # We have to define a subset ourselves
            relevant_index, *_ = self.prepare_subset_for_vetoes(trigger)
        else:
            relevant_index, _ = self.get_time_index(trigger)

        # Define position of trigger relative to subset of data
        rel_ind_sub = relevant_index + self.offset_sub

        # Get bestfit waveform if passed, else compute it
        if test_injection:
            bestfit_wf = \
                self.injected_wf_whitened[:,
                    relevant_index - bank.support_whitened_wf:relevant_index]
        elif bestfit_wf is not None:
            bestfit_wf = bestfit_wf[:,-bank.support_whitened_wf:]
        else:
            bestfit_wf = self.get_bestfit_wf(trigger, individual_modes=True,
                            physical_mode_ratio=physical_mode_ratio)

        bestfit_wf_combined = np.sum(bestfit_wf, axis=0)
        bestfit_amp = np.linalg.norm(bestfit_wf, axis=-1)
        bestfit_amp_combined = np.linalg.norm(bestfit_wf_combined, axis=-1)

        # Subtract the bestfit waveform
        # -----------------------------
        # Indices that contain the GW signal
        wf_end_ind = rel_ind_sub
        wf_start_ind = rel_ind_sub - len(bestfit_wf_combined)

        # Remove bestfit waveform from data
        strain_sub = self.strain_sub.copy()
        strain_sub[wf_start_ind:wf_end_ind] -= bestfit_wf_combined

        # Fix oversubtraction inside holes
        trig_mask = self.mask_sub * self.valid_mask_sub
        strain_sub *= trig_mask

        globals()["strain_sub"] = strain_sub

        # Look for glitches in subtracted data
        # ------------------------------------
        # Compute glitch thresholds for the bestfit waveform, and indices where
        # the bestfit waveform has power in each band. Pass waveform with SNR=1
        # and length support_whitened_wf, so indices are relative to its start
        new_glitch_thresholds = self.get_glitch_thresholds(
            (bestfit_wf_combined / bestfit_amp_combined), self.dt,
            self.preserve_max_snr * template_max_mismatch,
            self.sine_gaussian_intervals,
            self.bandlim_transient_intervals,
            self.excess_power_intervals)

        globals()["new_glitch_thresholds"] = new_glitch_thresholds

        # Check whether the glitch tests jumped near where the waveform has power
        # Look for outliers in whitened strain data without touching mask, as we
        # are not filling holes
        _, outlier_mask, _, emesg = d_ops.find_whitened_outliers(
            strain_sub, self.mask_sub, self.valid_mask_sub, self.dt,
            params.MIN_CLIPWIDTH,
            sigma_clipping_threshold=new_glitch_thresholds[0][0],
            zero_mask=False, nfire=nfire, renorm_wt=self.renorm_wt,
            sc_n01=self.sc_n01, mask_save=self.mask_save, verbose=False)
        wf_sc_ind = wf_start_ind + new_glitch_thresholds[0][1]
        if not outlier_mask[wf_sc_ind]:
            glitch_mask[0] = False
            if verbose > 0:
                print("Glitch test failed, found outlier in whitened data")
                if verbose > 1:
                    print(emesg)
            if lazy:
                return False, 1.0, glitch_mask

        # Look for sine gaussian transients without touching mask, as we are not
        # filling holes
        _, outlier_mask, emesg = d_ops.find_sine_gaussian_transients(
            strain_sub, self.mask_sub, self.valid_mask_sub, self.dt,
            params.MIN_CLIPWIDTH, self.freqs_lines, self.mask_lines,
            sine_gaussian_intervals=self.sine_gaussian_intervals,
            sine_gaussian_thresholds=new_glitch_thresholds[1][0],
            edgesafety=1, fftsize=utils.next_power(len(strain_sub)),
            zero_mask=False, nfire=nfire, renorm_wt=self.renorm_wt,
            mask_save=self.mask_save, verbose=False)

        for ind, (fc_sg, df_sg) in enumerate(self.sine_gaussian_intervals):
            wf_sg_ind = wf_start_ind + new_glitch_thresholds[1][1][ind]
            if not outlier_mask[ind, wf_sg_ind]:
                glitch_mask[1 + ind] = False
                if verbose > 0:
                    print("Glitch test failed for sine-Gaussian around " +
                          f"{fc_sg} Hz, bandwidth {df_sg} Hz")
                    if verbose > 1:
                        print(emesg)
                if lazy:
                    return False, 1.0, glitch_mask

        # Check for bandlimited excess power without touching mask, as we are
        # not filling holes
        _, outlier_mask, emesg = d_ops.find_excess_power_transients(
            strain_sub, self.mask_sub, self.valid_mask_sub, self.dt,
            excess_power_intervals=self.bandlim_transient_intervals,
            excess_power_thresholds=new_glitch_thresholds[2][0], edgesafety=1,
            freqs_lines=self.freqs_lines, mask_freqs=self.mask_lines,
            zero_mask=False, nfire=nfire, fmax=self.fmax,
            mask_save=self.mask_save, verbose=False)

        for ind, (interval, frng) in enumerate(
                self.bandlim_transient_intervals):
            wf_power_ind = wf_start_ind + new_glitch_thresholds[2][1][ind]
            if not outlier_mask[ind, wf_power_ind]:
                glitch_mask[1 + len(self.sine_gaussian_intervals) + ind] = False
                if verbose > 0:
                    print("Glitch test failed for excess power in band " +
                          f"{interval} s, {frng} Hz")
                    if verbose > 1:
                        print(emesg)
                if lazy:
                    return False, 1.0, glitch_mask

        # Check for excess power without touching the mask, as we
        # aren't filling holes
        power_intervals_inp = [
            [t, [0, np.inf]] for t in self.excess_power_intervals]
        _, outlier_mask, emesg = d_ops.find_excess_power_transients(
            strain_sub, self.mask_sub, self.valid_mask_sub, self.dt,
            excess_power_intervals=power_intervals_inp,
            excess_power_thresholds=new_glitch_thresholds[3][0], edgesafety=1,
            freqs_lines=self.freqs_lines, mask_freqs=self.mask_lines,
            zero_mask=False, nfire=nfire, fmax=self.fmax,
            mask_save=self.mask_save, verbose=verbose)

        for ind, interval in enumerate(self.excess_power_intervals):
            wf_power_ind = wf_start_ind + new_glitch_thresholds[3][1][ind]
            if not outlier_mask[ind, wf_power_ind]:
                glitch_mask[1 + len(self.sine_gaussian_intervals) +
                            len(self.bandlim_transient_intervals) + ind] = False
                if verbose > 0:
                    print("Glitch test failed for excess power over " +
                          f"{interval} s")
                    if verbose > 1:
                        print(emesg)
                if lazy:
                    return False, 1.0, glitch_mask

        # Check if we had glitches that weren't resolved by the PSD drift
        # correction, record correction factor for finer PSD drift (only if
        # it is significant)
        bestfit_wf_padded = utils.FFTIN((3,bank.fftsize))
        bestfit_wf_padded[:,-bank.support_whitened_wf:] = \
            (bestfit_wf[:].T / bestfit_amp[:].T).T
        bestfit_wf_fd = utils.RFFT(bestfit_wf_padded)
        # Note: By the time we reach this point, we should have computed
        # the safemean
        snr2_corr_factor = self.finer_psd_drift(
            trigger, wf_whitened_fd=bestfit_wf_fd, average='safemean')

        # Check if the trigger is pulled below threshold
        if (trig_snr2 * snr2_corr_factor) < self.threshold_chi2:
            glitch_mask[-1] = False
            finer_psd_drift_correction = \
                trig_psd_drift_correction / (snr2_corr_factor**0.5)
            if verbose > 0:
                print("Unresolved PSD drift!")
                if verbose > 1:
                    print("PSD drift correction: " +
                          f"{trig_psd_drift_correction} (old), " +
                          f"{finer_psd_drift_correction} (new)")

        return np.all(glitch_mask), snr2_corr_factor, glitch_mask

    def veto_trigger_phase(
            self, trigger, dcalphas=None, n_chunk=params.N_CHUNK,
            cov_degrade=params.COV_DEGRADE,
            chi2_min_eigval=params.CHI2_MIN_EIGVAL,
            pthresh_chi2=params.THRESHOLD_CHI2,
            split_chunks=params.SPLIT_CHUNKS,
            pthresh_split=params.THRESHOLD_SPLIT, subset_details=None,
            bestfit_wf=None, test_injection=False, physical_mode_ratio=False,
            verbose=True, lazy=True):
        """
        Chi2 and tail veto using phase
        :param trigger: Trigger that jumped (suggest using optimized trigger,
                        but should be OK otherwise)
        :param dcalphas: Array with spacings in calphas to allow mismatch in
        :param n_chunk: Number of chunks to split the whitened waveform into
        :param cov_degrade:
            Keep eigenvectors with eigenvalue > (max eigenvalue of covariance
            matrix * cov_degrade)
        :param chi2_min_eigval:
            If the highest eigenvalue of the covariance matrix is below this,
            do not perform chi-squared test
        :param pthresh_chi2:
            Probability of the overall chi2 veto firing due to Gaussian noise
        :param split_chunks:
            List of pairs of lists of chunk indices (each entry of the form
            [[ind11, ind12, ...], [ind21, ind22, ...]]) with the members of
            each pair a set of chunks to contrast with the others)
        :param pthresh_split:
            Probability of each split veto firing due to Gaussian noise
        :param subset_details:
            If we have already defined an appropriate subset, 3-tuple as in
            the output of prepare_subset_for_vetoes
        :param bestfit_wf:
            If known, pass the bestfit waveform (we can use any other waveform,
            as long as we take care of the indices). Pass 3 modes
        :param physical_mode_ratio: require A_33/A_22 and A_44/A_22 of the best-fit wf
                                    to be in physically allowed region
        :param test_injection: Use the injected waveform as the bestfit waveform
        :param verbose: Flag indicating whether to print warnings
        :param lazy: Flag indicating whether to stop after a test fails
        :return: 0. True/False if the trigger passes/fails the veto
                 1. Boolean array of size 2 + len(split_chunks) with zeros
                    marking glitch tests that fired
                    The indices correspond to:
                    0: No chunks present
                    1: Overall chi^2 test
                    2: 2 + len(split_chunks): Split tests
        """
        # Mask into tests that failed
        glitch_mask = np.ones(2 + len(split_chunks), dtype=bool)

        # Cosine and sine overlaps
        z_cos = trigger[self.psd_drift_pos] * trigger[self.rezpos]
        z_sin = trigger[self.psd_drift_pos] * trigger[self.imzpos]
        
        # Get bestfit waveform if passed, else compute it
        bank = self.templatebank
        if test_injection:
            raise NotImplementedError('Injections not implemented for higher modes')
            #bestfit_wf = \
            #    self.injected_wf_whitened[:,
            #        relative_index - bank.support_whitened_wf:relative_index]
        elif bestfit_wf is not None:
            bestfit_wf = bestfit_wf[:,-bank.support_whitened_wf:]
        else:
            bestfit_wf = self.get_bestfit_wf(trigger, individual_modes=True,
                                            physical_mode_ratio=physical_mode_ratio)
                                            
        bestfit_wf, bestfit_wf_33_44 = \
            bestfit_wf[0], bestfit_wf[1] + bestfit_wf[2]

        # Define splits and subset of data
        # --------------------------------------------
        # --------------------------------------------
        if subset_details is None:
            # We have to define the subset ourselves
            relative_index_ref, nscores_left_ref, nscores_right_ref = \
                self.prepare_subset_for_vetoes(trigger)
        else:
            relative_index_ref, nscores_left_ref, nscores_right_ref = \
                subset_details

        # Define position of trigger relative to subset of data
        relative_index, dt_subgrid = self.get_time_index(trigger)
        rel_index_sub = relative_index + self.offset_sub
        nscores_left = nscores_left_ref + relative_index - relative_index_ref
        nscores_right = nscores_right_ref - relative_index + relative_index_ref
        
        # Indices that contain the GW signal
        wf_end_ind = rel_index_sub
        wf_start_ind = rel_index_sub - len(bestfit_wf)

        # Remove bestfit waveform from data
        self.strain_sub[wf_start_ind:wf_end_ind] -= bestfit_wf_33_44
        fftsize_sub = utils.next_power(self.left_inds + self.right_inds)
        self.chunked_data_sub = d_ops.chunkedfft( self.strain_sub, fftsize_sub, 1)
        
        # Split template (with SNR=1) into chunks with roughly equal SNR
        fftsize_sub = len(self.time_sub)
        temp_td_sub = utils.FFTIN(fftsize_sub)
        temp_td_sub[-bank.support_whitened_wf:] = \
            bestfit_wf[:] / np.linalg.norm(bestfit_wf)
        temp_td_split = bank.split_whitened_wf_td(temp_td_sub, nchunk=n_chunk)
        temp_fd_split = utils.RFFT(temp_td_split, axis=-1)
        temp_fd_sub = np.sum(temp_fd_split, axis=0)

        # Generate split scores and estimate their statistics
        # ---------------------------------------------------
        # ---------------------------------------------------
        # Generate scores for the split templates
        overlaps_split, hole_corrections_split, valid_inds_split = \
            self.gen_scores(wfs_whitened_fd=temp_fd_split, subset=True, only_22=True)
        scores_split = overlaps_split / \
            (hole_corrections_split + params.HOLE_EPS)
        
        # We directly modified self.strain_sub, so just changing it back
        self.strain_sub[wf_start_ind:wf_end_ind] += bestfit_wf_33_44
        self.chunked_data_sub = d_ops.chunkedfft(self.strain_sub, fftsize_sub, 1)

        # Don't veto on chunks whose scores don't exist at the relevant index
        chi2mask = valid_inds_split[..., rel_index_sub]
        nchunk_exist = np.count_nonzero(chi2mask)

        if nchunk_exist < 2:
            glitch_mask[0] = False
            if verbose:
                warnings.warn("Not enough chunks for chi2 test!", Warning)
            # Cannot proceed, so end gracefully here
            return False, glitch_mask

        # Define index subsets for split tests
        # Define list of submasks of size nchunk_exist (into chi2mask=True)
        # that are true at (union of L and H subspaces), and index arrays into
        # the further restricted subspace (into submask=True) for the L and H
        # subspaces
        splitmasks = []
        for ind_arrs in split_chunks:
            linds_arr = np.asarray(ind_arrs[0])
            hinds_arr = np.asarray(ind_arrs[1])

            # First check that each of these subsets has atleast one member
            # with a score at the relevant index, else skip the test
            if ((np.sum(chi2mask[linds_arr]) == 0) or
                    (np.sum(chi2mask[hinds_arr]) == 0)):
                continue

            # Create mask defining the union of the L and H subspaces within
            # the set of valid chunks
            splitmask_union = utils.submask(chi2mask, linds_arr, hinds_arr)

            # Create individual masks defining the L and H subspaces within
            # the set of valid chunks
            splitmask_l = utils.submask(chi2mask, linds_arr)
            splitmask_h = utils.submask(chi2mask, hinds_arr)

            # Create masks defining the individual subspaces within the union
            # of the L and H subspaces
            submask_l = utils.submask(splitmask_union, splitmask_l)
            submask_h = utils.submask(splitmask_union, splitmask_h)

            splitmasks.append([splitmask_union, submask_l, submask_h])

        # Keep only the chunks whose scores exist at the relevant index
        scores_split = scores_split[chi2mask, :]
        valid_inds_split = valid_inds_split[chi2mask, :]
        temp_td_split = temp_td_split[chi2mask, :]
        temp_td_split_sin = utils.IRFFT(
            1j * temp_fd_split[chi2mask, :], n=fftsize_sub, axis=-1)

        # Mask indicating where all the split scores are valid, used to define
        # data to estimate the covariance matrix, avoid the event to be safe
        valid_inds = np.prod(valid_inds_split, axis=0).astype(bool)
        ninds_avoid = int(params.DT_AVOID / self.dt)
        valid_inds[...,rel_index_sub - ninds_avoid:rel_index_sub + ninds_avoid] = 0

        # Avoid scores corresponding to waveforms that wrap around
        valid_inds[...,:max(0, rel_index_sub - nscores_left)] = 0
        valid_inds[...,(rel_index_sub + nscores_right):] = 0

        # Estimate the covariance matrix of the split scores
        cov_split = np.cov(scores_split[..., valid_inds])

        # Predictors of amplitudes in chunks given the sum of existing chunks
        # in the noiseless case
        temp_td_split_comp = temp_td_split + 1j * temp_td_split_sin
        v_alphas = np.dot(temp_td_split_comp, temp_td_sub)
        v_alphas /= np.sum(v_alphas)

        # Compute quantities needed to define observables in tests
        # --------------------------------------------------------
        # --------------------------------------------------------
        # Start with split tests
        # ----------------------
        # Create list with [function to apply test, variance of result]
        split_test_details = []
        for maskset in splitmasks:
            splitmask_union, submask_l, submask_h = maskset

            # Covariance matrix restricted to the union of L and R
            cov_union = cov_split[np.ix_(splitmask_union, splitmask_union)]

            # Subblocks of the restricted covariance matrix
            # corresponding to the L and H subspaces
            cov_ll = cov_union[np.ix_(submask_l, submask_l)]
            cov_hh = cov_union[np.ix_(submask_h, submask_h)]
            cov_hl = cov_union[np.ix_(submask_h, submask_l)]
            cov_lh = cov_union[np.ix_(submask_l, submask_h)]

            # Matrix to multiply scores in L before subtracting from scores in
            # H to orthogonalize the latter
            proj_mat_l = np.dot(cov_hl, np.linalg.inv(cov_ll))

            # Covariance matrix of the scores in H orthogonalized w.r.t L
            cov_h_orthogonalized = cov_hh - np.dot(proj_mat_l, cov_lh)

            # Predictors of amplitudes in chunks given the total amplitude
            v_alphas_split = np.dot(
                temp_td_split_comp[splitmask_union], temp_td_sub)

            # Properties of predictors of total scores from L
            # -----------------------------------------------
            # Multiplicative factors from scores in L to predictions
            # of total score
            mul_facs_l = 1 / v_alphas_split[submask_l]

            # Covariance matrix of predictors of total score from L
            cov_est_l = cov_ll * mul_facs_l[:, np.newaxis] * mul_facs_l.conj()

            # Coefficients of minimal-variance combination of predictors from L
            minvarcoeff_l = np.linalg.solve(
                cov_est_l, np.ones(np.count_nonzero(submask_l))).conj()

            # Variance of the minimal-variance combination (<real^2 + imag^2>)
            var_predictor_l = 1 / np.real(np.sum(minvarcoeff_l))

            # Make the predictor unbiased
            minvarcoeff_l *= var_predictor_l

            # Properties of predictors of total scores from orthogonalized
            # scores in H
            # ------------------------------------------------------------
            # Multiplicative factors from orthogonalized scores in H to
            # predictions of total score
            mul_facs_h = 1 / \
                (v_alphas_split[submask_h] -
                 np.dot(proj_mat_l, v_alphas_split[submask_l]))

            # Covariance matrix of predictors of total score from
            # orthogonalized scores in H
            cov_est_h = (cov_h_orthogonalized * mul_facs_h[:, np.newaxis] *
                         mul_facs_h.conj())

            # Coefficients of minimal-variance combination of predictors from
            # orthogonalized scores in H
            minvarcoeff_h = np.linalg.solve(
                cov_est_h, np.ones(np.count_nonzero(submask_h))).conj()

            # Variance of the minimal-variance combination (<real^2 + imag^2>)
            var_predictor_h = 1 / np.real(np.sum(minvarcoeff_h))

            # Make the predictor unbiased
            minvarcoeff_h *= var_predictor_h

            # Create function to apply this split test
            # Get around late-binding by using default arguments
            def residual_split(
                    split_scores, mask_union=splitmask_union, mask_l=submask_l,
                    mask_h=submask_h, mul_l=mul_facs_l, mul_h=mul_facs_h,
                    coeff_l=minvarcoeff_l, coeff_h=minvarcoeff_h,
                    proj_l=proj_mat_l):
                """
                Applies the split test to scores and returns the residual
                Accepts multiple sets of scores for debugging purposes
                """
                if len(split_scores.shape) > 1:
                    # Many scores
                    # Restrict to union of L and R subsets inside valid chunks
                    split_scores_union = split_scores[mask_union, :]

                    # Compute predictions from the L chunks
                    prediction_l = np.sum(
                        split_scores_union[mask_l, :] *
                        mul_l[:, np.newaxis] * coeff_l[:, np.newaxis], axis=0)

                    # Compute predictions from the H chunks
                    split_scores_h_orthogonalized = utils.orthogonalize_split(
                        split_scores_union, proj_l, mask_l, mask_h)
                    prediction_h = np.sum(
                        split_scores_h_orthogonalized *
                        mul_h[:, np.newaxis] * coeff_h[:, np.newaxis], axis=0)
                else:
                    # Just one set of scores
                    split_scores_union = split_scores[mask_union]

                    # Compute prediction from the L chunks
                    prediction_l = np.sum(
                        split_scores_union[mask_l] * mul_l * coeff_l)

                    # Compute prediction from the H chunks
                    split_scores_h_orthogonalized = utils.orthogonalize_split(
                        split_scores_union, proj_l, mask_l, mask_h)
                    prediction_h = np.sum(
                        split_scores_h_orthogonalized * mul_h * coeff_h)

                return prediction_l - prediction_h

            split_test_details.append(
                [residual_split, var_predictor_l + var_predictor_h])

        # Compute changes due to shifts in time and mismatches in calpha
        # --------------------------------------------------------------
        # Time-lag
        # --------
        def shifted_time(dt):
            # Don't need to get phase correct, since multiplication by a
            # constant complex number doesn't bias our result
            snr = np.sqrt(z_cos ** 2 + z_sin ** 2)
            # Apply time shift
            fs_fft = np.fft.rfftfreq(n=fftsize_sub, d=bank.dt)
            temp_fd_sub_shifted = utils.FFTIN(len(fs_fft), dtype=np.complex128)
            temp_fd_sub_shifted[:] = temp_fd_sub[:]
            temp_fd_sub_shifted[:] *= np.exp(- 2. * np.pi * 1j * fs_fft * dt)
            # Scale to right amplitude
            temp_td_sub_shifted = snr * utils.IRFFT(
                temp_fd_sub_shifted, n=fftsize_sub)
            # Compute split scores
            split_scores_shifted = np.dot(
                temp_td_split_comp, temp_td_sub_shifted)

            # Compute the change in the unbiased split scores (chi2 test)
            shifted_chi2_var = utils.unbias_split(
                split_scores_shifted, v_alphas)

            # Compute the residuals between the predicted total scores from
            # L and R subsets (split tests)
            shifted_split_vars = []
            for test_params in split_test_details:
                shifted_split_vars.append(test_params[0](split_scores_shifted))

            return shifted_chi2_var, shifted_split_vars

        # Time resolution is self.dt/sub_fac
        n_interp = int(np.round(np.log2(self.dt / params.DT_FINAL)))
        sub_fac = 2**n_interp

        # Lag of 1/sub_fac index
        shifted_chi2_var_plus, shifted_split_residuals_plus = \
            shifted_time(bank.dt / sub_fac)

        # Lag of -1/sub_fac index
        shifted_chi2_var_minus, shifted_split_residuals_minus = \
            shifted_time(-bank.dt / sub_fac)

        # Define projection operator to remove the linear piece w/ lag
        # First derivative w.r.t shift (1 x n_chunk)
        shift_lin = ((shifted_chi2_var_plus -
                      shifted_chi2_var_minus)[np.newaxis, :] / 0.5)
        proj_lin = np.eye(nchunk_exist) - \
            (np.dot(shift_lin.T, shift_lin.conj()) /
             np.linalg.norm(shift_lin) ** 2)

        # Mismatch in calpha
        # ------------------
        def mismatch_calpha(calpha):
            # Create mismatched waveform with right phase and amplitude
            curr_wf_cos_td = bank.gen_whitened_wfs_td(calpha)[0]
            curr_wf_fd = utils.RFFT(curr_wf_cos_td)
            curr_wf_fd[:] *= (z_cos + 1j * z_sin) * np.exp(
                - 2. * np.pi * 1j * bank.fs_fft * dt_subgrid)
            curr_wf_td = utils.IRFFT(curr_wf_fd, n=bank.fftsize)

            # Define it on the right subdomain
            curr_wf_td_trunc = utils.FFTIN(fftsize_sub)
            curr_wf_td_trunc[-bank.support_whitened_wf:] = \
                curr_wf_td[-bank.support_whitened_wf:]

            # # Compute noiseless split scores allowing for extra shifts, and
            # # including any missing chunks
            # curr_wf_fd_trunc = utils.RFFT(curr_wf_td_trunc)
            # split_scores_mismatched = \
            #     utils.IRFFT(temp_fd_split.conj() * curr_wf_fd_trunc, axis=-1,
            #                 n=fftsize_sub) + \
            #     1j * utils.IRFFT(-1j * temp_fd_split.conj() * curr_wf_fd_trunc,
            #                      axis=-1, n=fftsize_sub)
            # # Pick shift that maximizes the overall overlap
            # ibest = np.argmax(
            #     utils.abs_sq(np.sum(split_scores_mismatched, axis=0)))
            # print(f"best overlap at {ibest}: ", (np.max(
            #     np.abs(np.sum(split_scores_mismatched, axis=0))) /
            #       (z_cos**2 + z_sin**2)**0.5))
            # split_scores_mismatched = split_scores_mismatched[chi2mask, ibest]

            # Compute noiseless split scores
            split_scores_mismatched = np.dot(
                temp_td_split_comp, curr_wf_td_trunc)

            # Compute the change in the unbiased split scores (chi2 test)
            mismatched_chi2_var = utils.unbias_split(
                split_scores_mismatched, v_alphas)

            # Compute the residuals between the predicted total scores from
            # L and R subsets (split tests)
            mismatched_split_vars = []
            for test_params in split_test_details:
                mismatched_split_vars.append(
                    test_params[0](split_scores_mismatched))

            # print(mismatched_chi2_var)
            # print(mismatched_split_vars)

            return mismatched_chi2_var, mismatched_split_vars, \
                split_scores_mismatched

        # Containers for bias due to mismatch in calpha
        trig_calpha = trigger[self.c0_pos:].copy()
        if utils.checkempty(dcalphas):
            n_calpha = 0
        else:
            n_calpha = min(len(dcalphas), len(trig_calpha))

        mismatched_chi2_vars = np.zeros(
            (n_calpha, 2, nchunk_exist), dtype=np.complex128)
        mismatched_split_residuals = [
            np.zeros((n_calpha, 2), dtype=np.complex128) for _ in
            split_test_details]

        # Projection operators to remove linear piece in calpha for chi2 test
        proj_calpha = utils.FFTIN(
            (n_calpha, nchunk_exist, nchunk_exist), dtype=np.complex128)

        for ind_calpha in range(n_calpha):
            dcalpha = dcalphas[ind_calpha]
            # Ensure that we don't do anything for trivial directions
            if dcalpha > 0:
                for ind_direction, dir_dcalpha in enumerate([1, -1]):
                    # Change the calpha in either direction keeping
                    # everything else fixed
                    curr_calpha = trig_calpha.copy()
                    curr_calpha[ind_calpha] += dir_dcalpha * dcalpha

                    # Define template in the same manner and compute
                    # changes in the chi2 and split observables
                    mismatches = mismatch_calpha(curr_calpha)
                    mismatch_chi2 = mismatches[0].copy()
                    mismatch_splits = mismatches[1].copy()

                    # Record changes in observables
                    mismatched_chi2_vars[ind_calpha, ind_direction, :] = \
                        mismatch_chi2[:]

                    for residual, mismatch_split in \
                            zip(mismatched_split_residuals, mismatch_splits):
                        residual[ind_calpha, ind_direction] = mismatch_split

            # Define projection operator to remove linear piece for this calpha
            # First derivative w.r.t dcalpha (1 x n_chunk)
            ds_calpha = \
                (mismatched_chi2_vars[ind_calpha, 0, :] -
                 mismatched_chi2_vars[ind_calpha, 1, :])[np.newaxis, :] / \
                (2 * dcalpha)
            proj_calpha[ind_calpha] = np.eye(nchunk_exist) - \
                (np.dot(ds_calpha.T, ds_calpha.conj()) /
                 np.linalg.norm(ds_calpha) ** 2)

        # Compute bias due to mismatches in multiple calphas (i.e., cross-terms)
        ind_sets = [[0, 1] for _ in range(n_calpha)]
        # Define offset directions w.r.t trigger calphas
        dir_dcalphas_all = [[-1, 1] for _ in range(n_calpha)]
        # dir_dcalphas_eval[i] is a 2x2..x2 hypercube with offset directions
        # for the i^th coefficient
        dir_dcalphas_eval = np.meshgrid(*dir_dcalphas_all)

        # Containers for bias due to cross-terms
        cross_mismatched_chi2_vars = None
        cross_mismatched_split_residuals = None
        if n_calpha > 0:
            cross_mismatched_chi2_vars = np.zeros(
                list(dir_dcalphas_eval[0].shape) + [nchunk_exist],
                dtype=np.complex128)
            cross_mismatched_split_residuals = [np.zeros_like(
                dir_dcalphas_eval[0], dtype=np.complex128) for _ in
                split_test_details]
            for offset_inds in itertools.product(*ind_sets):
                # Define offset trigger
                curr_calpha = trig_calpha.copy()
                for ind_calpha in range(n_calpha):
                    curr_calpha[ind_calpha] += dcalphas[ind_calpha] * \
                        dir_dcalphas_eval[ind_calpha][offset_inds]

                # Define template in the same manner and compute change in
                # chi2 and split observables
                mismatches = mismatch_calpha(curr_calpha)
                mismatch_chi2 = mismatches[0].copy()
                mismatch_splits = mismatches[1].copy()

                # Record changes in observables
                cross_mismatched_chi2_vars[offset_inds] = mismatch_chi2

                for residual, mismatch_split in \
                        zip(cross_mismatched_split_residuals, mismatch_splits):
                    residual[offset_inds] = mismatch_split

        # Define total projection matrix with linear pieces of all biases
        # projected out
        proj_tot = proj_lin.copy()
        for ind_calpha in range(n_calpha):
            proj_tot = np.dot(proj_calpha[ind_calpha], proj_tot)

        # Perform chi2 test
        # -----------------
        # -----------------
        # Covariance matrix of unbiased split scores
        v_alphas_vec = v_alphas[:, np.newaxis]
        v_ones = np.ones_like(v_alphas_vec)
        cov_unbiased = cov_split - \
            np.dot(np.dot(cov_split, v_ones), v_alphas_vec.conj().T) - \
            np.dot(v_alphas_vec, np.dot(v_ones.T, cov_split)) + \
            np.dot(v_alphas_vec, v_alphas_vec.conj().T) * np.sum(cov_split)

        # Covariance matrix with directions allowed by shifts and calpha
        # mismatches projected out
        proj_cov = np.dot(np.dot(proj_tot, cov_unbiased), proj_tot.T.conj())
        
        # Pick most significant remaining eigenvalues
        # Sometimes there's an error in calculating eigvals so we use a try statement
        try:
            eigvals, eigvecs = np.linalg.eigh(proj_cov)
        except:
            #np.save('/data/jayw/IAS/GW/Data/Triggers/Debug/'+\
            #        f'error_veto_trigger_{np.random.randint(100)}.npy', trigger)
            return np.all(glitch_mask), glitch_mask

        cov_ind = np.searchsorted(eigvals, eigvals[-1] * cov_degrade)

        # Define projection matrix into subspace with significant eigenvalues
        eproj = eigvecs[:, cov_ind:].T.conj()
        evals_proj = eigvals[cov_ind:]
        ndof = 2 * len(evals_proj)

        # Perform chi-squared test only if we have any discriminating power left
        continue_to_chi2 = (eigvals[-1] > 0) and (np.max(evals_proj) > chi2_min_eigval)
        globals()["continue_to_chi2"] = continue_to_chi2
        globals()["evals_proj"] = evals_proj
        globals()["norms2_split"] = np.linalg.norm(temp_td_split, axis=-1) ** 2
        globals()["scores_split"] = scores_split
        globals()["valid_inds"] = valid_inds
        globals()["rel_index_sub"] = rel_index_sub
        globals()["ndof"] = ndof

        if continue_to_chi2:
            # Assess significance of residuals at event
            # -----------------------------------------
            # Make function to apply chi2 test, so as to return for debugging
            def scores_to_chi2(split_scores):
                # Compute the unbiased split scores
                split_scores_unbiased = utils.unbias_split(
                    split_scores, v_alphas)

                # Project out mismatch, restrict to subspace with significant
                # eigenvalues, and compute variable that should be distributed
                # according to a chi2 dist
                if split_scores.ndim > 1:
                    # Multiple sets of scores
                    unitnormals = \
                        (np.dot(eproj, np.dot(proj_tot, split_scores_unbiased)) /
                         np.sqrt(evals_proj[:, np.newaxis] / 2.))
                    chi2_result = np.sum(utils.abs_sq(unitnormals), axis=0)
                else:
                    # Single set of scores
                    unitnormals = \
                        (np.dot(eproj, np.dot(proj_tot, split_scores_unbiased)) /
                         np.sqrt(evals_proj / 2.))
                    chi2_result = np.sum(utils.abs_sq(unitnormals))

                return chi2_result

            # Compute the unbiased split scores at the relevant index
            chi2var_chi2 = scores_to_chi2(scores_split[:, rel_index_sub])

            # Guarantee FAR for waveforms shifted by +/-1/sub_fac index, and
            # mismatched by +/-delta_calpha at measured SNR in all dimensions
            chi2_var_thresh = np.c_[
                shifted_chi2_var_plus,
                shifted_chi2_var_minus,
                mismatched_chi2_vars.reshape((-1, nchunk_exist)).T]
            if n_calpha > 0:
                # Add cross terms to safety
                chi2_var_thresh = np.c_[
                    chi2_var_thresh,
                    cross_mismatched_chi2_vars.reshape((-1, nchunk_exist)).T]
            meannormals = (np.dot(
                eproj, np.dot(proj_tot, chi2_var_thresh)) /
                           np.sqrt(evals_proj[:, np.newaxis] / 2.))
            ncpar_chi2 = np.max(np.sum(utils.abs_sq(meannormals), axis=0))

            # Global variables to debug chi2 test
            # chi2var_chi2 = trig.scores_to_chi2(trig.scores_split)
            globals()["scores_to_chi2"] = scores_to_chi2
            globals()["ncpar_chi2"] = ncpar_chi2

            if chi2var_chi2 > stats.ncx2.isf(pthresh_chi2, ndof, ncpar_chi2):
                glitch_mask[1] = False
                if verbose:
                    print("Failed chi^2 veto!")
                if lazy:
                    return False, glitch_mask

        # Perform split tests
        # -------------------
        # -------------------
        split_test_results = []
        bias_split_thresh = []
        for ind, test_details in enumerate(split_test_details):
            # Compute residual and its variance (<real^2> + <imag^2>)
            residual = test_details[0](scores_split[:, rel_index_sub])
            var_residual = test_details[1]
            chi2var_split = utils.abs_sq(residual) * 2 / var_residual

            # Guarantee FAR for waveforms shifted by +/-1/sub_fac index, and
            # mismatched by +/-delta_calpha/2 at measured SNR in all dimensions
            # Teja: I think this maps to +/-delta_calpha, which is the desired
            # output anyway
            chi2_var_split_thresh = np.array(
                [shifted_split_residuals_plus[ind],
                 shifted_split_residuals_minus[ind],
                 *(mismatched_split_residuals[ind].flatten())])
            if n_calpha > 0:
                # Add cross terms to safety
                chi2_var_split_thresh = np.r_[
                    chi2_var_split_thresh,
                    cross_mismatched_split_residuals[ind].flatten()]

            bias_split_thresh.append(chi2_var_split_thresh)

            # Non-centrality parameter for safety
            ncpar_split = np.max(utils.abs_sq(chi2_var_split_thresh)) * \
                (2 / var_residual)

            # Collect results and non-centrality parameters
            split_test_results.append([chi2var_split, ncpar_split])

        globals()["bias_split_thresh"] = bias_split_thresh.copy()

        # Global variables to debug split tests
        # Debugging split test #i
        # residuals = trig.split_test_details[i][0](trig.scores_split)
        # chi2var_split = utils.abs_sq(residuals) * 2 / trig.split_test_details[i][1]
        globals()["split_test_details"] = split_test_details.copy()
        globals()["split_test_results"] = split_test_results.copy()

        # Apply thresholds
        for ind, split_test_result in enumerate(split_test_results):
            chi2var_split, ncpar_split = split_test_result
            if chi2var_split > stats.ncx2.isf(pthresh_split, 2, ncpar_split):
                glitch_mask[2 + ind] = False
                if verbose:
                    print(f"Failed split test #{ind}")
                if lazy:
                    return False, glitch_mask

        return np.all(glitch_mask), glitch_mask

    def veto_trigger_all(
            self, trigger, dcalphas=None, short_wf_limit=params.SHORT_WF_LIMIT,
            subset_details=None, verbose=False, lazy=True, do_costly_vetos=True,
             **kwargs):
        """
        Convenience function to apply all vetos
        :param trigger: Arrow of length len(processedclist[0]) with trigger
        :param dcalphas: Array with spacings in calphas to allow mismatch in
        :param short_wf_limit: Waveform duration below which we avoid holes
        :param subset_details: If available, output of prepare_subset_for_vetoes
        :param verbose: Flag indicating whether to print details
        :param lazy: Flag indicating whether to stop after a test fails
        :param do_costly_vetos: Flag indicating whether we are doing 
                                veto_trigger_phase and veto_trigger_power
        :param kwargs: Arguments to pass to vetos
        :return: 1. Flag that is true/false if the trigger passed/failed
                 2. Correction factor to multiply snr^2 due to finer PSD drift
                 3. Boolean array of size
                    len(self.outlier_reasons) + 3 + len(split_chunks)
                    with zeros marking glitch tests that fired
                    The indices correspond to:
                    0: len(self.outlier_reasons): index into outlier reasons
                       for excess-power-like tests
                    len(self.outlier_reasons): Finer PSD drift killed it
                    len(self.outlier_reasons) + 1: No chunks present
                    len(self.outlier_reasons) + 2: Overall chi-2 test
                    len(self.outlier_reasons) + 3:
                        len(self.outlier_reasons) + 3 + len(split_chunks):
                            Split tests
        """
        # Mask into tests that failed
        split_chunks = kwargs.get("split_chunks", params.SPLIT_CHUNKS)
        glitch_mask = np.ones(
            len(self.outlier_reasons) + 3 + len(split_chunks), dtype=bool)

        # 1. First check that we are not close to `bad' times, to account for
        # not making good enough holes
        # -------------------------------------------------------------------
        # Locate waveform in data
        time_ind, _ = self.get_time_index(trigger)

        # Define safety to avoid holes
        ninds_avoid = int(params.DT_CLEAN_MASK / self.dt)

        # Get indices of important parts of waveform, relative to right edge
        physical_mode_ratio = kwargs.get('physical_mode_ratio', False)
        bestfit_wf_modes = self.get_bestfit_wf(
                trigger,physical_mode_ratio=physical_mode_ratio,
                individual_modes=True)
        bestfit_wf = np.sum(bestfit_wf_modes, axis=0)
        bestfit_wf_cmsm = np.cumsum(bestfit_wf[::-1]**2)
        bestfit_wf_cmsm /= bestfit_wf_cmsm[-1]
        aind, bind = np.searchsorted(bestfit_wf_cmsm, [0.05, 0.95])

        # passed_mask_check = True
        # TODO: avoid vetoing because of valid mask at the edge of the file
        if ((bind - aind) * self.dt) < short_wf_limit:
            # Avoid holes and bad seconds of data
            passed_mask_check = bool(
                np.prod(self.mask[time_ind - bind - ninds_avoid:
                                  time_ind - aind + ninds_avoid]) *
                np.prod(self.valid_mask[time_ind - bind - ninds_avoid:
                                        time_ind - aind + ninds_avoid]))

            if not passed_mask_check:
                # 0 is LIGO HOLE in self.outlier_reasons
                ifailed = 0
                glitch_mask[ifailed] = False
                if verbose:
                    print("Trigger too close to a hole!", flush=True)
                if lazy:
                    return False, 1.0, glitch_mask
        
        if not do_costly_vetos:
            return np.all(glitch_mask), 1.0, glitch_mask

        # 2. Subtract bestfit waveform and look for excess power
        # ------------------------------------------------------
        # Define subset if not available
        if subset_details is None:
            subset_details = self.prepare_subset_for_vetoes(trigger)
            
        # Check if we are vetoing the injected waveform
        test_injection = kwargs.get("test_injection", False)

        # Overwrite parameters of excess power test if passed
        template_max_mismatch = kwargs.get(
            "template_max_mismatch", params.DEF_TEMPLATE_MAX_MISMATCH)
        nfire = kwargs.get("nfire", params.NFIRE)

        vpower = 1 if verbose else 0
        passed_power_veto, snr2_corr_factor, power_veto_details = \
            self.veto_trigger_power(
                trigger, template_max_mismatch=template_max_mismatch,
                nfire=nfire, subset_defined=True, bestfit_wf=bestfit_wf_modes,
                test_injection=test_injection, lazy=lazy, verbose=vpower)
        glitch_mask[1:len(self.outlier_reasons) + 1] = power_veto_details[:]

        if (not passed_power_veto) and lazy:
            return False, snr2_corr_factor, glitch_mask
            
        # 3. Perform chi2 and split tests
        # -------------------------------
        # Overwrite parameters of chi2 and split tests if passed
        n_chunk = kwargs.get("n_chunk", params.N_CHUNK)
        cov_degrade = kwargs.get("cov_degrade", params.COV_DEGRADE)
        chi2_min_eigval = kwargs.get("chi2_min_eigval", params.CHI2_MIN_EIGVAL)
        pthresh_chi2 = kwargs.get("pthresh_chi2", params.THRESHOLD_CHI2)
        pthresh_split = kwargs.get("pthresh_split", params.THRESHOLD_SPLIT)

        passed_phase_veto, phase_veto_details = self.veto_trigger_phase(
            trigger, dcalphas=dcalphas, n_chunk=n_chunk,
            cov_degrade=cov_degrade, chi2_min_eigval=chi2_min_eigval,
            pthresh_chi2=pthresh_chi2, split_chunks=split_chunks,
            pthresh_split=pthresh_split, subset_details=subset_details,
            test_injection=test_injection,
            verbose=verbose, lazy=lazy)
        glitch_mask[len(self.outlier_reasons)+1:] = phase_veto_details[:]

        if (not passed_phase_veto) and lazy:
            return False, snr2_corr_factor, glitch_mask

        return np.all(glitch_mask), snr2_corr_factor, glitch_mask

    # Functions used to optimize triggers
    # -----------------------------------
    def prepare_subset_for_optimization(
            self, trigger=None, location=None, dt=params.DT_OPT,
            subset_defined=False, zero_pad=True):
        """
        Defines subset if needed, and returns parameters to locate trigger in it
        :param trigger: Trigger in the form of a row of a processed clist
        :param location:
            Tuple of length 2 with (linear-free time, calphas), used if trigger
            is not given
        :param dt: Shift in time (s) to allow
        :param subset_defined:
            Flag indicating if we already defined the required subset of data,
            used to save on FFTs
        :param zero_pad:
            Flag indicating whether to zero pad, or pad with existing data
        :return:
            1. calphas of trigger
            2. Index of trigger score in global time
            3. Index of trigger into subset
        """
        # Define trigger index and calphas
        if (location is None) and (trigger is None):
            raise RuntimeError("I need to know a location!")
        elif trigger is not None:
            trig_calpha = trigger[self.c0_pos:].copy()
            relevant_index, _ = self.get_time_index(trigger)
        else:
            bank = self.templatebank
            trig_time_lf, trig_calpha = location
            trig_time = trig_time_lf - bank.shift_whitened_wf * bank.dt
            relevant_index = np.searchsorted(self.time, trig_time)

        # Fix for bank 4, 3 with no important calphas
        if len(trig_calpha) == 0:
            trig_calpha = np.zeros(1)

        # Define subset of data, if required
        if not subset_defined:
            # Define number of indices to allow
            ninds = int(np.ceil(dt / self.dt))
            # Prepare subset of data
            self.prepare_subset_for_triggers(
                relevant_index, ninds, ninds, zero_pad=zero_pad)

        # Define the position of the zero-lag score of the
        # (non-sinc interpolated) trigger in the subset of data
        relevant_index_sub = relevant_index + self.offset_sub

        return trig_calpha, relevant_index, relevant_index_sub

    def define_finer_grid_func(self, dcalpha_coarse=None, trim_dims=True):
        """
        :param dcalpha_coarse: calpha_spacing of coarse grid
        :param trim_dims:
            Flag indicating whether to cut dimensions at self.dcalphas
        :return:
            1. Function that returns calphas in a finer grid around a given
               calpha
            2. Array with spacings of finer grid in all dimensions, used for
               safeties
        """
        bank = self.templatebank

        if dcalpha_coarse is None:
            dcalpha_coarse = self.delta_calpha

        # Define coarse calpha grid
        coarse_axes = bank.make_grid_axes(
            dcalpha_coarse, self.template_safety, self.force_zero)

        # If requested, trim the coarse grid at len(self.dcalphas)
        if trim_dims:
            dcalphas = self.dcalphas
            for ind_axes in range(len(coarse_axes)):
                if ind_axes >= len(dcalphas):
                    coarse_axes[ind_axes] = np.zeros(1)

        # Define finer calpha grid
        fine_grid_axes, fine_dims, spacings = bank.define_important_grid(
            dcalpha_coarse / 2, fudge=self.template_safety,
            force_zero=self.force_zero)
        max_ind = len(spacings)

        # Function giving the calphas of finer grid around a trigger,
        # including the trigger itself
        def finer_grid_func(trig_calpha):

            trig_calpha = np.asarray(trig_calpha)

            # Set the default to be one point at 0 for irrelevant dimensions
            opt_grid_axes = [[0.] for _ in range(max_ind)]

            # Extend the coordinates to have the same number of dimensions
            if len(trig_calpha) < max_ind:
                trig_calpha = np.concatenate(
                    [trig_calpha, np.zeros(max_ind - len(trig_calpha))])

            # Set the other dimensions in the finer grid
            for dim_ind, fine_coords in fine_dims:
                # First add the trigger's calpha to the finer grid
                if trig_calpha[dim_ind] not in fine_coords:
                    fine_coords = np.r_[fine_coords, trig_calpha[dim_ind]]
                # For every dimension, find the fine grid points that are
                # the responsibility of this trigger, i.e., closer to this point
                # than any other in the coarse grid
                # Index of calpha in coarse grid
                my_ind_coarse = np.argmin(
                    np.abs(trig_calpha[dim_ind] - coarse_axes[dim_ind]))
                # All fine coordinates that are closer to this coarse point than
                # any other
                opt_grid_axes[dim_ind] = \
                    [x for x in fine_coords if
                     np.argmin(np.abs(x - coarse_axes[dim_ind])) == my_ind_coarse]

            finer_calphas = bank.remove_nonphysical_templates(opt_grid_axes)

            return finer_calphas

        return finer_grid_func, spacings

    def prepare_summary_stats(
            self, trigger=None, location=None, relative_freq_bins=None,
            dcalphas=None, dt=params.DT_OPT, delta=0.1, subset_defined=False,
            relative_binning=True, zero_pad=True, calc_mode_covariance=True,
            use_HM=True):
        """
        Defines summary statistics for convolution with relative waveforms
        The statistics correspond to convolution with the calpha waveform,
        without any subgrid shifts
        :param trigger: Trigger in the form of a row of a processed clist
        :param location:
            Tuple of length 2 with (linear-free time, calphas), used if trigger
            is not given
        :param relative_freq_bins:
            Array with bin edges for frequency interpolation
        :param dcalphas: Array with calpha spacings of finer grid to optimize
        :param dt: Shift in time (s) to allow
        :param delta: Phase allowed to accumulate within each bin
        :param subset_defined:
            Flag indicating if we already defined the required subset of data,
            used to save on FFTs
        :param relative_binning:
            Flag indicating whether we are using relative binning
        :param zero_pad:
            Flag indicating whether to zero pad, or pad with existing data
        :param calc_mode_covariance:
            Flag indicating whether to calculate the summary stats for the
            covariance matrix of the modes
            (later used for orthogonalizing scores of modes)
        :param use_HM: Not yet implemented
        :return:
            If do_relative_binning
                1. Array with summary stats for constant phase
                2. Array with summary stats for linear phase
                3. Array with bin edges adjusted to lie on a FFT grid
                4. Absolute GPS time of right edge of waveform + 1 index = score
                5. Array with hole corrections on self.time_sub
                6. Array with valid inds on self.time_sub
            Else:
                1. Array with fft of overlaps with fiducial one set to zero-lag
                2. Frequencies of fft grid
                3. Absolute GPS time of right edge of waveform + 1 index = score
                4. Array with hole corrections on self.time_sub
                5. Array with valid inds on self.time_sub
        """
        trig_calpha, relevant_index, relevant_index_sub = \
            self.prepare_subset_for_optimization(
                trigger=trigger, location=location, dt=dt,
                subset_defined=subset_defined, zero_pad=zero_pad)

        # Define the global GPS time, safe to the edge case when the score is
        # achieved at the end
        score_time = self.time[0] + relevant_index * self.dt

        # Frequency domain conjugate to time_sub
        fs_sub = np.fft.rfftfreq(len(self.time_sub), d=self.dt)

        # Compute the overlaps, hole corrections, and valid inds for the
        # given calphas without sinc interpolation. Avoid zeroing invalid
        # scores because it is nasty in the Fourier domain
        overlaps, hole_corrections, valid_inds = self.gen_scores(
                calphas=trig_calpha, subset=True, zero_invalid=False,
                orthogonalize_modes=False)
        # Note that the modes are not orthogonalized at this point
        # we will calculate the covariance matrix of the modes
        # below if needed, which we will later use to orthogonalize the modes

        # The complex overlaps have only positive frequencies
        overlaps_rfft = utils.FFTIN((3,len(fs_sub)), dtype=np.complex128)
        overlaps_rfft[:] = utils.RFFT(overlaps.real) + \
                            1j * utils.RFFT(overlaps.imag)

        # Shift the zero-lag overlap to the zeroth index
        overlaps_rfft[:] = overlaps_rfft * np.exp(
            2. * np.pi * 1j * fs_sub * relevant_index_sub * self.dt)

        CovMat_rfft = []
        if calc_mode_covariance:
            wfs = self.templatebank.gen_whitened_wfs_td(
                        calpha=trig_calpha, orthogonalize=False, return_cov=False)
            wfs = wfs[:,-len(self.time_sub):]
            wfs = utils.RFFT(wfs, axis=-1)
            CovMat_rfft = np.array([wfs[0]* np.conj(wfs[1]),
                             wfs[0]* np.conj(wfs[2]), wfs[1]* np.conj(wfs[2])])
            CovMat_rfft /= len(fs_sub)

        if relative_binning:
            # Define frequency bins for 22 for linear interpolation, if required
            if relative_freq_bins is None:
                if dcalphas is None:
                    _, dcalphas = self.define_finer_grid_func()
                relative_freq_bins = self.templatebank.def_relative_bins(
                    dcalphas, dt=dt, delta=delta)

            # This can get triggered if the template bank uses a wider
            # frequency range than the analysis code, should be dealt with
            # before it reaches this stage!
            if np.any(relative_freq_bins > fs_sub[-1]):
                raise RuntimeError("FMAX is definitely not enough for this bank, "\
                                           + "try to increase it")

            # Find the best match to relative_freq_bins for different modes
            fs_rb = []
            constant_stats = []; linear_stats=[]
            closest_inds_modes = []
            for mode in range(3):
                mask = relative_freq_bins * (mode+2)/2 <= fs_sub[-1]
                relative_freq_bins_masked = relative_freq_bins[mask] * (mode+2) / 2
                fs_inds = np.searchsorted(fs_sub, relative_freq_bins_masked)

                if len(set(fs_inds)) < len(fs_inds):
                    raise RuntimeError(
                        "Frequency resolution not enough, " +
                        "increase the subset size!")
                
                closest_inds = fs_inds.copy()
                for bin_ind, (fs_ind, relative_freq_bin) \
                        in enumerate(zip(fs_inds, relative_freq_bins_masked)):
                    if ((fs_ind > 0) and
                        (np.abs(fs_sub[fs_ind-1] - relative_freq_bin) <
                         np.abs(fs_sub[fs_ind] - relative_freq_bin))):
                        closest_inds[bin_ind] = fs_ind - 1
                
                # Prepare binned summary data for convolution with
                # relative waveforms
                overlaps_rfft_split = np.split(overlaps_rfft[mode], closest_inds)
                closest_inds_modes.append(closest_inds)
                constant_stats.append([np.sum(x) for x in overlaps_rfft_split][1:-1])
                linear_stats.append(
                    [np.sum(x * np.linspace(-0.5, 0.5, num=len(x)+1)[:-1])
                     for x in overlaps_rfft_split][1:-1])
                fs_rb.append(fs_sub[closest_inds])

            constant_stats_cov = [[0 for _ in range(3)] for _ in range(3)];
            linear_stats_cov = [[0 for _ in range(3)] for _ in range(3)];
            if calc_mode_covariance:
                # Summary stats corresponding to extra-diagonal elements of
                # the symmetric covariance matrix
                for ind, (mi, mj) in enumerate([(0,1), (0,2), (1,2)]):
                    CovMat_rfft_split = np.split(CovMat_rfft[ind],
                                                closest_inds_modes[mi])
                    constant_stats_cov[mi][mj] = [np.sum(x) for x in CovMat_rfft_split][1:-1]
                    linear_stats_cov[mi][mj] = [np.sum(x * np.linspace(-0.5, 0.5, num=len(x)+1)[:-1])
                                                for x in CovMat_rfft_split][1:-1]

            return constant_stats, linear_stats, fs_rb, score_time, \
                hole_corrections, valid_inds, constant_stats_cov, linear_stats_cov
        else:
            return overlaps_rfft, np.ones((3, len(fs_sub)))*fs_sub, score_time, \
                hole_corrections, valid_inds, CovMat_rfft

    def gen_triggers_local(
            self, trigger=None, location=None, dt_left=params.DT_OPT,
            dt_right=params.DT_OPT, subset_defined=False, avoid_ids=None,
            compute_calphas=None, apply_threshold=True, relative_binning=True,
            delta=0.1, relative_freq_bins=None, zero_pad=True, best_only=False,
            orthogonalize_modes=True, return_mode_covariance=False):
        """
        Generates triggers at/on a small calpha grid around a trigger
        Has some not so quantifiable losses/biases due to the truncation of
        the waveforms that are being compared, and interpolations of the hole
        and PSD drift corrections
        Note: If the location is beyond the length of the data, it actually generates
        triggers around the edge due to quirks of searchsorted
        :param trigger: Trigger in the form of a row of a processed clist
        :param location:
            Tuple of length 2 with (linear-free time, calphas), used if the
            trigger was not given
        :param dt_left: Keep triggers with (t_trig - dt_left) <= t_lf
        :param dt_right: Keep triggers with t_lf <= (t_trig + dt_right)
        :param subset_defined:
            Flag indicating if we already defined the required subset of data
        :param avoid_ids:
            If needed, pass list of trigger IDs to avoid computing
            triggers and save on computations during coincidence
        :param compute_calphas:
            If needed, pass list/2D np.array of calphas to compute triggers
            for, overrides the default set to iterate over
        :param apply_threshold: Flag indicating whether to apply SNR cut
        :param relative_binning:
            Flag indicating whether we are using relative binning
        :param delta: Phase allowed to accumulate within each bin
        :param relative_freq_bins:
            Array with bin edges for frequency interpolation
        :param zero_pad:
            Flag indicating whether to zero pad, or pad with existing data
        :param best_only:
            Flag whether to return only the best trigger for each calpha,
            useful for making heatmaps
        :param orthogonalize_modes: Orthogonalizes the scores from different modes
        :return: Processedclist with triggers within dt_allowed
        """
        # Define list of waveforms to iterate over
        # ----------------------------------------
        # Read off the calphas of the central waveform
        if (location is None) and (trigger is None):
            raise RuntimeError("I need to know a location!")
        elif trigger is not None:
            trig_calpha = trigger[self.c0_pos:]
        else:
            _, trig_calpha = location

        # Define list of calphas that we have to compute triggers for
        # May include repetitions
        if compute_calphas is not None:
            # Compute only where asked
            calphas_to_compute = np.asarray(compute_calphas)
            if (len(calphas_to_compute) > 0) and \
                    (calphas_to_compute.shape[-1] == 0):
                calphas_to_compute = np.c_[
                    calphas_to_compute, np.zeros(len(calphas_to_compute))]
        else:
            # Function that returns finer grid points given a coarse calpha
            finer_grid_func, _ = self.define_finer_grid_func()
            calphas_to_compute = finer_grid_func(trig_calpha)

        # TODO: Fix mess to work with avoid calphas
        unique_calphas = list(
            {tuple(calphas) for calphas in calphas_to_compute})
        # Fix mess, hash((-159, 9)) = hash((159, -9))
        # # Define list of hashes of template ids to avoid
        # if avoid_ids is None:
        #     ids_to_avoid = []
        # else:
        #     ids_to_avoid = avoid_ids.copy()
        # # avoid_ids = utils.make_template_ids(avoid_calphas)
        #
        # # Prune list of calphas to unique list
        # unique_calphas = []
        # template_ids = utils.make_template_ids(calphas_to_compute)
        # for template_id, calphas_nopt in zip(template_ids, calphas_to_compute):
        #     # Define shifted calphas
        #     # calpha_opt = np.r_[np.array(calphas_nopt), trig_calpha[nopt:]]
        #     # calpha_opt = np.asarray(calphas_nopt)
        #     # trig_id = utils.make_template_ids(calpha_opt)
        #
        #     if template_id not in ids_to_avoid:
        #         # Add to lists of computed calphas and ids
        #         # unique_calphas.append(calpha_opt.copy())
        #         unique_calphas.append(np.asarray(calphas_nopt))
        #         # Avoid further repetitions
        #         ids_to_avoid.append(template_id)

        if len(unique_calphas) == 0:
            # We don't need to do any work
            return np.array([])

        # Code to compute the overlaps
        # ----------------------------
        # Defines summary statistics for overlaps with relative
        # waveforms, and subset of data if not already done
        calc_mode_covariance = orthogonalize_modes or return_mode_covariance
        summary_stats = self.prepare_summary_stats(
            trigger=trigger, location=location,
            relative_freq_bins=relative_freq_bins,
            dt=max(dt_left, dt_right), delta=delta,
            subset_defined=subset_defined, relative_binning=relative_binning,
            zero_pad=zero_pad,
            calc_mode_covariance=calc_mode_covariance)

        # All waveforms in the bank have the same amplitude, so we only
        # need their phases
        bank = self.templatebank

        if relative_binning:
            constant_stats, linear_stats, fs_rb, gps_time_right, hole_corrections,\
                  valid_inds, constant_stats_cov, linear_stats_cov = summary_stats
            central_wf_phases = bank.gen_phases_from_calpha(
                    trig_calpha, fs_out=fs_rb)
        else:
            overlaps_rfft, fs_rb, gps_time_right, hole_corrections,\
                  valid_inds, CovMat_rfft = summary_stats
            central_wf_phases = bank.gen_phases_from_calpha(
                    trig_calpha, fs_out=fs_rb[0])

        # Array of shifts to allow
        # First adjust extremes so that they lie on the grid
        dt_left = np.ceil(dt_left / self.dt) * self.dt
        dt_right = np.ceil(dt_right / self.dt) * self.dt

        # Fix the sinc-interpolated time grid
        n_interp = int(np.round(np.log2(self.dt / params.DT_FINAL)))
        sub_fac = 2**n_interp
        dt_fine = self.dt / sub_fac
        # Right edge is not inclusive
        ntimes = (dt_right + dt_left) / dt_fine
        dt_arr = np.arange(ntimes) * dt_fine - dt_left

        # Define subgrid hole corrections and PSD drift corrections using
        # linear interpolation, not perfect but works
        gps_time_scores = dt_arr + gps_time_right
        hole_corrections_int = np.zeros((3,len(gps_time_scores)))
        for i in range(3):
            hole_corrections_int[i] = np.interp(
                gps_time_scores, self.time_sub, hole_corrections[i])
        psd_drift_correction_int = np.interp(
            gps_time_scores, self.time_sub, self.psd_drift_correction_sub)

        # Define valid mask on subgrid
        inds_mask = np.floor(
            ((gps_time_scores - self.time_sub[0]) / self.dt)).astype(int)
        # Ensure that we don't have an error due to boundary conditions
        inds_mask_to_fill = (inds_mask >= 0) * (inds_mask < len(self.time_sub))
        valid_inds_sub = utils.FFTIN((3,len(gps_time_scores)), dtype=bool)
        valid_inds_sub[:, inds_mask_to_fill] = \
            valid_inds[:, inds_mask[inds_mask_to_fill]]

        # Function to compute scores by convolving with relative waveforms
        def relative_waveform_scores(calphas, central_wf_phases):

            if relative_binning:
                scores = []
                # Define phases of zero-lag waveforms
                zero_lag_phases = bank.gen_phases_from_calpha(
                        calphas, fs_out=fs_rb)
                for mode in range(3):
                    zero_lag_relative_phase = zero_lag_phases[mode] - central_wf_phases[mode]
                    # Phases of shifted waveforms relative to the trigger waveform
                    shift_arrays = - 2. * np.pi * dt_arr[:, np.newaxis] * fs_rb[mode]
                    rel_phases = shift_arrays + zero_lag_relative_phase
                    rel_wfs = np.exp(1j * rel_phases)

                    constant_parts = 0.5 * (rel_wfs[:, 1:] + rel_wfs[:, :-1])
                    linear_parts = (rel_wfs[:, 1:] - rel_wfs[:, :-1])

                    # The complexified scores have only positive frequencies
                    scores.append(np.sum(constant_parts.conj() * constant_stats[mode], axis=-1) + \
                            np.sum(linear_parts.conj() * linear_stats[mode], axis=-1))

                CovMat = np.diag((1,1,1)).astype(np.complex128)
                if calc_mode_covariance:
                    # Finding off-diagonal elements of the covariance matrix
                    for mi in [0,1]:
                        central_wf_phases = bank.gen_phases_from_calpha(
                            trig_calpha, fs_out=fs_rb[mi])
                        zero_lag_phases = bank.gen_phases_from_calpha(
                            calphas, fs_out=fs_rb[mi])
                        for mj in range(mi+1, 3):
                            rel_phases = central_wf_phases[mi]-central_wf_phases[mj]\
                                                    - zero_lag_phases[mi] + zero_lag_phases[mj]
                            rel_wfs = np.exp(1j*rel_phases)
                            constant_parts = 0.5 * (rel_wfs[1:] + rel_wfs[:-1])
                            linear_parts = (rel_wfs[1:] - rel_wfs[:-1])

                            CovMat[mi, mj] = np.sum(
                                        constant_parts.conj() * constant_stats_cov[mi][mj], axis=-1) \
                                        + np.sum(linear_parts.conj() * linear_stats_cov[mi][mj], axis=-1)
                            CovMat[mj, mi] = np.conj(CovMat[mi, mj])

                # Fix IRFFT normalization
                scores = np.array(scores) / len(self.time_sub)
            else:
                # Define phases of zero-lag waveforms
                zero_lag_phases = bank.gen_phases_from_calpha(calphas, fs_out=fs_rb[0])
                zero_lag_relative_phase = zero_lag_phases - central_wf_phases
                # Make 2D array with subgrid shifts to use FFT
                dt_shift = np.arange(sub_fac) * dt_fine
                shift_arrays = - 2. * np.pi * dt_shift[:, np.newaxis, np.newaxis] \
                                * fs_rb
                rel_phases = shift_arrays + zero_lag_relative_phase
                rel_wfs = np.exp(1j * rel_phases)

                # Go into full frequency grid
                # The complexified scores have only positive frequencies
                overlaps_fft = utils.FFTIN((3,len(self.time_sub)), dtype=np.complex128)
                overlaps_fft[:,:len(overlaps_rfft[0])] = overlaps_rfft[:]

                # Extend the relative waveforms
                rel_wfs_fft = utils.FFTIN(
                    (len(rel_wfs), 3, len(self.time_sub)), dtype=np.complex128)
                rel_wfs_fft[:, :, :len(rel_wfs[0,0])] = rel_wfs[:,:,:]
                if len(self.time_sub) % 2 == 0:
                    # Leave Nyquist freq and zero unaffected
                    rel_wfs_fft[:, :, len(rel_wfs[0,0]):] = \
                        rel_wfs[:, :, 1:-1][:, :, ::-1].conj()
                else:
                    rel_wfs_fft[:, :, len(rel_wfs[0,0]):] = \
                        rel_wfs[:,:, 1:][:, :, ::-1].conj()

                scores = utils.IFFT(rel_wfs_fft.conj() * overlaps_fft, axis=-1)

                # Pick out the subset corresponding to our required lags,
                # in the right order
                ind_zero_lag = np.searchsorted(dt_arr, 0)
                n_nonnegative = len(dt_arr[ind_zero_lag::sub_fac])
                n_negative = len(dt_arr[:ind_zero_lag][::sub_fac])
                scores = np.array([np.c_[scores[:, mode, -n_negative:],
                               scores[:, mode, :n_nonnegative]].flatten(order='F')
                               for mode in range(3)])
                CovMat = np.diag((1,1,1)).astype(np.complex128)
                if calc_mode_covariance:
                    for ind, (mi, mj) in enumerate([(0,1), (0,2), (1,2)]):
                        rel_wfs = np.exp(1j*(zero_lag_relative_phase[mi]-zero_lag_relative_phase[mj]))
                        CovMat[mi, mj] = np.sum(rel_wfs * CovMat_rfft[ind])
                        CovMat[mj, mi] = np.conj(CovMat[mi, mj])

            return scores, CovMat

        # Define function to go from shifted scores to processedclists
        def scores_to_processed_clist(scores, CovMat, calphas):
            # Apply hole corrections from the central waveform
            h_corr_scores = scores / (hole_corrections_int + params.HOLE_EPS)

            # Retain only valid scores
            h_corr_scores *= valid_inds_sub

            if orthogonalize_modes:
                # _, CovMat = bank.gen_whitened_wfs_td(
                #                     calphas, orthogonalize=False, return_cov=True)
                L = np.linalg.cholesky(np.linalg.inv(CovMat)[::-1,::-1])[::-1,::-1]
                for i in range(len(h_corr_scores[0])):
                    h_corr_scores[:,i] = np.dot(L.T, h_corr_scores[:,i])

            # Define clist
            if self.save_hole_correction:
                trigger_block = np.c_[
                    gps_time_scores,
                    h_corr_scores[0].real,
                    h_corr_scores[0].imag,
                    h_corr_scores[1].real,
                    h_corr_scores[1].imag,
                    h_corr_scores[2].real,
                    h_corr_scores[2].imag,
                    hole_corrections_int[0],
                    hole_corrections_int[1],
                    hole_corrections_int[2],
                    psd_drift_correction_int,
                    np.repeat(np.array(calphas)[np.newaxis, :],
                              len(gps_time_scores), axis=0)]
            else:
                trigger_block = np.c_[
                        gps_time_scores,
                        h_corr_scores[0].real,
                        h_corr_scores[0].imag,
                        h_corr_scores[1].real,
                        h_corr_scores[1].imag,
                        h_corr_scores[2].real,
                        h_corr_scores[2].imag,
                        psd_drift_correction_int,
                        np.repeat(np.array(calphas)[np.newaxis, :],
                                  len(gps_time_scores), axis=0)]

            processedclist = self.process_clist(trigger_block)

            return processedclist

        # List of calphas that we end up computing triggers for
        processedclist_sub = []
        
        CovMat_calphas = []
        for calphas_nopt in unique_calphas:
            # Define shifted calphas
            calpha_opt = np.asarray(calphas_nopt)

            # Generate triggers
            int_scores, CovMat = relative_waveform_scores(calpha_opt, central_wf_phases)
            trigger_list = scores_to_processed_clist(int_scores, CovMat, calpha_opt)
            CovMat_calphas.append(CovMat)

            if len(trigger_list) > 0:
                # Keep only those above the threshold, if needed
                if apply_threshold:
                    trigger_list = self.filter_processed_clist(
                        trigger_list,
                        filters={'snr': (self.threshold_chi2 ** 0.5, np.inf)})
                if best_only and len(trigger_list) > 0:
                    # Pick only the best trigger
                    trigger_list = trigger_list[np.argmax(trigger_list[:, 1])]
                processedclist_sub.append(trigger_list)

        if len(processedclist_sub) > 0:
            processedclist_sub = np.vstack(processedclist_sub)

        # Ensure that we return a numpy array, always 2-D
        processedclist_sub = np.asarray(processedclist_sub)

        if return_mode_covariance:
            return processedclist_sub, CovMat_calphas
        else:
            return processedclist_sub

    def gen_triggers_local_pars(
            self, trigger=None, location=None, dt_left=params.DT_OPT,
            dt_right=params.DT_OPT, subset_defined=False,
            relative_binning=True, delta=0.1, relative_freq_bins=None,
            psd_drift_kwargs=None, **pars):
        """
        NOTE: This fn has not been modified by Jay for higher modes.
        Generates triggers at/on a small calpha grid around a trigger
        Has some not so quantifiable losses/biases due to the truncation of
        the waveforms that are being compared, and interpolations of the hole
        and PSD drift corrections
        :param trigger: Trigger in the form of a row of a processed clist
        :param location:
            Tuple of length 2 with (linear-free time, calphas), used if the
            trigger was not given
        :param dt_left: Keep triggers with (t_trig - dt_left) <= t_lf
        :param dt_right: Keep triggers with t_lf <= (t_trig + dt_right)
        :param subset_defined:
            Flag indicating if we already defined the required subset of data
        :param relative_binning:
            Flag indicating whether we are using relative binning
        :param delta: Phase allowed to accumulate within each bin
        :param relative_freq_bins:
            Array with bin edges for frequency interpolation
        :param psd_drift_kwargs:
            Dictionary with any extra arguments for recomputing PSD drift
        :param pars:
            Dictionary with list of parameters for generating waveforms
            (look at gen_wf_fd_from_pars)
        :return: Processedclist with triggers within dt_allowed
        """
        # Define central waveform, and subset if not defined
        trig_calpha, *_ = self.prepare_subset_for_optimization(
            trigger=trigger, location=location, dt=max(dt_left, dt_right),
            subset_defined=subset_defined, zero_pad=False)

        # Define summary statistics for overlaps with relative waveforms
        summary_stats = self.prepare_summary_stats(
            trigger=trigger, location=location,
            relative_freq_bins=relative_freq_bins,
            dt=max(dt_left, dt_right), delta=delta, subset_defined=True,
            relative_binning=relative_binning, zero_pad=False)

        if relative_binning:
            constant_stats, linear_stats, fs_rb, gps_time_right, \
                hole_corrections, valid_inds = summary_stats
        else:
            overlaps_rfft, fs_rb, gps_time_right, \
                hole_corrections, valid_inds = summary_stats

        # Define normfac for given waveform
        # TODO: Do this with relative binning too
        bank = self.templatebank
        wf_whitened_td_shifted = bank.gen_whitened_wf_td_from_pars(
            highpass=True, **pars)
        wf_normfac = np.linalg.norm(wf_whitened_td_shifted)

        # Define relative waveform
        # TODO: Avoid second waveform generation to save some computations
        wf_fd_unconditioned = \
            bank.gen_wf_fd_from_pars(fs_out=fs_rb, **pars) / wf_normfac
        central_wf = bank.gen_wfs_fd_from_calpha(
            trig_calpha, fs_out=fs_rb) / bank.normfac
        rel_wf = wf_fd_unconditioned / central_wf
        rel_wf[np.isinf(rel_wf)] = 0
        rel_wf[np.isnan(rel_wf)] = 0

        # Array of shifts to allow
        # First adjust extremes so that they lie on the grid
        dt_left = np.ceil(dt_left / self.dt) * self.dt
        dt_right = np.ceil(dt_right / self.dt) * self.dt
        dt_sub = self.dt / 4
        # Right edge is not inclusive
        ntimes = (dt_right + dt_left) / dt_sub
        dt_arr = np.arange(ntimes) * dt_sub - dt_left

        # Recompute PSD drift correction using the properly
        # conditioned whitened FD physical waveform
        if psd_drift_kwargs is None:
            psd_drift_kwargs = {}
        wf_whitened_fd_shifted = utils.RFFT(wf_whitened_td_shifted / wf_normfac)
        psd_drift_correction = self.gen_psd_drift_correction(
            wf_whitened_fd=wf_whitened_fd_shifted, **psd_drift_kwargs)
        # Use the right subset
        psd_drift_correction = \
            np.pad(
                psd_drift_correction[
                    self.relevant_index - self.left_inds:
                    self.relevant_index + self.right_inds],
                (0, len(self.time_sub) - self.left_inds - self.right_inds),
                mode='constant')

        # Define subgrid hole corrections and PSD drift corrections using
        # linear interpolation, not perfect but works
        gps_time_scores = dt_arr + gps_time_right
        hole_corrections_int = np.zeros((3,len(gps_time_scores)))
        for i in range(3):
            hole_corrections_int[i] = np.interp(
                gps_time_scores, self.time_sub, hole_corrections[i])
        psd_drift_correction_int = np.interp(
            gps_time_scores, self.time_sub, psd_drift_correction)

        # Define valid mask on subgrid
        inds_mask = np.floor(
            ((gps_time_scores - self.time_sub[0]) / self.dt)).astype(int)
        # Ensure that we don't have an error due to boundary conditions
        inds_mask_to_fill = (inds_mask >= 0) * (inds_mask < len(self.time_sub))
        valid_inds_sub = utils.FFTIN((3,len(gps_time_scores)), dtype=bool)
        valid_inds_sub[:,inds_mask_to_fill] = \
            valid_inds[:,inds_mask[inds_mask_to_fill]]

        # Compute scores by convolving with relative waveforms
        if relative_binning:
            # Phases of shifted waveforms relative to the trigger waveform
            shift_arrays = - 2. * np.pi * dt_arr[:, np.newaxis] * fs_rb
            rel_wfs = np.exp(1j * shift_arrays) * rel_wf

            constant_parts = 0.5 * (rel_wfs[:, 1:] + rel_wfs[:, :-1])
            linear_parts = (rel_wfs[:, 1:] - rel_wfs[:, :-1])

            # The complexified scores have only positive frequencies
            scores = np.dot(constant_parts.conj(), constant_stats) + \
                np.dot(linear_parts.conj(), linear_stats)

            # Fix IRFFT normalization
            scores /= len(self.time_sub)
        else:
            # Make 2D array with subgrid shifts to use FFT
            dt_shift = np.arange(4) * self.dt / 4
            shift_arrays = - 2. * np.pi * dt_shift[:, np.newaxis] * fs_rb
            rel_wfs = np.exp(1j * shift_arrays) * rel_wf

            # Go into full frequency grid
            # The complexified scores have only positive frequencies
            overlaps_fft = utils.FFTIN(
                len(self.time_sub), dtype=np.complex128)
            overlaps_fft[:len(overlaps_rfft)] = overlaps_rfft[:]

            # Extend the relative waveforms
            rel_wfs_fft = utils.FFTIN(
                (len(rel_wfs), len(self.time_sub)), dtype=np.complex128)
            rel_wfs_fft[:, :len(rel_wfs[0])] = rel_wfs[:, :]
            if len(self.time_sub) % 2 == 0:
                # Leave Nyquist freq and zero unaffected
                rel_wfs_fft[:, len(rel_wfs[0]):] = \
                    rel_wfs[:, 1:-1][:, ::-1].conj()
            else:
                rel_wfs_fft[:, len(rel_wfs[0]):] = \
                    rel_wfs[:, 1:][:, ::-1].conj()

            scores = utils.IFFT(rel_wfs_fft.conj() * overlaps_fft, axis=-1)

            # Pick out subset corresponding to our required lags, in the
            # right order
            ind_zero_lag = np.searchsorted(dt_arr, 0)
            n_nonnegative = len(dt_arr[ind_zero_lag::4])
            n_negative = len(dt_arr[:ind_zero_lag][::4])
            scores = np.c_[scores[:, -n_negative:],
                           scores[:, :n_nonnegative]].flatten(order='F')

        # Apply hole corrections from central waveform, and retain valid ones
        h_corr_scores = scores / (hole_corrections_int + params.HOLE_EPS)
        h_corr_scores *= valid_inds_sub

        # Apply PSD drift correction
        h_corr_scores /= psd_drift_correction_int

        # Return linear-free times corresponding to scores
        gps_time_scores += bank.shift_whitened_wf * bank.dt

        return gps_time_scores, h_corr_scores, hole_corrections_int, \
            psd_drift_correction_int

    # Functions to filter triggers
    # -------------------------------------------------------------------------
    def filter_triggers(self, filters=None, rejects=None, reset_filters=False):
        """Apply cuts on triggers in self.filteredclist. Call with filters=None,
        rejects=None, and reset_filters=True to clear filters

        PARAMETERS:
            filters: Dictionary with cuts in the form
                     {parameter name: (vmin1, vmax1)}. Make vmin1 or vmax1
                     -/+np.inf if you don't want to apply that
                     List of allowed parameters:
                     time: Offset of shift corrected trigger times from fiducial
                           start of central file
                     snr: Signal to noise (properly corrected by PSD drift etc)
                     cossnr: Cosine component of SNR
                     sinsnr: Sine component of SNR
                     calpha: Coefficients of basis functions in template bank,
                             Need to pass alpha
                     alpha: Index into calpha (start with zero)
                     If any of the above parameters is not set, the data is not
                     filtered by that criterion
            rejects: Dictionary with rejection criteria in the form
                     {parameter name: [(vmin1, vmax1),...]}, or
                     {parameter name: (vmin1, vmax1)} if only one
            reset_filters: Boolean flag to indicate whether filters/rejects are
                           to be applied on top of existing ones
        """
        # First update the criteria if needed
        if reset_filters:
            # We're creating new filters all over
            self.filters = {}
            self.rejects = defaultdict(list)

            # Pull in all the triggers
            self.filteredclist = copy.deepcopy(self.processedclist)

        # Now load the new filters
        # Filters overwrite the previous ones
        if filters is not None:
            self.filters = filters.copy()

        # Rejects are added to the previous ones
        if rejects is not None:
            for key, criteria in rejects.items():
                if type(criteria) == list:
                    self.rejects[key] += criteria
                else:
                    self.rejects[key].append(criteria)

        if ((filters is None) and (rejects is None)) or \
                (len(self.filteredclist) == 0):
            # Nothing to do
            return

        self.filteredclist = self.filter_processed_clist(
            self.filteredclist, filters=self.filters, rejects=self.rejects, 
            t0=self.t0, rezpos=self.rezpos, imzpos=self.imzpos, 
            c0_pos=self.c0_pos)

        return

    def clip_outliers(self, snr_thresh=10, time_tol=1., reset_filters=False):
        if reset_filters:
            self.clear_filter()
        self.filteredclist = self.remove_loud_times(
            self.filteredclist, snr_thresh=snr_thresh, time_tol=time_tol)
        return

    def clear_filter(self):
        self.filter_triggers(filters=None, rejects=None, reset_filters=True)
        return

    @staticmethod
    def filter_processed_clist(
            processedclist, filters=None, rejects=None, t0=0., c0_pos=None):
        """
        Filters a processed clist
        :param processedclist: Must be a numpy array, not a list
        :param filters: See filter_triggers for values
        :param rejects:
        :param t0:
        :param c0_pos:
        :return: filteredclist
        """
        if utils.checkempty(processedclist):
            return np.array([])

        # Make mask
        cmask = np.ones(len(processedclist), dtype=bool)

        # Apply filters
        if filters is not None and (len(filters) > 0):
            # Time
            if 'time' in filters.keys():
                cmask *= (processedclist[:, 0] >= (filters['time'][0] + t0))
                cmask *= (processedclist[:, 0] <= (filters['time'][1] + t0))
            # SNR
            if 'snr' in filters.keys():
                snrSQ = processedclist[:, 1]
                cmask *= (snrSQ >= filters['snr'][0]**2)
                cmask *= (snrSQ <= filters['snr'][1]**2)
            ## Cosine SNR
            #if 'cossnr' in filters.keys():
            #    cmask *= (processedclist[:, rezpos] >= filters['cossnr'][0])
            #    cmask *= (processedclist[:, rezpos] <= filters['cossnr'][1])
            #if 'sinsnr' in filters.keys():
            #    cmask *= (processedclist[:, imzpos] >= filters['sinsnr'][0])
            #    cmask *= (processedclist[:, imzpos] <= filters['sinsnr'][1])
            
            # coefficients
            if 'calpha' in filters.keys():
                ind = filters['alpha'] + c0_pos
                cmask *= (processedclist[:, ind] >= filters['calpha'][0])
                cmask *= (processedclist[:, ind] <= filters['calpha'][1])

        # Apply rejects
        if rejects is not None:
            # Time
            if 'time' in rejects.keys():
                for interval in rejects['time']:
                    cmask *= np.logical_or(
                        processedclist[:, 0] < (interval[0] + t0),
                        processedclist[:, 0] > (interval[1] + t0))
            # SNR
            if 'snr' in rejects.keys():
                for interval in rejects['snr']:
                    snrSQ = processedclist[:, 1]
                    cmask *= np.logical_or(
                        snrSQ < (interval[0] ** 2),
                        snrSQ > (interval[1] ** 2))
            ## Cosine SNR
            #if 'cossnr' in rejects.keys():
            #    for interval in rejects['cossnr']:
            #        cmask *= np.logical_or(
            #            processedclist[:, rezpos] < interval[0],
            #            processedclist[:, rezpos] > interval[1])
            #if 'sinsnr' in rejects.keys():
            #    for interval in rejects['sinsnr']:
            #        cmask *= np.logical_or(
            #            processedclist[:, imzpos] < interval[0],
            #            processedclist[:, imzpos] > interval[1])
            
            # coefficients
            if 'calpha' in rejects.keys():
                ind = rejects['alpha'] + c0_pos
                for interval in rejects['calpha']:
                    cmask *= np.logical_or(
                        processedclist[:, ind] < interval[0],
                        processedclist[:, ind] > interval[1])

        filteredclist = processedclist[cmask]

        return filteredclist

    @staticmethod
    def remove_loud_times(filteredclist, snr_thresh=10, time_tol=1.):
        # First find bad inds
        rejects = {'snr': [(0, snr_thresh)]}
        badclist = TriggerList.filter_processed_clist(
            filteredclist, rejects=rejects)
        if len(badclist) == 0:
            print("no bad indices, is this a joke?")
            return filteredclist
        else:
            badtimes = badclist[:, 0]
            rejects = {'time': [(t - time_tol / 2, t + time_tol / 2)
                                for t in badtimes]}
            loud_removed = TriggerList.filter_processed_clist(
                filteredclist, rejects=rejects)
            return loud_removed

    # Plotting functions
    # -------------------------------------------------------------------------
    def specgram(
            self, nfft=None, tmin=None, tmax=None, t0=0, ax=None, vmin=None,
            vmax=None, **kwargs):
        """
        Makes spectrogram of whitened strain data
        :param nfft: Number of data points to use per FFT, defaults to number
                     per second
        :param tmin: Minimum time for spectrogram, relative to t0 (s)
        :param tmax: Minimum time for spectrogram, relative to t0 (s)
        :param t0: Origin to measure time relative to, when specifying tmin-tmax
        :param vmin: Minimum value for colorbar
        :param vmax: Maximum value for colorbar
        :param ax: Axis to place plot in
        :return: Axis with spectrogram
        """
        # Import modules and define axis if needed
        import matplotlib.ticker as ticker
        rflag = False
        if ax is None:
            rflag = True
            _, ax = utils.import_matplotlib()

        fs = int(np.round(1/self.dt))
        if nfft is None:
            nfft = fs
        minind = 0
        maxind = len(self.strain)

        if tmin is not None:
            minind = np.searchsorted(self.time, tmin + t0)
        if tmax is not None:
            maxind = np.searchsorted(self.time, tmax + t0)

        if vmin is None:
            vmin = 0

        if vmax is None:
            vmax = 25 / fs

        ax.specgram(
            self.strain[minind:maxind], NFFT=nfft, Fs=fs, vmin=vmin, vmax=vmax,
            scale='linear', **kwargs)

        if (t0 is None) or (t0 == 0):
            # Show offset w.r.t self.t0 if no reference time was passed
            t0 = self.t0

        deltat = t0 - self.time[minind]

        def format_data(t, pos):
            return '{:.5g}'.format(t - deltat)

        formatter = ticker.FuncFormatter(format_data)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(f"Time (s) - {t0}", fontsize=14)
        ax.set_ylabel("Frequency (Hz)", fontsize=14)
        ax.set_title(
            os.path.splitext(os.path.basename(self.fname))[0],
            fontsize=14, fontweight="bold")

        if rflag:
            return ax

    def plot_teststat_hist(
            self, ax=None, undo_psd_drift=False, plot_chi2=True, **hist_kwargs):
        """
        Plots normalized overall histogram of test statistic
        :param ax: Axis to plot into, if available
        :param undo_psd_drift: Flag indicating whether to undo psd drift
        :param plot_chi2: Flag to plot chi2 line
        :return: Axis with histogram, if none passed in
        """
        rflag = False
        if ax is None:
            rflag = True
            _, ax = utils.import_matplotlib()
        # Test statistic is SNR^2
        if not undo_psd_drift:
            hists = ax.hist(
                self.filteredclist[:, 1], 100, histtype='step', **hist_kwargs)
        else:
            hists = ax.hist(
                self.filteredclist[:, 1] *
                self.filteredclist[:, self.psd_drift_pos]**2,
                100, histtype='step', **hist_kwargs)
        ax.set_yscale('log')
        if plot_chi2:
            # Plot chi-squared with 2 dof
            dbin = np.diff(hists[1])
            bcent = hists[1][:-1] + 0.5 * dbin
            ax.plot(bcent,
                    0.5 * np.exp(-0.5 * (bcent - self.threshold_chi2)) *
                    dbin * len(self.filteredclist))
        if rflag:
            return ax

    def plot_time_hist(self, ax=None, bins=1000, **hist_kwargs):
        """
        Plots histogram of trigger time offsets from start of file (self.t0)
        :param ax: Axis to plot into, if available
        :param bins: Number of bins for histogram
        :return: Axis with histogram, if none passed in
        """
        rflag = False
        if ax is None:
            rflag = True
            _, ax = utils.import_matplotlib()
        ax.hist(self.filteredclist[:, 0] - self.t0, bins=bins, **hist_kwargs)
        if rflag:
            return ax

    @staticmethod
    def plot_triggers_corner(
            trig=None, filteredclist=None, indlist=(0, 1, 7, 8), t0=None,
            **kwargs):
        """
        Make corner plot of parameters of triggers
        :param trig: Trigger object whose filteredclist to use
        :param filteredclist: filteredclist to use
        :param indlist: Tuple of indices into self.filteredclist to plot
        :param t0: Origin for time, if known (defaults to t0 of file)
        :return: Figure with plot
        """
        import corner

        if trig is None:
            if t0 is None:
                t0 = 0
            if filteredclist is None:
                raise RuntimeError("I need something to plot!")
        else:
            filteredclist = trig.filteredclist.copy()
            if t0 is None:
                t0 = trig.t0

        filteredclist[:, 0] -= t0
        fig = corner.corner(
            filteredclist[:, indlist], plot_contours=False,
            plot_datapoints=True, plot_density=False, **kwargs)

        return fig

    def plot_running_hist(self, deltat=10., n_bins=50):
        """
        TODO: Deprecation Warning
        Plots running histogram of test statistic in time-chunks
        :param deltat: Make running histogram every deltat seconds
        :param n_bins: Number of bins of test statistic
        :return: Axis with histogram
        """
        # Make a histogram of triggers within every cfac indices
        fig, ax = utils.import_matplotlib()
        # Sort according to times
        inds = np.argsort(self.filteredclist[:, 0])
        clist_sorted = self.filteredclist[inds]
        # Define common histogram bins
        bins = np.linspace(np.min(clist_sorted[:, 1]),
                           np.max(clist_sorted[:, 1]) * 1.1, n_bins)
        bincent = (bins[:-1] + bins[1:]) / 2.
        # Define time bins
        time_low = np.arange(clist_sorted[0, 0], clist_sorted[-1, 0], deltat)
        chop_index = np.r_[np.searchsorted(clist_sorted[:, 0], time_low),
                           len(clist_sorted)]
        # Array of histograms
        vals = np.array([np.histogram(
                clist_sorted[chop_index[i]:chop_index[i+1], 1], bins=bins)[0]
                         for i in range(len(chop_index)-1)])
        im = ax.pcolormesh(bincent, time_low + 0.5 * deltat, vals, cmap='Paired')
        fig.colorbar(im, ax=ax)
        return fig

    def plot_bestfit_waveform(
            self, triggers, wfs_wt_cos_td=None, plot_injection=False, 
            labels=None, ax=None, ref_time=None, individual_modes = False,
            physical_mode_ratio=False, data_plot_kwargs={}, wf_plot_kwargs={}):
        """
        Plot waveforms in trigger(s) against the data
        :param triggers:
            Processedclists of triggers to plot (can be row if n_trigs = 1)
        :param wfs_wt_cos_td:
            If known, array with cosine waveforms in rows
            (will infer from the bank otherwise)
        :param plot_injection: Flag whether to plot the injected waveform
        :param labels: If known, labels to add to the plot
        :param ax: If known, axis to plot into
        :return: Axis that was plotted into, if none was passed in
        """
        rflag = False
        if ax is None:
            rflag = True
            _, ax = utils.import_matplotlib()

        # Locate first trigger in data
        if hasattr(triggers[0], '__len__'):
            # Multiple triggers
            relative_index, _ = self.get_time_index(triggers[0])
            if ref_time is None:
                ref_time = triggers[0][0]
        else:
            relative_index, _ = self.get_time_index(triggers)
            if ref_time is None:
                ref_time = triggers[0]

        # Plot whitened strain
        bank = self.templatebank
        ax.plot(self.time[relative_index - bank.support_whitened_wf:
                          relative_index] - ref_time,
                self.strain[relative_index - bank.support_whitened_wf:
                            relative_index], lw=0.5, label='Strain', **data_plot_kwargs)

        # Plot bestfit waveform(s)
        if hasattr(triggers[0], '__len__'):
            for ind, trigger in enumerate(triggers):
                relative_index, _ = self.get_time_index(trigger)
                ref_time = trigger[0]
                if wfs_wt_cos_td is None:
                    bestfit_wf = self.get_bestfit_wf(trigger, 
                                        individual_modes=individual_modes,
                                        physical_mode_ratio=physical_mode_ratio)
                else:
                    bestfit_wf = self.get_bestfit_wf(
                        trigger, wf_wt_cos_td=wfs_wt_cos_td[ind], 
                                        individual_modes=individual_modes,
                                        physical_mode_ratio=physical_mode_ratio)
                if labels is None:
                    label = f"Waveform {ind}"
                else:
                    label = labels[ind]
                    
                if individual_modes:
                    ax.plot(self.time[
                     relative_index - bank.support_whitened_wf:relative_index]
                      - ref_time, bestfit_wf[0], label='22', **wf_plot_kwargs)
                    ax.plot(self.time[
                     relative_index - bank.support_whitened_wf:relative_index]
                      - ref_time, bestfit_wf[1], label='33', **wf_plot_kwargs)
                    ax.plot(self.time[
                     relative_index - bank.support_whitened_wf:relative_index]
                      - ref_time, bestfit_wf[2], label='44', **wf_plot_kwargs)
                else:
                    ax.plot(self.time[
                    relative_index - bank.support_whitened_wf:
                    relative_index] - ref_time, bestfit_wf, label=label, **wf_plot_kwargs)
        else:
            if wfs_wt_cos_td is None:
                bestfit_wf = self.get_bestfit_wf(triggers,
                                    individual_modes=individual_modes,
                                    physical_mode_ratio=physical_mode_ratio)
            else:
                bestfit_wf = self.get_bestfit_wf(
                    triggers, wf_wt_cos_td=wfs_wt_cos_td,
                                    individual_modes=individual_modes,
                                    physical_mode_ratio=physical_mode_ratio)
            if labels is None:
                labels = "Bestfit waveform"
                
            if individual_modes:
                ax.plot(self.time[
                relative_index - bank.support_whitened_wf:relative_index] - ref_time,
                bestfit_wf[0], label='22', **wf_plot_kwargs)
                ax.plot(self.time[
                relative_index - bank.support_whitened_wf:relative_index] - ref_time,
                bestfit_wf[1], label='33', **wf_plot_kwargs)
                ax.plot(self.time[
                relative_index - bank.support_whitened_wf:relative_index] - ref_time,
                bestfit_wf[2], label='44', **wf_plot_kwargs)
            else:
                ax.plot(self.time[
                relative_index - bank.support_whitened_wf:relative_index] - ref_time,
                bestfit_wf, label=labels, **wf_plot_kwargs)

        # Plot injected waveform, if applicable
        if plot_injection:
            ax.plot(
                self.time[relative_index - bank.support_whitened_wf:
                          relative_index] - ref_time,
                self.injected_wf_whitened[
                    relative_index - bank.support_whitened_wf:relative_index],
                c='k', ls='--', label='Injected waveform', **wf_plot_kwargs)
            
        ax.legend(fancybox=True, framealpha=0.5)
        if rflag:
            return ax


pass
