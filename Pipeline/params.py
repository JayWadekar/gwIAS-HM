# Parameters used in data analysis

# Template bank parameters
# -----------------------------------------------------------------------------
# Frequency range (Hz)
# Sets minimum frequency of waveform generation
FMIN = 20.

# HM_search = True

# if HM_search: FMAX = 512.
# else: FMAX = 1024.
FMAX = 1024.

NF_GEN = 10**7
SUB_FAC = 1000  # Factor to subsample after unwrapping

# Tolerance for retaining basis
TOL = 1.0E-8
# Factor to protect conversion from basis to phase is finite if weights vanish
EPS_CALPHA = 1e-50

# Fix support captured, (1 - Wt of whitened waveform within finite support)/2
# When testing the overlaps, the accuracy cannot surpass this.
WFAC_WF = 1.E-2
# Relative safety factor for time-domain shift to avoid wraparound artifacts due
# to different waveforms in bank
REL_SHIFT_SAFETY = 0.2
# Factor to discriminate between inspiral-dominated (BNS) and merger-dominated
# (BH) waveforms
MERGERTYPE_THRESH = 100
# Safety factor to inflate estimated time-domain support for BH-type and
# BNS-type waveforms to cover orbital hangup due to spin orientations
SUPPORT_SAFETY_BH = 2 + REL_SHIFT_SAFETY
SUPPORT_SAFETY_BNS = 1.2 + REL_SHIFT_SAFETY
# Minimum waveform duration
MIN_WFDURATION = 16.

# Safety factor to inflate calpha ranges by
TEMPLATE_SAFETY = 1.05
# -----------------------------------------------------------------------------

# Data processing parameters
# -----------------------------------------------------------------------------
# PSD estimation
# --------------
# Default length of chunk for PSD estimation (in seconds)
# DEF_CHUNKTIME_PSD = 64.
# Change in O3a
DEF_CHUNKTIME_PSD = 32.

# Minimum number of valid samples to average over to measure PSD
MINSAMP_PSD = 1

# Minimum and maximum frequencies for PSD computation and data-analysis
# ---------------------------------------------------------------------
# Measure the PSD only above this frequency, this also sets the minimum
# frequencies that will be in the data after highpassing
# Start a bit redward of analysis cutoff to avoid fake line-identification
DF_DATA = -5.
# DF_DATA = 0.
FMIN_PSD = FMIN + DF_DATA

# Measure the PSD only upto this frequency, this should above the Nyquist
# frequency corresponding to fmax_overlap below
# if HM_search: FMAX_PSD = 1024.
# else: FMAX_PSD = 2048.
FMAX_PSD = 2048.

# Measure power only above this frequency in excess power tests, this equals
# fmin since the waveforms only have frequencies >= fmin
FMIN_ANALYSIS = FMIN

# Maximum frequency for matched filtering (Hz)
# if HM_search: FMAX_OVERLAP = 512
# else: FMAX_OVERLAP = 1024
FMAX_OVERLAP = 1024

# Minimum file length needed in units of chunktime_psd, for this choice the
# accuracy of ASD ~ 1/\sqrt{32} ~ 16 % => The loss of sensitivity due to this
# is about 3%
MIN_FILELENGTH_FAC = 16
# NOTE: Check that this is always larger than
# support_wf + butterworth irl + support_wt - 2 in seconds

# Parameter for overlap save
# Used to save time in data processing, not necessary to edit
# DEF_FFTSIZE = 2 ** 20
# Change in O3a
# Intel python's linear scaling ends at lower intervals
# IMPORTANT: This can be shorter than BNS waveforms, in which case it might need
# to x2 the fftsize, the code should do it automatically, but not debugged
DEF_FFTSIZE = 2 ** 18

# Parameters for line detection
# -----------------------------
# Detect lines as deviations from smoothed ASD at this significance
LINE_SIGMA = 4

# Dectect loud lines as deviations from smoothed ASD at this significance
# LOUD_LINE_SIGMA = 1000
# Change in O3a
LOUD_LINE_SIGMA = 500

# Glitch rejection
# ----------------
# Number of seconds to destroy extra whenever LIGO has a super big hole that
# trims a file. Holes cannot be dealt with on the edges currently
IMPROPER_FLAGGING_SAFETY_DURATION = 2
# Number of passes for sigma clipping
N_GLITCHREMOVAL = 7
# If the outlier is more than 1/OUTLIER_FRAC x sigma_clipping_threshold,
# don't clip in an overzealous manner
OUTLIER_FRAC = 0.1
# Minimum width to clip around outlier (s) (can be larger due to Butterworth)
MIN_CLIPWIDTH = 0.1
# Default filelength (s) used to map probabilities to glitch thresholds
DEF_FILELENGTH = 4096
# Number of files in run (used to set global FAR)
NFILES = 2000
# Glitch detectors should fire this many times per perfect file
# Increase to catch glitches more aggressively
NPERFILE = 0.2
# Fraction of interval successive specgrams are allowed to overlap by
# Increase to catch glitches more aggressively
# OVERLAP_FAC = 1 / 2
# OVERLAP_FAC = 7 / 8
OVERLAP_FAC = 3 / 4
# Number of independent excess power measurements in moving average to remove
# long modes
N_INDEP_MOVING_AVG_EXCESS_POWER = 5
# Safety factor to restrict aggressive trimming of the file due to PSD drift
# Related to typical level of PSD drift we see in files (not Gaussian)
# Decrease to catch glitches more aggressively
# PSD_DRIFT_SAFETY = 0
PSD_DRIFT_SAFETY = 5e-2

# Hole filling
# ------------
# Start by filling all holes using brute force logic at the outset, if number
# of bad entries is below this
NHOLE_BF = 2000
# Can't fill with bruteface if more than this many samples are bad in a single
# filling chunk
# NHOLE_MAX = 10000
# Change in O3a
NHOLE_MAX = 20000
# Can't fill if more than this many samples are bad in a consecutive chunk
NHOLE_MAX_CONSECUTIVE = 2**19
# NOTE: Check that this is always larger than support_wf
# -----------------------------------------------------------------------------

# Filter parameters
# -----------------------------------------------------------------------------
# Parameters of high-pass Butterworth filter
ORDER = 4
DF = 0.  # The gain falls to 1/sqrt(2) that of the passband at FMIN + DF
# Fraction of max impulse response to capture (make very small, because bad
# stuff is happening at low frequencies)
IRL_EPS_HIGH = 1e-9
# Same for bandpass filter to measure band power of waveforms
IRL_EPS_BAND = 1e-5

# Truncate notch filter at this multiple of 1/bandwidth
NOTCH_TRUNC_FAC = 4

WFAC_FILT = 1.E-3   # (1 - Weight of the filter to capture)/2
# Support of sinc interpolation in units of indices. Empirically tested at
# f_max = 512 Hz
SUPPORT_SINC_FILTER = 200
# Amount of data we lose from each side in the limit of an infinite number of
# sinc interpolations
SUPPORT_EDGE_DATA = 2 * (SUPPORT_SINC_FILTER + 1)
# -----------------------------------------------------------------------------

# Trigger analysis parameters
# -----------------------------------------------------------------------------
# Maximum SNR below which waveforms are never clipped by glitch rejection
DEF_PRESERVE_MAX_SNR = 20
# DEF_PRESERVE_MAX_SNR = 12
FALSE_NEGATIVE_PROB_POWER_TESTS = 1e-4

# Factor to multiply PRESERVE_MAX_SNR for vetoing AFTER template subtraction
DEF_TEMPLATE_MAX_MISMATCH = 0.1

## WARNING: If you are analyzing higher modes and change DEF_SINE_GAUSSIAN_INTERVALS        
#           or DEF_BANDLIM_TRANSIENT_INTERVALS or DEF_EXCESS_POWER_INTERVALS,
#           you should compute which higher mode wfs give you the max
#           glitch thresholds and store the info. in bank metadata
#           See the 'Computing the thresholds for different glitch tests with HMs'
#           subsection in scratch_files/TemplateBank_HigherModes.ipynb

# Frequency bands within which we look for Sine-Gaussian noise transients
# [central frequency, df = (upper - lower frequency)] Hz
DEF_SINE_GAUSSIAN_INTERVALS = [[60., 10.],
                               [40, 40],
                               [120., 40],
                               [140, 80],
                               [100., 100],
                               [90, 40],
                               [70, 40],
                               [150, 50],
                               [100, 50]]

# Time-interval and frequency bands within which to look for excess power
# transients [0.5, [55, 65]]
DEF_BANDLIM_TRANSIENT_INTERVALS = [[1., [55, 65]],
                                   [1., [70, 80]],
                                   [1, [40, 60]],
                                   [1, [25, 50]],
                                   [0.5, [40, 60]],
                                   [0.25, [140, 160]],
                                   [1., [20, 50]],
                                   [1., [100, 180]],
                                   [0.05, [25, 70]],
                                   [0.1, [25, 70]],
                                   [0.05, [20, 180]],
                                   [0.025, [60, 180]],
                                   [0.2, [25, 70]]]

DEF_BANDLIM_TRANSIENT_INTERVALS_O3 = [[1., [55, 65]],
                                   [1., [70, 80]],
                                   [1, [40, 60]],
                                   [1, [25, 50]],
                                   [0.5, [40, 60]],
                                   [0.25, [140, 160]],
                                   [1., [100, 180]],
                                   [0.025, [60, 180]],
                                   [0.2, [25, 70]]]

DEF_SINE_GAUSSIAN_INTERVALS_O3 = [[60., 10.],
                               [120., 40],
                               [140, 80],
                               [100., 100],
                               [90, 40],
                               [70, 40],
                               [150, 50],
                               [100, 50]]

# Scales over which to look for excess power
# DEF_EXCESS_POWER_INTERVALS = [0.2, 1, 4, 10]
# Ensure that longest scale x N_MOVING_AVG_EXCESS_POWER * (1 - OVERLAP_FAC) is
# less than timescale where there is a cliff in the autocorrelation of |d(t)|^2
DEF_EXCESS_POWER_INTERVALS = [0.2, 1]

# Don't trust any triggers that were corrected below this level
HOLE_CORRECTION_MIN = 0.5
# Ensure that division returns a real number
HOLE_EPS = 1e-5

# Interval to update PSD drift correction (s)
DEF_PSD_DRIFT_INTERVAL = 1

# Do not capture psd drifts below this level
PSD_DRIFT_TOL = 2.E-2

# Threshold to sigma clip the overlaps when estimating the PSD drift correction
# (in units of number of times realized over the window due to Gaussian noise)
PSD_DRIFT_SAFEMEAN_THRESH = 1

# When we clip outliers, margin to increase clipping by (in s) to prevent
# events from biasing the PSD drift correction
PSD_DRIFT_SAFETY_LEN = 0.01

# New parameter from O3a
# Target time resolution after sinc-interpolation
DT_FINAL = 1/4096.

# Support of sinc filter to use when optimizing calpha
# (higher is more accurate, but slower)
SUPPORT_SINC_FILTER_OPT = 1024

# For HM, we create a separate file with downsampled triggers,
DOWNSAMPLE_TRIGGERS = 1000
# -----------------------------------------------------------------------------

# Parameters for tracking lines in specgram
# -----------------------------------------------------------------------------
LINE_TRACKING_DT = 2
LINE_TRACKING_TIME_SCALE = 64
LINE_TRACKING_DF = 2

# -----------------------------------------------------------------------------

# Coincidence parameters
# -----------------------------------------------------------------------------
MAX_FRIEND_DEGRADE_SNR2 = 0.1

# Half-Size of timeseries to save (s), should be large enough for any time
# differences
DT_BOUND_TIMESERIES = 0.1

# -----------------------------------------------------------------------------

# Veto parameters
# -----------------------------------------------------------------------------
# Waveform duration below which we avoid holes
SHORT_WF_LIMIT = 10

# Window (s) around short waveforms where we demand no bad time
# Warning: this can mess up events like GW170817!
DT_CLEAN_MASK = 1

# Excess power veto should fire this many times per perfect file
# Increase to make veto more aggressive
NFIRE = 0.1

# If change in finer PSD drift exceeds Gaussian sigma x this factor, we deem it
# significant when vetoing
PSD_DRIFT_VETO_THRESH = 6

# Number of chunks to split waveforms into for chi2 veto
N_CHUNK = 6

# Subsets of chunks to compare for tail/hump tests
SPLIT_CHUNKS = [[[0], [4, 5]],
                [[0, 1], [4, 5]],
                [[0, 5], [2, 3]],
                [[0, 1, 2], [3, 4, 5]]]

# Calculate chi^2 with 20 bins [Catches "dots" in time-freq]
N_CHUNK_2 = 20

# Threshold relative to highest eigenvalue to keep in the covariance matrix,
# i.e., retain eigenvectors with eigenvalue > (max eigenvalue of covariance
# matrix * cov_degrade)
COV_DEGRADE = 0.1

# If the highest eigenvalue of the covariance matrix is below this, do not
# perform chi-squared test
CHI2_MIN_EIGVAL = 1e-2

# Threshold for chi2 test
THRESHOLD_CHI2 = 1e-2

# Threshold for split test
THRESHOLD_SPLIT = 1e-2

# Window (s) around trigger to avoid while estimating statistics of the scores
# from the data
DT_AVOID = 0.1

# Amount to allow time to shift during calpha optimization
DT_OPT = 0.01

# -----------------------------------------------------------------------------

# Coherent score parameters
# -----------------------------------------------------------------------------
LOG2N_QMC, NPHI, MAX_LOG2N_QMC, MIN_N_EFFECTIVE = 12, 256, 16, 50

# Spacing of samples in ms
DEFAULT_DT = 1000/4096
# Least count of timeslides in ms
DEFAULT_TIMESLIDE_JUMP = 100

# safety factor for rank function for extremely loud triggers
SNR2_MAX_BOUND = 300

# -----------------------------------------------------------------------------
