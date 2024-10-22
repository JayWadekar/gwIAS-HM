import numpy as np
import os
from numba import float64, complex128
from numba import njit, vectorize
import utils
try:
    import triggers_single_detector as trig
except: pass
from numpy.random import choice

# Name of the detector -> order of get_detector_fnames
DET2GET_DETECTOR_FNAMES = \
    {'H1': 0, 'H2': 0, 'LHO': 0,
     'L1': 1, 'LLO': 1,
     'G1': None, 'GEO': None, 'GEO600': None,
     'V1': 2, 'VIRGO': 2,
     'T1': None, 'TAMA': None, 'TAMA300': None,
     'K1': None, 'KAGRA': None, 'LCGT': None,
     'I1': None, 'LIO': None,
     'E1': None, 'E2': None, 'E3': None}

try:
    import lal
    # create Detector dictionary
    DETMAP = {'H1': lal.LALDetectorIndexLHODIFF,
              'H2': lal.LALDetectorIndexLHODIFF,
              'LHO': lal.LALDetectorIndexLHODIFF,
              'L1': lal.LALDetectorIndexLLODIFF,
              'LLO': lal.LALDetectorIndexLLODIFF,
              'G1': lal.LALDetectorIndexGEO600DIFF,
              'GEO': lal.LALDetectorIndexGEO600DIFF,
              'GEO600': lal.LALDetectorIndexGEO600DIFF,
              'V1': lal.LALDetectorIndexVIRGODIFF,
              'VIRGO': lal.LALDetectorIndexVIRGODIFF,
              'T1': lal.LALDetectorIndexTAMA300DIFF,
              'TAMA': lal.LALDetectorIndexTAMA300DIFF,
              'TAMA300': lal.LALDetectorIndexTAMA300DIFF,
              'K1': lal.LALDetectorIndexKAGRADIFF,
              'KAGRA': lal.LALDetectorIndexKAGRADIFF,
              'LCGT': lal.LALDetectorIndexKAGRADIFF,
              'I1': lal.LALDetectorIndexLIODIFF,
              'LIO': lal.LALDetectorIndexLIODIFF,
              'E1': lal.LALDetectorIndexE1DIFF,
              'E2': lal.LALDetectorIndexE2DIFF,
              'E3': lal.LALDetectorIndexE3DIFF}
except ModuleNotFoundError:
    print('lal not found, generating samples will be unavailable.')


# Useful variables
# Loose upper bound to travel time between any detectors in ms
DEFAULT_DT_MAX = 1000 / 16
# Spacing of samples in ms
# 0.25 ms, corresponding to
# O1: 2 sinc interpolations from 1 ms
# O2: 1 sinc interpolation from 0.5 ms
DEFAULT_DT = 1000/4096
# Least count of timeslides in ms
DEFAULT_TIMESLIDE_JUMP = 100

# Path to file with precomputed samples
DEFAULT_SAMPLES_FNAME = {("H1", "L1"): os.path.join(
    utils.DATA_ROOT, 'RA_dec_grid_H1_L1_4096_1136574828.npz'),
    ("H1", "V1"): os.path.join(
    utils.DATA_ROOT, 'RA_dec_grid_H1_V1_4096_1136574828.npz'),
    ("L1", "V1"): os.path.join(
    utils.DATA_ROOT, 'RA_dec_grid_L1_V1_4096_1136574828.npz'),
    ("H1", "L1", "V1"): os.path.join(
    utils.DATA_ROOT, 'RA_dec_grid_H1_L1_V1_4096_1136574828.npz')}

# hardwired f1=1 and f2=1
# can generate other values with the G function
LG_FAST_XP = np.asarray(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2,
     1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5,
     2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
     3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 5.1,
     5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6., 6.1, 6.2, 6.3, 6.4,
     6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
     7.8, 7.9, 8., 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.,
     9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9])

LG_FAST_YP = np.asarray(
    [-0.1135959, -0.12088098, -0.1427278, -0.1791095,
     -0.22997564, -0.2952433, -0.37478329, -0.46840065,
     -0.57580762, -0.69658744, -0.83014698, -0.97565681,
     -1.13197891, -1.29758569, -1.4704808, -1.64814261,
     -1.82752466, -2.00515831, -2.1774003, -2.34083626,
     -2.49278514, -2.63177248, -2.75780503, -2.87233483,
     -2.97792671, -3.07776058, -3.17513488, -3.27308869,
     -3.37418153, -3.48041019, -3.59321753, -3.71354945,
     -3.84192885, -3.97852807, -4.12323147, -4.27568566,
     -4.43533834, -4.60146807, -4.77320804, -4.94956693,
     -5.12944979, -5.31168161, -5.49503558, -5.67826708,
     -5.86015327, -6.0395366, -6.2153689, -6.38675167,
     -6.55296781, -6.71350061, -6.86803769, -7.01645979,
     -7.15881685, -7.29529541, -7.42618219, -7.55182832,
     -7.67261759, -7.78894088, -7.9011773, -8.0096818,
     -8.11477823, -8.21675658, -8.3158732, -8.41235286,
     -8.5063919, -8.59816165, -8.68781196, -8.77547441,
     -8.86126519, -8.94528754, -9.0276338, -9.10838716,
     -9.18762299, -9.26541003, -9.34181128, -9.41688478,
     -9.4906842, -9.5632594, -9.63465681, -9.70491982,
     -9.77408908, -9.84220278, -9.90929682, -9.9754051,
     -10.04055961, -10.10479061, -10.1681268, -10.23059541,
     -10.29222228, -10.35303204, -10.41304811, -10.47229284,
     -10.53078756, -10.58855264, -10.64560757, -10.70197099,
     -10.75766078, -10.81269406, -10.86708726, -10.92085616])


# ############## Functions to create library of saved samples ##################
def delays(ra, dec, detectors, gps_time):
    """
    Computes delays in arrival times between detectors, referred to the first
    (beyond 2 delays for most, and 3 delays for all, they're overspecified,
    but that's OK)
    Always a 1D array
    """
    times = np.asarray([
        lal.TimeDelayFromEarthCenter(det.location, ra, dec, gps_time)
        for det in detectors])
    dt = times[1:] - times[0]
    return dt


def phase_lags(ra, dec, detectors, gps_time):
    """
    Computes phase lags between detectors, referred to the first
    (beyond 2 delays for most, and 3 delays for all, they're overspecified,
    but that's OK)
    """
    # Greenwich mean sidereal time corresponding to the given GPS time
    gmst = lal.GreenwichMeanSiderealTime(gps_time)

    # Detector responses for phi = zero
    fs = [lal.ComputeDetAMResponse(det.response, ra, dec, 0, gmst)
          for det in detectors]

    # Phases
    # TODO: Check with Matias about the order
    phis = np.asarray([np.arctan2(f[1], f[0]) for f in fs])
    # Phase differences w.r.t the first detector (H1 for H1, L1)
    dphis = (phis[1:] - phis[0]) % (2 * np.pi)
    return dphis


@njit
def dt2key(dt, dt_sinc=DEFAULT_DT, dt_max=DEFAULT_DT_MAX):
    """
    Jitable key for the dictionary from the time delays
    This is clunky but faster than the nice expression by O(1) when vectorized
    :param dt:
        (ndelays = ndetectors-1) x nsamp array with time delays in milliseconds
        (should be 2d)
    :param dt_sinc: Time resolution in milliseconds
    :param dt_max: Maximum time delay in milliseconds
    :return: n_samp array of keys (always 1D)
    """
    dt = np.asarray(dt)
    nbase = int(np.floor(dt_max / dt_sinc)) * 2
    exparr = nbase ** np.arange(dt.shape[0])
    ind_arr = np.zeros(dt.shape[1], dtype=np.int32)
    for i0 in range(dt.shape[0]):
        for i1 in range(dt.shape[1]):
            ind_arr[i1] += \
                int(np.floor(dt[i0, i1] / dt_sinc + 0.5)) % nbase * exparr[i0]
    return ind_arr


def create_time_dict(
        nra, ndec, detectors, gps_time=1136574828.0, dt_sinc=DEFAULT_DT,
        dt_max=DEFAULT_DT_MAX):
    """
    Creates dictionary indexed by time, giving RA-Dec pairs for montecarlo
    integration
    :param nra: number of ra points in grid
    :param ndec: number of declinations in grid
    :param detectors:
        List of detectors, each with a location and response as given by LAL
    :param gps_time: Reference GPS time to generate the dictionary for
        (arbitrary for our use case)
    :param dt_sinc:
        size of the time binning used in ms; it must coincide with the time
        separation of samples in the overlap if this dictionary will be used
        for doing the marginalization
    :param dt_max: Rough upper bound on the individual delays
    :return:
        0. Dictionary indexed by the dts key, returning n_sky x 2 array with
           indices into ras and decs for each allowed dt tuple
        1. List of ras
        2. List of decs
        3. n_ra x n_dec x n_detector x 2 array with responses
        4. n_ra x n_dec x (n_detector - 1) array with delta ts
        5. n_ra x n_dec x (n_detector - 1) array with delta phis
        6. n_ra x n_dec array with network sensitivity (sum_{det, pol} resp^2)
    """
    ra_grid = np.linspace(0, 2.0 * np.pi, nra, endpoint=False)
    # Declination minus the poles
    sin_dec_grid = np.linspace(-1.0, 1.0, ndec + 1, endpoint=False)
    sin_dec_grid = sin_dec_grid[1:]
    dec_grid = np.arcsin(sin_dec_grid)

    # Greenwich mean sidereal time corresponding to the given GPS time
    gmst = lal.GreenwichMeanSiderealTime(gps_time)

    # Compute grids of response and phase difference for debugging
    # deltats contains time difference in milliseconds
    # We compute the other variables only for checking purposes
    deltats = []
    dphases = []
    responses = []
    rtot2s = []
    for ra in ra_grid:
        # Delays, phase differences, vector responses, scalar responses
        arrs_ra = [[], [], [], []]
        for dec in dec_grid:
            # Time delays in milliseconds
            deltat = 1000 * delays(ra, dec, detectors, gps_time)
            # Detector responses for phi = zero
            fs = [lal.ComputeDetAMResponse(det.response, ra, dec, 0, gmst)
                  for det in detectors]
            # Phases
            # TODO: Check with Matias about the order
            phis = np.asarray([np.arctan2(f[1], f[0]) for f in fs])
            # Phase differences w.r.t the first detector (e.g., H1 for H1, L1)
            dphis = (phis[1:] - phis[0]) % (2 * np.pi)
            # Network responses
            xrs = [np.linalg.norm(f) for f in fs]
            xrtot2 = sum(xr ** 2 for xr in xrs)
            arrs_ra[0].append(deltat)
            arrs_ra[1].append(dphis)
            arrs_ra[2].append(fs)
            arrs_ra[3].append(xrtot2)

        deltats.append(arrs_ra[0])
        dphases.append(arrs_ra[1])
        responses.append(arrs_ra[2])
        rtot2s.append(arrs_ra[3])

    # Define return values for debugging
    # n_ra x n_dec x (n_det - 1)
    deltats = np.asarray(deltats)
    dphases = np.asarray(dphases)
    # n_ra x n_dec x n_det x 2
    responses = np.asarray(responses)
    # n_ra x n_dec
    rtot2s = np.asarray(rtot2s)

    # Make `contour maps' of delta t, with entries = index of ra, index of dec
    dt_dict = {}
    for i in range(len(ra_grid)):
        for j in range(len(dec_grid)):
            key = dt2key(deltats[i, j][:, None], dt_sinc=dt_sinc, dt_max=dt_max)[0]
            if key in dt_dict:
                dt_dict[key].append((i, j))
            else:
                dt_dict[key] = [(i, j)]

    # Make the elements numpy arrays to pick and save easily
    for key in dt_dict.keys():
        dt_dict[key] = np.asarray(dt_dict[key], dtype=np.int32)

    return dt_dict, ra_grid, dec_grid, responses, deltats, dphases, rtot2s


def create_samples(
        fname, nra=100, ndec=100, detnames=('H1', 'L1'), gps_time=1136574828.0,
        dt_sinc=DEFAULT_DT, dt_max=DEFAULT_DT_MAX, nsamples=100000):
    """
    Create samples and save to file
    :param fname: File name of archive to create
    :param nra: Number of right ascensions
    :param ndec: Number of declinations
    :param detnames: Names of detectors for the structure
    :param gps_time:
        Fiducial GPS time for mapping between the RA-DEC and time delays
        (arbitrary for typical usage)
    :param dt_sinc: Time resolution for marginalization
    :param dt_max: Rough upper bound on the individual delays
    :param nsamples:
        Number of random samples of the inclination and the polarization
    :return:
    """
    # Create detectors
    detectors = [lal.CachedDetectors[DETMAP[det]] for det in detnames]
    # Create structures to deal with the mapping of the sphere to delays
    dt_dict, ra_grid, dec_grid, responses, deltats, dphases, rtot2s = \
        create_time_dict(
            nra, ndec, detectors, gps_time=gps_time, dt_sinc=dt_sinc,
            dt_max=dt_max)
    # Create random samples of the cosine of the inclination, and the
    # polarization
    psis = np.random.uniform(0, 2 * np.pi, size=nsamples)
    mus = np.random.uniform(-1, 1, size=nsamples)  # cos(inclination)

    filename = utils.rm_suffix(fname, suffix='.npz', new_suffix="_") + \
        "_".join(detnames) + f"_{int(1000/dt_sinc)}" + f"_{int(gps_time)}.npz"
    np.savez(filename,
             dt_dict=dt_dict, ra_grid=ra_grid, dec_grid=dec_grid,
             responses=responses, deltats=deltats, dphases=dphases,
             rtot2s=rtot2s, psis=psis, mus=mus, gps_time=gps_time,
             dt_sinc=dt_sinc, dt_max=dt_max)
    return


# ############################ Compiled functions ###########################
@vectorize([complex128(float64, float64, float64, float64)], nopython=True)
def gen_sample_amps_from_fplus_fcross(fplus, fcross, mu, psi):
    """
    :param fplus: Response to the plus polarization for psi = 0
    :param fcross: Response to the cross polarization for psi = 0
    :param mu: Inclination
    :param psi: Polarization angle
    :returns A_p + 1j * A_c
    ## Note that this seems to have the wrong convention for mu
    """
    twopsi = 2. * psi
    c2psi = np.cos(twopsi)
    s2psi = np.sin(twopsi)
    fp = c2psi * fplus + s2psi * fcross
    fc = -s2psi * fplus + c2psi * fcross
    # Strain amplitude at the detector
    ap = fp * (1. + mu ** 2) / 2.
    ac = -fc * mu
    return ap + 1j * ac


@njit
def lgg(zthatthatz, gtype):
    """
    param: zthatthatz
    """
    ## Formula obtained by bringing the prior to the exponent, expanding it to
    # quadratic order and integrating, used for the marginal likelihood
    ## The power law terms come from the prior and are not very important in
    # practice. zthatthatz has size of SNR^2
    if gtype == 0:
        ## Turn off distance prior
        logg = np.zeros_like(zthatthatz)
    else:
        logg = lg_fast(zthatthatz ** 0.5)
    return logg


@vectorize(nopython=True)
def lg_approx(x, f1, f2):
    anorm = 9. / 2. / (1 + f1 + f2)
    term1 = 1 / x ** 5 * np.exp(12 / x ** 2 + 136 / x ** 4)
    return np.log(anorm * term1)


@njit
def lg_fast(x):
    """Fast evaluation of the marginalized piece via an interpolation table for
    a vector x, much faster via vectors than many scalars due to vectorized
    interp"""
    x = np.atleast_1d(np.asarray(x))
    out = np.zeros_like(x, dtype=np.float64)
    imask = x < 10
    if np.any(imask):
        out[imask] = np.interp(x[imask], LG_FAST_XP, LG_FAST_YP)
    imask = np.logical_not(imask)
    if np.any(imask):
        out[imask] = lg_approx(x[imask], 1, 1)
    return out


@njit
def marg_lk(zz, tt, gtype=1, nsamp=None):
    """
    Computes likelihood marginalized over distance and orbital phase
    :param zz:
        nsamp x n_detector array with rows having complex overlaps for
        each detector
    :param tt: nsamp x n_detector array with predicted overlaps in each
        detector for fiducial orbital phase and distance
        (upto an arbitrary scaling factor)
    :param gtype: passed to function to compute marginalization over distance
        and phase, determines which one to run
    :param nsamp: Pass to only sum over part of the arrays
    :returns:
        nsamp array of likelihoods marginalized over the orbital phase, and
        distance if needed (always 1D)
    """
    if nsamp is None:
        nsamp = len(zz)

    # Sums over detectors, do it this way to save on some allocations
    z2 = np.zeros(nsamp)
    t2_pow = np.zeros(nsamp)
    zthatthatz = np.zeros(nsamp)
    for i in range(nsamp):
        ztbar = 0. + 0.j
        t2 = 0
        for j in range(zz.shape[1]):
            z2[i] += zz[i, j].real**2 + zz[i, j].imag**2
            t2 += tt[i, j].real**2 + tt[i, j].imag**2
            ztbar += zz[i, j] * np.conj(tt[i, j])
        zttz = ztbar.real**2 + ztbar.imag**2
        zthatthatz[i] = zttz / t2
        t2_pow[i] = t2 ** 1.5

    logg = lgg(zthatthatz, gtype)

    lk = np.zeros(nsamp)
    for i in range(nsamp):
        dis_phase_marg = t2_pow[i] * np.exp(logg[i])
        lk[i] = np.exp(-0.5 * (z2[i] - zthatthatz[i])) * dis_phase_marg

    return lk


@njit
def rand_choice_nb(arr, cprob, nvals):
    """
    :param arr: A nD numpy array of values to sample from
    :param cprob:
        A 1D numpy array of cumulative probabilities for the given samples
    :param nvals: Number of samples desired
    :return: nvals random samples from the given array with a given probability
    """
    rsamps = np.random.random(size=nvals)
    return arr[np.searchsorted(cprob, rsamps, side="right")]


@njit
def coherent_score_montecarlo_sky(
        timeseries, offsets, nfacs, dt_dict_keys, dt_dict_items, responses,
        t3norm, musamps=None, psisamps=None, gtype=1, dt_sinc=DEFAULT_DT,
        dt_max=DEFAULT_DT_MAX, nsamples=10000):
    """
    # TODO: Avoid tuples, make varargs?
    Evaluates the coherent score integral by montecarlo sampling all
    relevant variables
    :param timeseries:
        Tuple with n_samp x 3 arrays with times, Re(z), Im(z) in each detector
    :param offsets:
        (n_det - 1) array with offsets to add to the detectors > first one
        (e.g., H1 for H1, L1) to bring them to zero lag
    :param nfacs:
        n_det array of instantaneous sensitivities in each detector
        (normfac/psd_drift x hole_correction)
    :param dt_dict_keys:
        Keys to dictionary computed using the delays in each dt_tuple, sorted
    :param dt_dict_items:
        Values in dictionary, tuple of n_sky x 2 arrays with indices into
        ras, decs for each allowed dt tuple
    :param responses:
        n_ra x n_dec x n_detector x 2 array with detector responses
    :param t3norm:
        normalization constant such that prior integrated over the sky = 1
    :param musamps: If available, array with samples of mu (cos inclination)
    :param psisamps: If available, array with samples of psi (pol angle)
    :param gtype: 0/1 to not marginalize/marginalize over distance
    :param dt_sinc: Time resolution for the dictionary (ms)
    :param dt_max: Rough upper bound on the individual delays (ms)
    :param nsamples: Number of samples for montecarlo evaluation
    :returns:
        Montecarlo evaluation of complete coherent score
        (including the incoherent part)
    """
    # Set the seed from the first H1 time (hardcoded)
    np.random.seed(int(timeseries[0][0, 0]))

    # Define some required variables
    ndet = len(timeseries)
    n_sky_pos = responses.shape[0] * responses.shape[1]
    if musamps is None:
        musamps = np.random.uniform(-1, 1, size=nsamples)
    if psisamps is None:
        psisamps = np.random.uniform(0, 2 * np.pi, size=nsamples)

    # Pick samples of data points in each detector
    # -----------------------------------------------------------------------
    # Normalization factor for monte-carlo over times in the detectors
    twt = 1.

    # Pick samples ahead of time, since np.interp is faster when vectorized
    # Samples picked according to e^{\rho^2}/2
    tz_samples = np.zeros((ndet, 3, nsamples))
    # Go over each detector
    for ind_d in range(ndet):
        pclist_d = timeseries[ind_d]

        # Define the cumsum of weights for the random sampling over times
        cwts_d = np.zeros(len(pclist_d))
        twt_d = 0  # Total weight
        for ind_s in range(len(pclist_d)):
            twt_d += np.exp(
                0.5 * (pclist_d[ind_s, 1]**2 + pclist_d[ind_s, 2]**2))
            cwts_d[ind_s] = twt_d
        cwts_d /= twt_d

        # Record in the total weight factor
        twt *= twt_d

        # Pick samples according to the correct probabilities
        tz_samples[ind_d] = rand_choice_nb(pclist_d, cwts_d, nsamples).T

        # Add offsets to detectors > H1
        if ind_d > 0:
            tz_samples[ind_d, 0] += offsets[ind_d - 1]

    # Generate keys into the RA-Dec dictionary from the delays, do them
    # at once to save some time
    # Delays in ms ((ndet - 1) x nsamples)
    dts = tz_samples[1:, 0, :] - tz_samples[0, 0, :]
    dts *= 1000
    keys = dt2key(dt=dts, dt_sinc=dt_sinc, dt_max=dt_max)

    # Populate the structures to evaluate the marginalized likelihood with
    # samples that have allowed delays
    nsamp_phys = 0
    zzs = np.zeros((nsamples, ndet), dtype=np.complex128)
    tts = np.zeros((nsamples, ndet), dtype=np.complex128)
    fskys = np.zeros(nsamples, dtype=np.int32)
    dt_dict_key_inds = np.searchsorted(dt_dict_keys, keys)
    for ind_s in range(nsamples):
        dt_dict_key_ind = dt_dict_key_inds[ind_s]
        key = keys[ind_s]
        if (dt_dict_key_ind < len(dt_dict_keys)) and \
                (dt_dict_keys[dt_dict_key_ind] == key):
            # Add to list of samples of z
            zzs[nsamp_phys] = \
                tz_samples[:, 1, ind_s] + 1j * tz_samples[:, 2, ind_s]

            # Pick RA, Dec to get f_+, f_x
            radec_indlist = dt_dict_items[dt_dict_key_ind]
            radec_ind = np.random.choice(len(radec_indlist))
            ra_ind, dec_ind = radec_indlist[radec_ind]

            # Record fsky (normalization factor for monte-carlo over ra and dec)
            fskys[nsamp_phys] = len(radec_indlist)

            # Pick mu and psi
            mu = np.random.choice(musamps)
            psi = np.random.choice(psisamps)

            # Add to list of predicted z
            for ind_d in range(ndet):
                tts[nsamp_phys, ind_d] = \
                    nfacs[ind_d] * gen_sample_amps_from_fplus_fcross(
                        responses[ra_ind, dec_ind, ind_d, 0],
                        responses[ra_ind, dec_ind, ind_d, 1],
                        mu, psi)

            nsamp_phys += 1

    if nsamp_phys > 0:
        # Generate list of unnormalized marginalized likelihoods
        marg_lk_list = \
            marg_lk(zzs, tts, gtype=gtype, nsamp=nsamp_phys)

        # Sum with the right weights to get the net marginalized likelihood
        # The nsamples is not a bug, it needs to be to use weight = twt
        wfac = twt / nsamples / t3norm / n_sky_pos
        s = 0
        for i in range(nsamp_phys):
            s += marg_lk_list[i] * fskys[i]
        s *= wfac
        score = 2. * np.log(s)
        return score
    else:
        return float(-10**5)


# ###############################################################################
class CoherentScoreMZ(object):
    def __init__(
            self, samples_fname=DEFAULT_SAMPLES_FNAME[("H1", "L1")], detnames=None,
            norm_angles=False, run='O2', empty_init=False):
        """
        :param samples_fname:
            Path to file with samples, created by create_samples
        :param detnames:
            If known, pass the list of detector names for plot labels.
            If not given, we will infer it from the filename, assuming it was
            generated by create_samples
        :param norm_angles: Flag to recompute normalization w.r.t angles
        :param run: Which run we're computing the coherent score for
        :param empty_init:
            Flag to return an empty instance, useful for alternate start
        """
        if empty_init:
            return

        # Read the contents of the sample file
        npzfile = np.load(samples_fname, allow_pickle=True)

        # Set some scalar parameters
        # Time resolution of dictionary in ms
        self.dt_sinc = float(npzfile['dt_sinc'])
        # Upper bound on the delays in ms
        self.dt_max = float(npzfile['dt_max'])
        # The epoch at which the delays were generated
        self.gps = float(npzfile['gps_time'])

        # Arrays of RA and dec
        self.ra_grid = npzfile['ra_grid']
        self.dec_grid = npzfile['dec_grid']

        # Dictionary indexed by keys for delta_ts, containing
        # n_skypos x 2 arrays with indices into ra_grid and dec_grid
        dt_dict = npzfile['dt_dict'].item()
        self.dt_dict_keys = np.fromiter(dt_dict.keys(), dtype=np.int32)
        self.dt_dict_keys.sort()
        self.dt_dict_items = tuple([dt_dict[key] for key in self.dt_dict_keys])

        # n_ra x n_dec x n_detectors x 2 array with f_+/x for phi = 0
        self.responses = npzfile['responses']
        # Convenience for later
        self.ndet = self.responses.shape[2]
        if detnames is not None:
            self.detnames = detnames
        else:
            self.detnames = \
                samples_fname.split("RA_dec_grid_")[1].split(
                    f"_{int(1000/self.dt_sinc)}")[0].split("_")

        # Samples of mu (cos inclination) and psis
        self.mus = npzfile['mus']
        self.psis = npzfile['psis']

        # Arrays for debugging purpose
        # n_ra x n_dec x (n_detectors - 1) array with delays w.r.t
        # the first detector
        self.deltats = npzfile['deltats']
        # n_ra x n_dec x (n_detectors - 1) array with phases w.r.t
        # the first detector
        self.dphases = npzfile['dphases']
        # n_ra x n_dec array with total network response
        self.rtot2s = npzfile['rtot2s']

        # choose a normalization constant
        # To recompute normalization use norm_angles=True
        self.T3norm = 0.2372446769308674
        if norm_angles:
            self.gen_t3_norm()

        ## if self.Gtype=0 then distance-phase is not prior
        self.Gtype = 1

        ## Choose a reference normfac for the run
        ## accounts for the relative sensitivity of the detectors
        ## chosen so that spinning guy has normalization=1
        ## chosen so that Liang's trigger has normalization=1
        self.run = run
        if run.lower() == 'o1':
            self.norm_h1_normalization = 7.74908546328925
            self.norm_l1_normalization = 7.377041393512488
            self.normfac_pos = 2
            self.hole_correction_pos = None
            self.psd_drift_pos = 3
            self.rezpos = 4
            self.imzpos = 5
            self.c0_pos = 6
        else:
            # Set default to new runs
            self.norm_h1_normalization = 0.03694683692493602
            self.norm_l1_normalization = 0.042587464435623064
            self.normfac_pos = 2
            self.hole_correction_pos = 3
            self.psd_drift_pos = 4
            self.rezpos = 5
            self.imzpos = 6
            self.c0_pos = 7

        return

    @classmethod
    def from_new_samples(
            cls, nra=100, ndec=100, detnames=('H1', 'L1'),
            gps_time=1136574828.0, dt_sinc=DEFAULT_DT, dt_max=DEFAULT_DT_MAX,
            nsamples=100000, run="O2"):
        # Create detectors
        detectors = [lal.CachedDetectors[DETMAP[det]] for det in detnames]

        # Create structures to deal with the mapping of the sphere to delays
        dt_dict, ra_grid, dec_grid, responses, deltats, dphases, rtot2s = \
            create_time_dict(
                nra, ndec, detectors, gps_time=gps_time, dt_sinc=dt_sinc,
                dt_max=dt_max)

        # Create random samples of the cosine of the inclination, and the
        # polarization
        psis = np.random.uniform(0, 2 * np.pi, size=nsamples)
        mus = np.random.uniform(-1, 1, size=nsamples)  # cos(inclination)

        # Create an empty instance and read in the parameters
        instance = cls(empty_init=True)

        # Set parameters
        instance.dt_sinc = dt_sinc  # Time resolution of dictionary in ms
        instance.dt_max = dt_max  # Upper bound on the delays in ms
        instance.gps = gps_time  # The epoch at which the delays were generated

        # Arrays of RA and DEC
        instance.ra_grid = ra_grid
        instance.dec_grid = dec_grid

        # Dictionary indexed by keys for delta_ts, containing
        # n_skypos x 2 arrays with indices into ra_grid and dec_grid
        instance.dt_dict_keys = np.fromiter(dt_dict.keys(), dtype=np.int32)
        instance.dt_dict_keys.sort()
        instance.dt_dict_items = tuple(
            [dt_dict[key] for key in instance.dt_dict_keys])

        # n_ra x n_dec x n_detectors x 2 array with f_+/x for phi = 0
        instance.responses = responses
        # Convenience for later
        instance.ndet = responses.shape[2]
        instance.detnames = detnames

        # Samples of mu (cos inclination) and psis
        instance.mus = mus
        instance.psis = psis

        # Arrays for debugging purpose
        # n_ra x n_dec x (n_detectors - 1) array with delays w.r.t
        # the first detector
        instance.deltats = deltats
        # n_ra x n_dec x (n_detectors - 1) array with phases w.r.t
        # the first detector
        instance.dphases = dphases
        # n_ra x n_dec array with total network response
        instance.rtot2s = rtot2s

        # Generate norm to be safe
        instance.T3norm = 0.2372446769308674
        instance.gen_t3_norm()

        ## if self.Gtype=0 then distance-phase is not prior
        instance.Gtype = 1

        ## Choose value to normalize T
        ## accounts for the reltive sensitivity of the detectors
        ## chosen so that spinning guy has normalization=1
        ## chosen so that Liang's trigger has normalization=1
        instance.run = run
        if run == 'O1':
            instance.norm_h1_normalization = 7.74908546328925
            instance.norm_l1_normalization = 7.377041393512488
            instance.normfac_pos = 2
            instance.hole_correction_pos = None
            instance.psd_drift_pos = 3
            instance.rezpos = 4
            instance.imzpos = 5
            instance.c0_pos = 6
        else:
            instance.norm_h1_normalization = 0.03694683692493602
            instance.norm_l1_normalization = 0.042587464435623064
            instance.normfac_pos = 2
            instance.hole_correction_pos = 3
            instance.psd_drift_pos = 4
            instance.rezpos = 5
            instance.imzpos = 6
            instance.c0_pos = 7

        return instance

    def save_samples(self, fname):
        filename = utils.rm_suffix(fname, suffix='.npz', new_suffix="_") + \
            f"{int(1000 / self.dt_sinc)}" + f"_{int(self.gps)}.npz"
        dt_dict = dict(zip(self.dt_dict_keys, self.dt_dict_items))
        np.savez(filename,
                 dt_dict=dt_dict, ra_grid=self.ra_grid,
                 dec_grid=self.dec_grid, responses=self.responses,
                 deltats=self.deltats, dphases=self.dphases,
                 rtot2s=self.rtot2s, psis=self.psis, mus=self.mus,
                 gps_time=self.gps, dt_sinc=self.dt_sinc, dt_max=self.dt_max)

        return

    def gen_t3_norm(self, nsamples=10 ** 6):
        print('Old T3norm', self.T3norm)
        mus = np.random.choice(self.mus, size=nsamples, replace=True)
        psis = np.random.choice(self.psis, size=nsamples, replace=True)
        ra_inds = np.random.randint(0, len(self.ra_grid), size=nsamples)
        dec_inds = np.random.randint(0, len(self.dec_grid), size=nsamples)
        t_list = np.zeros(nsamples)
        for det_ind in range(self.ndet):
            responses = self.responses[ra_inds, dec_inds, det_ind, :]
            fplus, fcross = responses[:, 0], responses[:, 1]
            avec = gen_sample_amps_from_fplus_fcross(fplus, fcross, mus, psis)
            t_list += utils.abs_sq(avec)
        t3mean = np.mean(t_list ** 1.5)
        self.T3norm = t3mean
        print('New T3norm', self.T3norm)

        return

    def dt_dict(self, keyval):
        """Convenience function to simulate a dictionary"""
        ind = np.searchsorted(self.dt_dict_keys, keyval)
        if ((ind < len(self.dt_dict_keys)) and
                (self.dt_dict_keys[ind] == keyval)):
            return self.dt_dict_items[ind]
        else:
            raise KeyError(f"invalid key {keyval}")

    def get_all_prior_terms(
            self, events, timeseries=None, loc_id=None, ref_normfac=1,
            time_slide_jump=DEFAULT_TIMESLIDE_JUMP/1000, **score_kwargs):
        """
        :param events:
            (n_cand x (n_det=2) x processedclist)/((n_det=2) x processedclist)
            array with coincidence/background candidates
        :param timeseries: If known, list of lists/tuples of length n_detectors
            with n_samp x 3 array with t, Re(z), Im(z)
            (can be single list/tuple if n_events=1)
        :param loc_id:
            Tuple with (bank_id, subbank_id), used if we're reading files
        :param ref_normfac: Reference normfac to scale the values relative to
        :param time_slide_jump: Least count of time slides (s)
        :param score_kwargs: Extra arguments to trigger2comblist or comblist2cs
        :return: Computes the coherent score minus the rho^2 piece
        """
        if events.ndim == 2:
            # We're dealing with a single event
            events = events[None, :]
            if timeseries is not None:
                timeseries = [timeseries]

        if len(events) == 0:
            return np.zeros((0, 2)), np.zeros(events.shape[:2] + (4,)), \
                timeseries

        if utils.checkempty(timeseries):
            compute_ts = True
            timeseries = []
        else:
            compute_ts = False

        # n_cand x n_detector x 4
        params = self.get_params(events, time_slide_jump=time_slide_jump)

        # Some useful parameters
        rhosq = utils.incoherent_score(events)
        offsets = params[:, 1:, 0] - events[:, 1:, 0]
        nfacs = params[:, :, 3] / ref_normfac

        prior_terms = np.zeros((len(events), 2))
        for ind in range(len(events)):
            if compute_ts:
                # Define the timeseries ourself
                timeseries_ev, *_ = \
                    self.trigger2comblist(
                        trigger=events[ind], timeseries=None, loc_id=loc_id,
                        ref_normfac=ref_normfac, **score_kwargs)
                timeseries.append(timeseries_ev)
            else:
                timeseries_ev = timeseries[ind]

            if not isinstance(timeseries_ev, tuple):
                timeseries_ev = tuple(timeseries_ev)

            prior_terms[ind, 0] = self.comblist2cs(
                timeseries_ev, offsets[ind], nfacs[ind], **score_kwargs)
            # Extra term to remove the Gaussian part before adding the
            # rank function
            prior_terms[ind, 1] = - rhosq[ind]

        return prior_terms, params, timeseries

    def trigger2comblist(
            self, trigger=None, timeseries=None, loc_id=None, ref_normfac=None,
            time_slide_jump=DEFAULT_TIMESLIDE_JUMP/1000, dt=0.1, lk_cut=0.01,
            inject=(False, []), plot=False, ax=None, **kwargs):
        """
        Takes a trigger from plots_publication and produces input to the
        function for determining the coherent score, this can can inject,
        compute missing quantities, and make plots
        (assumes the trigger is from the class runtype)
        :param trigger: Array of size n_detector x row of processedclist
        :param timeseries: If known, tuple of length n_detectors with
            n_samp x 3 array with Re(z), Im(z) (the output shares memory)
        :param loc_id:
            Tuple with (bank_id, subbank_id), pass in if we're reading files
        :param ref_normfac:
            Reference normfac if known, else we use the reference value
            from the saved run
        :param time_slide_jump: Least count of time slides (s)
        :param dt: length of time around trigger time to compute the overlaps
        :param lk_cut: Will only keep times when the likelihood is below the
            peak by this factor
        :param inject:
            If desired, tuple with True/False, the list of injection
            parameters for each detector, and the calphas to use for MF
        :param plot: True produces a plot for debugging purposes
        :param ax: If known, axis to put the plot in
        :param kwargs: Generic variable to capture any extra arguments
        :return: 1. Timeseries (if we had input, we pass it out)
                 2. Incoherent score of trigger
                 3. Shifts to apply to the timeseries > H1 to bring to zero lag
                 4. Array of length n_detector with detector sensitivities
                    (useful when subtracting the incoherent piece)
        """
        if inject[0]:
            ndet = len(inject[1])
            t_event_dets = \
                np.asarray([pars_det['time'] for pars_det in inject[1]])
            calpha_event = inject[2]
            trigger = []
            timeseries = []
        elif trigger is not None:
            ndet = len(trigger)
            t_event_dets = trigger[:, 0]
            calpha_event = trigger[0, self.c0_pos:]
        else:
            raise RuntimeError("I need something to start with!")

        if ndet != self.ndet:
            print(f"The class is set up to deal with {self.ndet} " +
                  f"detectors, but we were passed {ndet} triggers")
            return (), np.zeros(0)

        # Get offsets to apply to the triggers in the detectors > H1
        dt_shift = self.dt_sinc / 1000  # in s
        dts = t_event_dets[1:] - t_event_dets[0]
        offsets = utils.offset_background(dts, time_slide_jump, dt_shift)

        if plot:
            # Create the figure and keep adding to it
            if ax is None:
                _, ax = utils.import_matplotlib(figsize=(12, 4))

        if utils.checkempty(timeseries):
            timeseries = []
            # We have to compute the timeseries ourselves
            if loc_id is None:
                raise RuntimeError("We need bank information to find the files")
            else:
                bank_id, subbank_id = loc_id

            for ind_det in range(self.ndet):
                # Load the file for the detector
                # -----------------------------------------------------------
                t_event_det = t_event_dets[ind_det]
                source = kwargs.get("source", "BBH")
                ind_fnames = DET2GET_DETECTOR_FNAMES[self.detnames[ind_det]]
                fname = utils.get_detector_fnames(
                    t_event_det, chirp_mass_id=bank_id,
                    subbank=subbank_id, run=self.run, source=source)[ind_fnames]

                if fname is None:
                    print("I couldn't find the file for detector " +
                          f"{self.detnames[ind_det]} at t = {t_event_dets[ind_det]}")
                    return (), np.zeros(0), np.zeros(0), 0
                else:
                    trig_obj = trig.TriggerList.from_json(
                        fname, load_trigs=False)

                # Inject if needed
                # -----------------------------------------------------------
                if inject[0]:
                    pars_det = inject[1][ind_det]
                    _ = trig_obj.inject_wf_into_data(
                        trig_obj.time, trig_obj.strain, trig_obj.templatebank,
                        trig_obj.templatebank.asdf, whitened=True,
                        min_filt_trunc_time=trig_obj.min_filt_trunc_time, **pars_det)

                # Generate triggers around the location
                # -----------------------------------------------------------
                pclist = trig_obj.gen_triggers_local(
                    location=(t_event_det, calpha_event), dt_left=dt,
                    dt_right=dt, compute_calphas=[calpha_event],
                    apply_threshold=False, relative_binning=False,
                    zero_pad=False)

                # Read off and save properties of the triggers
                # -----------------------------------------------------------
                # TODO: There are small differences w/ using the safemean PSD
                #  drift in the trigger (O2), and the subset of data used
                #  For faint injections, this can also pick up a nearby peak
                if inject[0]:
                    # We weren't passed in a trigger, so save the computed one
                    imax = np.argmax(pclist[:, 1])
                    maxsnr2 = pclist[imax, 1]
                    trigger.append(pclist[imax])
                else:
                    maxsnr2 = trigger[ind_det, 1]

                inds_good = np.where(
                    pclist[:, 1] > maxsnr2 + 2 * np.log(lk_cut))[0]
                rezpos = trig_obj.rezpos
                imzpos = trig_obj.imzpos
                timeseries.append(
                    pclist[np.ix_(inds_good, (0, rezpos, imzpos))])

                if plot:
                    # Add the entire timeseries to the plot
                    offset = -t_event_dets[0]
                    if ind_det > 0:
                        offset += offsets[ind_det - 1]
                    ax.plot(
                        pclist[:, 0] + offset, pclist[:, 1], c=f"C{ind_det}")

        if plot:
            # Add the filtered timeseries to the plot
            for ind_det in range(self.ndet):
                tzs_det = timeseries[ind_det]
                # Offset to bring everything to zero-lag
                offset = -t_event_dets[0]
                if ind_det > 0:
                    offset += offsets[ind_det - 1]
                times = tzs_det[:, 0] + offset
                snr2 = tzs_det[:, 1]**2 + tzs_det[:, 2]**2
                maxsnr2 = np.max(snr2)
                ax.plot(times, snr2, c=f"C{ind_det}", ls='None', marker="o",
                        label=self.detnames[ind_det])
                # Plot e^{\rho^2/2}
                if ind_det == 0:
                    ax.plot(times, maxsnr2 * np.exp(0.5 * (snr2 - maxsnr2)),
                            c='k', ls=":", label="$\exp{-Z^2/2}$")
                else:
                    ax.plot(times, maxsnr2 * np.exp(0.5 * (snr2 - maxsnr2)),
                            c='k', ls=":")

            ax.set_xlabel("$t-t_{event, H}$")
            ax.set_ylabel("$SNR^2$")
            ax.set_xlim(-0.02, 0.02)
            ax.set_title("Points have likelihood bigger than 1% of peak value")
            ax.legend()

        # No need to have an extra combined cut with the new computation
        trigger = np.asarray(trigger)
        if not isinstance(timeseries, tuple):
            timeseries = tuple(timeseries)

        # Compute the sensitivities for all detectors
        if self.hole_correction_pos is not None:
            # New format
            nfacs = trigger[:, self.normfac_pos] * \
                trigger[:, self.hole_correction_pos] / \
                trigger[:, self.psd_drift_pos]
        else:
            # Old format
            # Saved normfac = psd_drift x normfac
            nfacs = trigger[:, self.normfac_pos] / \
                trigger[:, self.psd_drift_pos]**2

        if ref_normfac is None:
            # The h1 is not a bug, as the normalization is only for a
            # fiducial value, and we want to preserve the asymmetry
            ref_normfac = self.norm_h1_normalization
        nfacs /= ref_normfac

        return timeseries, utils.incoherent_score(trigger), offsets, nfacs

    def comblist2cs(
            self, timeseries, offsets, nfacs, gtype=1, nsamples=10000,
            **kwargs):
        """
        Takes the return value of trigger2comblist and returns the coherent
        score by calling the jitted function
        :param timeseries:
            Tuple of length n_detectors with n_samp x 3 array with Re(z), Im(z)
        :param offsets:
            Array of length n_detector -1 with shifts to apply to the
            timeseries > H1 to bring them to zero lag
        :param nfacs: Array of length n_detector with detector sensitivities
        :param gtype: If 0, turns off the integration over distance
        :param nsamples: Number of samples to use in the monte carlo
        :param kwargs: Generic variable to capture any extra arguments
        :return: Coherent score, defined as 2 * log (p(H_1) / p(H_0))
        """
        # if timeseries is empty, return -10^5 (see coherent_score_montecarlo_sky()) and continue
        if any([len(x) == 0 for x in timeseries]):
            return -100000
        return coherent_score_montecarlo_sky(
            timeseries, offsets, nfacs, self.dt_dict_keys, self.dt_dict_items,
            self.responses, self.T3norm, musamps=self.mus, psisamps=self.psis,
            gtype=gtype, dt_sinc=self.dt_sinc, dt_max=self.dt_max,
            nsamples=nsamples)

    def get_params(self, events, time_slide_jump=DEFAULT_TIMESLIDE_JUMP/1000):
        """
        :param events:
            (n_cand x (n_det = 2) x processedclist)/
            ((n_det = 2) x processedclist) array with coincidence/background
            candidates
        :param time_slide_jump: Units of jumps (s) for timeslides
        :return: n_cand x (n_det = 2) x 4 array (always 3D) with
            shifted_ts, re(z), im(z), effective_sensitivity in each detector
        """
        if events.ndim == 2:
            # We're dealing with a single event
            events = events[None, :]

        if len(events) == 0:
            return np.zeros(events.shape[:2] + (4,))

        dt_shift = self.dt_sinc / 1000  # in s

        # Add shifts to each detector to get to zero lag
        # n_cand x n_det
        ts_out = events[:, :, 0].copy()
        shifts = utils.offset_background(
            ts_out[:, 1:] - ts_out[:, 0][:, None], time_slide_jump, dt_shift)
        ts_out[:, 1:] += shifts

        # Overlaps
        # n_cand x n_det
        rezs = events[:, :, self.rezpos]
        imzs = events[:, :, self.imzpos]

        # Sensitivity
        # The hole correction is a number between 0 and 1 reflecting the
        # sensitivity after some parts of the waveform fell into a hole
        # asd drift is the effective std that the score was divided with
        # => the bigger it is, the less the sensitivity
        asd_corrs = events[:, :, self.psd_drift_pos]
        ns = events[:, :, self.normfac_pos]
        if self.hole_correction_pos is not None:
            # New format
            # Hole corrections
            hs = events[:, :, self.hole_correction_pos]
            n_effs = ns / asd_corrs * hs
        else:
            # Old format in O1
            # Normfacs are actually normfac x psd_drift
            n_effs = ns / asd_corrs ** 2

        return np.stack((ts_out, rezs, imzs, n_effs), axis=2)


pass
