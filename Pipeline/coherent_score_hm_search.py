"""
Define class ``SearchCoherentScoreHMAS`` that computes the coherent
score (marginalized likelihood over extrinsic parameters) for
aligned-spin, (l,m) = [(2, 2), (3, 3), (4, 4)] waveforms.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import params
from scipy.special import logsumexp
import utils

path_to_cogwheel = Path(__file__).parent.parent / 'cogwheel'
sys.path.append(path_to_cogwheel.as_posix())

import cogwheel.utils
from cogwheel.likelihood.marginalization.base import (BaseCoherentScoreHM,
                                                      MarginalizationInfoHM)


class SearchCoherentScoreHMAS(BaseCoherentScoreHM):
    """
    Class to marginalize the likelihood over extrinsic parameters,
    intended for the search pipeline.

    Assumptions:
        * Quasicircular
        * Aligned spins,
        * (l, m) = [(2, 2), (3, 3), (4, 4)] harmonics.
    """
    M_ARR = np.array([2, 3, 4])  # Also assume l = m

    def __init__(self, *, sky_dict, m_arr=M_ARR, lookup_table=None,
                 log2n_qmc: int = 11, nphi=128, seed=0,
                 beta_temperature=.1, n_qmc_sequences=128,
                 min_n_effective=50, max_log2n_qmc: int = 15):
        if not np.array_equal(m_arr, self.M_ARR):
            raise ValueError(f'`m_arr` must be {self.M_ARR} in this class.')

        super().__init__(m_arr=self.M_ARR,
                         sky_dict=sky_dict,
                         lookup_table=lookup_table,
                         log2n_qmc=log2n_qmc,
                         nphi=nphi,
                         seed=seed,
                         beta_temperature=beta_temperature,
                         n_qmc_sequences=n_qmc_sequences,
                         min_n_effective=min_n_effective,
                         max_log2n_qmc=max_log2n_qmc)

    @property
    def _qmc_range_dic(self):
        """
        Parameter ranges for the QMC sequence.
        The sequence explores the cumulatives of the single-detector
        (incoherent) likelihood of arrival times, the polarization, the
        fine (subpixel) time of arrival and the cosine inclination.
        """
        return super()._qmc_range_dic | {'cosiota': (-1, 1)}

    def _create_qmc_sequence(self):
        """
        Return a dictionary whose values are arrays corresponding to a
        Quasi Monte Carlo sequence that explores parameters per
        ``._qmc_range_dic``.
        The arrival time cumulatives are packed in a single entry
        'u_tdet'. An entry 'rot_psi' has the rotation matrices to
        transform the antenna factors between psi=0 and psi=psi_qmc.
        Also, entries for 'response' are provided. The response is defined
        so that:

          total_response :=
            := (1+cosiota**2)/2*fplus - 1j*cosiota*fcross
            = ((1+cosiota**2)/2, - 1j*cosiota) @ (fplus, fcross)
            = ((1+cosiota**2)/2, - 1j*cosiota) @ rot @ (fplus0, fcross0)
            = response @ (fplus0, fcross0)
        for the (2, 2) mode; the (3, 3) mode has an extra siniota; and
        the (4, 4) a siniota^2.
        """
        qmc_sequence = super()._create_qmc_sequence()
        siniota = np.sin(np.arccos(qmc_sequence['cosiota']))
        qmc_sequence['response'] = np.einsum(
            'Pq,qPp,qm->qpm',
            ((1 + qmc_sequence['cosiota']**2) / 2,
             - 1j * qmc_sequence['cosiota']),
            qmc_sequence['rot_psi'],
            np.power.outer(siniota, np.arange(3)))
        return qmc_sequence

    def get_marginalization_info(self, dh_mtd, hh_md, times,
                                 incoherent_lnprob_td, mode_ratios_qm):
        """
        Return a MarginalizationInfoHM object with extrinsic parameter
        integration results, ensuring that one of three conditions
        regarding the effective sample size holds:
            * n_effective >= .min_n_effective; or
            * n_qmc == 2 ** .max_log2n_qmc; or
            * n_effective = 0 (if the first proposal only gave
                               unphysical samples)
        """
        self.sky_dict.set_generators()  # For reproducible output

        # Resample to match sky_dict's dt:
        dh_mtd, _ = self.sky_dict.resample_timeseries(
            dh_mtd, times, axis=-2)
        t_arrival_lnprob, times = self.sky_dict.resample_timeseries(
            incoherent_lnprob_td.T, times, axis=-1)

        self.sky_dict.apply_tdet_prior(t_arrival_lnprob)
        t_arrival_prob = cogwheel.utils.exp_normalize(t_arrival_lnprob, axis=1)

        return self._get_marginalization_info(
            dh_mtd, hh_md, times, t_arrival_prob,
            mode_ratios_qm=mode_ratios_qm)

    def _get_marginalization_info_chunk(self, dh_mtd, hh_md, times,
                                        t_arrival_prob, i_chunk,
                                        mode_ratios_qm):
        q_inds = self._qmc_ind_chunks[i_chunk]  # Will update along the way
        n_qmc = len(q_inds)
        tdet_inds = self._get_tdet_inds(t_arrival_prob, q_inds)

        sky_inds, sky_prior, physical_mask \
            = self.sky_dict.get_sky_inds_and_prior(
                tdet_inds[1:] - tdet_inds[0])  # q, q, q

        # Apply physical mask (sensible time delays):
        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]

        if not any(physical_mask):
            return MarginalizationInfoHM(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                q_inds=np.array([], int),
                o_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                proposals_n_qmc=[n_qmc],
                proposals=[t_arrival_prob],
                flip_psi=np.array([], bool)
                )

        t_first_det = (times[tdet_inds[0]]
                       + self._qmc_sequence['t_fine'][q_inds])

        dh_qo, hh_qo = self._get_dh_hh_qo(sky_inds, q_inds, t_first_det,
                                          times, dh_mtd, hh_md, mode_ratios_qm)

        ln_numerators, important, flip_psi \
            = self._get_lnnumerators_important_flippsi(dh_qo, hh_qo, sky_prior)

        # Keep important samples (lnl above threshold):
        q_inds = q_inds[important[0]]
        sky_inds = sky_inds[important[0]]
        t_first_det = t_first_det[important[0]]
        tdet_inds = tdet_inds[:, important[0]]

        return MarginalizationInfoHM(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators=ln_numerators,
            q_inds=q_inds,
            o_inds=important[1],
            sky_inds=sky_inds,
            t_first_det=t_first_det,
            d_h=dh_qo[important],
            h_h=hh_qo[important],
            tdet_inds=tdet_inds,
            proposals_n_qmc=[n_qmc],
            proposals=[t_arrival_prob],
            flip_psi=flip_psi,
            )

    def lnlike_marginalized(self, dh_mtd, hh_md, times,
                            incoherent_lnprob_td, mode_ratios_qm):
        """
        Return log of marginalized likelihood over inclination, sky
        location, orbital phase, polarization, time of arrival and
        distance.

        Parameters
        ----------
        dh_mtd: (n_modes, n_times, n_det) complex array
            Timeseries of the inner product (d|h) between data and
            template, where the template is evaluated at a distance
            ``self.lookup_table.REFERENCE_DISTANCE`` Mpc.
            The convention in the inner product is that the second
            factor (i.e. h, not d) is conjugated.

        hh_md: (n_modes*(n_modes-1)/2, n_det) complex array
            Covariance between the different modes, i.e. (h_m|h_m'),
            with the off-diagonal entries (m != m') multiplied by 2.
            The ordering of the modes can be found with
            `self.m_arr[self.m_inds], self.m_arr[self.mprime_inds]`.
            The same template normalization and inner product convention
            as for ``dh_mtd`` apply.

        times: (n_times,) float array
            Times corresponding to the (d|h) timeseries (s).

        incoherent_lnprob_td: (n_times, n_det) float array
            Incoherent proposal for log probability of arrival times at
            each detector.

        mode_ratios_qm: (2**self.max_log2n_qmc, n_modes-1) float array
            Samples of mode amplitude ratio to the first mode (the part
            independent of inclination). These samples are used to
            marginalize over intrinsic parameters, mainly mass ratio.
        """
        marg_info = self.get_marginalization_info(
            dh_mtd, hh_md, times, incoherent_lnprob_td,
            mode_ratios_qm=mode_ratios_qm)
        return marg_info.lnl_marginalized

    def _get_dh_hh_qo(self, sky_inds, q_inds, t_first_det, times,
                      dh_mtd, hh_md, mode_ratios_qm):
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dmq = np.array(
            [self._interp_locally(times, dh_mtd[..., i_det], t_det[i_det])
             for i_det in range(len(self.sky_dict.detector_names))])

        # h_qdm = factor_qdm * h0_qm
        factor_qdm = (self.sky_dict.fplus_fcross_0[sky_inds, ]
                      @ self._qmc_sequence['response'][q_inds]
                      )  # qdp @ qpm -> qdm
        factor_qdm[..., 1:] *= mode_ratios_qm[q_inds, np.newaxis, :]

        dh_qm = np.einsum('dmq,qdm->qm', dh_dmq, factor_qdm.conj())  # qm
        hh_qm = np.einsum('md,qdm,qdm->qm',
                          hh_md,
                          factor_qdm[..., self.m_inds],
                          factor_qdm.conj()[..., self.mprime_inds])

        dh_qo = cogwheel.utils.real_matmul(dh_qm, self._dh_phasor)  # qo
        hh_qo = cogwheel.utils.real_matmul(hh_qm, self._hh_phasor)  # qo
        return dh_qo, hh_qo


def initialize_cs_instance(
        trig1=None, trig2=None, detectors=('H1', 'L1'), seed=0,
        n_qmc_sequences=1, log2n_qmc=params.LOG2N_QMC, nphi=params.NPHI,
        max_log2n_qmc=params.MAX_LOG2N_QMC,
        min_n_effective=params.MIN_N_EFFECTIVE, **cs_kwargs):

    detectors = tuple(sorted(detectors))

    cs_instance = SearchCoherentScoreHMAS(
                    sky_dict=cogwheel.likelihood.marginalization.SkyDictionary(
                            detectors[0][0]+detectors[1][0]), seed=seed,
                    log2n_qmc=log2n_qmc, nphi=nphi, max_log2n_qmc=max_log2n_qmc,
                    n_qmc_sequences=n_qmc_sequences, **cs_kwargs)
    cs_instance.min_n_effective = min_n_effective
    # Calculating reference distance to SNR=1 template and rescaling
    # by normfac
    cs_instance.dist_factor_ref = cs_instance.lookup_table.d_luminosity_max\
        / cs_instance.lookup_table.REFERENCE_DISTANCE
    if trig1 is not None:
        cs_instance.mode_ratios_qm_subbank = \
            create_mode_ratios_qm_subbank_new([trig1, trig2], 2**cs_instance.max_log2n_qmc)
    else:
        cs_instance.mode_ratios_qm_subbank = []

    return cs_instance


def create_mode_ratios_qm_subbank(trig1, trig2, size):
    mode_ratios_unsampled = trig1.templatebank.HM_amp_ratio_samples[1, :, :2].copy()
    weights = trig1.templatebank.HM_amp_ratio_samples[1, :, 2].copy()

    # TODO_LessImp: The detector with better normfac gets to decide the Rij samples
    # (note the Rij samples are scaled slightly based on the detector PSD)
    # In the future, one could use Rij samples from both detectors to calculate
    # the marginalized integral over mode amp ratios.
    if trig1.normfac >= trig2.normfac:
        mode_ratios_unsampled *= trig1.templatebank.HM_amp_ratio_PSD_factor
    else:
        mode_ratios_unsampled *= trig2.templatebank.HM_amp_ratio_PSD_factor
    weights = weights/np.sum(weights)

    np.random.seed(seed=0)
    mode_ratios_inds = np.random.choice(
        len(mode_ratios_unsampled), size=size, p=weights)
    return mode_ratios_unsampled[mode_ratios_inds]


def compute_coherent_scores(
        cs_instance, coincident_trig_list, trig1, trig2,
        minimal_time_slide_jump=0.1, score_reduction_timeseries=10,
        output_timeseries=False, output_coherent_score=True,
        coh_score_iterations=1, return_inputs=False):
    """
    :param cs_instance: instance of SearchCoherentScoreHMAS
    :param coincident_trig_list: list of coincident triggers
    :param trig1: trigger object for detector 1
    :param trig2: trigger object for detector 2
    :param minimal_time_slide_jump: minimal time slide jump
    :param score_reduction_timeseries:
    :param output_timeseries:
    :param output_coherent_score:
    :param coh_score_iterations: to improve convergence, one could average
            over multiple iterations of the coherent score calculation
    :param return_inputs: for debugging purposes, return the inputs to the
            coherent score calculation (only for the first trigger in the list)
    """
    timeseries = []
    coherent_scores = []
    for ind, (trig_h1, trig_l1) in enumerate(coincident_trig_list):
        if ind % 10 == 0:
            print('Coh score frac complete:', ind / len(coincident_trig_list), flush=True)
        tdiff = (trig_l1[0] - trig_h1[0]) % minimal_time_slide_jump
        if tdiff > (minimal_time_slide_jump/2):
            tdiff -= minimal_time_slide_jump
        tdiff_l1 = tdiff / (1 + np.exp((trig_l1[1]-trig_h1[1])/2))
        tdiff_l1 = np.round((tdiff_l1 - (tdiff/2))/trig1.dt)*trig1.dt\
            + (tdiff/2)
        tdiff_h1 = tdiff - tdiff_l1
        tdiff_l1 = np.round(tdiff_l1, 13)
        tdiff_h1 = np.round(tdiff_h1, 13)
        # Rounding is a temporary patch as of now, properly look into this later

        # The time difference in the detectors in the zero lag case
        # is weighted by the likelihood in each detector
        if trig1.dt != trig2.dt:
            raise NotImplementedError('Currently only works when dt of' +
                                      'both detectors are the same')

        # Compute SNR time series near the peak
        trigs_calpha_h1, CovMat_H = trig1.gen_triggers_local(
            trigger=trig_h1,
            dt_left=params.DT_BOUND_TIMESERIES - tdiff_h1,
            dt_right=params.DT_BOUND_TIMESERIES + tdiff_h1,
            compute_calphas=[trig_h1[trig1.c0_pos:]],
            apply_threshold=False, relative_binning=False,
            zero_pad=False, orthogonalize_modes=False,
            return_mode_covariance=True)
        trigs_calpha_l1, CovMat_L = trig2.gen_triggers_local(
            trigger=trig_l1,
            dt_left=params.DT_BOUND_TIMESERIES + tdiff_l1,
            dt_right=params.DT_BOUND_TIMESERIES - tdiff_l1,
            compute_calphas=[trig_l1[trig2.c0_pos:]],
            apply_threshold=False, relative_binning=False,
            zero_pad=False, orthogonalize_modes=False,
            return_mode_covariance=True)
        CovMat_H = CovMat_H[0]
        CovMat_L = CovMat_L[0]

        if output_timeseries:
            # Keep only near the peak
            # This errs on the side of keeping more triggers in
            mask_h1 = trigs_calpha_h1[:, 1] > trig_h1[1] - score_reduction_timeseries
            mask_l1 = trigs_calpha_l1[:, 1] > trig_l1[1] - score_reduction_timeseries
            timeseries_h1 = np.c_[
                trigs_calpha_h1[mask_h1, 0],
                trigs_calpha_h1[mask_h1, trig1.rezpos],
                trigs_calpha_h1[mask_h1, trig1.imzpos],
                trigs_calpha_h1[mask_h1, trig1.rezpos+2],
                trigs_calpha_h1[mask_h1, trig1.imzpos+2],
                trigs_calpha_h1[mask_h1, trig1.rezpos+4],
                trigs_calpha_h1[mask_h1, trig1.imzpos+4]]
            timeseries_l1 = np.c_[
                trigs_calpha_l1[mask_l1, 0],
                trigs_calpha_l1[mask_l1, trig2.rezpos],
                trigs_calpha_l1[mask_l1, trig2.imzpos],
                trigs_calpha_l1[mask_l1, trig2.rezpos+2],
                trigs_calpha_l1[mask_l1, trig2.imzpos+2],
                trigs_calpha_l1[mask_l1, trig2.rezpos+4],
                trigs_calpha_l1[mask_l1, trig2.imzpos+4]]

            timeseries.append((timeseries_h1, timeseries_l1))
            # There is a subtle bug with np.array() when the length of the two
            # timeseries is indentical for all candidates, where the array shape
            # is different in an unexpected way, leave it in the form of a list
            # of tuples, and we will deal with it when saving

        if not output_coherent_score:
            continue

        # if this is computationally expensive
        # precompute the mode_ratios_qm for the calphas in the bank
        if trig1.templatebank.Rij_Coh_Score_NF is not None:

            # if the calpha is at the extremes of the bank, the NF may not extrapolate
            # well, so we first add the following check
            if (trig1.templatebank.Template_Prior_NF.log_prior(
                                            [trig_h1[trig1.c0_pos:]])
                    < (trig1.templatebank.Template_Prior_NF.calpha_reject_threshold + 1.5)):
                mode_ratios_qm = cs_instance.mode_ratios_qm_subbank.copy()

            else:
                mode_ratios_unsampled = \
                    trig1.templatebank.Rij_Coh_Score_NF.generate_samples(
                        trig_h1[trig1.c0_pos:], set_seed=True,
                        num_samples=2**(cs_instance.max_log2n_qmc-3))
                weights = mode_ratios_unsampled[:, 2]
                if trig_h1[1] >= trig_l1[1]:
                    mode_ratios_unsampled = mode_ratios_unsampled[:, :2] * \
                            trig1.templatebank.HM_amp_ratio_PSD_factor
                else:
                    mode_ratios_unsampled = mode_ratios_unsampled[:, :2] * \
                            trig2.templatebank.HM_amp_ratio_PSD_factor
                weights = weights/np.sum(weights)

                np.random.seed(seed=0)
                mode_ratios_inds = np.random.choice(
                    len(mode_ratios_unsampled),
                    size=2**cs_instance.max_log2n_qmc, p=weights)
                mode_ratios_qm = mode_ratios_unsampled[mode_ratios_inds]
        else:
            mode_ratios_qm = cs_instance.mode_ratios_qm_subbank.copy()

        if len(mode_ratios_qm) == 0:
            mode_ratios_qm = create_mode_ratios_qm_subbank(
                trig1, trig2, 2**cs_instance.max_log2n_qmc)

        times = trigs_calpha_h1[:, 0]

        dh_mtd = np.zeros((3, len(times), 2), dtype=complex)
        dh_mtd[:, :, 0] = (trigs_calpha_h1[:, trig1.rezpos:trig1.rezpos+5:2] +
                           1j*trigs_calpha_h1[:, trig1.imzpos:trig1.imzpos+5:2]).T
        dh_mtd[:, :, 1] = (trigs_calpha_l1[:, trig2.rezpos:trig2.rezpos+5:2] +
                           1j*trigs_calpha_l1[:, trig2.imzpos:trig2.imzpos+5:2]).T

        incoherent_lnprob_td = np.zeros((len(times), 2))
        L_H = np.linalg.cholesky(np.linalg.inv(CovMat_H)[::-1, ::-1])[::-1, ::-1]
        L_L = np.linalg.cholesky(np.linalg.inv(CovMat_L)[::-1, ::-1])[::-1, ::-1]
        scores_H = dh_mtd[:, :, 0].T
        scores_L = dh_mtd[:, :, 1].T
        for i in range(len(times)):
            incoherent_lnprob_td[i, 0] = np.sum(np.abs(
                                            np.dot(L_H.T, scores_H[i]))**2)/2.
            incoherent_lnprob_td[i, 1] = np.sum(np.abs(
                                            np.dot(L_L.T, scores_L[i]))**2)/2.

        # Rescaling distance factors using psd drift correction
        # and the 22 hole correction
        dist_factor_h1 = cs_instance.dist_factor_ref * trig1.normfac \
            / trig_h1[trig1.psd_drift_pos] * trig_h1[trig1.hole_correction_pos]
        dist_factor_l1 = cs_instance.dist_factor_ref * trig2.normfac \
            / trig_l1[trig2.psd_drift_pos] * trig_l1[trig2.hole_correction_pos]

        dh_mtd[:, :, 0] *= dist_factor_h1
        dh_mtd[:, :, 1] *= dist_factor_l1

        hh_md = np.zeros((6, 2), dtype=complex)
        hh_md[:, 0] = CovMat_H[np.triu_indices(3)] * dist_factor_h1**2
        hh_md[:, 1] = CovMat_L[np.triu_indices(3)] * dist_factor_l1**2
        hh_md[[1, 2, 4]] *= 2.  # convention by Javier
        if coh_score_iterations > 1:
            coherent_scores.append(
                np.average([cs_instance.lnlike_marginalized(
                    dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm)
                    for _ in range(coh_score_iterations)]))
        else:
            coherent_scores.append(cs_instance.lnlike_marginalized(
                dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm))
        if return_inputs:
            dh_mtd[:, :, 0] /= dist_factor_h1
            dh_mtd[:, :, 1] /= dist_factor_l1
            hh_md[:, 0] /= dist_factor_h1**2
            hh_md[:, 1] /= dist_factor_l1**2
            return(dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm,
                   tdiff, dist_factor_h1, dist_factor_l1)

    return np.array(coherent_scores), timeseries


def create_mode_ratios_qm_subbank_new(trigs, size):
    mode_ratios_unsampled = \
        trigs[0].templatebank.HM_amp_ratio_samples[1, :, :2].copy()
    weights = trigs[0].templatebank.HM_amp_ratio_samples[1, :, 2].copy()

    # TODO_LessImp: The detector with better normfac gets to decide the Rij samples
    # (note the Rij samples are scaled slightly based on the detector PSD)
    # In the future, one could use Rij samples from both detectors to calculate
    # the marginalized integral over mode amp ratios.
    normfacs = np.array([trig.normfac for trig in trigs])
    ibest = np.argmax(normfacs)
    mode_ratios_unsampled *= trigs[ibest].templatebank.HM_amp_ratio_PSD_factor
    weights = weights / np.sum(weights)

    np.random.seed(seed=0)
    mode_ratios_inds = np.random.choice(
        len(mode_ratios_unsampled), size=size, p=weights)
    return mode_ratios_unsampled[mode_ratios_inds]


def compute_coherent_scores_new(
        cs_instance, coincident_pclists, trig_objs,
        minimal_time_slide_jump=0.1, score_reduction_timeseries=10,
        output_timeseries=False, output_coherent_score=True,
        coh_score_iterations=1, return_inputs=False):
    """
    :param cs_instance: instance of SearchCoherentScoreHMAS
    :param coincident_pclists:
        list of coincident triggers, each entry is has n_det pclists
    :param trig_objs: Iterable with n_det trigger objects
    :param minimal_time_slide_jump: minimal time slide jump (in seconds)
    :param score_reduction_timeseries:
    :param output_timeseries:
    :param output_coherent_score:
    :param coh_score_iterations: to improve convergence, one could average
            over multiple iterations of the coherent score calculation
    :param return_inputs: for debugging purposes, return the inputs to the
            coherent score calculation (only for the first trigger in the list)
    """
    coincident_pclists = np.asarray(coincident_pclists)
    if coincident_pclists.ndim < 3:
        # We're dealing with a single event, and we were given a list of them
        new_shape = (1,) * (3 - coincident_pclists.ndim) + \
                    coincident_pclists.shape
        # n_cand x n_det x len(processedclist)
        coincident_pclists = np.reshape(coincident_pclists, new_shape)

    if not hasattr(trig_objs, '__iter__'):
        if coincident_pclists.shape[1] == 1:
            # One detector
            trig_objs = [trig_objs]
        else:
            utils.close_hdf5()
            raise ValueError('We need one trig_obj per detector')

    # Get the time resolution for the timeseries
    dt_coarse = trig_objs[0].dt
    for trig_obj in trig_objs[1:]:
        if trig_obj.dt != dt_coarse:
            utils.close_hdf5()
            raise ValueError('All trigger objects must have the same dt')
    n_interp = int(np.round(np.log2(dt_coarse / params.DT_FINAL)))
    sub_fac = 2 ** n_interp
    dt_fine = dt_coarse / sub_fac

    # Get the implied shifts for trigger times in all detectors
    # n_cand x n_det
    ts_shifted = coincident_pclists[:, :, 0].copy()
    if ts_shifted.shape[1] > 1:
        # n_cand x n_det
        shifts = utils.offset_background(
            ts_shifted - ts_shifted[:, 0][:, None],
            minimal_time_slide_jump, dt_fine)
        ts_shifted += shifts
    else:
        shifts = np.zeros_like(ts_shifted)

    # Find the 'center of mass' of the shifted trigger times in all detectors,
    # with each trigger weighted by its likelihood
    # logsumexp does this in a numerically stable manner
    # n_cand x n_det
    lnlikes = coincident_pclists[:, :, 1] / 2
    # n_cand x 1 (easy to broadcast)
    ts_center_cands_shifted = np.exp(
        logsumexp(lnlikes, axis=1, b=ts_shifted, keepdims=True) -
        logsumexp(lnlikes, axis=1, keepdims=True))
    # Let's round the centers to the nearest time bin on the fine grid,
    # since that is our resolution on the shifts
    ts_center_cands_shifted = \
        np.round(ts_center_cands_shifted / dt_fine) * dt_fine

    # For each candidate, find the centers of the time series in each detector
    # by shifting back
    # n_cand x n_det
    ts_center_cands = ts_center_cands_shifted - shifts

    timeseries = []
    coherent_scores = []
    for ind, (pclists, ts_center_cand) in \
            enumerate(zip(coincident_pclists, ts_center_cands)):
        if ind % 10 == 0:
            print('Coh score frac complete:', ind / len(coincident_pclists),
                  flush=True)

        # Compute SNR time series near the peak for each detector
        trigs_calpha_cand = []
        covmats_cand = []
        # There is a subtle bug with np.array() when the length of two
        # timeseries is indentical for all candidates, where the array shape is
        # different in an unexpected way, leave it in the form of a list of
        # tuples, and we will deal with it when saving
        timeseries_cand = ()
        for trig_obj, pclist, t_center_cand in \
                zip(trig_objs, pclists, ts_center_cand):
            tdiff = t_center_cand - pclist[0]
            dt_left = params.DT_BOUND_TIMESERIES - tdiff
            # Round to the nearest time bin on the coarse grid
            # overcomes issues with np.ceil inside gen_triggers_local()
            dt_left = np.round(dt_left / trig_obj.dt) * trig_obj.dt
            # Compute SNR time series near the peak
            trigs_calpha, covmat = trig_obj.gen_triggers_local(
                trigger=pclist,
                dt_left=dt_left,
                dt_right=2*params.DT_BOUND_TIMESERIES -dt_left,
                compute_calphas=[pclist[trig_obj.c0_pos:]],
                apply_threshold=False, relative_binning=False,
                zero_pad=False, orthogonalize_modes=False,
                return_mode_covariance=True)
            covmat = covmat[0]

            # If we want to retain the required lengths of timeseries
            # in case np.ceil expanded the range
            # mask = np.abs(trigs_calpha[:, 0] - t_center_cand) <= \
            #     params.DT_BOUND_TIMESERIES
            # trigs_calpha = trigs_calpha[mask]

            trigs_calpha_cand.append(trigs_calpha)
            covmats_cand.append(covmat)

            if output_timeseries:
                # Keep only near the peak
                # This errs on the side of keeping more triggers in
                # TODO: Hardcoded that we're using 3 modes. Fix this.
                mask = trigs_calpha[:, 1] > pclist[1] - score_reduction_timeseries
                timeseries_det = np.c_[
                    trigs_calpha[mask, 0],
                    trigs_calpha[mask, trig_obj.rezpos],
                    trigs_calpha[mask, trig_obj.imzpos],
                    trigs_calpha[mask, trig_obj.rezpos + 2],
                    trigs_calpha[mask, trig_obj.imzpos + 2],
                    trigs_calpha[mask, trig_obj.rezpos + 4],
                    trigs_calpha[mask, trig_obj.imzpos + 4]]
                timeseries_cand += (timeseries_det,)

        if output_timeseries:
            timeseries.append(timeseries_cand)

        if not output_coherent_score:
            continue

        # if this is computationally expensive
        # precompute the mode_ratios_qm for the calphas in the bank
        if trig_objs[0].templatebank.Rij_Coh_Score_NF is not None:
            # if the calpha is at the extremes of the bank, the NF may not
            # extrapolate well, so we first add the following check
            prior_NF = trig_objs[0].templatebank.Template_Prior_NF
            if (prior_NF.log_prior([pclists[0, trig_objs[0].c0_pos:]])
                    < (prior_NF.calpha_reject_threshold + 1.5)):
                mode_ratios_qm = cs_instance.mode_ratios_qm_subbank.copy()
            else:
                mode_ratios_unsampled = \
                    trig_objs[0].templatebank.Rij_Coh_Score_NF.generate_samples(
                        pclists[0, trig_objs[0].c0_pos:], set_seed=True,
                        num_samples=2 ** (cs_instance.max_log2n_qmc - 3))
                weights = mode_ratios_unsampled[:, 2]
                # Use the loudest detector to decide the mode ratios we want
                # TODO: Use the detector with the best sensitivity SNR combo
                ibest = np.argmax(pclists[:, 1])
                mode_ratios_unsampled = mode_ratios_unsampled[:, :2] * \
                    trig_objs[ibest].templatebank.HM_amp_ratio_PSD_factor
                weights /= np.sum(weights)

                np.random.seed(seed=0)
                mode_ratios_inds = np.random.choice(
                    len(mode_ratios_unsampled),
                    size=2 ** cs_instance.max_log2n_qmc, p=weights)
                mode_ratios_qm = mode_ratios_unsampled[mode_ratios_inds]
        else:
            mode_ratios_qm = cs_instance.mode_ratios_qm_subbank.copy()

        if len(mode_ratios_qm) == 0:
            mode_ratios_qm = create_mode_ratios_qm_subbank_new(
                trig_objs, 2**cs_instance.max_log2n_qmc)

        # Assemble the data for the coherent score calculation
        times = trigs_calpha_cand[0][:, 0]
        dh_mtd = np.zeros((3, len(times), len(trig_objs)), dtype=complex)
        incoherent_lnprob_td = np.zeros((len(times), len(trig_objs)))
        hh_md = np.zeros((6, len(trig_objs)), dtype=complex)
        dist_factors = np.zeros(len(trig_objs))
        # Loop over detectors
        for i, (trigs_calpha, trig_obj, covmat) in \
                enumerate(zip(trigs_calpha_cand, trig_objs, covmats_cand)):
            dh_mtd[:, :, i] = (
                trigs_calpha[:, trig_obj.rezpos:trig_obj.rezpos + 5:2]
                + 1j * trigs_calpha[:, trig_obj.imzpos:trig_obj.imzpos + 5:2]).T

            L = np.linalg.cholesky(
                np.linalg.inv(covmat)[::-1, ::-1])[::-1, ::-1]
            scores = dh_mtd[:, :, i].T
            for j in range(len(times)):
                incoherent_lnprob_td[j, i] = np.sum(
                    np.abs(np.dot(L.T, scores[j]))**2) / 2.

            # Rescaling distance factors using psd drift correction
            # and the 22 hole correction
            dist_factor = cs_instance.dist_factor_ref * trig_obj.normfac \
                / pclist[trig_obj.psd_drift_pos] \
                * pclist[trig_obj.hole_correction_pos]

            dh_mtd[:, :, i] *= dist_factor
            hh_md[:, i] = covmat[np.triu_indices(3)] * dist_factor**2
            dist_factors[i] = dist_factor

        hh_md[[1, 2, 4]] *= 2.  # convention by Javier
        if coh_score_iterations > 1:
            coherent_scores.append(
                np.average([
                    cs_instance.lnlike_marginalized(
                        dh_mtd, hh_md, times, incoherent_lnprob_td,
                        mode_ratios_qm) for _ in range(coh_score_iterations)]))
        else:
            coherent_scores.append(
                cs_instance.lnlike_marginalized(
                    dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm))

        if return_inputs:
            dh_mtd /= dist_factors
            hh_md /= dist_factors**2
            return dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm, \
                ts_shifted[0, 1:] - ts_shifted[0, 0], *dist_factors

    return np.array(coherent_scores), timeseries
