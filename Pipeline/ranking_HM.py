import numpy as np
import triggers_single_detector_HM as trig
import utils
import os, glob, shutil, copy
import coincidence_HM as coin
import coherent_score_mz_fast as cs_mz
import coherent_score_hm_search as cs_JR
import json
from scipy.interpolate import interp1d
import dill as pickle
from numba import njit, vectorize
import template_bank_generator_HM as tg
import params
import scipy
import scipy.stats as ss
import h5py
import ast
from functools import partial
import itertools
import pathlib

try: import ranking as rank22
except: pass

# Note: For running this code, refer to the notebook
# 5.Ranking_candidates.ipynb

# Injection details
INJECTION_STARTS, INJECTION_ENDS, *_ = utils.get_injection_details()
LSC_EVENT_TIMES = utils.get_lsc_event_times()

# HDF5 variables
# Keys for objects in bg_fg_by_subbank
FOBJ_KEYS = ['background', 'zero-lag', 'LVC', 'injections']
# Variable length datatype for timeseries
VLEN_TIMESERIES = h5py.special_dtype(vlen=np.dtype(np.float64))

# %% General functions
def correct_snr2sumcut_safe(snr2sumcut_safe, snr2sumcut, snr2min, max_rhosq_limit):
    if (snr2sumcut_safe - snr2min) > max_rhosq_limit:
        dsnr2_safety = snr2sumcut_safe - snr2sumcut
        snr2sumcut_safe = snr2min + max_rhosq_limit - dsnr2_safety
        if snr2sumcut_safe < snr2sumcut:
            snr2sumcut_safe = snr2sumcut
            if (snr2sumcut_safe - snr2min) > max_rhosq_limit:
                raise ValueError("Check if there are there enough samples " +
                                 "to fit a distribution to")

    return snr2sumcut_safe

# %% Functions to veto triggers
def veto_subthreshold_top_cands(
        rank_obj, ncores=1, num_redo_bkg=5000, triggers_dir=None,
        rerun_coh_score=True, fname_to_save=None, score_final_object=True,
        **ranking_kwargs):
    """
    Top candidates in the cands_preveto_max list which are subthreshold
    (i.e. have SNR below min_veto_chi2)
    are passed through all the veto tests again
    
    :param rank_obj: Individual Rank object (not a list)
    :param ncores:
    :param num_redo_bkg:
        How many background triggers (ranked by IFAR starting from the highest)
        to redo vetos for
    :param triggers_dir: Directory where the triggers are stored
        e.g., '/data/jayw/IAS/GW/Data/HM_O3b_search/OutputDir/'
    :param rerun_coh_score:
        Flag whether to rerun the coherent score for all the top candidates
        (not just the subthreshold ones).
        The reason to do this is that coherent score is now calculated at a
        higher resolution compared the version in coincidence_HM.py
        (the coh score is now averaged over 10 iterations and calculated with
        slightly larger log2n_qmc and nphi)
    :param fname_to_save:
        If known, filename to save the Rank object with the new vetoes to.
        If None, we apply the vetoes in place, but don't save the object
    :param score_final_object: Flag whether to score the final object
    :param ranking_kwargs:
        Any extra arguments to pass to the ranking function. Used only if
        score_final_object is True

    Below is an example script for running this code on a cluster:
        salloc --nodes=1 --cpus-per-task=24 --time=12:00:00 --mem-per-cpu=4GB
        python
        import sys
        sys.path.insert(0,".../gw_detection_ias")
        import ranking_HM as rank
        # Need to load in the mode r+ as we will modify veto metadatas of triggers
        rank_obj = rank.Rank.from_hdf5('--.hdf5', mode="r+")
        rank.veto_subthreshold_top_cands(rank_obj, ncores=24, num_redo_bkg=5000,
            triggers_dir='--', fname_to_save='---.hdf5')
        # Safely close the file
        rank_obj.fobj.close()
    """
    # Initialize the saved object and load the necessary modules
    # ----------------------------------------------------------
    # Record the maximization options
    new_maxopts = locals().copy()
    options_to_remove = \
        ['rank_obj', 'triggers_dir', 'ncores', 'fname_to_save',
         'score_final_object']
    for option in options_to_remove:
        _ = new_maxopts.pop(option)

    mp = utils.load_module('multiprocess')

    # Define some parameters that the veto function needs
    # -----------------------------------------------------
    veto_params = dict()
    veto_params['triggers_dir'] = triggers_dir
    veto_params['rerun_coh_score'] = rerun_coh_score

    # Index of veto_metadata in scores_(non)vetoed_max after accounting for
    # the removal of prior terms (added in case we change order in the future)
    idx_veto_metadata = utils.index_after_removal(
        rank_obj.extra_array_names, 'veto_metadata', 'prior_terms')
    if idx_veto_metadata is None:
        print("We don't have veto_metadata in the ranking object")
        return
    # The preceding elements in an entry of scores_(non)vetoed_max are
    # prior terms, processed_clist, loc_id, sensitivity params
    idx_veto_metadata += 4
    veto_params['idx_veto_metadata'] = idx_veto_metadata

    # List of vetoes we want to apply to label a trigger as passing
    veto_inds = np.where(rank_obj.mask_veto_criteria)[0]
    veto_params['veto_inds'] = veto_inds

    # Read the minimum veto chi2 value from the first subbank
    cand_collection_dics = [json.load(
        open(os.path.join(x[0], 'coincidence_parameters_H1_L1.json'), 'r'))
        for x in rank_obj.cand_dirs_mcbin]
    min_veto_chi2 = cand_collection_dics[0]['min_veto_chi2']
    veto_params['min_veto_chi2'] = min_veto_chi2

    # Read the source class and runs
    veto_params['source'] = rank_obj.source
    veto_params['runs'] = rank_obj.runs
    veto_params['detectors'] = rank_obj.detectors

    if rerun_coh_score:
        # Create cs_instance for the coherent score
        # To be explicit, we should pass these parameters in, but for now we'll
        # do the same hack as for the num_redos
        n_qmc_sequences = ranking_kwargs.pop('n_qmc_sequences', 100)
        log2n_qmc = ranking_kwargs.pop('log2n_qmc', params.LOG2N_QMC + 2)
        nphi = ranking_kwargs.pop('nphi', params.NPHI * 2)
        new_maxopts['ranking_kwargs'].update(
            {'n_qmc_sequences': n_qmc_sequences,
             'log2n_qmc': log2n_qmc,
             'nphi': nphi})
        cs_instance = rank_obj.create_coh_score_instance(
            n_qmc_sequences=n_qmc_sequences, log2n_qmc=log2n_qmc, nphi=nphi,
            **rank_obj.cs_kwargs)
        veto_params['cs_instance'] = cs_instance

        coh_score_iterations = ranking_kwargs.pop('coh_score_iterations', 10)
        new_maxopts['ranking_kwargs'].update(
            {'coh_score_iterations': coh_score_iterations})
        veto_params['coh_score_iterations'] = coh_score_iterations
        veto_params['time_slide_jump'] = rank_obj.time_slide_jump

    # Define the set of triggers to revisit
    # -------------------------------------
    # We want to revisit triggers that
    # (a) previously passed vetoes
    # (b) have high values of the ranking statistic
    # Mask into subset that satisfy condition (a)
    # Index into bg_fg_by_subbank entries
    idx_veto_metadata_bg_fg_by_subbank = utils.index_after_removal(
        rank_obj.extra_array_names, 'veto_metadata')
    if idx_veto_metadata_bg_fg_by_subbank is not None:
        idx_veto_metadata_bg_fg_by_subbank += 1
    _, veto_masks, _ = rank_obj.compute_function_on_subbank_data(
        'veto_metadata', idx_veto_metadata_bg_fg_by_subbank, True,
        lambda arr: np.all(arr[..., veto_inds], axis=(1, 2)))

    # Compile the ranking statistics we have - these were computed on
    # scores_(non)vetoed_max if old_include_vetoed_triggers was True(False)
    ranking_stat_lists = [
        rank_obj.coherent_scores_bg + rank_obj.back_0_score + rank_obj.back_1_score,
        rank_obj.coherent_scores_cand + rank_obj.cand_0_score + rank_obj.cand_1_score,
        rank_obj.coherent_scores_lsc + rank_obj.lsc_0_score + rank_obj.lsc_1_score,
        rank_obj.coherent_scores_inj + rank_obj.inj_0_score + rank_obj.inj_1_score]
    ranking_stat_lists = [
        np.nan_to_num(x, copy=False, nan=-1e5) for x in ranking_stat_lists]

    # The value of include_vetoed_triggers we previously used for the ranking
    old_include_vetoed_triggers = False
    for maxopt in rank_obj.maxopts[::-1]:
        old_flag = maxopt.get('include_vetoed_triggers', None)
        if old_flag is None:
            old_flag = maxopt.get(
                'ranking_kwargs', {}).get('include_vetoed_triggers', None)
        if old_flag is not None:
            old_include_vetoed_triggers = old_flag
            break

    # TODO: Teja: This is not a consistent procedure
    #  We should have background num = foreground num x Nsim for redoing vetos
    num_redo_bkg = num_redo_bkg
    # For completeness, load and save these numbers in a reusable manner
    num_redo_foreground = ranking_kwargs.pop('num_redo_foreground', 200)
    num_redo_LVC = ranking_kwargs.pop('num_redo_LVC', 100)
    num_redo_inj = ranking_kwargs.pop('num_redo_inj', 100)
    new_maxopts['ranking_kwargs'].update(
        {'num_redo_foreground': num_redo_foreground,
         'num_redo_LVC': num_redo_LVC,
         'num_redo_inj': num_redo_inj})
    nums_redo = [num_redo_bkg, num_redo_foreground, num_redo_LVC, num_redo_inj]

    inds_unvetoed_lists = []
    new_veto_metadatas_lists = []
    new_coherent_scores_lists = []
    for cand_type, (veto_mask, ranking_stat_list, num_redo) in \
            enumerate(zip(veto_masks, ranking_stat_lists, nums_redo)):
        print(f'Running for {FOBJ_KEYS[cand_type]} triggers', flush=True)
        if not np.any(veto_mask):
            inds_unvetoed_lists.append([])
            new_veto_metadatas_lists.append([])
            new_coherent_scores_lists.append([])
            continue

        with mp.Pool(ncores) as p:
            # Define the chunks to process
            # Indices of triggers in cands_preveto_max that passed the vetoes
            inds_unvetoed = np.where(veto_mask)[0]
            # Pick the top num_redo triggers within these triggers
            if old_include_vetoed_triggers:
                # The ranking statistics were for all triggers and not just the
                # ones that passed the vetoes, pick out the latter subset
                ranking_stat_list = ranking_stat_list[inds_unvetoed]
            inds_unvetoed = \
                inds_unvetoed[np.argsort(ranking_stat_list)[-num_redo:]]

            rank_entry_list = copy.deepcopy([rank_obj.cands_preveto_max[cand_type][ind]
                               for ind in inds_unvetoed])
            inds_entry_list = np.arange(len(inds_unvetoed))
            rank_entry_chunks = [rank_entry_list[i:i + ncores * 16] for i in
                                 range(len(rank_entry_list))[::ncores * 16]]
            inds_entry_chunks = [inds_entry_list[i:i + ncores * 16] for i in
                                 range(len(inds_entry_list))[::ncores * 16]]

            new_veto_metadatas = []
            new_coherent_scores = []
            for ichunk, (rank_entry_chunk, inds_entry_chunk) in \
                    enumerate(zip(rank_entry_chunks, inds_entry_chunks)):
                results_chunk = \
                    p.map_async(
                        partial(veto_rank_entry_wrapper, veto_params=veto_params),
                        zip(rank_entry_chunk, inds_entry_chunk), chunksize=16)
                utils.track_job(
                    results_chunk, 'veto_rank_entry', 16 * ncores,
                    n_tasks_prev=ichunk * 16 * ncores,
                    n_tasks_tot=len(rank_entry_list), update_interval=30)
                results_chunk = results_chunk.get()
                for new_veto_metadata, new_coherent_score in results_chunk:
                    new_veto_metadatas.append(new_veto_metadata)
                    new_coherent_scores.append(new_coherent_score)

            inds_unvetoed_lists.append(inds_unvetoed)
            new_veto_metadatas_lists.append(new_veto_metadatas)
            new_coherent_scores_lists.append(new_coherent_scores)

    # If saving to a new file, create it and work with the new file instead
    if fname_to_save is not None:
        rank_obj.to_hdf5(path=fname_to_save, overwrite=True)
        os.chmod(fname_to_save, 0o755)
        rank_obj = Rank.from_hdf5(fname_to_save)

    # Write the new vetoes and coherent scores back into the rank object
    for cand_type, (inds_unvetoed_list,
                    new_veto_metadatas_list,
                    new_coherent_scores_list) in enumerate(
            zip(inds_unvetoed_lists,
                new_veto_metadatas_lists,
                new_coherent_scores_lists)):
        for ind_unvetoed, new_veto_metadata, new_coherent_score in zip(
                inds_unvetoed_list,
                new_veto_metadatas_list,
                new_coherent_scores_list):
            rank_obj.cands_preveto_max[
                cand_type][ind_unvetoed][idx_veto_metadata][...] = \
                new_veto_metadata
            rank_obj.cands_preveto_max[cand_type][ind_unvetoed][0][0] = \
                new_coherent_score

    if score_final_object:
        if ranking_kwargs.get('apply_veto_before_scoring', False):
            print("We'll apply vetoes since we already shuffled between banks")
        ranking_kwargs['apply_veto_before_scoring'] = True
        rank_obj.score_bg_fg_lists(**ranking_kwargs)

    # Before saving, add the new maxopts to the existing ones
    rank_obj.maxopts = list(rank_obj.maxopts)
    rank_obj.maxopts.append(new_maxopts)

    if fname_to_save is not None:
        rank_obj.to_hdf5(path=fname_to_save, overwrite=True)
        # Having saved it, close it to avoid memory leaks
        if rank_obj.fobj is not None:
            rank_obj.fobj.close()

    return


def veto_rank_entry_wrapper(args, veto_params):
    rank_entry, ind = args
    return veto_rank_entry(rank_entry, ind, **veto_params)

def veto_rank_entry(
        rank_entry, ind, triggers_dir=None, source='BBH', 
        runs=('hm_o3a', 'hm_o3b'), min_veto_chi2=30,
          idx_veto_metadata=-2, detectors=('H1', 'L1'),
        veto_inds=None, rerun_coh_score=False, cs_instance=None,
        time_slide_jump=params.DEFAULT_TIMESLIDE_JUMP/1000.,
        coh_score_iterations=10, return_trig_obj_info=False):
    """
    :param rank_entry: Entry of one of the arrays in scores_(non)vetoed_max
    :param ind:
        Index into cands_preveto_max[cand_type], only used to debug which
        trigger failed
    :param triggers_dir:
        Directory where the triggers are stored, exposed for debugging in case
        we're using a non-standard directory
    :param source: Should be one of 'BBH', 'BNS', or 'NSBH'
    :param runs: Iterable with runs being searched in
    :param min_veto_chi2: Minimum chi2 value to veto
    :param idx_veto_metadata: Index of veto_metadata in the rank_entry
    :param detectors:
        Iterable of detectors, should match the order in the processedclists
    :param veto_inds: Indices of vetoes to apply (only used for check)
    :param rerun_coh_score:
        Flag to rerun the coherent score computation
        (only for triggers that passed the vetoes)
    :param cs_instance: Coherent score instance, used only if rerun_coh_score
    :param time_slide_jump: The least count of the timeslides
    :param coh_score_iterations: Number of iterations for the coherent score
    :param return_trig_obj_info:
        Flag to return the trigger objects (only used for debugging)
    :return: New veto metadata and coherent score
    """
    pclists = rank_entry[1][...][()]
    loc_id = rank_entry[2]
    new_veto_metadata = rank_entry[idx_veto_metadata][...][()].copy()

    rerun_vetoes = True
    if np.all(pclists[:, 1] > min_veto_chi2) and not return_trig_obj_info:
        rerun_vetoes = False

    if not (rerun_vetoes or rerun_coh_score):
        # We already vetoed the trigger and are happy with the coherent score
        return new_veto_metadata, rank_entry[0][0]

    # We need to rerun the vetoes and/or the coherent score
    # Define some useful common variables, and load the trig objects
    # Get the directory with the json files for the run, source, and loc_id
    run = None
    for run_to_use in runs:
        if utils.is_in_run(pclists[0, 0], run_to_use):
            # We'll miss an event that spans across runs :)
            run = run_to_use
            break

    def handle_failure():
        if return_trig_obj_info:
            return new_veto_metadata, rank_entry[0][0], \
                *[None] * (2 * len(detectors) + 1)
        else:
            return new_veto_metadata, rank_entry[0][0]

    if run is None:
        # We couldn't find the run, return gracefully
        print(f"Couldn't find the run in {runs} for trigger at " +
              f"{pclists[0, 0]}, loc_id: {loc_id}")
        return handle_failure()

    if triggers_dir is None:
        dname = utils.get_dirs(
            dtype='trigs', source=source, runs=[run])[0][loc_id[0]][loc_id[1]]
    else:
        prefix_dict = utils.SOURCE_TO_PREFIX.get(source.lower(), None)
        if prefix_dict is None:
            print(f"Could not recognize source {source}!")
            return handle_failure()
        prefix = prefix_dict.get(run.lower(), None)
        if prefix is None:
            print(f"Could not find prefix for {source} in {run}!")
            return handle_failure()

        dname = utils.create_chirp_mass_directory_dict(
            triggers_dir, prefix, '')[loc_id[0]][loc_id[1]]

    # To return in case return_trig_obj_info == True, or useful for the
    # coherent score recomputation
    trig_objs = []
    veto_spacings = []
    for i, detector in enumerate(detectors):
        # metadata = np.ones(26 + 6 + len(params.SPLIT_CHUNKS), dtype=bool)
        fpath = utils.get_detector_fnames(
            pclists[i, 0], dname=dname, detectors=(detector,),
            return_only_existing=False)[0]
        flag = os.path.exists(fpath)
        trig_obj = None
        if flag:
            trig_obj = trig.TriggerList.from_json(fpath, load_trigs=False)
        else:
            # It could have been pulled in from the left or right
            left_right_fnames = utils.get_left_right_fnames(fpath)
            for fname in left_right_fnames:
                if fname is not None:
                    trig_obj = trig.TriggerList.from_json(
                        fname, load_trigs=False)
                    if ((pclists[i, 0] > trig_obj.time[0]) and
                            (pclists[i, 0] < trig_obj.time[-1])):
                        flag = True
                        break
            if not flag:
                # Exit gracefully
                print(f"Event is outside files for trigger at " +
                      f"{pclists[i, 0]}, loc_id: {loc_id}")
                return handle_failure()

        trig_objs.append(trig_obj)

        if pclists[i, 1] <= min_veto_chi2:
            # We didn't veto this trigger earlier, let's do it now
            # Redo the steps in coincidence_HM.py
            _, dcalphas_veto = trig_obj.define_finer_grid_func(
                    dcalpha_coarse=trig_obj.delta_calpha / 2, trim_dims=False)
            veto_spacing = dcalphas_veto.copy()

            # Half-width of bank in dimensions
            extent = (trig_obj.templatebank.bounds[:len(dcalphas_veto), 2] -
                      trig_obj.templatebank.bounds[:len(dcalphas_veto), 0]) / 2
            extent *= trig_obj.template_safety
            snr_trig = np.sqrt(pclists[i, 1])
            veto_spacing[
                np.logical_and(
                    extent > 1/snr_trig,
                    1/snr_trig > veto_spacing)] = 1/snr_trig
            veto_spacings.append(veto_spacing)

            nsimple = len(trig_obj.outlier_reasons) + 6 + \
                len(params.SPLIT_CHUNKS)
            nstringent = 4 + len(params.SPLIT_CHUNKS)
            # Populate simple vetoes
            # We can trust the metadata at place 0, 1, and nsimple - 1,
            # as those entries are CBC_CAT flags and sinc_interpolation
            # failures that were populated for everyone
            new_veto_metadata[i, 2:nsimple-1] = trig_obj.veto_trigger_all(
                pclists[i], dcalphas=veto_spacing, subset_details=None,
                lazy=True)[2]
            # Perform stringent vetoes
            new_veto_metadata[i, nsimple:nsimple + nstringent] = \
                coin.stringent_veto(
                    pclists[None, :], i, trig_obj, min_veto_snr2=None,
                    group_duration=time_slide_jump, origin=0, n_cores=1)[1][0]
            # We can trust the metadata for vetoes at other places, as
            # those are found another louder trigger in the same
            # time-shift-tol window, secondary peak, and trigger below the
            # coincidence cut which were done for everyone

    if veto_inds is None:
        veto_inds = np.arange(len(new_veto_metadata[0]))

    # If it failed the vetoes, or we're not rerunning the coherent score, return
    if not (bool(apply_vetos(new_veto_metadata, veto_inds))
            or rerun_coh_score):
        if return_trig_obj_info:
            return new_veto_metadata, rank_entry[0][0], *trig_objs, *pclists, \
                veto_spacings
        else:
            return new_veto_metadata, rank_entry[0][0]

    # We're recomputing the coherent score
    new_coherent_score = 2 * cs_JR.compute_coherent_scores_new(
        cs_instance, np.array([pclists]), trig_objs,
        minimal_time_slide_jump=time_slide_jump,
        coh_score_iterations=coh_score_iterations)[0][0]

    diff = new_coherent_score - rank_entry[0][0]
    if abs(diff) > 3:
        print(f'New CS, increase/decrease: {new_coherent_score, diff}, for ind:{ind}')

    if return_trig_obj_info:
        return new_veto_metadata, new_coherent_score, *trig_objs, *pclists, \
            veto_spacings
    else:
        return new_veto_metadata, new_coherent_score

def parabolic_func(x, a_fit):
        '''
        Define a parabolic function to fit
        downsampling corrections
        '''
        return a_fit*x**2

def generate_downsampling_correction_fits(Z_gauss_complex=None, bank_id=None, tbp_dir=None,
             bank_obj=None, n_SNRsq_bins=15, SNRsq_cutoff=20, return_plot=False):
    '''
    We want to calculate the non-Gaussian correction
    to the ranking statistic (see arXiv: 2405.17400).
    We therefore want to compare the empirical histogram of the SNR^2
    to that expected from the Gaussian noise hypothesis (see Fig.3 of the paper).
    The empirical histogram is unfortunately also affected by downsampling
    at the low SNR^2 end due to using HM marginalized statistic based threshold
    in the matched_filtering stage (instead of using rho_incoherent^2 as the threshold).
    To remedy this, we do a quick and rough calculation of the downsampling correction.
    and fit it by a parabolic function to avoid noise at the high SNR^2 end.
    The fit is then used to correct the empirical histogram or rank function.
    :param Z_gauss_complex: Complex Gaussian noise triggers [n_triggers,3]
        They take time to generate, so try generate in a notebook once and
        supply them for all banks
    :param bank_id: The bank id of the bank to be used for the correction
                    (if bank_obj is not provided)
    :param tbp_dir: The directory where the template banks are stored 
                    (you could load temp bank params obj and use tbp.DIR)
    :param bank_obj: The bank object to be used for the correction
    :param n_SNRsq_bins: Number of bins to use for the histogram
                        (the histogram can be noisy if the number of bins is too large)
    :param SNRsq_cutoff: The cutoff SNR^2 value above which the triggers are considered.
                        This parameter needs further investigation,
                        Either provide the lowest SNR^2 cutoff among all banks used
                        during triggering (recommended),
                        or provide a median value across all banks.
    :param return_plot: Whether to return the plot (for debugging)
    '''
    _, ax = utils.import_matplotlib(figsize=(7,4))
    if Z_gauss_complex is None:
        Z_gauss_complex = np.empty((0,6), dtype=float)
        for i in range(100):
            Z_sample = np.random.normal(size=(1000000,6))
            Z_gauss_complex = np.r_[Z_gauss_complex,
                    Z_sample[np.linalg.norm(Z_sample, axis=1)**2>SNRsq_cutoff]]
        Z_gauss_complex = Z_gauss_complex[:,0::2] + 1j*Z_gauss_complex[:,1::2]
        print(f'{Z_gauss_complex.shape} triggers were generated above the cutoff SNR^2 value')
    if bank_obj is None:
        if bank_id is None:
            raise ValueError('Either bank_id or bank_obj must be provided')
        bank_obj = tg.TemplateBank.from_json(os.path.join(
            tbp_dir,f'BBH_{bank_id[0]}/bank_{bank_id[1]}/metadata.json'))
        bank_obj.set_waveform_conditioning(2**18, 1/2048)
    Z_gaussian = copy.deepcopy(Z_gauss_complex)
    Z_gaussian = Z_gaussian[np.linalg.norm(Z_gaussian, axis=1)**2>SNRsq_cutoff]
    Z_marg = bank_obj.marginalized_HM_scores(Z_gaussian, input_Z=True)
    Z_gaussian = np.linalg.norm(Z_gaussian, axis=1)**2
    Z_marg = Z_gaussian[Z_marg>SNRsq_cutoff]
    val_gauss,bins,_ = ax.hist(Z_gaussian, bins=n_SNRsq_bins, histtype='step', cumulative=-1,log=True,
                                label='Gaussian noise triggers');
    val_marg,_,_ = ax.hist(Z_marg, bins=bins, histtype='step', cumulative=-1,log=True,
                            label='Using marginalized HM statistic');
    bincent = utils.bincent(bins)
    if return_plot:
      chisq_values = (1-ss.chi2.cdf(bincent,6))/(1-ss.chi2.cdf(bincent[10],6))*val_gauss[10]
      ax.plot(bincent,chisq_values,
            ls='--', label=r'$\chi^2$ (6 d.o.f)')
    rank_corr = 2*np.log(val_marg/val_gauss)
    rank_corr = np.nan_to_num(rank_corr, nan=0, posinf=0, neginf=0)

    # we require the correction to be zero at the last bin
    xarr = bincent - bincent[-1]
    popt, pcov = scipy.optimize.curve_fit(parabolic_func, xarr, rank_corr, sigma=10/np.sqrt(val_marg))
    a_fit = popt[0]
    # We fit a parabolic function to the difference between the two cumulative histograms
    rank_fn_difference = parabolic_func(xarr, a_fit)
    if return_plot:
        ax.plot(bincent, chisq_values*np.exp(rank_fn_difference/2),ls='--',
          label=r'$\chi^2 * \mathrm{Exp[}-\frac{1}{2}a_\mathrm{fit}(\rho^2-\rho^2_0)^2] $')
        ax.set_xlabel(r'$\rho^2\ \ (=|\rho^\perp_{22}|^2+|\rho^\perp_{33}|^2+|\rho^\perp_{44}|^2)$');
        ax.set_ylabel('Cumulative counts (right to left)');
        ax.legend()
    return([a_fit, bincent[-1]])

@njit
def apply_vetos(veto_metadata, inds_test):
    flag = True
    for ind_det in range(len(veto_metadata)):
        for ind_test in inds_test:
            flag *= veto_metadata[ind_det, ind_test]
    return flag


@njit
def apply_vetos_det(veto_metadata, inds_test, ind_det):
    flag = True
    for ind_test in inds_test:
        flag *= veto_metadata[ind_det, ind_test]
    return flag


# %% Functions to populate bg and fg lists
def collect_all_subbanks(
        cand_dirs, chirp_mass_id, subbank_subset=None, snr2min=None,
        snr2min_marg=None, snr2sumcut=None, max_time_slide_shift=None,
        minimal_time_slide_jump=0.1, max_zero_lag_delay=0.015,
        score_reduction_max=5, collect_rerun=False, collect_before_veto=True,
        collect_timeseries=False, coinc_ftype="npz", detectors=('H1', 'L1'),
        ncores=1, trigger_objs=None):
    # cand_dirs is nrun x n_subbank
    if subbank_subset is None:
        subbank_subset = np.arange(len(cand_dirs[0]))

    # Four elements for pure background, our events, LSC events, injected events
    # Each dict is indexed by (bank_id, subbank_id)
    bg_fg_by_subbank = [{}, {}, {}, {}]

    # Ensure return value is defined
    veto_metadata_keys = None
    extra_array_names = []

    # Do it run by run
    for cand_dirs_run in cand_dirs:
        for ind, (subbank_id, cand_dir) in \
                enumerate(zip(subbank_subset, cand_dirs_run)):
            print("Initializing subbank:", subbank_id)

            loc_id = (chirp_mass_id, subbank_id)
            trig_obj = trigger_objs[ind]

            # Collect all vetoed/nonvetoed as desired and optimized events
            if coinc_ftype.lower() == "npz":
                # Assume we always collect coherent scores with npz!
                zz = coin.collect_all_candidate_files_npz(
                        cand_dir,
                        collect_before_veto=collect_before_veto,
                        collect_rerun=collect_rerun,
                        collect_timeseries=True,
                        detectors=detectors,
                        ncores=ncores)
                events, veto_metadata, coherent_scores, veto_metadata_keys, \
                    timeseries = zz
                # JR coherent score gives ln Z but MZ coherent score gave
                # 2 * ln Z and the current code uses the MZ format
                coherent_scores *= 2
            else:
                utils.close_hdf5()
                raise NotImplementedError
            
            # Enforce any desired additional cuts
            if snr2min is None:
                snr2min = 0
            if snr2sumcut is None:
                snr2sumcut = 0

            # TODO: Ensure these are done in injection_loader_HM.py
            # Apply PSD drift correction and coincident SNR cuts
            # n_events x n_det
            masks_psd_drift = events[:, :, trig_obj.psd_drift_pos] >= 0.5
            # n_events
            mask_coinc = utils.incoherent_score(events) >= snr2sumcut
            ndet = masks_psd_drift.shape[1]
            # n_events x n_det x 2
            threshold_vetos = np.stack([
                masks_psd_drift, np.tile(mask_coinc, (ndet, 1)).T], axis=-1)
            threshold_veto_keys = ['Big PSD drift correction', 'Coincident SNR']

            if snr2min_marg is not None:
                # Cut on the SNR (marginalized over HM amplitudes and phases)
                masks_marginalized_snr = \
                    np.zeros((len(events), ndet), dtype=bool)
                for idet in range(ndet):
                    zsq = trig_obj.templatebank.marginalized_HM_scores(
                        events[:, idet], marginalized=True)
                    masks_marginalized_snr[:, idet] = zsq >= snr2min_marg

                # n_events x n_det x 3
                threshold_vetos = np.append(
                    threshold_vetos, masks_marginalized_snr[..., None], axis=-1)
                threshold_veto_keys.append('Marginalized SNR')

            veto_metadata = np.append(veto_metadata, threshold_vetos, axis=-1)
            veto_metadata_keys = np.append(
                veto_metadata_keys, threshold_veto_keys)

            # # Add the secondary peak reject veto if we didn't apply it earlier
            # if 'Secondary_peak_timeseries' not in veto_metadata_keys:
            #     # We only use the 22 timeseries info for the secondary peak vetoveto_timeseries = np.array([
            #     veto_timeseries = np.array([
            #         [[not coin.secondary_peak_reject(
            #             events[i][0], timeseries[i][0],
            #             score_reduction_max=score_reduction_max)],
            #          [not coin.secondary_peak_reject(
            #             events[i][1], timeseries[i][1],
            #             score_reduction_max=score_reduction_max)]]
            #         for i in range(len(events))])
            #     veto_metadata = np.append(veto_metadata, veto_timeseries, axis=-1)
            #     veto_metadata_keys = np.append(
            #         veto_metadata_keys, 'Secondary_peak_timeseries')

            if not collect_timeseries:
                timeseries = None

            # (Re)apply the timeslides, in case we are demanding a different
            # max_time_slide_shift from what was collected
            events, (timeseries, coherent_scores, veto_metadata) = \
                coin.create_shifted_observations(
                    events, max_time_slide_shift, minimal_time_slide_jump,
                    max_zero_lag_delay, timeseries, coherent_scores, veto_metadata)

            # Make lists with vetoed and optimized events for the subbank
            add_to_bg_and_fg_dicts(
                events, loc_id, bg_fg_by_subbank,
                extra_arrays=[timeseries, veto_metadata, coherent_scores])
            # Record the names of the extra arrays we saved
            extra_array_names = []
            if timeseries is not None:
                extra_array_names.append('timeseries')
            if veto_metadata is not None:
                extra_array_names.append('veto_metadata')
            if coherent_scores is not None:
                extra_array_names.append('prior_terms')

    # print('Combining subbanks together and maximizing')
    # Avoid maximizing over subbanks, we'll do that later
    bg_fg_max = []
    for cand_dict in bg_fg_by_subbank:
        bg_fg_max.append(combine_subbank_dict(cand_dict))

    return bg_fg_max, bg_fg_by_subbank, veto_metadata_keys, extra_array_names


def add_to_bg_and_fg_dicts(
        events, loc_id, bg_fg_dicts, extra_arrays=(), separate_injections=True,
        separate_lsc_events=True, inj_ends=INJECTION_ENDS,
        inj_starts=INJECTION_STARTS, eps=4, output_inj_mask=False):
    """
    Populates events in lists of pure background, zero-lag candidates,
    LSC events, and injected events
    :param events: Array of shape n_events x (n_det=2) x row of processedclist
    :param loc_id: Tuple with (chirp_mass_id, subbank ID)
    :param bg_fg_dicts:
        List of dicts with pure background, zero-lag candidates,
         (and separate LSC events and injected events, if needed)
    :param extra_arrays:
        If known, iterable with extra arrays for the events
        (timeseries, veto_metadata, coherent scores)
    :param separate_injections: Flag to collect injections separately
    :param separate_lsc_events: Flag to collect lsc events separately
    :param inj_ends: Array with end times of injections
    :param inj_starts: Array with start times of injections
    :param eps: Tolerance for checking if an event is close to an injection
    :return:
        Adds an entry to each dict in bg_fg_dicts with key = loc_id, and
        item = events, or (events, timeseries) if we have timeseries
    """
    times_h1 = events[:, 0, 0]
    times_l1 = events[:, 1, 0]

    # Mask into events associated with coincident events
    coincident_mask = np.abs(times_h1 - times_l1) <= 0.015

    # Mask into events associated with LSC events or injections
    loud_mask = np.zeros(len(times_h1), dtype=bool)
    if separate_injections:
        inj_mask = np.logical_or(
            utils.is_close_to(inj_ends, times_h1, t_1_start=inj_starts, eps=eps),
            utils.is_close_to(inj_ends, times_l1, t_1_start=inj_starts, eps=eps))
        loud_mask = np.logical_or(loud_mask, inj_mask)
    else:
        inj_mask = None

    if separate_lsc_events:
        lsc_mask = np.logical_or(
            utils.is_close_to(LSC_EVENT_TIMES, times_h1, eps=eps),
            utils.is_close_to(LSC_EVENT_TIMES, times_l1, eps=eps))
        loud_mask = np.logical_or(loud_mask, lsc_mask)
    else:
        lsc_mask = None

    # Pure background not associated with loud places
    # -----------------------------------------------
    mask_bg = np.logical_not(np.logical_or(coincident_mask, loud_mask))
    bg_non_loud = events[mask_bg]
    if extra_arrays is not None:
        extra_arrays_bg = [pick_mask(x, mask_bg) for x in extra_arrays]
    else:
        extra_arrays_bg = []
    add_or_vstack_to_dic(
        bg_fg_dicts[0], loc_id, bg_non_loud, extra_arrays=extra_arrays_bg)

    # Zero-lag events
    # ---------------
    # Our events
    mask_fg = np.logical_and(coincident_mask, np.logical_not(loud_mask))
    our_events = events[mask_fg]
    if extra_arrays is not None:
        extra_arrays_fg = [pick_mask(x, mask_fg) for x in extra_arrays]
    else:
        extra_arrays_fg = []
    add_or_vstack_to_dic(
        bg_fg_dicts[1], loc_id, our_events, extra_arrays=extra_arrays_fg)

    # LSC events
    if separate_lsc_events:
        mask_lsc = np.logical_and(coincident_mask, lsc_mask)
        lsc_events = events[mask_lsc]
        if extra_arrays is not None:
            extra_arrays_lsc = [pick_mask(x, mask_lsc) for x in extra_arrays]
        else:
            extra_arrays_lsc = []
        add_or_vstack_to_dic(
            bg_fg_dicts[2], loc_id, lsc_events, extra_arrays=extra_arrays_lsc)

    # Injected events
    if separate_injections:
        mask_inj = np.logical_and(coincident_mask, inj_mask)
        injected_events = events[mask_inj]
        if extra_arrays is not None:
            extra_arrays_inj = [pick_mask(x, mask_inj) for x in extra_arrays]
        else:
            extra_arrays_inj = []
        add_or_vstack_to_dic(
            bg_fg_dicts[3], loc_id, injected_events,
            extra_arrays=extra_arrays_inj)
        if output_inj_mask:
            return mask_inj
    return


def pick_mask(arr, mask):
    return arr[mask] if not utils.checkempty(arr) else arr


def add_or_vstack_to_dic(dic, key, arr, extra_arrays=()):
    inds_extra = []
    for ind, extra_array in enumerate(extra_arrays):
        # if not utils.checkempty(extra_array):
        if extra_array is not None:
            # Allowing for empty arrays maintains a coherent structure, e.g.,
            # when there are no injections
            inds_extra.append(ind)
    if len(inds_extra) == 0:
        dic[key] = utils.safe_concatenate(dic.get(key, None), arr)
    else:
        old_entry = dic.get(key, [None] + [None] * len(inds_extra))
        new_entry = [utils.safe_concatenate(old_entry[0], arr)]
        for i_save, ind in enumerate(inds_extra):
            new_entry.append(
                utils.safe_concatenate(
                    old_entry[i_save + 1], extra_arrays[ind]))
        dic[key] = new_entry
    return


def combine_subbank_dict(dic_by_subbank):
    """
    Populates a list of bg_fg_max
    :param dic_by_subbank:
        Dict with keys = loc_ids, and values = list of arrays with each array
        containing a property of the events (an element of bg_fg_by_subbank)
    :return:
    """
    bg_fg_max_list = []
    for loc_id in sorted(dic_by_subbank.keys()):
        events_info = dic_by_subbank[loc_id]
        if isinstance(events_info, list):
            bg_fg_max_list += \
                [[event, loc_id, *extra_arrays_entries, iev]
                 for iev, (event, *extra_arrays_entries) in
                 enumerate(zip(*events_info))]
        else:
            bg_fg_max_list += [[event, loc_id, iev] for iev, event in
                               enumerate(events_info)]

    return bg_fg_max_list


def split_into_subbank_dicts(event_list, loc_id_index, skip_index=True):
    """
    :param event_list:
        One of the lists in bg_fg_max/cands_preveto_max/cands_postveto_max
    :param loc_id_index: Index of the loc_id in each element of the list
    :param skip_index: Flag to skip the loc_id in the resulting dictionary
    :return:
        Dict with keys = loc_ids, and values = list of lists with each list
        containing a property of the events other than loc_id.
        This is like an entry of bg_fg_by_subbank
    """
    if utils.checkempty(event_list):
        return dict()

    # First sort the event list by loc_id
    event_list.sort(key=lambda x: tuple(x[loc_id_index]))

    dic_by_subbank = {}
    subbank_len = 0
    loc_id_current = tuple(event_list[0][loc_id_index])
    leventprops = len(event_list[0])
    for iev in range(len(event_list) + 1):
        # We go one index extra to treat the last subbank
        if (iev == len(event_list) or
                tuple(event_list[iev][loc_id_index]) != loc_id_current):
            # The past subbank_len elements all had the same loc_id, and we're
            # starting a new loc_id. Save the events with the past loc_id
            dic_by_subbank[loc_id_current] = []
            for iprop in range(leventprops):
                if skip_index and iprop == loc_id_index:
                    continue
                dic_by_subbank[loc_id_current].append(
                    [ev[iprop] for ev in event_list[iev - subbank_len:iev]])

            if iev == len(event_list):
                # We don't need to continue beyond this point
                break

            # Start a new subbank
            subbank_len = 0
            loc_id_current = tuple(event_list[iev][loc_id_index])

        subbank_len += 1

    return dic_by_subbank


def check_and_fix_bg_fg_lists(
        bg_fg_max, bg_fg_by_subbank, extra_array_names, fobj=None,
        raise_error=False):
    """
    Our injection codes populate bg_fg_max, but not bg_fg_by_subbank.
    We also want to save indices into bg_fg_by_subbank in bg_fg_max to avoid
    duplication in memory, but old codes might not have saved it in this
    format. This function fixes both issues. Run this before scoring, and
    re-score after.
    :param bg_fg_max: List of four lists with entries like bg_fg_max
    :param bg_fg_by_subbank:
        List of four dicts with entries like bg_fg_by_subbank
    :param extra_array_names: List of names of the extra arrays in bg_fg_max
    :param fobj:
        If bg_by_subbank is read from a hdf5 file, the File object
        (must be writeable)
    :param raise_error:
        Flag to raise an error if the number of triggers don't match, just use
        as a check
    :return:
        Fixes bg_fg_max and bg_fg_by_subbank in place, and also fixes the data
        inside fobj
    """
    extra_array_names = list(extra_array_names)
    idx_prior = extra_array_names.index('prior_terms') if \
        'prior_terms' in extra_array_names else None

    if (fobj is not None) and (fobj.mode not in utils.HDF5_MODE_DICT['a']):
        utils.close_hdf5()
        raise RuntimeError("fobj must be writeable")

    for i, bg_fg_list in enumerate(bg_fg_max):
        bg_fg_subbank_len = sum(
            [len(events_info) if isinstance(events_info, np.ndarray)
             else len(events_info[0]) for loc_id, events_info in
             bg_fg_by_subbank[i].items()])

        if utils.checkempty(bg_fg_list):
            # Nothing to do here, unless the bg_fg_max was never populated
            if bg_fg_subbank_len > 0:
                bg_fg_max[i] = combine_subbank_dict(bg_fg_by_subbank[i])
            continue

        if (isinstance(bg_fg_list[-1][-1], (int, np.integer)) and
                (len(bg_fg_list) == bg_fg_subbank_len)):
            # The int checks whether we saved the indices into bg_fg_by_subbank
            # We have the correct format and number of items
            continue

        if raise_error:
            utils.close_hdf5()
            raise RuntimeError(
                "The number of events doesn't match, rerun the scoring routine")

        # If we got here, we added some new injections
        # The number of extra entries in bg_fg_by_subbank[i][loc_id]
        # -2 for pclist and loc_id, omit the index to subbank if it exists
        nextra = (len(bg_fg_list[0]) - 3) if \
            isinstance(bg_fg_list[0][-1], (int, np.integer)) else \
            (len(bg_fg_list[0]) - 2)
        nextra_final = (len(bg_fg_list[-1]) - 3) if \
            isinstance(bg_fg_list[-1][-1], (int, np.integer)) else \
            (len(bg_fg_list[-1]) - 2)

        # Check some edge cases
        if nextra != len(extra_array_names):
            utils.close_hdf5()
            raise RuntimeError(
                "The extra arrays don't match the list of expected names")
        if nextra != nextra_final:
            utils.close_hdf5()
            raise RuntimeError(
                "We need a homogenous format, did you add a few injections " +
                "without extra data?")
        if fobj is not None:
            loc_id_old = tuple(bg_fg_list[0][1])
            if (('prior_terms' not in extra_array_names) and
                    (os.path.join(FOBJ_KEYS[i],
                                  str(loc_id_old),
                                  'prior_terms') in fobj)):
                utils.close_hdf5()
                raise RuntimeError(
                    "We stored the prior terms separately for existing " +
                    "events, but not for the new injections you're trying " +
                    "to add. Redo after storing everything together.")

        # Reconstruct bg_fg_by_subbank[i]
        # First split by loc_id and collect different properties together
        dic_by_subbank = \
            split_into_subbank_dicts(bg_fg_list, 1, skip_index=True)

        # Handle an edge case of an edge case: the prior term might have
        # inconsistent shapes, since we previously computed all terms only for
        # the LVC injections, but might have only dealt with the coherent term
        # for our injections. Recompute the prior terms after this!
        if idx_prior is not None:
            for loc_id, events_infos in dic_by_subbank.items():
                events_infos[idx_prior + 1] = \
                    [utils.scalar(x) for x in events_infos[idx_prior + 1]]

        # Construct the entries of bg_fg_by_subbank[i] to have contiguous memory
        for loc_id, events_infos in dic_by_subbank.items():
            if fobj is None:
                if nextra == 0:
                    bg_fg_by_subbank[i][loc_id] = np.asarray(events_infos[0])
                else:
                    # Populate the extra arrays as well
                    bg_fg_by_subbank[i][loc_id] = [
                        np.asarray(events_infos[i]) for i in range(nextra + 1)]
            else:
                # If possible, edit fobj in place and refer to its memory
                outformat = type(bg_fg_list[0][0])
                event_struct = utils.write_hdf5_node(
                    fobj, [FOBJ_KEYS[i], str(loc_id), 'events'],
                    np.array(events_infos[0]), overwrite=True, outformat=outformat)  #Ajit modification
                if nextra == 0:
                    bg_fg_by_subbank[i][loc_id] = event_struct
                else:
                    # Populate the extra arrays as well
                    bg_fg_by_subbank[i][loc_id] = [event_struct]
                    for iextra in range(nextra):
                        if extra_array_names[iextra] == 'timeseries':
                            # Variable length datatype, we cannot use mmap
                            dtype = VLEN_TIMESERIES
                            outformat_to_use = h5py.Dataset
                            # Dereference the array once
                            # array_to_write = events_infos[iextra + 1][...][()]
                            # array_to_write = [
                            #     [p.ravel() for p in t] for t in array_to_write]
                            array_to_write = [
                                [p.ravel() for p in t] for t in events_infos[iextra+1]]
                        else:
                            dtype = None
                            outformat_to_use = outformat
                            array_to_write = events_infos[iextra + 1]
                        bg_fg_by_subbank[i][loc_id].append(
                            utils.write_hdf5_node(
                                fobj,
                                [FOBJ_KEYS[i],
                                 str(loc_id),
                                 extra_array_names[iextra]],
                                np.array(array_to_write),          ##Ajit modification
                                dtype=dtype, overwrite=True,
                                outformat=outformat_to_use))

        # Fix bg_fg_max to have shared memory and indices into bg_fg_by_subbank
        bg_fg_max[i] = combine_subbank_dict(bg_fg_by_subbank[i])

    return


# %% Functions to maximize over banks
def maximize_using_saved_options(
        list_of_rank_objs, maxopts_filepath, keys_to_pop=None, **kwargs):
    """
    Convenience function to use saved options to maximize_over_banks
    :param list_of_rank_objs: List of Rank objects
    :param maxopts_filepath: Path to the hdf5 file with the maximization options
    :param keys_to_pop:
        Keys to pop from the arguments, if needed
        (adding this in case we lose some options in the future)
    :param kwargs:
        Any extra arguments we want to override or pass to the maximization
    :return: Performs the maximization over banks
    """
    # Read from the file
    with h5py.File(maxopts_filepath, 'r') as fobj:
        maxopts = utils.load_dict_from_hdf5_attrs(fobj)

    # Pop the keys we don't want to pass
    if keys_to_pop is not None:
        for key in keys_to_pop:
            _ = maxopts.pop(key, None)

    # Print what we maximized using these options previously
    previous_banks = maxopts.pop('list_of_rank_objs', [])
    print("We previously used these options to maximize over:")
    for bank in previous_banks:
        print(bank)

    # Override any of the kwargs, or add new arguments
    for key, val in kwargs.items():
        maxopts[key] = val

    # Pop ranking_kwargs if it's there
    ranking_kwargs = maxopts.pop('ranking_kwargs', {})

    # Perform the maximization
    maximize_over_banks(list_of_rank_objs, **maxopts, **ranking_kwargs)


def maximize_over_banks(
        list_of_rank_objs, maxopts_filepath=None,
        incoherent_score_func=utils.incoherent_score,
        coherent_score_func=utils.coherent_score, mask_veto_criteria=None,
        apply_veto_before_scoring=False, matching_point=None,
        downsampling_correction=True, include_vetoed_triggers=False,
        p_veto_real_event=(lambda x: 0.05, lambda x: 0.05), **ranking_kwargs):
    """
    Note that Seth vetoed the triggers before the bank assignment
    :param list_of_rank_objs: List of Rank objects
    :param maxopts_filepath:
        Path to a hdf5 file to save the maximization options to.
        If it doesn't exist, it's created and populated with the given options.
        If it exists, it's overwritten.
    :param incoherent_score_func:
        Function that accepts two processedclists for a trigger and returns an
        incoherent score
    :param coherent_score_func:
        Function that accepts the coherent terms for a trigger and returns
        the coherent score (just the sum by default)
    :param mask_veto_criteria:
        If needed, pass in a mask on rank_obj.veto_metadata_keys to
        identify glitch tests that we use, to override the default (everything)
    :param apply_veto_before_scoring:
        Whether to apply the vetoes before scoring (we always apply them after
        scoring anyway)
        Can be a boolean variable, or a mask on rank_obj.veto_metadata_keys
        To reproduce O3a, i.e., 2201.02252, pass a mask with ones everywhere
        except at the entry corresponding to 'Secondary_peak_timeseries'
        False is the recommended input for all future catalogs
    :param downsampling_correction:
        If the triggers were downsampled compared to a chi-sq distribution
        because of an additional cut (e.g., based on whether the mode ratios
        A33/A22 or A44/A22 are physical). This flag corrects the rank function
        so that it follows the chi-sq behavior again. This flag needs a file
        downsamp_corr_path.npy to be input when creating the Rank instance
    :param matching_point: Where we match the rank functions
    :param include_vetoed_triggers: Flag whether to include the triggers which
        failed the vetos in our final list
    :param p_veto_real_event:
        Tuple with functions for the probability that a real event fails the
        vetoes in each detector, which in the most general case can be a
        function of all properties of the trigger. They should accept a list of
        entries of scores_(non)vetoed_max and yield an array of probabilities
    :return:
        Considers all the banks in list_of_rank_objs together, assigns each
        trigger to a single (bank, subbank) pair and populates
        cands_preveto_max in all the banks
    """
    # TODO: Doesn't work for BBH + BNS together because chirp mass ID is not
    #  unique, fix later
    if utils.checkempty(list_of_rank_objs):
        return

    # Make a list of options to save
    maxopts = locals().copy()
    maxopts['list_of_rank_objs'] = \
        [x.fobj.filename if x.fobj is not None else x.chirp_mass_id
         for x in list_of_rank_objs]
    _ = maxopts.pop('maxopts_filepath', None)

    if maxopts_filepath is not None:
        # Save the maximization options to a file
        maxopts_filepath = utils.rm_suffix(
            maxopts_filepath, suffix='.*', new_suffix='.hdf5')

        if os.path.isfile(maxopts_filepath):
            print(f"Warning: {maxopts_filepath} already exists, overwriting.")

        # Save the options to the file
        with h5py.File(maxopts_filepath, 'w') as fobj:
            utils.save_dict_to_hdf5_attrs(fobj, maxopts, overwrite=True)
        os.chmod(maxopts_filepath, 0o755)

    nbanks = len(list_of_rank_objs)
    nlists = len(list_of_rank_objs[0].bg_fg_by_subbank)
    chirp_mass_ids = [x.chirp_mass_id for x in list_of_rank_objs]
    if matching_point is None:
        # Use the maximimum trigger collection SNRsq among banks as the matching_point
        matching_point = max([rank_obj.snr2min for rank_obj in list_of_rank_objs])

    n_templates_by_major_bank = dict()
    bank_details_by_subbank = dict()
    # Score all banks first before we assign triggers to banks
    for rank_obj in list_of_rank_objs:
        # Record the options in the Rank object
        rank_obj.maxopts = list(rank_obj.maxopts)
        rank_obj.maxopts.append(maxopts)

        if mask_veto_criteria is not None:
            rank_obj.mask_veto_criteria = mask_veto_criteria

        # Score all banks first before we assign triggers to banks
        # Redo the foreground in case we have injections, and background to
        # forget what we did if we maximized before
        print(f"Scoring bank {rank_obj.chirp_mass_id}")
        # Note: If we want to use rank functions to decide between banks, we
        # need to use score_triggers=True, and pass the other arguments, and
        # find a way to pass the results to maximize_over_groups!
        rank_obj.score_bg_fg_lists(
            redo_bg=True, redo_fg=True,
            apply_veto_before_scoring=apply_veto_before_scoring,
            score_triggers=False, **ranking_kwargs)

        # Record some details
        n_templates_by_major_bank[rank_obj.chirp_mass_id] = \
            np.sum(rank_obj.ntemplates_by_subbank)

        for subbank_id, bank_details in zip(
                rank_obj.subbank_subset, rank_obj.bank_details_by_subbank):
            loc_id = (rank_obj.chirp_mass_id, subbank_id)
            bank_details_by_subbank[loc_id] = bank_details

    # Collect cands_preveto_max from each Rank object into a master list
    # (prior terms + trigger + loc_id + sensitivity params + timeseries)
    print(f"Creating a master list of triggers")
    scores_nonvetoed_master = [[] for _ in range(nlists)]
    for rank_obj in list_of_rank_objs:
        # Add bg, zero-lag candidates, lsc events, and injections
        # Precompute information for maximization to append to the lists as it
        # is faster than dereferencing them one by one
        idx_prior_terms = utils.index_after_removal(
            rank_obj.extra_array_names, 'prior_terms')
        if idx_prior_terms is not None:
            # Index into the entries of bg_fg_by_subbank
            idx_prior_terms += 1
        _, pclist_info_lists, _ = rank_obj.compute_function_on_subbank_data(
            'events', 0, True, maximization_info_from_pclists,
            incoherent_score_func=incoherent_score_func)
        _, cscores_lists, _ = \
            rank_obj.compute_function_on_subbank_data(
                'prior_terms', idx_prior_terms, True, coherent_score_func)
        for i, (cands_preveto_max_list, pclist_info_list, cscores_list) in \
                enumerate(zip(rank_obj.cands_preveto_max,
                              pclist_info_lists,
                              cscores_lists)):
            scores_nonvetoed_master[i] += \
                [[*entry, np.array([*pclist_info, cscore])] for
                 entry, pclist_info, cscore in
                 zip(cands_preveto_max_list, pclist_info_list, cscores_list)]

    print(f"Maximizing within the master trigger list")

    # Maximize over (bank_id, subbank_id)
    # Returns cands_preveto_max_like lists
    scores_nonvetoed_master_max = [[] for _ in range(nlists)]
    for i, scores_nonvetoed_list in enumerate(scores_nonvetoed_master):
        scores_nonvetoed_master_max[i] = maximize_over_groups(
            scores_nonvetoed_list, incoherent_score_func=incoherent_score_func,
            coherent_score_func=coherent_score_func,
            n_templates_by_major_bank=n_templates_by_major_bank,
            bank_details_by_subbank=bank_details_by_subbank,
            template_prior_applied=list_of_rank_objs[0].template_prior_applied,
            precomputed_details=True)

    print(f"Splitting the maximized master list back to banks")

    # Split according to chirp_mass_id
    # n_chirp_mass_bank x n_list
    scores_nonvetoed_master_max_split = \
        [[[] for _ in range(nlists)] for _ in range(nbanks)]
    for i, scores_nonvetoed_list in enumerate(scores_nonvetoed_master_max):
        if len(scores_nonvetoed_list) == 0:
            continue
        for entry in scores_nonvetoed_list:
            chirp_mass_id = entry[2][0]
            idx = chirp_mass_ids.index(chirp_mass_id)
            # Remove the precomputed maximization info
            scores_nonvetoed_master_max_split[idx][i].append(entry[:-1])

    # Overwrite the appropriate elements of each cands_preveto_max
    for rank_obj, split_lists in zip(
            list_of_rank_objs, scores_nonvetoed_master_max_split):
        rank_obj.cands_preveto_max = split_lists

    # Score again
    for rank_obj in list_of_rank_objs:
        print(f"Scoring bank {rank_obj.chirp_mass_id}")
        rank_obj.score_bg_fg_lists(
            apply_veto_before_scoring=True,
            score_triggers=True,
            coherent_score_func=coherent_score_func,
            include_vetoed_triggers=include_vetoed_triggers,
            matching_point=matching_point,
            downsampling_correction=downsampling_correction,
            p_veto_real_event=p_veto_real_event, **ranking_kwargs)

    return


def maximization_info_from_pclists(
        pclists, incoherent_score_func=utils.incoherent_score):
    """
    Returns information useful for maximize_over_groups
    :param pclists: n_candidates x n_det x len(processedclist) array
    :param incoherent_score_func:
        Function that accepts processedclists for trigger(s) and returns
        incoherent score(s) used for ranking
    :return:
        n_candidates x ... array with
        t_H, t_L, .., incoherent score used for ranking, rho^2
    """
    times_det = pclists[..., 0]
    incoherent_score = incoherent_score_func(pclists)
    rhosq = utils.incoherent_score(pclists)
    return np.column_stack((times_det, incoherent_score, rhosq))


def maximize_over_groups(
        events_list, incoherent_score_func=utils.incoherent_score,
        coherent_score_func=utils.coherent_score,
        n_templates_by_major_bank=None, bank_details_by_subbank=None,
        template_prior_applied=False, precomputed_details=False):
    """
    :param events_list:
        List of tuples with (pclists, (chirp_mass_id, subbank id)) if incoherent
        (prior_terms, pclists, (chirp_mass_id, subbank id)) if coherent
    :param incoherent_score_func:
        Function that accepts two processedclists for a trigger and returns an
        incoherent score (the idea was that we can use rank score if we want)
    :param coherent_score_func:
        Function that accepts the coherent terms for a trigger and returns
        the coherent score (just the sum by default)
    :param n_templates_by_major_bank:
        Dictionary giving the number of templates in each chirp mass bank
    :param bank_details_by_subbank:
        Dictionary giving the calpha_dimensionality, and the grid spacing of
        each subbank
    :param template_prior_applied:
        Flag indicating whether the template prior was applied
    :param precomputed_details:
        Flag to indicate that we precomputed the details using bg_fg_by_subbank,
        we added an entry to each event info with t_H, t_L, ... (more if more
        detectors), incoherent score used for ranking, rho^2, and coherent score
        used for ranking
    :return: List in the same format, maximized over pair of 0.1 s groups
    """
    if n_templates_by_major_bank is None:
        n_templates_by_major_bank = {}

    def get_event_details(event_info):
        """
        :param event_info: Entry of cands_preveto_max
        :return: 1. t_H
                 2. t_L
                 ... (more if more detectors)
                 3. incoherent score used for ranking
                 4. coherent score used for ranking
                 5. any extra template penalties due to maximization over
                    parameters that wasn't included in the coherent score
        """
        # Consider the coherent score when deciding
        # The convention is that the coherent terms are the excess over
        # the Gaussian case, so they are written to exclude the rho_sq
        # and the power-law prefactors due to the integration over the DOF
        if precomputed_details:
            *times_det, incoherent_score, rhosq, coherent_score = \
                event_info[-1]
        else:
            coherent_terms = event_info[0]
            pclist = event_info[1]

            times_det = pclist[:, 0]
            incoherent_score = incoherent_score_func(pclist)
            rhosq = utils.incoherent_score(pclist)
            coherent_score = coherent_score_func(coherent_terms)

        # TODO_Jay
        loc_id = event_info[2]
        n_dims, delta_c_alpha = bank_details_by_subbank[loc_id]
        # The log factors due to the maximization over times and phases
        # are already included in the coherent score
        template_penalty = n_dims * np.log(2 * np.pi / rhosq)
        if not template_prior_applied:
            n_templates = n_templates_by_major_bank.get(loc_id[0], 1)
            template_penalty += -2 * np.log(n_templates * delta_c_alpha**n_dims)

        return *times_det, incoherent_score, coherent_score, template_penalty

    event_dic = {}
    for event in events_list:
        *times_det_ev, incoherent_score_ev, coherent_score_ev, \
            template_penalty_ev = get_event_details(event)
        base_key = tuple(int(x / 0.1) for x in times_det_ev)
        derived_keys = list(
            itertools.product(*[(x - 1, x, x + 1) for x in base_key]))
        merged_to_existing = False
        for key in derived_keys:
            dic_event = event_dic.get(key, None)
            if dic_event is not None:
                *times_det_dic_ev, incoherent_score_dic_ev, \
                    coherent_score_dic_ev, template_penalty_dic_ev = \
                    get_event_details(dic_event)
                # Can incorrectly merge limiting timeslides in edge case?
                if np.all(
                        np.abs(
                            np.asarray(times_det_ev) -
                            np.asarray(times_det_dic_ev)) < 0.1):
                    merged_to_existing = True
                    # Condolences, see Eq. (B7) of 1904.07214
                    if ((incoherent_score_ev +
                         coherent_score_ev +
                         template_penalty_ev) >
                            (incoherent_score_dic_ev +
                             coherent_score_dic_ev +
                             template_penalty_dic_ev)):
                        event_dic[key] = event
        if not merged_to_existing:
            event_dic[base_key] = event

    return list(event_dic.values())


# %% Functions to score lists
def compute_coherent_scores(
        cands_by_subbank, extra_array_names, clist_pos, time_slide_jump=0.1,
        mask_veto_criteria=None, median_normfacs_by_subbank=None,
        template_prior_funcs=None, fobj=None, candtype=FOBJ_KEYS[0]):
    """
    1. Computes coherent scores if not computed, and prior and sensitivity terms
    2. Edits hdf5 file object fobj in place to include prior terms if applicable
    3. Edits cands_by_subbank in place to include extra prior and sensitivity
       terms if coherent scores were saved
    4. Returns entries like those of scores_bg_by_subbank_nonvetoed (for fg,
       combine outside this function) and median_normfacs_by_subbank
    :param cands_by_subbank:
        An element of bg_fg_by_subbank - a dictionary with loc_id as keys, with
        n_cand x (n_det = 2) x row of processedclist arrays with triggers in the
        subbank (also w/ timeseries, veto_metadata, coherent scores if we have
        them)
    :param extra_array_names: List of names of the extra arrays in bg_by_subbank
    :param clist_pos:
        Dictionary with the name of trigger attributes as keys and the index of
        the attributes in the processedclist as values
    :param time_slide_jump: The least-count of timeslides (s)
    :param mask_veto_criteria:
        If needed, mask on veto_metadata.shape[-1] that we use to pick criteria
        to veto candidates on (used only if veto_metadata is available)
    :param median_normfacs_by_subbank:
        If known, dictionary with loc_id as keys, and n_det array of median
        normfacs as values. If None, we estimate them from the data (should only
        estimate for the background!)
    :param template_prior_funcs:
        If known, dictionary indexed by subbank_id, with
        (function that returns template prior given calpha, # input dimensions)
    :param fobj:
        If bg_by_subbank is read from a hdf5 file, the File object (must be
        writeable)
    :param candtype: Entry into FOBJ_KEYS for the type of candidates
    :return: 1. List of lists (n_subbank x
                 4 + n_extra_arrays - 1 (if prior is within) + 1) with each
                subbank's contribution to the lists in scores_(non)vetoed_max
                (for the background, this is scores_bg_by_subbank_nonvetoed)
                Each entry of a subbank's list is composed of
                    a) The prior terms
                        (coherent score,
                        -rho^2,
                        2 log(1/median normfac^3)),
                        2 log(template prior))
                    b) (n_det=2) x row of processedclist array with the trigger
                    c) (bank_id, subbank_id)
                    b) 4 x (n_det=2) array with t, Re(rho_22), Im(rho_22), sensitivity ratio
                    e)... extra info that was passed in about each trigger minus
                       the coherent score which is already in the prior terms
                    f) The index of the trigger into bg_by_subbank, useful as
                       things can be reordered when we maximize over banks
             2. Dictionary giving median H1 and L1 normfacs, with
                subbank_ids as keys (median_normfacs_by_subbank)
    """
    if (fobj is not None) and (fobj.mode not in utils.HDF5_MODE_DICT['a']):
        utils.close_hdf5()
        raise RuntimeError("fobj must be writeable")

    extra_array_names = list(extra_array_names)
    idx_veto_metadata = extra_array_names.index('veto_metadata') if \
        'veto_metadata' in extra_array_names else None
    idx_prior = extra_array_names.index('prior_terms') if \
        'prior_terms' in extra_array_names else None

    if not idx_prior:
        utils.close_hdf5()
        raise RuntimeError("Prior terms must have been already computed")

    scores_by_subbank = []
    if median_normfacs_by_subbank is None:
        median_normfacs_by_subbank = {}

    # Computes the coherent score, subbank by subbank
    for loc_id in sorted(cands_by_subbank.keys()):
        cand_arr = cands_by_subbank[loc_id]
        if isinstance(cand_arr, list):
            cand_arr, *extra_arrays = cand_arr
        else:
            extra_arrays = []

        if len(extra_arrays) != len(extra_array_names):
            utils.close_hdf5()
            raise RuntimeError(
                f"Extra array names don't match the data for {loc_id}")

        # Initialize memory for the prior terms
        if extra_arrays[idx_prior].ndim > 1:
            # We already computed and saved all prior terms, so let's just
            # reuse the array space
            prior_terms = extra_arrays[idx_prior]
        else:
            # Create a prior terms array, and overwrite the entries of
            # bg_by_subbank later
            if template_prior_funcs is not None:
                prior_terms = np.zeros((len(cand_arr), 4))
            else:
                prior_terms = np.zeros((len(cand_arr), 3))

        mask_nonvetoed = np.ones(len(cand_arr), dtype=bool)

        # We already computed the coherent scores in coincidence.py
        # Just read them in
        # TODO: Revisit for fishing
        # Pick only the coherent scores even if we saved all prior terms
        # previously
        coherent_scores = extra_arrays[idx_prior][:, 0] if \
            extra_arrays[idx_prior].ndim > 1 else extra_arrays[idx_prior]
        prior_terms[:, 0] = coherent_scores
        prior_terms[:, 1] = -utils.incoherent_score(cand_arr)

        # Read off the sensitivity params using the given coherent score
        # instance, this is n_event x n_det x 4 array with
        #  t, Re(rho_22), Im(rho_22), nfac
        # in each detector
        # TODO: Revisit for fishing
        sensitivity_params = get_params(
            cand_arr, clist_pos, time_slide_jump=time_slide_jump)

        # Apply the veto criteria if we were asked to
        if mask_veto_criteria is not None and idx_veto_metadata is not None:
            veto_metadata = extra_arrays[idx_veto_metadata]
            if np.all(mask_veto_criteria):
                # Keep triggers that pass all vetoes in both detectors
                mask_nonvetoed = np.all(np.all(veto_metadata, axis=-1), axis=-1)
            else:
                # Keep triggers that pass requested vetoes in both detectors
                # veto_metadata is n_cand x n_det x n_veto_criteria
                mask_nonvetoed = \
                    np.all(
                        np.all(
                            veto_metadata[:, :, mask_veto_criteria], axis=-1),
                        axis=-1)

        # Read the median normfacs and make the H1 median the reference
        median_normfacs = median_normfacs_by_subbank.get(loc_id, None)

        if median_normfacs is None:
            # Compute them from what we have
            if np.all(mask_nonvetoed):
                median_normfacs = np.median(sensitivity_params[:, :, 3], axis=0)
            else:
                median_normfacs = np.median(
                    sensitivity_params[mask_nonvetoed, :, 3], axis=0)
            median_normfacs_by_subbank[loc_id] = median_normfacs

        # The first term in prior_terms has the part that's relevant to p(t)
        # The median normfacs[0] is not a bug, it just sets the scale to H1
        # prior_terms[:, 0] += 2 * np.log(1 / median_normfacs[0]**3)
        prior_terms[:, 2] = 2 * np.log(1 / median_normfacs[0] ** 3)

        # Apply the template prior
        if template_prior_funcs is not None:
            template_prior_func, ndim = template_prior_funcs[loc_id[1]]
            calphas = \
                cand_arr[:, 0, clist_pos['c0_pos']:clist_pos['c0_pos'] + ndim]
            if calphas.shape[-1] < ndim:
                # Edge case of an edge case
                calphas = np.pad(
                    calphas, pad_width=((0, 0), (0, ndim - calphas.shape[-1])))
            # prior_terms[:, 0] += 2 * np.log(template_prior_func(calphas))
            if len(calphas) > 0:
                prior_terms[:, 3] = 2 * np.log(template_prior_func(calphas))

        # Update prior terms and sensitivity_params on disk if needed
        if fobj is not None:
            outformat = type(cand_arr)
            if prior_terms is not extra_arrays[idx_prior]:
                # We weren't just changing a prior dataset's entries, so we
                # should save it
                prior_terms = utils.write_hdf5_node(
                    fobj, [candtype, str(loc_id), 'prior_terms'],
                    prior_terms, overwrite=True, outformat=outformat)

            sensitivity_params = utils.write_hdf5_node(
                fobj, [candtype, str(loc_id), 'sensitivity_params'],
                sensitivity_params, overwrite=True, outformat=outformat)

        # We were saving the prior terms in cands_by_subbank
        # Edit in place to include the (new) prior terms
        cands_by_subbank[loc_id] = [cand_arr]
        for iextra in range(len(extra_arrays)):
            if iextra != idx_prior:
                cands_by_subbank[loc_id].append(extra_arrays[iextra])
            else:
                cands_by_subbank[loc_id].append(prior_terms)

        # We're already saving the coherent scores in prior_terms in
        # scores_(bg/fg)_by_subbank so remove the corresponding entry from
        # extra arrays
        extra_arrays = [x for i, x in enumerate(extra_arrays) if i != idx_prior]

        # Add to data structures
        scores_list = []
        if not utils.checkempty(extra_arrays):
            for ind, (veto_result, ph1, event, param, *extra_arrays_entries) in \
                    enumerate(zip(mask_nonvetoed,
                                  prior_terms,
                                  cand_arr,
                                  sensitivity_params,
                                  *extra_arrays)):
                if veto_result:
                    scores_list.append(
                        [ph1, event, loc_id, param, *extra_arrays_entries, ind])
        else:
            for ind, (veto_result, ph1, event, param) in \
                    enumerate(zip(mask_nonvetoed,
                                  prior_terms,
                                  cand_arr,
                                  sensitivity_params)):
                if veto_result:
                    scores_list.append([ph1, event, loc_id, param, ind])

        scores_by_subbank.append(scores_list)

    return scores_by_subbank, median_normfacs_by_subbank


def reform_coherent_scores(cands_preveto_max, median_normfacs_by_subbank):
    """
    Edits input structures in place via aliasing and returns
    scores_bg_by_subbank
    :param cands_preveto_max:
        List (bg/fg/lvc/inj) of lists (n_trigger) with each element having a set
        of event attributes
        (prior terms, pclists, loc_id, sensitivity params, other stuff)
    :param median_normfacs_by_subbank:
        Dictionary with n_det array of median normfacs for each subbank ID
    :return:
        Recomputes the median, and edits the prior terms and median normfacs in
        place. It also returns the background split by subbank
    """
    scores_by_subbank_lists = [[] for _ in cands_preveto_max]
    # Go over the list subbank by subbank
    for loc_id in sorted(median_normfacs_by_subbank.keys()):
        # n_cand_types x n_events in the subbank
        scores_lists = \
            [[ev for ev in evlist if tuple(ev[2]) == loc_id]
             for evlist in cands_preveto_max]

        # TODO: Revisit for fishing
        # n_det array of median normfacs from the background events
        median_normfacs = np.median(
            np.array([ev[3][:, 3] for ev in scores_lists[0]]), axis=0)

        # Fix the median sensitivity penalty with this new median
        # The median normfacs[0] is not a bug, H1 sets a scale
        ptoffset = 2 * np.log(
            (median_normfacs_by_subbank[loc_id][0] / median_normfacs[0])**3)

        # Fix the median sensitivity penalty for all the lists
        for evlist in scores_lists:
            for ev in evlist:
                ev[0][2] += ptoffset

        # Having applied the new median, overwrite the saved one
        median_normfacs_by_subbank[loc_id] = median_normfacs

        # # Save the background to the structure by subbank
        # scores_bgs_by_subbank.append(subbank_scores_bg)
        # Save the lists to structure by subbank
        for i, evlist in enumerate(scores_lists):
            scores_by_subbank_lists[i].append(evlist)

    # Each entry of cands_preveto_max is no longer in order of subbanks
    # after bank reassignment, so rearrange to fix that
    # cands_preveto_max[0] = sum(scores_bgs_by_subbank, [])
    for i, evlist in enumerate(scores_by_subbank_lists):
        cands_preveto_max[i] = sum(evlist, [])

    # return scores_bgs_by_subbank
    return scores_by_subbank_lists[0]


@vectorize(nopython=True)
def offset_background(dt, time_slide_jump, dt_shift):
    """
    Finds the amount to shift the detectors' data streams by
    :param dt: Time delays (s)
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


def get_params(
        events, clist_pos, time_slide_jump=params.DEFAULT_TIMESLIDE_JUMP/1000):
    """
    TODO_Jay: We only need the sensitivity (where?) and not ts_out, rezs, imzs
    :param events:
        (n_cand x (n_det = 2) x processedclist)/
        ((n_det = 2) x processedclist) array with coincidence/background
        candidates
    :param clist_pos:
        Dictionary with the name of trigger attributes as keys and the index of
        the attributes in the processedclist as values
    :param time_slide_jump: Units of jumps (s) for timeslides
    :return: n_cand x (n_det = 2) x 4 array (always 3D) with
        shifted_ts, re(rho_22), im(rho_22), effective_sensitivity in each detector
    """
    if events.ndim == 2:
        # We're dealing with a single event
        events = events[None, :]

    if len(events) == 0:
        return np.zeros(events.shape[:2] + (4,))

    dt_shift = params.DEFAULT_DT / 1000  # in s

    # Add shifts to each detector to get to zero lag
    # n_cand x n_det
    ts_out = events[:, :, 0].copy()
    shifts = offset_background(
        ts_out[:, 1:] - ts_out[:, 0][:, None], time_slide_jump, dt_shift)
    ts_out[:, 1:] += shifts

    # Overlaps
    # n_cand x n_det
    rezs = events[:, :, clist_pos['rezpos']]
    imzs = events[:, :, clist_pos['imzpos']]

    # Sensitivity
    # The hole correction is a number between 0 and 1 reflecting the
    # sensitivity after some parts of the waveform fell into a hole
    # asd drift is the effective std that the score was divided with
    # => the bigger it is, the less the sensitivity
    asd_corrs = events[:, :, clist_pos['psd_drift_pos']]
    ns = events[:, :, clist_pos['normfac_pos']]

    # Hole corrections
    hs = events[:, :, clist_pos['hole_correction_pos']]
    n_effs = ns / asd_corrs * hs

    return np.stack((ts_out, rezs, imzs, n_effs), axis=2)


# %% Core Rank class
class Rank(object):
    def __init__(
            self, chirp_mass_id=3, cver='cand0', source='BBH', runs=('hm_o3a',),
            subbank_subset=None, collect_rerun=True, collect_before_veto=True,
            collect_timeseries=False, coinc_ftype="npz", snr2min=None,
            snr2min_marg=None, template_prior_data_file_path=None, override_dic=None,
            empty_init=False, ncores=1, cand_dirs_all=None, outputdirs_all=None,
            detectors=('H1', 'L1'), downsamp_corr_path=None, **cs_kwargs):
        """
        Assumes two dominant detectors
        :param chirp_mass_id: Chirp mass ID of bank
        :param cver: Version used in candidate collection
        :param source: Should be one of 'BBH', 'BNS', or 'NSBH'
        :param runs:
            List of runs being searched in, elements should be keys of
            utils.(BBH|BNS|NSBH)_PREFIXES and give sensible results for
            utils.(OUTPUT|CAND)_DIR. Since this is a higher modes version,
            we'll prepend hm_ if the names doesn't already have it
        :param subbank_subset: If desired, subset of subbanks to include
        :param collect_rerun:
            Flag indicating whether to collect failed files that we discovered
            and reran after papers, pass False to preserve old runs
            (no meaningful change to the background, did the BBH0 guy pop up
            like this?)
        :param collect_before_veto:
            Flag indicating whether we want to collect before veto instead
        :param collect_timeseries:
            FLag indicating whether we should collect saved SNR timeseries
        :param coinc_ftype:
            Type of the files saved by the coincidence script, can be
            "npy" (old) or "npz" (new)
        :param snr2min:
            Minimum single detector incoherent SNRsq
            (if none, this is inferred from a particular trigger file)
        :param snr2min_marg: Minimum single detector marginalized SNRsq
        :param template_prior_data_path:
            path str. if prior_data not passed, will load from here.
            contains template prior_data dictionary, with structure:
            (properties dictionary)->(multibank dictionary)->(sub-bank-list)
        :param override_dic:
            Dictionary with any parameters to override before collecting
            candidates
        :param empty_init: Flag to create an empty version
        :param ncores: Number of cores to use for candidate collection
        :param cand_dirs_all:
            n_runs list of dictionary of dictionaries with the outer dict's keys
            being the chirp mass id, and the inner dict's key being the
            subbank id, and the values being the directories with coincident
            candidates, used to "cheat"
        :param outputdirs_all:
            n_runs list of dictionary of dictionaries with the outer dict's keys
            being the chirp mass id, and the inner dict's key being the
            subbank id, and the values being the directories with individual
            triggers, used to "cheat"
        :param detectors:
            Tuple with names of the two detectors we will be loading results for
        :param downsamp_corr_path:
            Kept only for backwards compatibility (remove later),
            the path to the downsampling corrections .npy file
        :param cs_kwargs: Other arguments to pass to coherent score instance
        """
        if empty_init:
            return

        self.fobj = None   # To populate if loading from hdf5 file
        self.maxopts = []  # List of maximizations done on this object

        self.chirp_mass_id = chirp_mass_id
        self.cver = cver
        self.source = source
        # Ensure higher mode runs are prefixed with 'hm_'
        runs = [s if s.startswith('hm_') else 'hm_' + s for s in runs]
        # Sort the runs by the order of the start time
        runs = sorted(
            runs, key=lambda x: utils.BOUNDS_RUNS[utils.BASE_RUNS[x].lower()][0])
        self.runs = runs
        self.detectors = detectors

        # ------------------------------------------------
        print(f"Defining candidate and single_det_trigger directories for " +
              f"{source} {chirp_mass_id}")
        if cand_dirs_all is None:
            # List of dict of dicts
            cand_dirs_all = utils.get_dirs(
                dtype='cand', vers_suffix=cver, source=source, runs=runs)
        if outputdirs_all is None:
            # List of dict of dicts
            outputdirs_all = utils.get_dirs(
                dtype='trigs', source=source, runs=runs)

        # Define the subbank subset
        if subbank_subset is not None:
            self.subbank_subset = sorted(subbank_subset)
        else:
            self.subbank_subset = np.arange(
                len(cand_dirs_all[0][chirp_mass_id]))
        self.n_subbanks = len(self.subbank_subset)

        # n_runs x n_subbank
        self.cand_dirs_mcbin = \
            [[x[chirp_mass_id][subbank_id]
              for subbank_id in self.subbank_subset] for x in cand_dirs_all]
        self.outputdirs = \
            [[x[chirp_mass_id][subbank_id]
              for subbank_id in self.subbank_subset] for x in outputdirs_all]

        if downsamp_corr_path is not None:
            self.downsampling_corrections = np.load(downsamp_corr_path)
        else:
            self.downsampling_corrections = None

        # ------------------------------------------------
        print('Fixing candidate collection parameters')
        # Read example trigger objects for each subbank
        # n_runs x n_subbank: assumes that subbanks were run with the same
        # thresholds
        example_jsons = [[glob.glob(
            os.path.join(outputdir, "*config.json"))[0]
                         for outputdir in x] for x in self.outputdirs]
        example_trigs = [
            [trig.TriggerList.from_json(
                example_json, load_trigs=False, do_ffts=False)
                for example_json in x] for x in example_jsons]
        for example_trigs_run in example_trigs:
            for example_trig in example_trigs_run:
                example_trig.templatebank.HM_amp_ratio_PSD_factor = \
                    np.array([1, 1])

        self.clist_pos = {'c0_pos': example_trigs[0][0].c0_pos,
                          'rezpos': example_trigs[0][0].rezpos,
                          'imzpos': example_trigs[0][0].imzpos,
                          'psd_drift_pos': example_trigs[0][0].psd_drift_pos,
                          'normfac_pos': example_trigs[0][0].normfac_pos,
                          'hole_correction_pos': example_trigs[0][0].hole_correction_pos}

        # Different runs were run with the same template bank
        self.ntemplates_by_subbank = \
            [T.templatebank.ntemplates(T.delta_calpha, T.template_safety)
             for T in example_trigs[0]]

        # Lists of length 2, with dimension and delta_calpha for the subbanks
        self.bank_details_by_subbank = \
            [[T.templatebank.ndims, T.delta_calpha] for T in example_trigs[0]]

        # We choose the most stringent single detector snr^2 cut
        if snr2min is not None:
            self.snr2min = snr2min
        else:
            self.snr2min = np.max(
                [[example_trig.threshold_chi2 for example_trig in x]
                 for x in example_trigs])

        self.snr2min_marg = snr2min_marg

        # Save the example trigs which can be useful later
        self.example_trigs = example_trigs[0]

        # Read coincidence parameters
        # List of length n_run with dictionaries, assumes the subbanks
        # were collected with the same collection parameters
        # TODO: Cannot collect O3a and O2 together, if we ever need then
        coinc_fname = ('coincidence_parameters_' + "_".join(detectors)
                       if "o3" in runs[0].lower() else
                       'coincidence_params') + '.json'
        cand_collection_dics = [json.load(
            open(os.path.join(x[0], coinc_fname), 'r'))
            for x in self.cand_dirs_mcbin]

        # We choose the most stringent double detector snr^2 cut
        self.snr2sumcut = np.max(
            [x['threshold_chi2'] for x in cand_collection_dics])

        # Record some collection parameters
        self.time_slide_jump = \
            cand_collection_dics[0]['minimal_time_slide_jump']
        # TODO: Revisit if fishing
        max_zero_lag_delay = cand_collection_dics[0].get(
            "max_zero_lag_delay", cand_collection_dics[0]['time_shift_tol'])

        # If the runs were collected with different numbers
        # of timeslides, pick the lowest number
        self.Nsim = np.min([
            (2 * x['max_time_slide_shift'] / x['minimal_time_slide_jump'])
            for x in cand_collection_dics])
        max_time_slide_shifts = \
            [x['max_time_slide_shift'] for x in cand_collection_dics]
        if len(set(max_time_slide_shifts)) == 1:
            max_time_slide_shift = None
        else:
            max_time_slide_shift = np.min(max_time_slide_shifts)
        print("Applying max_time_slide_shift = ", max_time_slide_shift)

        # ------------------------------------------------
        print('Instantiating template prior')
        # Creates a dictionary indexed by subbank id, with (func, num_calphas)
        if template_prior_data_file_path is None:
            if example_trigs[0][0].templatebank.Template_Prior_NF is None:
                self.template_prior_funcs = {
                    subid: (lambda x: 1., 1) for subid in self.subbank_subset}
                self.template_prior_applied = False
            else:
                self.template_prior_applied = True
                self.template_prior_funcs = {
                    subid: (lambda x, i=i: np.exp(
                        example_trigs[0][i].templatebank.Template_Prior_NF.log_prior(x)),
                            self.bank_details_by_subbank[i][0])
                    for i, subid in enumerate(self.subbank_subset)}
        else:
            # TODO: Can it happen that we have a singleton subbank?
            # How does the code deal with this?
            self.template_prior_applied = True
            self.template_prior_funcs = {subid: tg.get_prior_interp_func(
                source + f"_{chirp_mass_id}", subid,
                ndim=self.bank_details_by_subbank[i][0],
                prior_data_file_path=template_prior_data_file_path)
                for i, subid in enumerate(self.subbank_subset)}
            # overriding ndim
            for i, subid in enumerate(self.subbank_subset):
                self.bank_details_by_subbank[i][0] = \
                    self.template_prior_funcs[subid][1]
        # ------------------------------------------------
        # Override any parameters that the user wants before
        # collecting candidates
        if override_dic is not None:
            self.__dict__.update(override_dic)

        # ------------------------------------------------
        print('Collecting candidates from subbanks')
        # Pure background, our events, LSC events, injected events
        # It's cheap to retain the bg_fg_by_subbank
        # since the numpy arrays share memory
        self.bg_fg_max, self.bg_fg_by_subbank, self.veto_metadata_keys, \
            self.extra_array_names = \
            collect_all_subbanks(
                self.cand_dirs_mcbin,
                chirp_mass_id,
                subbank_subset=self.subbank_subset,
                snr2min=self.snr2min,
                snr2min_marg=self.snr2min_marg,
                snr2sumcut=self.snr2sumcut,
                max_time_slide_shift=max_time_slide_shift,
                minimal_time_slide_jump=self.time_slide_jump,
                max_zero_lag_delay=max_zero_lag_delay,
                collect_rerun=collect_rerun,
                collect_before_veto=collect_before_veto,
                collect_timeseries=collect_timeseries,
                coinc_ftype=coinc_ftype,
                detectors=detectors,
                score_reduction_max=cand_collection_dics[0][
                    "score_reduction_max"],
                ncores=ncores,
                trigger_objs=example_trigs[0])

        # Initialize quantities that will be defined when lists are scored
        # Mask on self.veto_metadata_keys to identify glitch tests that we use
        self.mask_veto_criteria = None
        if self.veto_metadata_keys is not None:
            self.mask_veto_criteria = \
                np.ones_like(self.veto_metadata_keys, dtype=bool)
        # TODO: Remember the first two are CBC_CAT2 and CBC_CAT3 and don't apply later

        # Needed to define the rank score
        self.scores_bg_by_subbank_nonvetoed = None
        self.cands_preveto_max = None
        # Dictionary indexed by the loc_id, list in the old one
        self.median_normfacs_by_subbank = None

        # Structure that will hold the final candidates/background
        self.scores_bg_by_subbank = None
        self.cands_postveto_max = None
        self.cands_postveto_max_keys = {'Description':
            'List containing the candidates after maximization over banks '\
            +'which cleared the veto',
            'ind_0':{0:'Background (timeslides)',
                                        1:'Coincidence (non-LVK)',
                                        2:'Coincidence (LVK)',
                                        3:'Injections'},
            'ind_1':'candidate_index',
            'ind_2':{0:{0:'Coherent score',
                        1:'-rho^2',
                        2:'2 log(1/median normfac^3)',
                        3:'2 log(template prior)'},
                    1:{0:'H_trigger_processedclist',
                        1:'L_trigger_processedclist'},
                    2:{'bank_id',
                        'subbank_id'},
                    3:{0:{'H':'t',
                        1:'Re(rho_22)',
                        2:'Im(rho_22)',
                        3:'Det. sensitivity (Normfac/ASD drift*hole correction)'},
                    1:'L'},
                    4:{0:'Veto_metadata_H',
                    1:'Veto_metadata_L'},
                    5:'Index into bg_by_subbank'}}
        
        self.cands_preveto_max_keys = self.cands_postveto_max_keys
        self.cands_preveto_max_keys['Description'] = \
        'List containing both vetoed and non-vetoed candidates after '\
        +'maximization over banks'

        self.bg_fg_by_subbank_keys = {'Description':
            'List containing candidates before maximization over subbanks (and banks)',
            'ind_0':{0:'Background (timeslides)',
                                        1:'Coincidence (non-LVK)',
                                        2:'Coincidence (LVK)',
                                        3:'Injections'},
            'ind_1':'(bank_id, subbank_id)',
            'ind_2':{0:'n_cand x n_det x processedclist',
                     1:'n_cand x n_det x veto_metadata',
                     2:{'n_cand x n_det x': 
                     {0:'Coherent score',
                        1:'-rho^2',
                        2:'2 log(1/median normfac^3)',
                        3:'2 log(template prior)'}}}}

        self.processedclist_keys = self.example_trigs[0].processedclist_keys


        self.snr2max = None
        self.matching_point = None

        # List of size n_subbanks with (score_h1, score_l1)
        self.score_funcs = None

        # Individual incoherent scores per detector and coherent scores
        # Vetoed triggers
        self.back_0_score_vetoed = None
        self.back_1_score_vetoed = None
        self.cand_0_score_vetoed = None
        self.cand_1_score_vetoed = None
        self.lsc_0_score_vetoed = None
        self.lsc_1_score_vetoed = None
        self.inj_0_score_vetoed = None
        self.inj_1_score_vetoed = None
        # Nonvetoed triggers
        self.back_0_score = None
        self.back_1_score = None
        self.cand_0_score = None
        self.cand_1_score = None
        self.lsc_0_score = None
        self.lsc_1_score = None
        self.inj_0_score = None
        self.inj_1_score = None

        self.coherent_scores_bg = None
        self.coherent_scores_cand = None
        self.coherent_scores_lsc = None
        self.coherent_scores_inj = None

        self._avg_sensitive_volume = None
        self.cs_kwargs = cs_kwargs
        return

    @classmethod
    def from_pickle(cls, path):
        # deprecated on 6/2024
        # loading the class dictionary
        bytes_in = bytearray(0)
        max_bytes = 2 ** 30
        input_size = os.path.getsize(path)
        with open(path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        d = pickle.loads(bytes_in)

        # Preserve old saved structures
        if d.get('subbank_subset', None) is None:
            d['subbank_subset'] = np.arange(d['n_subbanks'])

        # setting up the class structure
        instance = cls(empty_init=True)
        # replacing the class dictionary with the loaded one.
        instance.__dict__ = d
        return instance

    def to_pickle(self, path, overwrite=False):
        """
        Deprecated on 6/2024
        Saves the class to a pickle
        :param path: Path to the pickle file to save to
        :param overwrite: Flag to overwrite if file(s) exist
        :return:
        """
        if os.path.isfile(path):
            print(f"{path} already exists!")
            if not overwrite:
                return

        # Better to not save the coherent score instance
        # as it is dependent of the cogwheel version
        # and can cause version conflicts
        # cs_instance can later easily be loaded using create_coh_score_instance()
        d_to_save = self.__dict__.copy()
        d_to_save['cs_instance'] = None
        bytes_out = pickle.dumps(d_to_save)

        # write
        max_bytes = 2 ** 30
        with open(path, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx + max_bytes])

    def to_hdf5(self, path=None, overwrite=False):
        """
        Saves the class to a hdf5 file. Ensure that you ran fix_bg_fg_lists if
        you added extra injections (scoring does it automatically)
        Note that saving to a new path doesn't replace what is in the instance.
        If you want to use the new hdf5 file created, load it using from_hdf5
        :param path:
            Path to the hdf5 file to save to, can also be a File object.
            If None, defaults to self.fobj
        :param overwrite: Flag to overwrite data in the file if it exists
        :return:
        """
        overwrite = utils.bool2int(overwrite)
        if path is None:
            if self.fobj is not None:
                f = self.fobj
                toclose = False
            else:
                utils.close_hdf5()
                raise ValueError("No path or fobj given, cannot save!")
        else:
            # Note: Raises an error if the file is already open in readonly mode
            path = utils.rm_suffix(path, suffix='.*', new_suffix='.hdf5')
            if os.path.isfile(path):
                if overwrite == 0:
                    print(f"{path} already exists, pass overwrite=True to overwrite!")
                    return
                else:
                    # We create a new file as there is an issue with loading from the 
                    # previous file and then replacing datasets, somehow the memory
                    # starts to blow up
                    path = utils.rm_suffix(path, suffix='.*', new_suffix='.temporary_hdf5')

            f, toclose = utils.get_hdf5_file(path, 'a')

        extra_array_names = list(self.extra_array_names)
        idx_prior = extra_array_names.index('prior_terms') if \
            'prior_terms' in extra_array_names else None

        # Raise an error if we added extra injections without scoring them
        check_and_fix_bg_fg_lists(
            self.bg_fg_max, self.bg_fg_by_subbank, self.extra_array_names,
            raise_error=True)

        # First save the big data to the hdf5 file
        keys_to_skip = []
        # Start with bg_fg_by_subbank which contains all the loaded trigger data
        for ind, candtype in enumerate(FOBJ_KEYS):
            # Save the entries of bg_fg_by_subbank[ind]
            for loc_id, subbank_data in self.bg_fg_by_subbank[ind].items():
                # First save the processedclists
                if utils.checkempty(self.extra_array_names):
                    # We only have the processedclists
                    pclists = subbank_data
                else:
                    pclists = subbank_data[0]
                _ = utils.write_hdf5_node(
                    f, [candtype, str(loc_id), 'events'], pclists,
                    overwrite=overwrite)

                # Then save the extra arrays we read
                for iextra, extra_array_name in enumerate(
                        self.extra_array_names):
                    if extra_array_name == 'timeseries':
                        # Variable length timeseries
                        dtype = VLEN_TIMESERIES
                        # Dereference the array once
                        array_to_write = subbank_data[iextra + 1][...][()]
                        array_to_write = [
                            [p.ravel() for p in t] for t in array_to_write]
                    else:
                        dtype = None
                        array_to_write = subbank_data[iextra + 1]
                    _ = utils.write_hdf5_node(
                        f, [candtype, str(loc_id), extra_array_name],
                        array_to_write, dtype=dtype,
                        overwrite=overwrite)

                if self.cands_preveto_max is not None:
                    # We separately saved the sensitivity params, and in an
                    # edge case, the prior terms
                    h5path = [candtype, str(loc_id), 'sensitivity_params']
                    sensitivity_params_path = os.path.join(*h5path)
                    if (self.fobj is None or
                            sensitivity_params_path not in self.fobj):
                        utils.close_hdf5()
                        raise ValueError(
                            "We had to save the sensitivity params " +
                            "separately but we didn't, rerun " +
                            "score_bg_fg_lists before saving.")
                    _ = utils.write_hdf5_node(
                        f, h5path, self.fobj[sensitivity_params_path],
                        overwrite=overwrite)

                    if idx_prior is None:
                        h5path = [candtype, str(loc_id), 'prior_terms']
                        prior_path = os.path.join(*h5path)
                        if self.fobj is None or prior_path not in self.fobj:
                            utils.close_hdf5()
                            raise ValueError(
                                "We had to save the prior terms separately " +
                                "but we didn't, rerun score_bg_fg_lists")
                        else:
                            _ = utils.write_hdf5_node(
                                f, h5path, self.fobj[prior_path],
                                overwrite=overwrite)

            # Save indices into bg_fg_by_subbank[ind] for cands_preveto_max
            if self.cands_preveto_max is not None:
                # Check that the last entry is an integer index
                if len(self.cands_preveto_max[ind]) > 0 and not isinstance(
                        self.cands_preveto_max[ind][0][-1],
                        (int, np.integer)):
                    utils.close_hdf5()
                    raise ValueError(
                        "cands_preveto_max is in its old format, we need " +
                        "to save the indices into bg_fg_by_subbank")

                # Save n x 3 array with loc_id and indices into bg_fg_by_subbank
                index_array = \
                    [(*ev[2], ev[-1]) for ev in self.cands_preveto_max[ind]]
                _ = utils.write_hdf5_node(
                    f, [candtype, "cands_preveto_max"], index_array,
                    dtype=int, overwrite=overwrite)

            # Save indices into bg_fg_by_subbank[ind] for cands_postveto_max
            if self.cands_postveto_max is not None:
                # Check that the last entry is an integer index
                if len(self.cands_postveto_max[ind]) > 0 and not isinstance(
                        self.cands_postveto_max[ind][0][-1], (int, np.integer)):
                    utils.close_hdf5()
                    raise ValueError(
                        "cands_postveto_max is in its old format, we need " +
                        "to save the indices into bg_fg_by_subbank")

                # Save n x 3 array with loc_id and indices into bg_fg_by_subbank
                index_array = \
                    [(*ev[2], ev[-1]) for ev in self.cands_postveto_max[ind]]
                _ = utils.write_hdf5_node(
                    f, [candtype, "cands_postveto_max"], index_array,
                    dtype=int, overwrite=overwrite)

        keys_to_skip.append('bg_fg_by_subbank')
        keys_to_skip.append('cands_preveto_max')
        keys_to_skip.append('cands_postveto_max')

        # Don't save fobj within fobj!
        keys_to_skip.append('fobj')

        # Don't bother saving duplicated entries, we can reconstruct them
        # bg_fg_by_subbank -> bg_fg_max
        # cands_preveto_max[0] -> scores_bg_by_subbank_nonvetoed
        # cands_postveto_max[0] -> scores_bg_by_subbank
        keys_to_skip.append('bg_fg_max')
        keys_to_skip.append('scores_bg_by_subbank_nonvetoed')
        keys_to_skip.append('scores_bg_by_subbank')

        # Don't save example_trigs, they are large, and we can reconstruct them
        keys_to_skip.append('example_trigs')

        # Save the rest as attributes
        utils.save_dict_to_hdf5_attrs(f, self.__dict__, keys_to_skip, overwrite)

        if toclose:
            f.close()
            os.chmod(path, 0o755)
            if overwrite != 0:
                shutil.move(path, utils.rm_suffix(path,
                         suffix='.*', new_suffix='.hdf5'))

        return

    @classmethod
    def from_hdf5(
            cls, path, outformat=h5py.Dataset, mode='r',
            load_example_trigs=True, outputdir_to_use=None):
        """
        Load the class from a hdf5 file
        :param path: Path to the hdf5 file, can also be a File object
        :param outformat:
            Format of the big data in the output (default is dataset, can also
            be mmap). It can be a string or a type
        :param mode: Mode to open the hdf5 file or mmap in
        :param load_example_trigs: Boolean flag to load the example trigs
        :param outputdir_to_use:
            If given, replace the outputdir with this path for the example trigs
        :return: Rank instance with self.fobj set to the hdf5 file
        """
        fobj, _ = utils.get_hdf5_file(path, mode=mode, raise_error=True)

        # Define format to use for the big data
        if ((isinstance(outformat, str) and
             outformat.lower() in ("mmap", "memmap")) or
                (isinstance(outformat, type) and
                 issubclass(outformat, np.memmap))):
            outformat = np.memmap
            if mode == 'a':
                # We can't write to a memmap in append mode
                mode = 'r+'
        else:
            outformat = h5py.Dataset

        # Set up the class structure
        instance = cls(empty_init=True)

        # Load the attributes
        utils.load_dict_from_hdf5_attrs(fobj, outdict=instance.__dict__)

        # Load the big data
        # First, the example trigs
        if load_example_trigs:
            # Replace the outputdir with the one given if needed
            if outputdir_to_use is not None:
                for run, rundirs in enumerate(instance.outputdirs):
                    for subbank, outputdir in enumerate(rundirs):
                        instance.outputdirs[run][subbank] = pathlib.PosixPath(
                            os.path.join(outputdir_to_use,
                                         os.path.basename(outputdir)))
            # Only the first run is needed to get the example trigs
            example_jsons = [glob.glob(
                os.path.join(outputdir, "*config.json"))[0]
                             for outputdir in instance.outputdirs[0]]
            example_trigs = [trig.TriggerList.from_json(
                    example_json, load_trigs=False, do_ffts=False)
                    for example_json in example_jsons]
            instance.example_trigs = example_trigs
        else:
            instance.example_trigs = None

        instance.bg_fg_by_subbank = [{} for _ in FOBJ_KEYS]
        instance.cands_preveto_max = None
        instance.cands_postveto_max = None

        extra_array_names = list(instance.extra_array_names)
        idx_prior = extra_array_names.index('prior_terms') if \
            'prior_terms' in extra_array_names else None
        for icand, candtype in enumerate(FOBJ_KEYS):
            # Load the entries of bg_fg_by_subbank[icand]
            cdict = instance.bg_fg_by_subbank[icand]
            prior_terms = {}
            sensitivity_params = {}
            for key, val in fobj[candtype].items():
                # Loop over the subbanks
                if not isinstance(val, h5py.Group):
                    continue

                loc_id = ast.literal_eval(key)
                if not isinstance(loc_id, tuple):
                    print(f"Skipping {key} in {candtype}")
                    continue

                # First load the processedclists
                if outformat == np.memmap:
                    pclists = utils.mmap_h5(
                        fobj, [candtype, key, 'events'], mode=mode)
                else:
                    pclists = utils.read_hdf5_node(val, 'events')

                # Then load the extra arrays we saved
                extra_arrays = []
                for extra_array_name in instance.extra_array_names:
                    if (outformat == h5py.Dataset or
                            extra_array_name == 'timeseries'):
                        # mmap isn't supported for variable length timeseries
                        extra_arrays.append(
                            utils.read_hdf5_node(val, extra_array_name))
                    else:
                        extra_arrays.append(utils.mmap_h5(
                            fobj, [candtype, key, extra_array_name], mode=mode))

                # Populate the entries of bg_fg_by_subbank[icand]
                if utils.checkempty(instance.extra_array_names):
                    cdict[loc_id] = pclists
                else:
                    cdict[loc_id] = [pclists, *extra_arrays]

                # Load the sensitivity params and prior terms if they exist
                if idx_prior is not None:
                    prior_terms[loc_id] = extra_arrays[idx_prior]
                elif 'prior_terms' in val:
                    if outformat == h5py.Dataset:
                        prior_terms[loc_id] = \
                            utils.read_hdf5_node(val, 'prior_terms')
                    else:
                        prior_terms[loc_id] = utils.mmap_h5(
                            fobj, [candtype, key, 'prior_terms'], mode=mode)

                if 'sensitivity_params' in val:
                    if outformat == h5py.Dataset:
                        sensitivity_params[loc_id] = \
                            utils.read_hdf5_node(val, 'sensitivity_params')
                    else:
                        sensitivity_params[loc_id] = utils.mmap_h5(
                            fobj, [candtype, key, 'sensitivity_params'],
                            mode=mode)

            # Load cands_preveto_max if saved
            if 'cands_preveto_max' in fobj[candtype]:
                # Create the structure if it doesn't exist
                if instance.cands_preveto_max is None:
                    instance.cands_preveto_max = [[] for _ in FOBJ_KEYS]

                # Read n x 3 array with loc_id and indices into a
                # bg_fg_by_subbank entry
                index_array = utils.read_hdf5_node(
                    fobj, [candtype, "cands_preveto_max"])[...][()]
                for *loc_id, idx in index_array:
                    loc_id = tuple(loc_id)
                    prior_terms_ev = prior_terms[loc_id][idx]
                    if utils.checkempty(instance.extra_array_names):
                        pclist_ev = cdict[loc_id][idx]
                        extra_arrays_ev = []
                    else:
                        pclist_ev = cdict[loc_id][0][idx]
                        # Skip the prior terms as we already loaded them
                        extra_arrays_ev = [
                            cdict[loc_id][i + 1][idx] for i, label in
                            enumerate(instance.extra_array_names)
                            if label != 'prior_terms']
                    sensitivity_params_ev = sensitivity_params[loc_id][idx]
                    event_entry = [
                        prior_terms_ev,
                        pclist_ev,
                        loc_id,
                        sensitivity_params_ev,
                        *extra_arrays_ev,
                        idx]
                    instance.cands_preveto_max[icand].append(event_entry)

            # Load cands_postveto_max if saved
            if 'cands_postveto_max' in fobj[candtype]:
                # Create the structure the first time
                if instance.cands_postveto_max is None:
                    instance.cands_postveto_max = [[] for _ in FOBJ_KEYS]

                # Read n x 3 array with loc_id and indices into a
                # bg_fg_by_subbank entry
                index_array = utils.read_hdf5_node(
                    fobj, [candtype, "cands_postveto_max"])[...][()]
                for *loc_id, idx in index_array:
                    loc_id = tuple(loc_id)
                    prior_terms_ev = prior_terms[loc_id][idx]
                    if utils.checkempty(instance.extra_array_names):
                        pclist_ev = cdict[loc_id][idx]
                        extra_arrays_ev = []
                    else:
                        pclist_ev = cdict[loc_id][0][idx]
                        # Skip the prior terms as we already loaded them
                        extra_arrays_ev = [
                            cdict[loc_id][i + 1][idx] for i, label in
                            enumerate(instance.extra_array_names)
                            if label != 'prior_terms']
                    sensitivity_params_ev = sensitivity_params[loc_id][idx]
                    event_entry = [
                        prior_terms_ev,
                        pclist_ev,
                        loc_id,
                        sensitivity_params_ev,
                        *extra_arrays_ev,
                        idx]
                    instance.cands_postveto_max[icand].append(event_entry)

        # Create derived structures
        instance.bg_fg_max = []
        for cand_dict in instance.bg_fg_by_subbank:
            instance.bg_fg_max.append(combine_subbank_dict(cand_dict))

        # cands_preveto_max[0] -> scores_bg_by_subbank_nonvetoed
        if instance.cands_preveto_max is not None:
            dic_by_subbank = split_into_subbank_dicts(
                instance.cands_preveto_max[0], 2, skip_index=False)
            instance.scores_bg_by_subbank_nonvetoed = [
                [list(x) for x in zip(*dic_by_subbank[loc_id])]
                for loc_id in sorted(dic_by_subbank.keys())]

        # cands_postveto_max[0] -> scores_bg_by_subbank
        if instance.cands_postveto_max is not None:
            dic_by_subbank = split_into_subbank_dicts(
                instance.cands_postveto_max[0], 2, skip_index=False)
            instance.scores_bg_by_subbank = [
                [list(x) for x in zip(*dic_by_subbank[loc_id])]
                for loc_id in sorted(dic_by_subbank.keys())]

        # Finally, load fobj into the class instance
        instance.fobj = fobj

        return instance

    def create_coh_score_instance(
            self, cs_ver='JR', cs_table=None, example_trigs=None, **cs_kwargs):
        """
        Removed from init to make saved rank objects independent of 
        the cogwheel version
        :param cs_ver:
            Version of coherent score to use, can be 'JR', 'O2', 'mz'
        :param cs_table:
            Path to file with coherent score table (can be dictionary mapping
            (bank_id, subbank ID) to file), in case cs_ver == 'mz', it is
            instead a path to the appropriate coherent score npz file
        :param example_trigs: List of example triggers to use for coherent score
        :param cs_kwargs: Additional arguments to pass to the coherent score
        :return: Instance of coherent score
        """
        # Use self.cs_kwargs updated with what we pass here
        cs_kwargs_to_use = self.cs_kwargs.copy()
        cs_kwargs_to_use.update(cs_kwargs)
        if 'cs_ver' in self.cs_kwargs:
            cs_kwargs_to_use.pop('cs_ver')
            if self.cs_kwargs['cs_ver'] != cs_ver:
                raise RuntimeWarning(f"Changing coherent score version from\
                   {self.cs_kwargs['cs_ver']} to {cs_ver}")

        print('Creating coherent score instance(s)')
        # Assumes all runs had the same format of their processedclists
        # Warning: precludes combined analysis of old O1 triggers and new ones
        if cs_ver.lower() == 'jr':
            # import coherent_score_hm_search as cs_JR
            cs_instance = cs_JR.initialize_cs_instance(
                detectors=self.detectors, **cs_kwargs_to_use)

        elif cs_ver.lower() == 'mz':
            cs_instance = cs_mz.CoherentScoreMZ(
                samples_fname=cs_table, run=self.runs[0])
        else:
            raise ValueError(f"Unknown coherent score version {cs_ver}")

        return cs_instance

    # Functions to score lists
    # --------------------------------------------------------------
    def reform_scores_lists(self):
        """
        After maximizing over banks, there are fewer entries in
        cands_preveto_max, which are no longer in order of subbank_id. Apart
        from the return, this function
        1. Recomputes the median normfacs for the subbanks
        2. Reevaluates the sensitivity penalties for all candidates
        3. Reorders the entries in cands_preveto_max
        :return: A list of len(nsubbanks) with scores_bg_by_subbank_nonvetoed
        """
        if self.cands_preveto_max is None:
            # Nothing to do
            return

        scores_by_subbank_lists = [[] for _ in self.cands_preveto_max]

        if self.fobj is not None:
            # We can do it the fast way, read the sensitivity params
            nfacs_dicts = self.compute_function_on_subbank_data(
                'sensitivity_params', None, False, lambda arr: arr[..., 3][()])
            # Don't dereference the prior terms as we'll update them
            sensitivity_penalty_dicts = self.compute_function_on_subbank_data(
                'prior_terms', None, False, lambda arr: arr[..., 2])

            # Evaluate the median H1 normfac for background triggers and update
            # the sensitivity penalties
            for loc_id, nfacs in sorted(nfacs_dicts[0].items()):
                # Get the indices for the entries in cands_preveto_max[0]
                # with the same loc_id
                inds = [x[-1] for x in self.cands_preveto_max[0]
                        if tuple(x[2]) == loc_id]
                new_median_normfacs = np.median(nfacs[inds], axis=0)

                # Fix the median sensitivity penalty with this new median
                # The median normfacs[0] is not a bug, H1 sets a scale
                ptoffset = 2 * np.log(
                    (self.median_normfacs_by_subbank[loc_id][0] /
                     new_median_normfacs[0]) ** 3)

                # Fix the median sensitivity penalty for all the candidate types
                for sensitivity_penalty_dict in sensitivity_penalty_dicts:
                    sensitivity_penalty_dict[loc_id] += ptoffset

                # Having applied the new median, overwrite the saved one
                self.median_normfacs_by_subbank[loc_id] = new_median_normfacs

                # For each candidate type, collect the triggers with the same
                # loc_id into our emerging list
                # n_cand_types x n_events in the subbank
                scores_lists_loc_id = \
                    [[ev for ev in evlist if tuple(ev[2]) == loc_id]
                     for evlist in self.cands_preveto_max]
                # Save the lists to structure by subbank
                for icand, evlist_loc_id in enumerate(scores_lists_loc_id):
                    scores_by_subbank_lists[icand].append(evlist_loc_id)

            # We edited the hdf5 arrays, so let's flush to disk
            self.fobj.flush()
        else:
            # Do it the slow way
            # Go over the list subbank by subbank
            for loc_id in sorted(self.median_normfacs_by_subbank.keys()):
                # n_cand_types x n_events in the subbank
                scores_lists_loc_id = \
                    [[ev for ev in evlist if tuple(ev[2]) == loc_id]
                     for evlist in self.cands_preveto_max]

                # TODO: Revisit for fishing
                # n_det array of median normfacs from the background events
                new_median_normfacs = np.median(np.array(
                    [ev[3][:, 3] for ev in scores_lists_loc_id[0]]), axis=0)

                # Fix the median sensitivity penalty with this new median
                # The median normfacs[0] is not a bug, H1 sets a scale
                ptoffset = 2 * np.log(
                    (self.median_normfacs_by_subbank[loc_id][0] /
                     new_median_normfacs[0]) ** 3)

                # Fix the median sensitivity penalty for all the lists
                for evlist_loc_id in scores_lists_loc_id:
                    for ev in evlist_loc_id:
                        ev[0][2] += ptoffset

                # Having applied the new median, overwrite the saved one
                self.median_normfacs_by_subbank[loc_id] = new_median_normfacs

                # Save the lists to structure by subbank
                for icand, evlist_loc_id in enumerate(scores_lists_loc_id):
                    scores_by_subbank_lists[icand].append(evlist_loc_id)

        # Each entry of cands_preveto_max is no longer in order of subbanks
        # after bank reassignment, so rearrange to fix that
        for icand, evlist_loc_ids in enumerate(scores_by_subbank_lists):
            self.cands_preveto_max[icand] = sum(evlist_loc_ids, [])

        # return scores_bgs_by_subbank_nonvetoed
        return scores_by_subbank_lists[0]

    def compute_function_on_subbank_data(
            self, leafname, entry_idx, return_lists, func, *args, **kwargs):
        """
        It can be slow to loop over the entries in scores_(non)vetoed_max, read
        them into memory, and compute something, in bulk. It is faster to
        dereference the hdf5 arrays in bg_fg_by_subbank once, compute what we
        need and read the values relevant to scores_(non)vetoed_max
        :param leafname: Name of the hdf5 node to read (if known)
        :param entry_idx:
            Index into an entry in bg_fg_by_subbank (used if leafname is None)
        :param return_lists:
            Flag to return lists for cands_preveto_max and cands_postveto_max
        :param func: Function to apply to the data
        :param args: Arguments to pass to the function
        :param kwargs: Keyword arguments to pass to the function
        :return:
            1. If not return_lists, a list of dicts with func evaluated for the
               entries in bg_fg_by_subbank
            2. If return_lists, a list of dicts with func evaluated for the
               entries in bg_fg_by_subbank, and lists of func evaluated for the
               entries in cands_preveto_max, and cands_postveto_max
        """
        if leafname is None and entry_idx is None:
            # Nothing to do
            return

        # Go cand type by cand type
        dicts_all = []
        for candtype, bg_fg_dict in zip(FOBJ_KEYS, self.bg_fg_by_subbank):
            dict_candtype = {}
            for loc_id, entries in bg_fg_dict.items():
                h5path = [candtype, str(loc_id), leafname if leafname else '']
                prior_path = os.path.join(*h5path)
                if (self.fobj is not None) and (leafname is not None) and \
                        (prior_path in self.fobj):
                    # Look up the saved array
                    saved_array = utils.read_hdf5_node(
                        self.fobj, h5path, create=False)
                elif entry_idx is not None:
                    saved_array = entries[entry_idx]
                else:
                    print(f"Couldn't find {leafname} for {candtype} {loc_id}")
                    return
                dict_candtype[loc_id] = func(saved_array, *args, **kwargs)

            dicts_all.append(dict_candtype)

        if not return_lists:
            return dicts_all

        fvals_cands_preveto_max = []
        if self.cands_preveto_max is not None:
            for dict_candtype, bg_fg_list in zip(
                    dicts_all, self.cands_preveto_max):
                # 2 for loc_id and -1 for index into bg_fg_by_subbank
                return_val = [dict_candtype[x[2]][x[-1]] for x in bg_fg_list]
                try:
                    return_val = np.array(return_val)
                except ValueError:
                    pass
                fvals_cands_preveto_max.append(return_val)

        fvals_cands_postveto_max = []
        if self.cands_postveto_max is not None:
            for dict_candtype, bg_fg_list in zip(
                    dicts_all, self.cands_postveto_max):
                # 2 for loc_id and -1 for index into bg_fg_by_subbank
                return_val = [dict_candtype[x[2]][x[-1]] for x in bg_fg_list]
                try:
                    return_val = np.array(return_val)
                except ValueError:
                    pass
                fvals_cands_postveto_max.append(return_val)

        return dicts_all, fvals_cands_preveto_max, fvals_cands_postveto_max

    def score_bg_fg_lists(
            self, redo_bg=False, redo_fg=False, apply_veto_before_scoring=True,
            score_triggers=True, coherent_score_func=utils.coherent_score,
            include_vetoed_triggers=False, safety_factor=4, matching_point=None,
            scoring_method="old", downsampling_correction=True,
            min_trigs_per_grp=500,
            p_veto_real_event=(lambda x: 0.05, lambda x: 0.05),
            **ranking_kwargs):
        """
        Compute quantities needed to assign scores to the triggers
        Responsible for populating:
         1. coherent scores
         2. rank functions (CDF of P(SNR^2|H0) in each detector)
        If we are rerunning with a different subset of vetoes, update the
        desired set in self.mask_veto_criteria and, if
        apply_veto_before_scoring is True, set redo_bg = True
        :param redo_bg:
            Flag whether to override the saved coherent scores for background
            triggers and recompute them (also recomputes for the foreground)
        :param redo_fg:
            Flag whether to override the saved coherent scores for foreground
            triggers and recompute them
        :param apply_veto_before_scoring:
            Flag whether to apply the vetoes before scoring. The vetoes to
            apply are set by self.mask_veto_criteria on self.veto_metadata_keys
            (we apply them after scoring anyway)
        :param score_triggers: Boolean flag to compute the scores
        :param coherent_score_func:
            Function that accepts the coherent terms for a trigger and returns
            the coherent score
        :param include_vetoed_triggers:
            Flag to include vetoed triggers with a penalty in the final list
        :param safety_factor:
            Add this to threshold_chi2 before estimating the rank functions
            to account for incompleteness related to optimization
        :param matching_point:
            Set SNR^2 at which the rank functions are matched, the default
            is threshold network SNR^2/2
        :param scoring_method:
            Flag to indicate whether we're using the old (mz/cdf) way of ranking
            vs new way (fitting for the pdf)
        :param downsampling_correction:
            If the triggers were downsampled compared to a chi-sq distribution
            because of an additional cut (e.g., based on whether the mode ratios
            A33/A22 or A44/A22 are physical). This flag corrects the rank
            function so that it follows the chi-sq behavior again. This flag
            needs a file downsamp_corr_path.npy to be input when creating
            Rank class object
        :param min_trigs_per_grp:
            To avoid pathologies with making the rank functions, we require that
            the templates in each group have more than a particular
            number of background triggers associated to them
        :param p_veto_real_event:
            Tuple with functions for the probability that a real event fails the
            vetoes in each detector, which in the most general case can be a
            function of all properties of the trigger. They should accept a list
            of entries of scores_(non)vetoed_max and yield an array of
            probabilities.
        :param ranking_kwargs:
            Any extra arguments we want to pass to the ranking function
        :return:
        """
        # ------------------------------------------------
        print('Computing coherent scores')

        # As good practice, first fix the bg_fg_by_subbank to read in any extra
        # entries in bg_fg_max that were added e.g. as injections
        check_and_fix_bg_fg_lists(
            self.bg_fg_max, self.bg_fg_by_subbank, self.extra_array_names,
            self.fobj, raise_error=False)

        mask_veto_criteria = None
        if apply_veto_before_scoring:
            if isinstance(apply_veto_before_scoring, bool):
                mask_veto_criteria = self.mask_veto_criteria
            else:
                mask_veto_criteria = apply_veto_before_scoring

        if (self.cands_preveto_max is None) or redo_bg:
            # Do the heavy lifting the first time
            # Per subbank for the background to get the median normfacs
            self.scores_bg_by_subbank_nonvetoed, \
                self.median_normfacs_by_subbank = compute_coherent_scores(
                    self.bg_fg_by_subbank[0], self.extra_array_names,
                    self.clist_pos,
                    time_slide_jump=self.time_slide_jump,
                    mask_veto_criteria=mask_veto_criteria,
                    template_prior_funcs=self.template_prior_funcs,
                    fobj=self.fobj, candtype=FOBJ_KEYS[0])

            # Start structure to save to
            self.cands_preveto_max = [
                sum(self.scores_bg_by_subbank_nonvetoed, [])]

            # Score the foreground events and save to the structure
            for ind, fg_dict in enumerate(self.bg_fg_by_subbank[1:]):
                scores_fg_by_subbank_nonvetoed, _ = compute_coherent_scores(
                    fg_dict, self.extra_array_names, self.clist_pos,
                    time_slide_jump=self.time_slide_jump,
                    mask_veto_criteria=mask_veto_criteria,
                    median_normfacs_by_subbank=self.median_normfacs_by_subbank,
                    template_prior_funcs=self.template_prior_funcs,
                    fobj=self.fobj, candtype=FOBJ_KEYS[ind + 1])
                self.cands_preveto_max.append(
                    sum(scores_fg_by_subbank_nonvetoed, []))

        else:
            # # We might have lost some events due to floating between banks,
            # # redo the subbank lists and medians, edits its inputs to
            # # preserve order
            # self.scores_bg_by_subbank_nonvetoed = reform_coherent_scores(
            #     self.cands_preveto_max, self.median_normfacs_by_subbank)
            # We might have lost some events due to floating between banks,
            # redo the subbank lists, ordering, and medians
            self.scores_bg_by_subbank_nonvetoed = self.reform_scores_lists()

            if redo_fg:
                # Forget old bank assignment for the FG
                # Warning: If you're doing this, redo bank assignment after!
                for ind, fg_dict in enumerate(self.bg_fg_by_subbank[1:]):
                    scores_fg_by_subbank_nonvetoed, _ = compute_coherent_scores(
                        fg_dict, self.extra_array_names, self.clist_pos,
                        time_slide_jump=self.time_slide_jump,
                        mask_veto_criteria=mask_veto_criteria,
                        median_normfacs_by_subbank=self.median_normfacs_by_subbank,
                        template_prior_funcs=self.template_prior_funcs,
                        fobj=self.fobj, candtype=FOBJ_KEYS[ind + 1])
                    self.cands_preveto_max[ind + 1] = sum(
                        scores_fg_by_subbank_nonvetoed, [])

        if not score_triggers:
            return

        # ------------------------------------------------
        # Since we finished collecting the triggers and applied whichever
        # subset of vetoes we asked for at that stage, let's revert to the
        # full set of vetos
        mask_veto_criteria = self.mask_veto_criteria
        # # Find the index of veto_metadata in scores_(non)vetoed_max after
        # # accounting for the removal of prior terms
        # # (added in case we change order in the future)
        # idx_veto_metadata = utils.index_after_removal(
        #     self.extra_array_names, 'veto_metadata', 'prior_terms')
        # if idx_veto_metadata is not None:
        #     # The preceding elements are
        #     # prior terms, processed_clist, loc_id, sensitivity params
        #     idx_veto_metadata += 4

        # Index of veto_metadata in bg_fg_by_subbank dictionary values
        idx_veto_metadata = utils.index_after_removal(
            self.extra_array_names, 'veto_metadata')
        if idx_veto_metadata is not None:
            idx_veto_metadata += 1
        # Index of secondary peak vetoes in the list of vetoes
        ind_second_peak = utils.index_after_removal(
            self.veto_metadata_keys, 'Secondary_peak_timeseries')

        # if (apply_veto_before_scoring and
        #         (len(self.cands_preveto_max[0][0]) > 5)):
        # if ((mask_veto_criteria is not None) and
        #         (len(self.cands_preveto_max[0][0]) > 5)):
        if apply_veto_before_scoring and (idx_veto_metadata is not None):
            print("Applying vetoes")
            # We're using the new format, apology to students
            veto_inds = np.where(mask_veto_criteria)[0]

            # For each cand type, define an n_events x (n_det=2) array with
            # overall pass/fail for all vetoes
            passed_veto_masks_dicts, passed_veto_masks_det, _ = \
                self.compute_function_on_subbank_data(
                    'veto_metadata', idx_veto_metadata, True,
                    lambda arr: np.all(arr[..., veto_inds], axis=-1))
            # For each cand type, define an n_events array with pass/fail for
            # the secondary peak veto
            if ind_second_peak is not None:
                fn_secondary_peak = \
                    lambda arr: np.all(arr[..., ind_second_peak], axis=-1)
            else:
                fn_secondary_peak = lambda arr: np.ones(len(arr), dtype=bool)
            _, passed_secondary_peak_masks, _ = \
                self.compute_function_on_subbank_data(
                    'veto_metadata', idx_veto_metadata, True, fn_secondary_peak)

            # Masks to identify triggers that passed vetoes in both detectors
            # 4 x n_events array
            # passed_veto_masks = \
            #     [np.all(x, axis=-1) for x in passed_veto_masks_det]
            passed_veto_masks = \
                [np.all(x, axis=-1) if x.size > 0 else np.array([], dtype=bool)
                 for x in passed_veto_masks_det]

            self.cands_postveto_max = [
                [x for passflag, x in zip(passed_veto_mask, bg_fg_list)
                 if passflag] for passed_veto_mask, bg_fg_list in
                zip(passed_veto_masks, self.cands_preveto_max)]
            self.scores_bg_by_subbank = [
                [x for x in subbank_list if
                 np.all(passed_veto_masks_dicts[0][loc_id][x[-1]])]
                for loc_id, subbank_list in
                zip(sorted(self.median_normfacs_by_subbank.keys()),
                    self.scores_bg_by_subbank_nonvetoed)]
        else:
            # No vetoes to apply
            self.scores_bg_by_subbank = self.scores_bg_by_subbank_nonvetoed
            self.cands_postveto_max = self.cands_preveto_max
            passed_veto_masks_det = [
                np.ones((len(bg_fg_list), len(self.detectors)), dtype=bool)
                for bg_fg_list in self.cands_preveto_max]
            passed_secondary_peak_masks = [
                np.ones(len(bg_fg_list), dtype=bool)
                for bg_fg_list in self.cands_preveto_max]
            passed_veto_masks = \
                [np.all(x, axis=-1) if x.size > 0 else np.array([], dtype=bool)
                 for x in passed_veto_masks_det]

        # Compute coherent scores for the background and events
        idx_prior_terms = utils.index_after_removal(
            self.extra_array_names, 'prior_terms')
        if idx_prior_terms is not None:
            # Index into the entries of bg_fg_by_subbank
            idx_prior_terms += 1
        # Much faster to dereference the hdf5 array once and compute the scores
        if (self.fobj is not None) or (idx_prior_terms is not None):
            _, cscores_cands_preveto_max, cscores_cands_postveto_max = \
                self.compute_function_on_subbank_data(
                    'prior_terms', idx_prior_terms, True, coherent_score_func)
        else:
            # Do it the slow way
            cscores_cands_preveto_max = [np.array(
                [coherent_score_func(event[0]) for event in bg_fg_list])
                for bg_fg_list in self.cands_preveto_max]
            cscores_cands_postveto_max = [np.array(
                [coherent_score_func(event[0]) for event in bg_fg_list])
                for bg_fg_list in self.cands_postveto_max]

        if include_vetoed_triggers:
            # ------------------------------------------------
            print('Computing coherent scores for background and events')
            # Store coherent scores for all triggers regardless of veto
            # We'll omit triggers with secondary peaks even if we're ignoring
            # vetoes since they are super coherent and ruin the procedure
            cscores_cands_preveto_max = [
                np.array([x if result else -10**5 for result, x in
                          zip(passed_mask, cscores_bg_fg_list)])
                for passed_mask, cscores_bg_fg_list in
                zip(passed_secondary_peak_masks, cscores_cands_preveto_max)]
            self.coherent_scores_bg, self.coherent_scores_cand, \
                self.coherent_scores_lsc, self.coherent_scores_inj = \
                cscores_cands_preveto_max
        else:
            # ------------------------------------------------
            print('Computing coherent scores for background and events')
            # Store coherent scores for only vetoed triggers
            self.coherent_scores_bg, self.coherent_scores_cand, \
                self.coherent_scores_lsc, self.coherent_scores_inj = \
                cscores_cands_postveto_max

        # ------------------------------------------------
        print('Defining rank-based scores and computing them ' +
              'for background and events')
        # Get rhosq and calpha arrays, the [()] dereferences the array once
        _, rhosqs_cands_preveto_max, _ = \
            self.compute_function_on_subbank_data(
                'events', 0, True, lambda arr: arr[..., 1][()])
        _, calphas_cands_preveto_max, _ = \
            self.compute_function_on_subbank_data(
                'events', 0, True,
                lambda arr: arr[..., 0, self.clist_pos['c0_pos']:][()])

        self.snr2max = int(np.max(np.sum(
            rhosqs_cands_preveto_max[0][passed_veto_masks[0], :],
            axis=-1))) + 1.0
        # self.snr2max = int(np.max(
        #     [event[1][0, 1] + event[1][1, 1]
        #      for event in self.cands_postveto_max[0]])) + 1.0

        # List of scores for vetoed triggers
        self.back_0_score_vetoed = []
        self.back_1_score_vetoed = []
        self.cand_0_score_vetoed, self.cand_1_score_vetoed = np.zeros(
            (2, len(self.cands_postveto_max[1])))
        self.lsc_0_score_vetoed, self.lsc_1_score_vetoed = np.zeros(
            (2, len(self.cands_postveto_max[2])))
        self.inj_0_score_vetoed, self.inj_1_score_vetoed = np.zeros(
            (2, len(self.cands_postveto_max[3])))

        # List of scores for all triggers
        self.back_0_score = []
        self.back_1_score = []
        self.cand_0_score, self.cand_1_score = np.zeros(
            (2, len(self.cands_preveto_max[1])))
        self.lsc_0_score, self.lsc_1_score = np.zeros(
            (2, len(self.cands_preveto_max[2])))
        self.inj_0_score, self.inj_1_score = np.zeros(
            (2, len(self.cands_preveto_max[3])))

        # List of size n_subbanks with (score_func_h1, score_func_l1)
        self.score_funcs = []
        for isb in range(len(self.scores_bg_by_subbank)):
            if scoring_method.lower() == "new":
                # TODO: Fix this function
                # First compute rank scores for triggers that passed the vetos
                # Populates self.back_0_score_vetoed, self.back_1_score_vetoed
                Rank.rank_scores_calc(
                    self, isb, safety_factor=safety_factor,
                    matching_point=matching_point, vetoed=True,
                    **ranking_kwargs)

                if include_vetoed_triggers:
                    # Then compute scores for all triggers and populate
                    # self.back_0_score, self.back_1_score
                    Rank.rank_scores_calc(
                        self, isb, safety_factor=safety_factor,
                        matching_point=matching_point, vetoed=False,
                        **ranking_kwargs)
            else:
                # First compute rank scores for triggers that passed the vetos
                # Populates self.back_0_score_vetoed, self.back_1_score_vetoed
                self.rank_scores_calc_MZ(
                    isb, safety_factor=safety_factor,
                    matching_point=matching_point,
                    downsampling_correction=downsampling_correction,
                    min_trigs_per_grp=min_trigs_per_grp, vetoed=True,
                    rhosqs_arrays=rhosqs_cands_preveto_max,
                    calphas_arrays=calphas_cands_preveto_max,
                    masks_vetoed=passed_veto_masks, **ranking_kwargs)

                if include_vetoed_triggers:
                    # Then compute scores for all triggers and populate
                    # self.back_0_score, self.back_1_score
                    self.rank_scores_calc_MZ(
                        isb, safety_factor=safety_factor,
                        matching_point=matching_point,
                        downsampling_correction=downsampling_correction,
                        min_trigs_per_grp=min_trigs_per_grp, vetoed=False,
                        rhosqs_arrays=rhosqs_cands_preveto_max,
                        calphas_arrays=calphas_cands_preveto_max,
                        **ranking_kwargs)

        # Always defined and populated
        self.back_0_score_vetoed = np.concatenate(self.back_0_score_vetoed)
        self.back_1_score_vetoed = np.concatenate(self.back_1_score_vetoed)

        if include_vetoed_triggers:
            utils.close_hdf5()
            raise NotImplementedError

            # self.back_0_score = np.concatenate(self.back_0_score)
            # self.back_1_score = np.concatenate(self.back_1_score)
            #
            # # Add a penalty for the possibility that real signals fail the veto
            # # Add to all the scores, we will overwrite the ones that passed the
            # # veto later
            # self.back_0_score += 2 * np.log(
            #     p_veto_real_event[0](self.cands_preveto_max[0]))
            # self.back_1_score += 2 * np.log(
            #     p_veto_real_event[1](self.cands_preveto_max[0]))
            # self.cand_0_score += 2 * np.log(
            #     p_veto_real_event[0](self.cands_preveto_max[1]))
            # self.cand_1_score += 2 * np.log(
            #     p_veto_real_event[1](self.cands_preveto_max[1]))
            # self.lsc_0_score += 2 * np.log(
            #     p_veto_real_event[0](self.cands_preveto_max[2]))
            # self.lsc_1_score += 2 * np.log(
            #     p_veto_real_event[1](self.cands_preveto_max[2]))
            # self.inj_0_score += 2 * np.log(
            #     p_veto_real_event[0](self.cands_preveto_max[3]))
            # self.inj_1_score += 2 * np.log(
            #     p_veto_real_event[1](self.cands_preveto_max[3]))
            #
            # # Replace the scores with the appropriate rank functions for the
            # # ones that passed the veto
            # # TODO: This is incorrect, passed_veto_masks_det[0][:, 0] is a mask
            # #  into triggers that failed the H1 veto, while
            # #  self.back_0_score_vetoed is computed on triggers that passed both
            # #  vetoes. Fix this...
            # self.back_0_score[passed_veto_masks_det[0][:, 0]] = \
            #     self.back_0_score_vetoed
            # self.back_1_score[passed_veto_masks_det[0][:, 1]] = \
            #     self.back_1_score_vetoed
            # self.cand_0_score[passed_veto_masks_det[1][:, 0]] = \
            #     self.cand_0_score_vetoed
            # self.cand_1_score[passed_veto_masks_det[1][:, 1]] = \
            #     self.cand_1_score_vetoed
            # if len(passed_veto_masks_det[2]) > 0:
            #     self.lsc_0_score[passed_veto_masks_det[2][:, 0]] = \
            #         self.lsc_0_score_vetoed
            #     self.lsc_1_score[passed_veto_masks_det[2][:, 1]] = \
            #         self.lsc_1_score_vetoed
            # if len(passed_veto_masks_det[3]) > 0:
            #     self.inj_0_score[passed_veto_masks_det[3][:, 0]] = \
            #         self.inj_0_score_vetoed
            #     self.inj_1_score[passed_veto_masks_det[3][:, 1]] = \
            #         self.inj_1_score_vetoed

        else:
            self.back_0_score = self.back_0_score_vetoed
            self.back_1_score = self.back_1_score_vetoed
            self.cand_0_score = self.cand_0_score_vetoed
            self.cand_1_score = self.cand_1_score_vetoed
            self.lsc_0_score = self.lsc_0_score_vetoed
            self.lsc_1_score = self.lsc_1_score_vetoed
            self.inj_0_score = self.inj_0_score_vetoed
            self.inj_1_score = self.inj_1_score_vetoed

        return
        
    def rank_scores_calc(
            self, i_subbank, safety_factor=4, matching_point=None,
            output_rank_func_temp_grps=False, snr2_ref=50,
            n_glitchy_groups_seed=20, min_trigs_per_grp=500,
            downsampling_correction=False, n_calpha_dim=2, vetoed=True):
        """
        TODO: Speed this function up like rank_scores_calc_MZ
        Calculating ranking scores for all the bg, fg, lsc, inj triggers.
        We first group the templates based on their "glitchiness" and then make
        separate rank functions for different groups.
        :param i_subbank: index of the subbank within self.scores_bg_by_subbank
        :param safety_factor:
            Add this to threshold_chi2 before estimating the rank functions
            to account for incompleteness related to optimization
        :param matching_point:
            Set SNR^2 at which the rank functions are matched, the default
            is threshold network SNR^2/2
            (the rank fns are made such that all template groups within the same
             subbank match at self.snr2min and the least glitchy template group
             in all subbanks match at matching_point)
        :param output_rank_func_temp_grps:
            Flag if you want to output the rank funcs for separate template groups
            for reference (typically used for debugging)
        :param snr2_ref:
            Quantify templates' glitchiness by the fraction of triggers with
            SNR>snr2_ref
        :param n_glitchy_groups_seed:
            Number of seed groups to classify the templates into based on
            their glitchiness
        :param min_trigs_per_grp:
            To avoid pathologies with making the rank functions, we require that
            the templates in each group have more than a particular
            number of background triggers associated to them
        :param downsampling_correction:
            If the triggers were downsampled compared to a chi-sq distribution
            because of an additional cut (e.g., based on whether
            the mode ratios A33/A22 or A44/A22 are physical). This flag corrects
            the rank function so that it follows the chi-sq behavior again. This
            flag needs a file downsamp_corr_path.npy to be input when creating
            Rank class object
        :param n_calpha_dim:
            Number of calpha dimensions to include while grouping triggers,
            some banks have extra dimensions that are out of control
        :param vetoed: Flag to indicate whether we're computing the rank scores
            for vetoed or non-vetoed triggers
        """
        # ---------------------------------------------------------------------
        # Divide the templates into groups based on their glitchiness
        # ---------------------------------------------------------------------
        # First group all templates according to the closest coarse template
        if hasattr(self, 'example_trigs') and self.example_trigs is not None:
            trig_obj = self.example_trigs[i_subbank]
        else:
            example_json = glob.glob(
                os.path.join(self.outputdirs[0][i_subbank], "*config.json"))[0]
            trig_obj = trig.TriggerList.from_json(
                example_json, load_trigs=False, do_ffts=False)

        sub_id = self.subbank_subset[i_subbank]
        bank = trig_obj.templatebank
        coarse_axes = bank.make_grid_axes(
            trig_obj.delta_calpha, trig_obj.template_safety,
            trig_obj.force_zero)
        coarse_axes = coarse_axes[:len(trig_obj.dcalphas)]

        if vetoed:
            bg_list = self.scores_bg_by_subbank[i_subbank]
            fg_list = self.cands_postveto_max
        else:
            bg_list = self.scores_bg_by_subbank_nonvetoed[i_subbank]
            fg_list = self.cands_preveto_max

        back_0_subbank = np.array([event[1][0, 1] for event in bg_list])
        back_1_subbank = np.array([event[1][1, 1] for event in bg_list])
        fine_calphas_arrays = np.array(
            [event[1][0, self.clist_pos['c0_pos']:] for event in bg_list])

        calphas_arrays = utils.find_closest_coarse_calphas(
            coarse_axes, fine_calphas_arrays)
        calphas_arrays = calphas_arrays[:, :n_calpha_dim]

        # Find the unique calphas, and indices to reconstruct the original array
        unique_calphas, template_inds = np.unique(
            calphas_arrays, axis=0, return_inverse=True)

        # We will split templates into groups based on the fraction of triggers
        # above a reference SNR
        # TODO: Define the total fraction instead of the sum of that in each detector
        glitch_fraction = np.zeros((2, len(unique_calphas)))
        for i in range(len(unique_calphas)):
            back_template_h = back_0_subbank[template_inds == i]
            back_template_l = back_1_subbank[template_inds == i]

            nglitch_h = np.count_nonzero(back_template_h > snr2_ref)
            glitch_fraction[0, i] = nglitch_h / len(back_template_h)
            nglitch_l = np.count_nonzero(back_template_l > snr2_ref)
            glitch_fraction[1, i] = nglitch_l / len(back_template_l)
        sum_glitch_fraction = np.sum(glitch_fraction, axis=0)

        # We split the templates based on the sum of glitch fractions in H and L
        num_groups = n_glitchy_groups_seed
        _, groups = np.histogram(sum_glitch_fraction, bins=num_groups)
        # Slightly shifting the right-edge so all elements are inside
        groups[-1] += 1e-4
        # Index into which group each unique calpha lies in
        templates_group_ind = np.digitize(sum_glitch_fraction, bins=groups) - 1

        # Assign each trigger to its group, going unique calpha by unique calpha
        groups_ind_bg = np.zeros(len(calphas_arrays), dtype=int)
        for t, g in enumerate(templates_group_ind):
            groups_ind_bg[template_inds == t] = g
        # Count the number of triggers in each group
        trigs_per_grp = np.array([
            np.count_nonzero(groups_ind_bg == i) for i in range(num_groups)])

        # Remake the groups so that at there are more than min_trigs_per_grp
        # triggers per group, just to avoid pathologies with making the rank
        # functions
        new_groups = [groups[0]]
        sum_trigs = 0
        for i in range(0, len(groups) - 1):
            sum_trigs += trigs_per_grp[i]
            if sum_trigs > min_trigs_per_grp:
                new_groups.append(groups[i + 1])
                sum_trigs = 0
        # The right edge of the new groups needs to be the same as that of the
        # old groups to ensure all templates are included
        new_groups[-1] = groups[-1]

        # Use these consolidated groups instead of the originals
        groups = new_groups
        num_groups = len(groups) - 1
        # Index into which group each unique calpha lies in
        templates_group_ind = np.digitize(sum_glitch_fraction, bins=groups) - 1
        # Assign each trigger to its group, going unique calpha by unique calpha
        for t, g in enumerate(templates_group_ind):
            groups_ind_bg[template_inds == t] = g

        print(f'Split the templates into {num_groups} groups for calculating ' +
              'the rank functions.')

        # Similar to bg triggers, we split the cand, lsc and inj triggers into
        # these groups and collect info for rank func application below
        # Sets of dictionaries with keys = 'cand', 'lsc', and 'inj'
        subbank_masks = {}  # Mask that picks out the members within the subbank
        groups_ind = {}  # Index into groups for the triggers picked by the mask
        # n_triggers x 2 array with rho_H^2, rho_L^2 for the triggers picked
        # by the mask
        snrsq = {}
        # 2 x n_triggers array with H1, L1 rank scores for the triggers picked
        # by the mask
        rank_scores = {}
        for i, category in enumerate(['cand', 'lsc', 'inj']):
            # Skipping bg as we already did it
            i += 1
            subbank_masks[category] = [
                event[2][1] == sub_id for event in fg_list[i]]

            calphas_arrays = np.array(
                [event[1][0, self.clist_pos['c0_pos']:] for event in fg_list[i]
                 if event[2][1] == sub_id])
            calphas_arrays = utils.find_closest_coarse_calphas(
                coarse_axes, calphas_arrays)
            calphas_arrays = calphas_arrays[:, :n_calpha_dim]
            if calphas_arrays.size == 0:
                calphas_arrays = np.zeros((0, n_calpha_dim))
            template_indices = [
                np.argmin(np.linalg.norm(unique_calphas - calpha_event, axis=-1))
                for calpha_event in calphas_arrays]
            groups_ind[category] = templates_group_ind[template_indices]

            snrsq[category] = np.array([
                event[1][:, 1] for event in fg_list[i] if event[2][1] == sub_id])

            rank_scores[category] = np.zeros(
                (2, np.count_nonzero(subbank_masks[category])))

        # ---------------------------------------------------------------------
        # Generate rank scores for each group
        # ---------------------------------------------------------------------
        # Set a fiducial matching point to match the rank functions at
        if matching_point is None:
            # matching_point = snr2sumcut_safe / 2.0
            if self.matching_point is None:
                matching_point = self.snr2min
                self.matching_point = matching_point
            else:
                matching_point = self.matching_point
        else:
            self.matching_point = matching_point
        if matching_point < self.snr2min:
            raise RuntimeError("Cannot match where we did not collect!")

        # Some broad dsnr2, only used in the first pass, so the value doesn't
        # matter very much
        dsnr2 = 0.5
        snr2sumcut_safe = self.snr2sumcut + safety_factor

        # Calculating scores for bg triggers first and then others for
        # foreground categories
        back_0_score, back_1_score = np.zeros((2, len(back_0_subbank)))
        rank_funcs_groups = []
        matching_val_0 = None
        matching_val_1 = None
        if downsampling_correction:
            corrections = self.downsampling_corrections[i_subbank]
        else:
            corrections = None
        # Do it group by group
        for g in range(num_groups):
            back_0_group = back_0_subbank[groups_ind_bg == g]
            back_1_group = back_1_subbank[groups_ind_bg == g]
            back_0_group[back_0_group > params.SNR2_MAX_BOUND] = params.SNR2_MAX_BOUND - 1e-3
            back_1_group[back_1_group > params.SNR2_MAX_BOUND] = params.SNR2_MAX_BOUND - 1e-3
            
            fitfunc_x, fitfunc_y = rank22.make_score_funcs(
                back_0_group, back_1_group, self.snr2min, dsnr2,
                self.snr2sumcut, snr2sumcut_safe, downsampling_corrections=corrections)

            print(f"Finished ranking group {g}")

            if g == 0:
                matching_val_0 = -2 * np.log(
                    fitfunc_x(matching_point)/fitfunc_x(self.snr2min)) \
                    + 4*np.log(matching_point/self.snr2min)
                matching_val_1 = -2 * np.log(
                    fitfunc_y(matching_point)/fitfunc_y(self.snr2min)) \
                    + 4*np.log(matching_point/self.snr2min)
                    
            score_0_func = rank22.make_score_func_from_fit_func(
                fitfunc_x, matching_val_0, self.snr2min, use_HM=True)
            score_1_func = rank22.make_score_func_from_fit_func(
                fitfunc_y, matching_val_1, self.snr2min, use_HM=True)

            back_0_score[groups_ind_bg == g] = score_0_func(back_0_group)
            back_1_score[groups_ind_bg == g] = score_1_func(back_1_group)

            # Now score the foregrounds
            for cat in ['cand', 'lsc', 'inj']:
                snrsq_category = snrsq[cat][groups_ind[cat] == g]
                if len(snrsq_category) > 0:
                    rank_scores[cat][:, groups_ind[cat] == g] = \
                        np.atleast_1d(score_0_func(snrsq_category[:, 0])), \
                        np.atleast_1d(score_1_func(snrsq_category[:, 1]))

            if output_rank_func_temp_grps:
                rank_funcs_groups.append((score_0_func, score_1_func))

        # Populating all arrays for rank score
        if vetoed:
            if (output_rank_func_temp_grps and
                    isinstance(self.back_0_score_vetoed, np.ndarray)):
                self.back_0_score_vetoed = []
                self.back_1_score_vetoed = []
            self.back_0_score_vetoed.append(back_0_score)
            self.back_1_score_vetoed.append(back_1_score)
            structs_to_fill = [
                [self.cand_0_score_vetoed, self.cand_1_score_vetoed],
                [self.lsc_0_score_vetoed, self.lsc_1_score_vetoed],
                [self.inj_0_score_vetoed, self.inj_1_score_vetoed]]
            for cat, structs in zip(['cand', 'lsc', 'inj'], structs_to_fill):
                structs[0][subbank_masks[cat]] = rank_scores[cat][0]
                structs[1][subbank_masks[cat]] = rank_scores[cat][1]
        else:
            if (output_rank_func_temp_grps and
                    isinstance(self.back_0_score, np.ndarray)):
                self.back_0_score = []
                self.back_1_score = []
            self.back_0_score.append(back_0_score)
            self.back_1_score.append(back_1_score)
            structs_to_fill = [
                [self.cand_0_score, self.cand_1_score],
                [self.lsc_0_score, self.lsc_1_score],
                [self.inj_0_score, self.inj_1_score]]
            for cat, structs in zip(['cand', 'lsc', 'inj'], structs_to_fill):
                structs[0][subbank_masks[cat]] = rank_scores[cat][0]
                structs[1][subbank_masks[cat]] = rank_scores[cat][1]

        # Finding rank function for the entire subbank now
        # (just for reference, as this is not used in the scores)
        if vetoed:
            fitfunc_x, fitfunc_y = rank22.make_score_funcs(
                back_0_subbank, back_1_subbank, self.snr2min, dsnr2,
                self.snr2sumcut, snr2sumcut_safe,
                downsampling_corrections=corrections)
            score_0_func = rank22.make_score_func_from_fit_func(
                        fitfunc_x, matching_val_0, self.snr2min, use_HM=True)
            score_1_func = rank22.make_score_func_from_fit_func(
                        fitfunc_y, matching_val_1, self.snr2min, use_HM=True)
            self.score_funcs.append((score_0_func, score_1_func))

        if output_rank_func_temp_grps:
            return rank_funcs_groups
        
    # Functions to compute FARs
    # --------------------------------------------------------------
    def compute_fars_rank_score(self, print_to_screen=True):
        """Compute FARs using ranking scores as test statistics"""
        # Sort background according to ranking score
        back_score_t = self.back_0_score + self.back_1_score
        # Due to an edge case in O1, BBH (2, 3) has so few templates that the
        # score funcs return NAN
        nanmask = np.isnan(back_score_t)
        if np.any(nanmask):
            print("Warning: there are NANs in the scores, can happen due to " +
                  "a small subbank, such as BBH (2, 3) in O1")
            back_score_t = np.nan_to_num(back_score_t, copy=False, nan=-np.inf)
        back_score_t.sort()

        # Compute ranking score for fg
        cand_t = self.cand_0_score + self.cand_1_score
        lsc_t = self.lsc_0_score + self.lsc_1_score
        inj_t = self.inj_0_score + self.inj_1_score

        # Compute FARs using the ranking scores
        # Number of background scores >= candidates and lsc events and injections
        nsc = len(back_score_t)
        cand_rank_score = nsc - np.searchsorted(back_score_t, cand_t)
        lsc_rank_score = nsc - np.searchsorted(back_score_t, lsc_t)
        inj_rank_score = nsc - np.searchsorted(back_score_t, inj_t)

        # For zero-lag and lsc events, the order wasn't changed
        if print_to_screen:
            print()
            print()
            print('Rank score (incoherent)  and FAR ')
            print('FAR is per bank per run time (or per time-chunk passed)')
            print()
            print('Zero-lag candidates:')
            for ind, entry in enumerate(self.cands_postveto_max[1]):
                prior_terms, event, *_ = entry
                if self.Nsim/cand_rank_score[ind] > 0.1:
                    print(f'ind: {ind}, Rank: {cand_rank_score[ind]}, '\
                    f'IFAR: {round(self.Nsim / cand_rank_score[ind], 2)}, '\
                    f'H1_time: {round(event[0, 0], 3)}, '\
                    f'SNR^2: {round(event[0, 1], 2), round(event[1, 1], 2)}')
            print()
            print('LSC events:')
            for ind, entry in enumerate(self.cands_postveto_max[2]):
                prior_terms, event, *_ = entry
                print(f'ind: {ind}, Rank: {lsc_rank_score[ind]}, '\
                    f'IFAR: {round(self.Nsim / lsc_rank_score[ind], 2)}, '\
                    f'H1_time: {round(event[0, 0], 3)}, '\
                    f'SNR^2: {round(event[0, 1], 2), round(event[1, 1], 2)}')
            print()
            print('Injections:')
            for ind, entry in enumerate(self.cands_postveto_max[3]):
                prior_terms, event, *_ = entry
                print(f'ind: {ind}, Rank: {inj_rank_score[ind]}, '\
                    f'IFAR: {round(self.Nsim / inj_rank_score[ind], 2)}, '\
                    f'H1_time: {round(event[0, 0], 3)}, '\
                    f'SNR^2: {round(event[0, 1], 2), round(event[1, 1], 2)}')

        return cand_rank_score, lsc_rank_score, inj_rank_score

    def compute_fars_coherent_score(self, print_to_screen=True):
        # Compute FARs using coherent score + ranking scores as test statistics
        # Sort background in increasing order of test statistic
        full_scores_bg = self.coherent_scores_bg + self.back_0_score + \
            self.back_1_score
        # Due to an edge case in O1, BBH (2, 3) has so few templates that the
        # score funcs return NAN
        nanmask = np.isnan(full_scores_bg)
        if np.any(nanmask):
            print("Warning: there are NANs in the scores, can happen due to " +
                  "a small subbank, such as BBH (2, 3) in O1")
            full_scores_bg = \
                np.nan_to_num(full_scores_bg, copy=False, nan=-np.inf)
        full_scores_bg.sort()

        # Compute coherent score for fg
        full_scores_cand = self.coherent_scores_cand + self.cand_0_score + \
            self.cand_1_score
        full_scores_lsc = self.coherent_scores_lsc + self.lsc_0_score + \
            self.lsc_1_score
        full_scores_inj = self.coherent_scores_inj + self.inj_0_score + \
            self.inj_1_score

        # Compute FARs using coherent + ranking scores
        nsc = len(full_scores_bg)
        cand_rank_full = nsc - np.searchsorted(full_scores_bg, full_scores_cand)
        lsc_rank_full = nsc - np.searchsorted(full_scores_bg, full_scores_lsc)
        inj_rank_full = nsc - np.searchsorted(full_scores_bg, full_scores_inj)

        # For zero-lag and lsc events, the order wasn't changed
        if print_to_screen:
            print()
            print()
            print('Rank New final score and IFAR ')
            print('IFAR is per bank per run time (or per time-chunk passed)')
            print()
            print('Zero-lag candidates:')
            order = np.argsort(cand_rank_full)
            for ind in order:
                prior_terms, event, *_ = self.cands_postveto_max[1][ind]
                print(f'ind: {ind}, Rank: {cand_rank_full[ind]}, '\
                    f'IFAR: {round(self.Nsim / cand_rank_full[ind], 2)}, '\
                    f'H1_time: {round(event[0, 0], 3)}, '\
                    f'SNR^2: {round(event[0, 1], 2), round(event[1, 1], 2)}')

            print()
            print('LSC events:')
            for ind, entry in enumerate(self.cands_postveto_max[2]):
                prior_terms, event, *_ = entry
                print(f'ind: {ind}, Rank: {lsc_rank_full[ind]}, '\
                    f'IFAR: {round(self.Nsim / lsc_rank_full[ind], 2)}, '\
                    f'H1_time: {round(event[0, 0], 3)}, '\
                    f'SNR^2: {round(event[0, 1], 2), round(event[1, 1], 2)}')
            print()
            print('Injections:')
            for ind, entry in enumerate(self.cands_postveto_max[3]):
                prior_terms, event, *_ = entry
                print(f'ind: {ind}, Rank: {inj_rank_full[ind]}, '\
                    f'IFAR: {round(self.Nsim / inj_rank_full[ind], 2)}, '\
                    f'H1_time: {round(event[0, 0], 3)}, '\
                    f'SNR^2: {round(event[0, 1], 2), round(event[1, 1], 2)}')

        return cand_rank_full, lsc_rank_full, inj_rank_full

    def avg_sensitive_volume(self, subbank_id):
        """
        Run after scoring the bank bg
        :return: Avg sensitive volume in relative units
        The reference point is at 1/1 normfac ratio and
        self.median_normfacs_by_subbank[subbank_id][0]
        """
        if self._avg_sensitive_volume is None:
            sensitivity_volumes = [[] for _ in range(self.n_subbanks)]
            # sensitivity_ratios = []

            # TODO: BARAK: Does this have added noise?
            # TODO: Only exists for O2
            table_H1, table_ranges_H1 = np.load(
                "/data/bzackay/GW/coherent_score_hist_rho_sq_monte_carlo_with_loud.npy",
                allow_pickle=True)
            rho_centers_H1 = utils.bincent(table_ranges_H1[-1][:-1])

            table_integrals = [
                np.sum([
                    table_H1[normfac_ind, :, :, x, y]
                    for x in range(len(rho_centers_H1))
                    for y in range(len(rho_centers_H1)) if
                    (rho_centers_H1[x] + rho_centers_H1[y]) > self.snr2sumcut and
                    (rho_centers_H1[x] > self.snr2min) and
                    (rho_centers_H1[y] > self.snr2min)])
                for normfac_ind in range(len(table_H1))]

            self.normfac_ratio_ref_ind = np.argmin(np.abs(1 - table_ranges_H1[0]))

            for prior_terms, event, loc_id, sensitivity_params in \
                    self.cands_postveto_max[0][::10]:
                subbank_id = loc_id[1]
                S_H = sensitivity_params[0, 3] / \
                    self.median_normfacs_by_subbank[loc_id][0]
                # Note: the [0] at the end is not a bug
                S_L = sensitivity_params[1, 3] / \
                    self.median_normfacs_by_subbank[loc_id][0]

                # Compute sensitivity
                # sensitivity_ratios.append(S_H / S_L)
                normfac_ind = np.argmin(np.abs(S_H/S_L - table_ranges_H1[0]))
                sensitivity_volumes[subbank_id].append(
                    (S_H ** 2 + S_L ** 2) ** 1.5 *
                    table_integrals[normfac_ind] /
                    table_integrals[self.normfac_ratio_ref_ind])

            self._avg_sensitive_volume = \
                [np.mean(sensitivity_volumes[i]) for i in range(self.n_subbanks)]

        return self._avg_sensitive_volume[subbank_id]

    def rank_scores_calc_MZ(
            self, i_subbank, safety_factor=4, matching_point=None,
            downsampling_correction=True, min_trigs_per_grp=500,
            output_rank_func_temp_grps=False, group_coarse_calphas=False,
            n_calpha_dim=2, snr2_ref=None, n_snr2_ref=1e-3, vetoed=True,
            rhosqs_arrays=None, calphas_arrays=None, masks_vetoed=None,
            **score_func_kwargs):
        """
        Calculating ranking scores for all the bg, fg, lsc, inj triggers.
        The rank functions are constructed in a sub-function
        make_score_funcs_MZ(). We first group the templates based on their
        "glitchiness" and then make separate rank functions for different groups
        :param i_subbank: index of the subbank of the bank
        :param safety_factor:
            Add this to threshold_chi2 before estimating the rank functions
            to account for incompleteness related to optimization
        :param matching_point:
            Set SNR^2 at which the rank functions are matched, the default
            is threshold network SNR^2/2
            (the rank fns are made such that all template groups within the same
             subbank match at self.snr2min and the least glitchy template group
             in all subbanks match at matching_point)
        :param downsampling_correction:
            If the triggers were downsampled compared to a chi-sq distribution
            because of an additional cut (e.g., based on whether the mode ratios 
            A33/A22 or A44/A22 are physical). This flag corrects the rank 
            function so that it follows the chi-sq behavior again. This flag
            needs a file downsamp_corr_path.npy to be input when creating
            Rank class object
        :param min_trigs_per_grp:
            To avoid pathologies with making the rank functions, we require that
            the templates in each group have more than a particular
            number of background triggers associated to them
        :param output_rank_func_temp_grps:
            Flag if you want to output the rank funcs for separate template groups
            for reference
        :param group_coarse_calphas:
            Flag whether we group the coarse or fine calphas
        :param n_calpha_dim: Number of dimensions to use for grouping calphas
        :param snr2_ref:
            Quantify templates' glitchiness by the fraction of triggers with
            SNR>snr2_ref
        :param n_snr2_ref:
            Set snr2_ref by demanding that in the Gaussian noise case, the
            expected number of triggers > snr2_ref is n_snr2_ref over all
            calphas. Used only if snr2_ref is None
        :param vetoed: Flag to compute scores only for vetoed triggers
        :param rhosqs_arrays:
            If known, 4 x n_event x 2 array of SNR^2 values for background,
            foreground, LVC, injections (pass to avoid looping over hdf5)
        :param calphas_arrays:
            If known, 4 x n_event x n_calpha_dim array of calphas
        :param masks_vetoed:
            If known, array of length 4 with masks for vetoed triggers
            This makes us set vetoed to True
        """
        if hasattr(self, 'example_trigs') and self.example_trigs is not None:
            trig_obj = self.example_trigs[i_subbank]
        else:
            example_json = glob.glob(
                os.path.join(self.outputdirs[0][i_subbank], "*config.json"))[0]
            trig_obj = trig.TriggerList.from_json(
                example_json, load_trigs=False, do_ffts=False)

        sub_id = self.subbank_subset[i_subbank]
        bank = trig_obj.templatebank
        coarse_axes = bank.make_grid_axes(
            trig_obj.delta_calpha, trig_obj.template_safety,
            trig_obj.force_zero)
        coarse_axes = coarse_axes[:len(trig_obj.dcalphas)]

        self.snr2max = min(params.SNR2_MAX_BOUND, self.snr2max)
        dsnr2 = 0.1
        nbins = int((self.snr2max - self.snr2min) / dsnr2)
        
        # Treat cuts
        # Add safety near the SNR^2 sumcut, since there is some incompleteness
        # related to optimization, coincidence logic and/or friends
        snr2sumcut_safe = self.snr2sumcut + safety_factor

        # Function that, given an index for a detector, gives the lowest index
        # in the other detector that is not affected by the snr^2 sumcut
        ind_complete = int((snr2sumcut_safe - 2 * self.snr2min) / dsnr2)
        low_ind_comp = lambda x:  int(
            np.heaviside(ind_complete-x, 0) * (ind_complete-x))
            
        # Set scale at fiducial matching point
        if matching_point is None:
            # matching_point = snr2sumcut_safe / 2.0
            matching_point = self.snr2min

        if matching_point < self.snr2min:
            raise RuntimeError("Cannot match where we did not collect!")

        # Get the SNR^2 values and calphas for the triggers
        if (rhosqs_arrays is not None) and (calphas_arrays is not None):
            back_0_bank = rhosqs_arrays[0][:, 0]
            back_1_bank = rhosqs_arrays[0][:, 1]
            if masks_vetoed is not None:
                vetoed = True
                back_0_bank = back_0_bank[masks_vetoed[0]]
                back_1_bank = back_1_bank[masks_vetoed[0]]
                calphas_arrays_vetoed = \
                    [x for result, x in zip(masks_vetoed[0], calphas_arrays[0])
                     if result]
                back_0_subbank = np.array(
                    [snrsq for snrsq, x in
                     zip(back_0_bank, self.cands_postveto_max[0])
                     if x[2][1] == sub_id])
                back_1_subbank = np.array(
                    [snrsq for snrsq, x in
                     zip(back_1_bank, self.cands_postveto_max[0])
                     if x[2][1] == sub_id])
                fine_calphas_arrays = np.array(
                    [calphas for calphas, x in
                     zip(calphas_arrays_vetoed, self.cands_postveto_max[0])
                     if x[2][1] == sub_id])
            else:
                vetoed = False
                # Warning: Silently works even if arrays have the wrong lengths
                back_0_subbank = np.array(
                    [snrsq for snrsq, x in
                     zip(back_0_bank, self.cands_preveto_max[0])
                     if x[2][1] == sub_id])
                back_1_subbank = np.array(
                    [snrsq for snrsq, x in
                     zip(back_1_bank, self.cands_preveto_max[0])
                     if x[2][1] == sub_id])
                fine_calphas_arrays = np.array(
                    [calphas for calphas, x in
                     zip(calphas_arrays[0], self.cands_preveto_max[0])
                     if x[2][1] == sub_id])
        else:
            # Do it the slow way
            if vetoed:
                back_0_subbank = np.array([
                    event[1][0, 1] for event in
                    self.scores_bg_by_subbank[i_subbank]])
                back_1_subbank = np.array([
                    event[1][1, 1] for event in
                    self.scores_bg_by_subbank[i_subbank]])
                fine_calphas_arrays = np.array([
                    event[1][0, self.clist_pos['c0_pos']:] for event in
                    self.scores_bg_by_subbank[i_subbank]])
            else:
                back_0_subbank = np.array([
                    event[1][0, 1] for event in
                    self.scores_bg_by_subbank_nonvetoed[i_subbank]])
                back_1_subbank = np.array([
                    event[1][1, 1] for event in
                    self.scores_bg_by_subbank_nonvetoed[i_subbank]])
                fine_calphas_arrays = np.array([
                    event[1][0, self.clist_pos['c0_pos']:] for event in
                    self.scores_bg_by_subbank_nonvetoed[i_subbank]])

        if group_coarse_calphas:
            calphas_arrays_to_use = utils.find_closest_coarse_calphas(
                coarse_axes, fine_calphas_arrays)
            calphas_arrays_to_use = calphas_arrays_to_use[:, :n_calpha_dim]
        else:
            calphas_arrays_to_use = fine_calphas_arrays
                
        calphas, template_inds = np.unique(
            calphas_arrays_to_use, axis=0, return_inverse=True)
                
        # We will split templates into groups based on the fraction of triggers
        # above a reference SNR
        num_groups = 20
        max_rhosq_limit = min(
            np.max(back_0_subbank), np.max(back_1_subbank))
        snr2sumcut_safe = correct_snr2sumcut_safe(
            self.snr2sumcut + safety_factor, self.snr2sumcut,
            self.snr2min, max_rhosq_limit)
        rhosqthresh = snr2sumcut_safe - self.snr2min
        n_trigs = np.count_nonzero(back_0_subbank > rhosqthresh) + \
            np.count_nonzero(back_1_subbank > rhosqthresh)
        # 6 + 1 = 7 for amplitude, phase, time
        ndof = 7 + n_calpha_dim
        p_all = ss.chi2.sf(rhosqthresh, ndof)
        
        if snr2_ref is None:
            # Pick a reference SNR2 that isn't in the Gaussian regime
            gaussian_safety_fraction = 1/num_groups
            snr2_ref = ss.chi2.isf(
                gaussian_safety_fraction /
                (n_trigs / len(back_0_subbank)) * p_all * n_snr2_ref, ndof)
            
            # Old code by Teja below
            # snr2_ref = ss.chi2.isf(n_snr2_ref / n_trigs * p_all, ndof)
        else:
            gaussian_safety_fraction = \
                ss.chi2.sf(snr2_ref, ndof) / p_all * n_trigs / \
                len(back_0_subbank) / n_snr2_ref
            
        print(f'snr2_ref for subbank {i_subbank} is {np.round(snr2_ref,2)}')
        print('and the safety lower limit of glitch fraction for templates is '
              + f'{np.round(gaussian_safety_fraction,3)}')
            
        glitch_fraction = np.zeros((2, len(calphas)))
        for i in range(len(calphas)):
            back_template = back_0_subbank[template_inds == i]
            _ = np.count_nonzero(back_template > snr2_ref)
            glitch_fraction[0, i] = _ / len(back_template)
            back_template = back_1_subbank[template_inds == i]
            _ = np.count_nonzero(back_template > snr2_ref)
            glitch_fraction[1, i] = _ / len(back_template)
            
        # We split the templates based on sum of glitch fractions in H and L
        groups_ind_bg = np.zeros((len(template_inds)))
        _, groups = np.histogram(
            glitch_fraction[0] + glitch_fraction[1], bins=num_groups)
        groups = np.array(groups)
        ind_safety = np.argmin(np.abs(groups - gaussian_safety_fraction))
        if ind_safety > 0:
            groups = np.r_[groups[0], groups[ind_safety:]]
        # slightly shifting right-edge so all elements are inside
        groups[-1] += 1e-4
        
        templates_group_ind = np.digitize(
            glitch_fraction[0] + glitch_fraction[1], bins=groups) - 1
        for t, g in enumerate(templates_group_ind):
            groups_ind_bg[template_inds == t] = g
        trigs_per_grp = np.array([
            np.count_nonzero(groups_ind_bg == i) for i in range(num_groups)])
        
        # Remaking the groups so that at there are more than ~500 triggers per
        # group, just to avoid pathologies with making the rank functions
        new_groups = [groups[0]]
        sum_trigs = 0
        for i in range(0, len(groups) - 1):
            sum_trigs += trigs_per_grp[i]
            if sum_trigs > min_trigs_per_grp:
                new_groups.append(groups[i + 1])
                sum_trigs = 0
        if len(new_groups)==1:
            new_groups.append(groups[-1])
        new_groups[-1] = groups[-1]
        # the right edge of the new groups needs to be the same as that of the
        # old groups to ensure all templates are included
        groups = new_groups
        num_groups = len(groups)-1
        templates_group_ind = np.digitize(
            glitch_fraction[0] + glitch_fraction[1], bins=groups) - 1
        ntemplates_groups = [
            np.count_nonzero(templates_group_ind == i)
            for i in range(num_groups)]
        for t, g in enumerate(templates_group_ind):
            groups_ind_bg[template_inds == t] = g
            
        print(f'Split the templates into {num_groups} groups for ' +
              'calculating the rank functions.')
        
        # Similar to bg triggers, we split the cand, lsc and inj triggers also
        # into these groups and collect info for rank func application below
        groups_ind = {}
        subbank_masks = {}
        snrsq = {}
        scores = {}
        for i, category in enumerate(['cand', 'lsc', 'inj']):
            # Index into the scores_(non)vetoed_max lists
            i += 1

            if calphas_arrays is not None:
                if masks_vetoed is not None:
                    calphas_arrays_vetoed = \
                        [x for result, x in
                         zip(masks_vetoed[i], calphas_arrays[i]) if result]
                    fine_calphas_arrays_cat = np.array(
                        [calphas for calphas, x in
                         zip(calphas_arrays_vetoed, self.cands_postveto_max[i])
                         if x[2][1] == sub_id])
                else:
                    fine_calphas_arrays_cat = np.array(
                        [calphas for calphas, x in
                         zip(calphas_arrays[i], self.cands_preveto_max[i])
                         if x[2][1] == sub_id])
            else:
                # Do it the slow way
                if vetoed:
                    fine_calphas_arrays_cat = np.array(
                        [event[1][0, self.clist_pos['c0_pos']:] for event in
                         self.cands_postveto_max[i] if event[2][1] == sub_id])
                else:
                    fine_calphas_arrays_cat = np.array(
                        [event[1][0, self.clist_pos['c0_pos']:] for event in
                         self.cands_preveto_max[i] if event[2][1] == sub_id])

            if group_coarse_calphas:
                calphas_arrays_to_use_cat = utils.find_closest_coarse_calphas(
                    coarse_axes, fine_calphas_arrays_cat)
                calphas_arrays_to_use_cat = \
                    calphas_arrays_to_use_cat[:, :n_calpha_dim]
                if calphas_arrays_to_use_cat.size == 0:
                    calphas_arrays_to_use_cat = np.zeros((0, n_calpha_dim))
            else:
                calphas_arrays_to_use_cat = fine_calphas_arrays_cat

            template_indices_cat = [
                np.argmin(np.linalg.norm(calphas - calpha_event, axis=-1))
                for calpha_event in calphas_arrays_to_use_cat]
            groups_ind[category] = templates_group_ind[template_indices_cat]

            if vetoed:
                subbank_masks[category] = \
                    [event[2][1] == sub_id
                     for event in self.cands_postveto_max[i]]
            else:
                subbank_masks[category] = \
                    [event[2][1] == sub_id
                     for event in self.cands_preveto_max[i]]

            if rhosqs_arrays is not None:
                snrsq[category] = rhosqs_arrays[i]
                if masks_vetoed is not None:
                    snrsq[category] = snrsq[category][masks_vetoed[i]]
                    snrsq[category] = np.array(
                        [snrsq for snrsq, x in
                         zip(snrsq[category], self.cands_postveto_max[i])
                         if x[2][1] == sub_id])
                else:
                    snrsq[category] = np.array(
                        [snrsq for snrsq, x in
                         zip(snrsq[category], self.cands_preveto_max[i])
                         if x[2][1] == sub_id])
            else:
                # Do it the slow way
                if vetoed:
                    snrsq[category] = np.array(
                        [event[1][:, 1] for event in self.cands_postveto_max[i]
                         if event[2][1] == sub_id])
                else:
                    snrsq[category] = np.array(
                        [event[1][:, 1] for event in self.cands_preveto_max[i]
                         if event[2][1] == sub_id])

            scores[category] = np.zeros(
                (2, np.count_nonzero(subbank_masks[category])))
            
        # Function for constructing interpolated rank functions
        # based on SNRsq of bg triggers
        def make_score_funcs_MZ(
                back_0, back_1, ntemplates, group_ind, matching_val_0=None,
                matching_val_1=None):
            # Make a 2-d histogram
            counts, xbins, ybins = np.histogram2d(
                back_0, back_1, bins=nbins,
                range=[[self.snr2min, self.snr2max],
                       [self.snr2min, self.snr2max]])

            counts /= ntemplates

            # Compute cumulative sums over both detectors, in two stages
            # The entry in (i, j) is the sum of all cells with (x >= i, y >= j)
            cumsum_mat = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
            cumsum_mat = np.cumsum(cumsum_mat[::-1, :], axis=0)[::-1, :]

            # For a given snr compute the ratio between the cumulative sum above
            # that snr and above the following bin using the lowest possible snr
            # in the other detector that is unaffected by the snr^2 sumcut
            num_0 = np.array(
                [cumsum_mat[j, low_ind_comp(j)] for j in range(ind_complete)])
            den_0 = np.array(
                [cumsum_mat[j + 1, low_ind_comp(j)] for j in range(ind_complete)])
            rat_0 = np.divide(
                num_0, den_0, out=np.ones_like(num_0), where=(den_0 != 0))

            num_1 = np.array(
                [cumsum_mat[low_ind_comp(j), j] for j in range(ind_complete)])
            den_1 = np.array(
                [cumsum_mat[low_ind_comp(j), j + 1] for j in range(ind_complete)])
            rat_1 = np.divide(
                num_1, den_1, out=np.ones_like(num_1), where=(den_1 != 0))

            # Compute the final cumulative sum by scaling the cumulative sum up
            # from the cumulative sum for the next bin, starting from the last
            # bin that can be trusted
            cumsum_0 = cumsum_mat[:, 0]
            cumsum_1 = cumsum_mat[0, :]
            for j in range(ind_complete)[::-1]:
                cumsum_0[j] = cumsum_0[j+1]*rat_0[j]
                cumsum_1[j] = cumsum_1[j+1]*rat_1[j]

            # Put a minimum number in all the empty bins
            cumsum_0 = np.maximum(cumsum_0, np.min(cumsum_0[cumsum_0 > 0]))
            cumsum_1 = np.maximum(cumsum_1, np.min(cumsum_1[cumsum_1 > 0]))

            # Define rank scores
            rankscore_0 = -2 * np.log(cumsum_0)
            rankscore_1 = -2 * np.log(cumsum_1)

            # Normalize ranking score at matching point
            xbincent = utils.bincent(xbins)
            ybincent = utils.bincent(ybins)

            if downsampling_correction:
                if self.downsampling_corrections is not None:
                    corrections = self.downsampling_corrections[i_subbank]
                    rankscore_0 += np.interp(
                        xbincent, corrections[:, 0], corrections[:, 1])
                    rankscore_1 += np.interp(
                        xbincent, corrections[:, 0], corrections[:, 2])
                else:
                    if self.downsampling_corr_fit_params is not None:
                        a_fit, rhoSq_0 = self.downsampling_corr_fit_params[i_subbank]
                    else:
                        raise NotImplementedError("I need downsampling_corr_fit_params\
                                or downsampling_corrections as input")
                    rankscore_0 += np.r_[parabolic_func(xbincent-rhoSq_0, a_fit)[xbincent<rhoSq_0],
                                        np.zeros(len(xbincent[xbincent>=rhoSq_0]))]
                    rankscore_1 += np.r_[parabolic_func(xbincent-rhoSq_0, a_fit)[xbincent<rhoSq_0],
                                        np.zeros(len(xbincent[xbincent>=rhoSq_0]))]
            
            # The rank fns are made such that all template groups within the same
            # subbank match at self.snr2min and the least glitchy template group
            # in all subbanks match at matching_point

            ind_x0_0 = np.searchsorted(xbincent, self.snr2min)
            ind_x0_1 = np.searchsorted(ybincent, self.snr2min)
            
            # Scale the values to the matching point for the least glitchy group (g=0)
            if group_ind == 0:
                matching_val_0 = \
                    rankscore_0[np.searchsorted(xbincent, matching_point)] \
                    - rankscore_0[ind_x0_0]
                matching_val_1 = \
                    rankscore_1[np.searchsorted(ybincent, matching_point)] \
                    - rankscore_1[ind_x0_1]
            elif (matching_val_0 is None) or (matching_val_1 is None):
                utils.close_hdf5()
                raise RuntimeError("Matching values weren't given")
                                 
            rankscore_0 -= rankscore_0[ind_x0_0] + matching_val_0
            rankscore_1 -= rankscore_1[ind_x0_1] + matching_val_1
            
            score_0_func = interp1d(xbincent, rankscore_0 + 4*np.log(xbincent),
                                    bounds_error=False, fill_value="extrapolate")
            # The log term is added because the rank fns should correspond to
            # the density of triggers in bins of (rho22^2, rho33^2, rho44^2)
            # but we instead average it over bins of
            # (rho22^2 + rho33^2 + rho44^2) = constant
                                    
            score_1_func = interp1d(ybincent, rankscore_1 + 4*np.log(ybincent),
                                    bounds_error=False, fill_value="extrapolate")
            
            return score_0_func, score_1_func, matching_val_0, matching_val_1
        
        # Calculating scores for background first and then others for all groups
        back_0_score, back_1_score = np.zeros((2, len(back_0_subbank)))
        rank_funcs_groups = []
        matching_value_0 = None
        matching_value_1 = None
        for g in range(num_groups):
            ntemplates_group = ntemplates_groups[g]
            back_0_group = back_0_subbank[groups_ind_bg == g]
            back_1_group = back_1_subbank[groups_ind_bg == g]
            back_0_group[back_0_group > params.SNR2_MAX_BOUND] = params.SNR2_MAX_BOUND-1e-3
            back_1_group[back_1_group > params.SNR2_MAX_BOUND] = params.SNR2_MAX_BOUND-1e-3
            score_func_0, score_func_1, matching_value_0, matching_value_1 = \
                make_score_funcs_MZ(
                    back_0_group, back_1_group, ntemplates_group, g,
                    matching_val_0=matching_value_0,
                    matching_val_1=matching_value_1)
        
            back_0_score[groups_ind_bg == g] = score_func_0(back_0_group)
            back_1_score[groups_ind_bg == g] = score_func_1(back_1_group)
            
            snrsq_category = snrsq['cand'][groups_ind['cand'] == g]
            if len(snrsq_category) > 0:
                scores['cand'][:, groups_ind['cand'] == g] = \
                    score_func_0(snrsq_category[:, 0]), \
                    score_func_1(snrsq_category[:, 1])
            snrsq_category = snrsq['lsc'][groups_ind['lsc'] == g]
            if len(snrsq_category) > 0:
                scores['lsc'][:, groups_ind['lsc'] == g] = \
                    score_func_0(snrsq_category[:, 0]), \
                    score_func_1(snrsq_category[:, 1])
            snrsq_category = snrsq['inj'][groups_ind['inj'] == g]
            if len(snrsq_category) > 0:
                scores['inj'][:, groups_ind['inj'] == g] = \
                    score_func_0(snrsq_category[:, 0]), \
                    score_func_1(snrsq_category[:, 1])
                    
            if output_rank_func_temp_grps:
                rank_funcs_groups.append((score_func_0, score_func_1))
                
        # Populating all arrays for rank score
        if vetoed:
            self.back_0_score_vetoed.append(back_0_score)
            self.back_1_score_vetoed.append(back_1_score)
            self.cand_0_score_vetoed[subbank_masks['cand']] = scores['cand'][0]
            self.cand_1_score_vetoed[subbank_masks['cand']] = scores['cand'][1]
            self.lsc_0_score_vetoed[subbank_masks['lsc']] = scores['lsc'][0]
            self.lsc_1_score_vetoed[subbank_masks['lsc']] = scores['lsc'][1]
            self.inj_0_score_vetoed[subbank_masks['inj']] = scores['inj'][0]
            self.inj_1_score_vetoed[subbank_masks['inj']] = scores['inj'][1]
        else:
            self.back_0_score.append(back_0_score)
            self.back_1_score.append(back_1_score)
            self.cand_0_score[subbank_masks['cand']] = scores['cand'][0]
            self.cand_1_score[subbank_masks['cand']] = scores['cand'][1]
            self.lsc_0_score[subbank_masks['lsc']] = scores['lsc'][0]
            self.lsc_1_score[subbank_masks['lsc']] = scores['lsc'][1]
            self.inj_0_score[subbank_masks['inj']] = scores['inj'][0]
            self.inj_1_score[subbank_masks['inj']] = scores['inj'][1]
        
        # Finding rank function for the entire subbank now
        # (just for reference, as this is not used in the scores)
        score_func_0, score_func_1, *_ = make_score_funcs_MZ(
            back_0_subbank, back_1_subbank,
            self.ntemplates_by_subbank[i_subbank], 1,
            matching_val_0=matching_value_0, matching_val_1=matching_value_1)
        self.score_funcs.append((score_func_0, score_func_1))

        if output_rank_func_temp_grps:
            return rank_funcs_groups
    
