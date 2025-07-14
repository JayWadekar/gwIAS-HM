"""
Module for Vetoing and Optimizing Coincident Gravitational Wave Candidates.

This module contains functions extracted and generalized from coincidence_HM.py.
The logic has been refactored to support an N-detector network (e.g., H1-L1-V1)
instead of being hardcoded for a 2-detector network.
"""
import copy
import itertools
import multiprocess as mp
import numpy as np
from numba import njit

# Local application imports
import coherent_score_hm_search as cs
import params
import triggers_single_detector_HM as trig
import utils


# --- Top-Level Orchestrator ---

def veto_and_optimize_coincidence_list(
        bg_events, list_of_trig_objects, time_shift_tol, threshold_chi2,
        minimal_time_slide_jump, veto_triggers=True, min_veto_chi2=32,
        apply_threshold=True, origin=0, n_cores=1, opt_format="new",
        output_timeseries=True, output_coherent_score=True,
        score_reduction_timeseries=10, score_reduction_max=5,
        detectors=('H1', 'L1'), score_func=utils.incoherent_score, **kwargs):
    """
    Returns list of vetoed and optimized coincident candidates for an N-detector network.
    
    This is a generalized version of the original function, handling N detectors.

    :param bg_events: n_candidate x N_det x len(Processedclist[0]) array with candidates.
    :param list_of_trig_objects: A list of N trigger objects, one for each detector.
    :param time_shift_tol: Time tolerance for bucketing triggers
    :param threshold_chi2: Threshold for chi2 filtering
    :param minimal_time_slide_jump: Minimum time slide jump
    :param veto_triggers: Whether to apply veto tests
    :param min_veto_chi2: Minimum chi2 for veto application
    :param apply_threshold: Whether to apply threshold filtering
    :param origin: Time origin for bucketing
    :param n_cores: Number of cores for parallel processing
    :param opt_format: Format for optimization ("new" or "old")
    :param output_timeseries: Whether to output timeseries
    :param output_coherent_score: Whether to output coherent scores
    :param score_reduction_timeseries: Score reduction for timeseries
    :param score_reduction_max: Maximum score reduction
    :param detectors: List of detector names
    :param score_func: Scoring function to use
    :return: Various outputs depending on flags
    """
    trig_obj_ref = list_of_trig_objects[0]  # Use first detector for metadata structure
    num_detectors = len(list_of_trig_objects)

    # Create key to interpret the metadata
    npower = len(trig_obj_ref.outlier_reasons)
    nsplit = len(params.SPLIT_CHUNKS)
    metadata_keys = np.zeros(npower + 11 + 2 * nsplit, dtype='<U64')
    
    # Build metadata keys
    metadata_keys[0] = "CBC_CAT2"
    metadata_keys[1] = "CBC_CAT3"
    metadata_keys[2:2 + npower] = trig_obj_ref.outlier_reasons[:]
    metadata_keys[2 + npower] = "Finer PSD drift"
    metadata_keys[3 + npower] = "Pass initial veto"
    metadata_keys[4 + npower] = "Pass optimization"
    metadata_keys[5 + npower] = "Pass finer PSD drift"
    metadata_keys[6 + npower] = "Pass duplicate removal"
    metadata_keys[7 + npower] = "Pass phase vetoes"
    metadata_keys[8 + npower] = "Pass autocorrelation veto"
    metadata_keys[9 + npower] = "Pass sine-Gaussian veto"
    metadata_keys[10 + npower] = "Pass slow sine-Gaussian veto"
    metadata_keys[11 + npower:11 + npower + nsplit] = [
        f"Pass {sc}s chunk veto" for sc in params.SPLIT_CHUNKS]
    metadata_keys[11 + npower + nsplit:11 + npower + 2 * nsplit] = [
        f"Pass {params.N_CHUNK_2}s chunk veto"
        for _ in range(nsplit)]
    metadata_keys[11 + npower + 2 * nsplit] = "Pass group optimization"
    
    if output_timeseries or output_coherent_score:
        metadata_keys = np.append(metadata_keys, 'Secondary_peak_timeseries')

    if utils.checkempty(bg_events):
        # Return empty structures with correct shapes
        mask_vetoed = np.zeros(0, dtype=bool)
        timeseries, coherent_scores = [], np.zeros(0)
        metadata_arr = np.ones((0, num_detectors, len(metadata_keys)), dtype=bool)
        
        if output_timeseries and output_coherent_score:
            return bg_events, mask_vetoed, metadata_arr, timeseries, coherent_scores, metadata_keys
        elif output_timeseries:
            return bg_events, mask_vetoed, metadata_arr, timeseries, metadata_keys
        elif output_coherent_score:
            return bg_events, mask_vetoed, metadata_arr, coherent_scores, metadata_keys
        else:
            return bg_events, mask_vetoed, metadata_arr, metadata_keys

    print(f"Going to veto and optimize {len(bg_events)} triggers for {num_detectors} detectors.", flush=True)

    # --- GENERALIZED VETO LOOP (from 2 to N detectors) ---
    # Veto and optimize triggers for each detector in a loop.
    veto_dicts = [
        veto_and_optimize_single_detector(
            bg_events[:, i, :], trig_obj, time_shift_tol,
            group_duration=minimal_time_slide_jump, veto_triggers=veto_triggers,
            min_veto_chi2=min_veto_chi2, apply_threshold=apply_threshold,
            origin=origin, n_cores=n_cores, opt_format=opt_format
        ) for i, trig_obj in enumerate(list_of_trig_objects)
    ]
    print(f"Finished initial vetoes. Picking optimal coincident tuples.", flush=True)

    # Pick best trigger tuples from N clouds
    bg_events_all, mask_vetoed, metadata_arr = \
        select_optimal_trigger_tuples(
            bg_events, veto_dicts, time_shift_tol,
            threshold_chi2, trig_obj_ref, origin=origin, score_func=score_func,
            **kwargs)
    print(f"Picked {len(bg_events_all)} coincident optimized candidates.", flush=True)

    del veto_dicts  # Free memory

    # Flag and handle duplicate candidates
    bg_events_all, mask_vetoed, metadata_arr, mask_retain = \
        flag_duplicates_per_group_pair(
            bg_events_all, minimal_time_slide_jump, trig_obj_ref.c0_pos, origin=origin,
            veto_mask=mask_vetoed, extra_arrays=[metadata_arr],
            remove=False, score_func=score_func, **kwargs)
    metadata_arr = metadata_arr[0]
    mask_vetoed *= np.all(mask_retain, axis=-1)

    # Prepare for stringent vetoes
    metadata_arr = np.pad(
        metadata_arr, ((0, 0), (0, 0), (0, 5 + nsplit)),
        mode="constant", constant_values=True)
    metadata_arr[:, :, -1] = mask_retain[:]

    # --- GENERALIZED STRINGENT VETO LOOP ---
    if veto_triggers and np.any(mask_vetoed):
        print(f"Applying stringent veto to {len(bg_events_all)} candidates.", flush=True)

        all_trigger_fates = []
        for i, trig_obj in enumerate(list_of_trig_objects):
            trigger_fates, metadata_stringent = stringent_veto(
                bg_events_all, i, trig_obj, min_veto_snr2=min_veto_chi2,
                group_duration=minimal_time_slide_jump, origin=origin,
                n_cores=n_cores)
            
            # Place stringent veto metadata in the correct slice
            metadata_arr[:, i, -(5 + nsplit):-1] = metadata_stringent[:]
            all_trigger_fates.append(trigger_fates)

        # A candidate survives only if it passes the stringent veto in ALL detectors
        mask_vetoed *= np.all(np.stack(all_trigger_fates, axis=0), axis=0)
        
        print(f"Saving {len(bg_events_all)} candidates ({np.sum(mask_vetoed)} survived vetoes).", flush=True)

    # --- GENERALIZED COHERENT SCORE and TIMESERIES ---
    if output_timeseries or output_coherent_score:
        print(f"Computing coherent scores and/or timeseries for {len(bg_events_all)} candidates.", flush=True)
        
        # Use the N-detector version of the coherent score functions
        cs_instance = cs.initialize_cs_instance_new(list_of_trig_objects, detectors=detectors)
        coherent_scores, timeseries = cs.compute_coherent_scores_new(
                        cs_instance, bg_events_all, list_of_trig_objects,
                        minimal_time_slide_jump=minimal_time_slide_jump,
                        score_reduction_timeseries=score_reduction_timeseries,
                        output_timeseries=output_timeseries,
                        output_coherent_score=output_coherent_score)
        
        print(f"Finished coherent analysis.", flush=True)

        # Veto based on secondary peaks in the timeseries for each detector
        veto_timeseries_per_det = []
        for i in range(num_detectors):
            veto_this_det = [
                not secondary_peak_reject(
                    bg_events_all[k][i], timeseries[k][i],
                    score_reduction_max=score_reduction_max
                ) for k in range(len(bg_events_all))
            ]
            veto_timeseries_per_det.append(veto_this_det)

        veto_timeseries = np.stack(veto_timeseries_per_det, axis=1)  # Shape (n_events, n_dets)
        
        # Append this veto result to the metadata array
        metadata_arr = np.append(metadata_arr, np.expand_dims(veto_timeseries, axis=-1), axis=-1)
        
        # A candidate survives only if it passes this check in ALL detectors
        mask_vetoed *= np.all(veto_timeseries, axis=1)

        if output_timeseries and output_coherent_score:
            return bg_events_all, mask_vetoed, metadata_arr, timeseries, coherent_scores, metadata_keys
        elif output_timeseries:
            return bg_events_all, mask_vetoed, metadata_arr, timeseries, metadata_keys
        else:  # only coherent score
            return bg_events_all, mask_vetoed, metadata_arr, coherent_scores, metadata_keys

    return bg_events_all, mask_vetoed, metadata_arr, metadata_keys


def select_optimal_trigger_tuples(
        candidates, veto_dicts, time_shift_tol, threshold_chi2, trig_obj_ref,
        origin=0, score_func=utils.incoherent_score, **kwargs):
    """
    Function to return an optimized N-tuple background candidate.
    
    Renamed from `select_optimal_trigger_pairs` and generalized for N detectors.
    """
    all_candidates = []
    mask_vetoed = []
    metadata_arr = []
    num_detectors = len(veto_dicts)

    _, spacings_opt = trig_obj_ref.define_finer_grid_func()
    
    for candidate in candidates:
        dict_keys = np.floor((candidate[:, 0] - origin) / time_shift_tol).astype(int)

        passed_veto = True
        skip = False
        
        # --- GENERALIZED LOOP ---
        friends_arrs = []
        snr2_corr_factors = []
        metadatas = []
        for i in range(num_detectors):
            opt_results = veto_dicts[i].get(dict_keys[i])

            if opt_results is None:
                # Handle missing key case
                pass_flag = False
                friends_to_optimize = np.array([])
                metadata = np.ones(len(trig_obj_ref.outlier_reasons) + 6 + len(params.SPLIT_CHUNKS), dtype=bool)
                metadata[-1] = False  # Mark as failed
                skip = True
            else:
                pass_flag, cloud, snr2_corr_factor, metadata = opt_results
                friends_to_optimize = trig.TriggerList.filter_processed_clist(
                    np.asarray(cloud),
                    filters={'time': (candidate[i, 0] - params.DT_OPT, candidate[i, 0] + params.DT_OPT)}
                )
                if utils.checkempty(friends_to_optimize):
                    pass_flag = False
                    metadata[-1] = False
                    skip = True

            passed_veto *= pass_flag
            friends_arrs.append(friends_to_optimize)
            snr2_corr_factors.append(snr2_corr_factor)
            metadatas.append(copy.deepcopy(metadata))

        if skip:
            # Pad original candidate to match finer grid dimension for consistency
            if candidate.shape[1] < len(spacings_opt) + trig_obj_ref.c0_pos:
                npad = len(spacings_opt) + trig_obj_ref.c0_pos - candidate.shape[1]
                best_trig_tuple = list(np.pad(candidate, ((0, 0), (0, npad)), 'constant'))
            else:
                best_trig_tuple = list(candidate)
        else:
            # Check for PSD drift veto
            finer_incoherent_score = np.dot(candidate[:, 1], np.asarray(snr2_corr_factors))
            if finer_incoherent_score < threshold_chi2:
                passed_veto = False
                # Mark PSD drift failure in metadata for all detectors
                for m in metadatas:
                    m[len(trig_obj_ref.outlier_reasons) + 2] = False
            
            # Use generalized helper function
            best_trig_tuple = get_best_candidate_tuple(
                friends_arrs, trig_obj_ref.c0_pos, score_func=score_func, **kwargs
            )

            if best_trig_tuple is None or any(t is None for t in best_trig_tuple):
                print("Bug or Feature?: No common element found in clouds!", flush=True)
                continue

        optimized_candidate = np.array(best_trig_tuple)
        all_candidates.append(optimized_candidate)
        mask_vetoed.append(passed_veto)
        metadata_arr.append(metadatas)

    if len(all_candidates) > 0:
        # Stack results, handling potential padding for different calpha dimensions
        try:
            all_candidates = np.stack(all_candidates)
        except ValueError:
            trig_dims = [x.shape[-1] for x in all_candidates]
            target_dim = np.max(trig_dims)
            for i in range(len(all_candidates)):
                if all_candidates[i].shape[-1] < target_dim:
                    npad = target_dim - all_candidates[i].shape[-1]
                    all_candidates[i] = np.pad(all_candidates[i], ((0, 0), (0, npad)), 'constant')
            all_candidates = np.stack(all_candidates)
            
        mask_vetoed = np.array(mask_vetoed, dtype=bool)
        metadata_arr = np.array(metadata_arr, dtype=bool)
    else:
        # Return empty arrays with correct shape
        cand_shape = candidates.shape
        all_candidates = np.zeros((0,) + cand_shape[1:])
        mask_vetoed = np.ones(0, dtype=bool)
        metadata_arr = np.zeros(
            (0, num_detectors, len(trig_obj_ref.outlier_reasons) + 6 + len(params.SPLIT_CHUNKS)),
            dtype=bool)

    return all_candidates, mask_vetoed, metadata_arr


def get_best_candidate_tuple(friends_arrs, c0_pos, score_func=utils.incoherent_score, **kwargs):
    """
    Get the best candidate tuple from multiple friend arrays.
    
    :param friends_arrs: List of friend arrays, one per detector
    :param c0_pos: Position of template parameters
    :param score_func: Scoring function to use
    :return: Best candidate tuple or None if no matches
    """
    if not friends_arrs or any(len(arr) == 0 for arr in friends_arrs):
        return None
    
    # Group by template ID for each detector
    template_groups = []
    for friends_arr in friends_arrs:
        if len(friends_arr) == 0:
            template_groups.append({})
            continue
        
        template_ids = utils.make_template_ids(friends_arr[:, c0_pos:])
        unique_ids, groups = group_by_id(friends_arr, c0_pos)
        template_groups.append(dict(zip(unique_ids, groups)))
    
    # Find common template IDs
    if not template_groups:
        return None
    
    common_ids = set(template_groups[0].keys())
    for group in template_groups[1:]:
        common_ids &= set(group.keys())
    
    if not common_ids:
        return None
    
    # Find best candidate for each common template
    best_candidate = None
    best_score = -np.inf
    
    for template_id in common_ids:
        candidate_tuple = []
        for group in template_groups:
            triggers = group[template_id]
            # Get best trigger for this template
            best_idx = np.argmax(triggers[:, 1])  # Assuming score is at position 1
            candidate_tuple.append(triggers[best_idx])
        
        candidate_tuple = np.array(candidate_tuple)
        total_score = score_func(candidate_tuple)
        
        if total_score > best_score:
            best_score = total_score
            best_candidate = candidate_tuple
    
    return best_candidate


# --- Extracted Worker and Helper Functions ---

def veto_and_optimize_single_detector(
        triggers, trig_obj, time_shift_tol, group_duration=0.1,
        veto_triggers=True, min_veto_chi2=None, apply_threshold=True,
        origin=0, n_cores=1, opt_format="new"):
    """
    Veto and optimize coincident candidates in a single detector.
    """
    if utils.checkempty(triggers):
        return {}

    index_groups = utils.splitarray(
        np.arange(len(triggers)), triggers[:, 0], group_duration,
        axis=0, origin=origin)

    def veto_opt_gp(inds_gp):
        trigs_gp = triggers[inds_gp]
        return veto_and_optimize_group(
            trigs_gp, trig_obj, time_shift_tol, veto_triggers=veto_triggers,
            min_veto_chi2=min_veto_chi2, apply_threshold=apply_threshold,
            origin=origin, opt_format=opt_format)

    opt_dic_file = {}
    if n_cores == 1:
        for i, inds_group in enumerate(index_groups):
            opt_dic_group = veto_opt_gp(inds_group)
            opt_dic_file.update(copy.deepcopy(opt_dic_group))
    else:
        with mp.Pool(n_cores) as p:
            results = p.map(veto_opt_gp, index_groups)
        for res in results:
            opt_dic_file.update(copy.deepcopy(res))

    return opt_dic_file


def veto_and_optimize_group(
        trigs_gp, trig_obj, time_shift_tol, veto_triggers=True,
        min_veto_chi2=None, apply_threshold=True, relative_binning=True,
        origin=0, opt_format='new'):
    """
    Vetoes and optimizes a group of triggers for a single detector.
    """
    if utils.checkempty(trigs_gp):
        return {}

    # Define subset of data to veto any trigger within this group
    subset_details = trig_obj.prepare_subset_for_vetoes(trigs_gp)

    # Extreme times in the group
    min_time_gp, max_time_gp = np.min(trigs_gp[:, 0]), np.max(trigs_gp[:, 0])

    # Define parameters useful for defining safety margins for vetoes
    _, dcalphas_veto = trig_obj.define_finer_grid_func(
        dcalpha_coarse=trig_obj.delta_calpha / 2, trim_dims=False)

    # Half-width of bank in dimensions
    extent = (trig_obj.templatebank.bounds[:len(dcalphas_veto), 2] -
              trig_obj.templatebank.bounds[:len(dcalphas_veto), 0]) / 2
    extent *= trig_obj.template_safety

    # Define parameters for optimization
    if opt_format.lower() == "old":
        # Define spacings of finer grid
        spacings_opt = []
        for grid in trig_obj.grid_axes:
            if len(grid) > 1:
                spacings_opt.append(trig_obj.delta_calpha / 2)
        spacings_opt = np.array(spacings_opt)
        if utils.checkempty(spacings_opt):
            spacings_opt = np.array([trig_obj.delta_calpha])

        # Define function that returns finer grid points
        example_trig_calpha = trigs_gp[0, trig_obj.c0_pos:]
        nopt = min(len(spacings_opt), len(example_trig_calpha))
        offset_axes = \
            [np.arange(-1, 2) * spacings_opt[ind] for ind in range(nopt)] + \
            [[0] for _ in range(nopt, len(example_trig_calpha))]
        offsets = np.array(list(itertools.product(*offset_axes)))

        def finer_grid_func(trig_calpha):
            return trig_calpha + offsets
    else:
        finer_grid_func, spacings_opt = trig_obj.define_finer_grid_func()

    # Define bin edges for relative binning
    if relative_binning:
        dt_rb = max_time_gp - min_time_gp + 2 * params.DT_OPT
        relative_freq_bins = trig_obj.templatebank.def_relative_bins(
            spacings_opt, dt=dt_rb, delta=0.1)
    else:
        relative_freq_bins = None

    # Optimize all triggers
    calphas_gp = trigs_gp[:, trig_obj.c0_pos:]
    calphas_gp = {tuple(row) for row in calphas_gp}

    finer_calphas = []
    for calpha in calphas_gp:
        finer_calphas.append(finer_grid_func(calpha))
    finer_calphas = np.unique(np.vstack(finer_calphas), axis=0)

    trigger = trigs_gp[np.argmax(trigs_gp[:, 1])]
    dt_left = trigger[0] - min_time_gp + params.DT_OPT
    dt_right = max_time_gp - trigger[0] + params.DT_OPT
    opt_triggers = trig_obj.gen_triggers_local(
        trigger=trigger, dt_left=dt_left, dt_right=dt_right,
        apply_threshold=apply_threshold, relative_binning=relative_binning,
        relative_freq_bins=relative_freq_bins, subset_defined=True,
        compute_calphas=finer_calphas, orthogonalize_modes=True)

    cloud_scratch_dict = {}
    if not utils.checkempty(opt_triggers):
        temp_ids, opt_trigger_groups = group_by_id(opt_triggers, trig_obj.c0_pos)
        for temp_id, opt_trigger_group in zip(temp_ids, opt_trigger_groups):
            cloud_scratch_dict[temp_id] = opt_trigger_group

    keys, indices_buckets = utils.splitarray(
        np.arange(len(trigs_gp)), trigs_gp[:, 0], time_shift_tol,
        origin=origin, return_split_keys=True)
    opt_dic_gp = {}

    for key, indices_bucket in zip(keys, indices_buckets):
        metadata = np.ones(
            len(trig_obj.outlier_reasons) + 6 + len(params.SPLIT_CHUNKS),
            dtype=bool)
        trigs_bucket = trigs_gp[indices_bucket]
        unique_trigs = np.unique(trigs_bucket, axis=0)
        lists_of_friends = []

        for unique_trig in unique_trigs:
            finer_calphas_local = finer_grid_func(unique_trig[trig_obj.c0_pos:])
            previous_opt_triggers = find_friends_in_cloud(
                cloud_scratch_dict, finer_calphas_local)
            previous_opt_triggers = trig_obj.filter_processed_clist(
                previous_opt_triggers,
                filters={'time': (unique_trig[0] - params.DT_OPT,
                                  unique_trig[0] + params.DT_OPT)})
            if len(previous_opt_triggers) > 0:
                lists_of_friends.append(previous_opt_triggers)

        if len(lists_of_friends) == 0:
            passed_veto = False
            list_of_friends = np.array([])
            snr2_corr_factor = 1
            metadata[0] = np.all(read_channel_dict(
                trig_obj, unique_trigs[:, 0], 'CBC_CAT2'))
            metadata[1] = np.all(read_channel_dict(
                trig_obj, unique_trigs[:, 0], 'CBC_CAT3'))
            metadata[-1] = False
        else:
            list_of_friends = np.unique(np.vstack(lists_of_friends), axis=0)
            metadata[0] = np.all(read_channel_dict(
                trig_obj, list_of_friends[:, 0], 'CBC_CAT2'))
            metadata[1] = np.all(read_channel_dict(
                trig_obj, list_of_friends[:, 0], 'CBC_CAT3'))

            trigger_to_veto = list_of_friends[np.argmax(list_of_friends[:, 1])]

            if (veto_triggers and ((min_veto_chi2 is None) or
                                   (trigger_to_veto[1] > min_veto_chi2))):
                veto_spacing = dcalphas_veto.copy()
                snr_trig = np.sqrt(trigger_to_veto[1])
                veto_spacing[
                    np.logical_and(extent > 1/snr_trig,
                                   1/snr_trig > veto_spacing)] = 1/snr_trig
                passed_veto, snr2_corr_factor, glitch_mask = \
                    trig_obj.veto_trigger_all(
                        trigger_to_veto, dcalphas=veto_spacing,
                        subset_details=subset_details, lazy=False)
                metadata[2:-1] = glitch_mask[:]
            else:
                passed_veto = True
                if veto_triggers:
                    passed_veto, _, glitch_mask = trig_obj.veto_trigger_all(
                        trigger_to_veto, do_costly_vetos=False,
                        subset_details=subset_details, lazy=False)
                    snr2_corr_factor = trig_obj.finer_psd_drift(
                        trigger_to_veto, average='safemean')
                else:
                    snr2_corr_factor = 1
        opt_dic_gp[key] = (passed_veto, list_of_friends,
                           snr2_corr_factor, metadata)
    return opt_dic_gp


def stringent_veto(
        triggers, det_ind, trig_obj, min_veto_snr2=None, group_duration=0.1,
        origin=0, n_cores=1):
    """
    Apply stringent vetoes to optimized candidates for a single detector.
    """
    if utils.checkempty(triggers):
        return np.zeros(0, dtype=bool), \
               np.ones((0, 4 + len(params.SPLIT_CHUNKS)), dtype=bool)

    _, dcalphas_veto = trig_obj.define_finer_grid_func(
        dcalpha_coarse=trig_obj.delta_calpha / 2, trim_dims=False)
    extent = (trig_obj.templatebank.bounds[:len(dcalphas_veto), 2] -
              trig_obj.templatebank.bounds[:len(dcalphas_veto), 0]) / 2
    extent *= trig_obj.template_safety

    indices_groups = utils.splitarray(
        np.arange(len(triggers)), triggers[:, det_ind, 0], group_duration,
        axis=0, origin=origin)

    def stringent_veto_gp(trigs_gp):
        trigs_gp_det = trigs_gp[:, det_ind, :]
        if ((min_veto_snr2 is not None) and
                (np.max(trigs_gp_det[:, 1]) <= min_veto_snr2)):
            return np.ones(len(trigs_gp), dtype=bool), \
                np.ones((len(trigs_gp), 4 + len(params.SPLIT_CHUNKS)),
                        dtype=bool)

        subset_details = trig_obj.prepare_subset_for_vetoes(trigs_gp_det)
        trigger_ids = np.asarray(
            utils.make_trigger_ids(trigs_gp_det, trig_obj.c0_pos),
            dtype=np.int64)
        unique_inds_gp = utils.splitarray(
            np.arange(len(trigger_ids)), trigger_ids, 1)

        trigger_fates_gp = []
        metadata_gp = []
        for unique_ind_gp in unique_inds_gp:
            trigger_to_veto = trigs_gp_det[unique_ind_gp[0]]
            glitch_mask = np.ones(4 + len(params.SPLIT_CHUNKS), dtype=bool)

            if (min_veto_snr2 is None) or (trigger_to_veto[1] > min_veto_snr2):
                veto_spacing = dcalphas_veto.copy()
                trigs_chi2 = trigs_gp[unique_ind_gp[0], :, 1]
                snr_trig = np.sqrt(np.sum(trigs_chi2))
                veto_spacing[
                    np.logical_and(
                        extent > 1 / snr_trig,
                        1 / snr_trig > veto_spacing)] = 1 / snr_trig

                passed_veto, phase_veto_mask = \
                    trig_obj.veto_trigger_phase(
                        trigger_to_veto, dcalphas=veto_spacing,
                        subset_details=subset_details, verbose=True, lazy=False)
                glitch_mask[:len(phase_veto_mask)] = phase_veto_mask[:]

                passed_veto_2, large_chunk_veto_mask = \
                    trig_obj.veto_trigger_phase(
                        trigger_to_veto, n_chunk=params.N_CHUNK_2,
                        split_chunks=[], dcalphas=veto_spacing,
                        subset_details=subset_details, verbose=True, lazy=False)
                glitch_mask[len(phase_veto_mask):] = large_chunk_veto_mask[:]
                passed_veto *= passed_veto_2
            else:
                passed_veto = True

            trigger_fates_gp.append(
                np.c_[unique_ind_gp, [passed_veto] * len(unique_ind_gp)])
            metadata_gp.append(
                np.c_[unique_ind_gp, [glitch_mask] * len(unique_ind_gp)])

        trigger_fates_gp = np.vstack(trigger_fates_gp)
        trigger_fates_gp = trigger_fates_gp[trigger_fates_gp[:, 0].argsort(), 1]
        metadata_gp = np.vstack(metadata_gp)
        metadata_gp = metadata_gp[metadata_gp[:, 0].argsort(), 1:]
        return trigger_fates_gp, metadata_gp

    trigger_fates_file = []
    metadata_file = []
    if n_cores == 1:
        for i, indices_group in enumerate(indices_groups):
            trigs_group = triggers[indices_group]
            trigger_fates_group, metadata_group = stringent_veto_gp(trigs_group)
            trigger_fates_file.append(np.c_[indices_group, trigger_fates_group])
            metadata_file.append(np.c_[indices_group, metadata_group])
    else:
        with mp.Pool(n_cores) as p:
            trigger_groups_chunk = [triggers[ids] for ids in indices_groups]
            results = p.map(stringent_veto_gp, trigger_groups_chunk)

        for indices_group, (trigger_fates_group, metadata_group) in zip(
                indices_groups, results):
            trigger_fates_file.append(np.c_[indices_group, trigger_fates_group])
            metadata_file.append(np.c_[indices_group, metadata_group])

    trigger_fates_file = np.vstack(trigger_fates_file)
    trigger_fates_file = trigger_fates_file[trigger_fates_file[:, 0].argsort(), 1]
    metadata_file = np.vstack(metadata_file)
    metadata_file = metadata_file[metadata_file[:, 0].argsort(), 1:]

    return np.array(trigger_fates_file, dtype=bool), \
        np.array(metadata_file, dtype=bool)


# --- Helper Functions ---

def group_by_id(clist, c0_pos, ncalpha=None):
    """
    Groups the triggers in a clist into sublists with common calphas.
    """
    if ncalpha is not None:
        template_ids = np.asarray(
            utils.make_template_ids(clist[:, c0_pos:c0_pos + ncalpha]),
            dtype=np.int64)
    else:
        template_ids = np.asarray(
            utils.make_template_ids(clist[:, c0_pos:]), dtype=np.int64)

    return utils.splitarray(
        clist, template_ids, 1, axis=0, return_split_keys=True)


def find_friends_in_cloud(cloud_dict, finer_calphas):
    """
    Helper to find triggers in an optimized cloud dictionary.
    """
    if len(cloud_dict.keys()) == 0:
        return []

    tg_ids_finer = np.array(
        utils.make_template_ids(finer_calphas), dtype=np.int64)
    members_of_cloud = [cloud_dict.get(tg_id) for tg_id in tg_ids_finer
                        if cloud_dict.get(tg_id) is not None]

    if len(members_of_cloud) > 0:
        return np.vstack(members_of_cloud)
    return []


def read_channel_dict(trig_obj, times, chan_name):
    """
    Reads off boolean channel dict entries for specified times.
    """
    inds = np.floor(np.asarray(times) - trig_obj.time[0]).astype(int)
    return trig_obj.channel_dict[chan_name][inds]


@njit
def secondary_peak_reject(
        processed_clist, timeseries,
        max_friend_degrade_snr2=params.MAX_FRIEND_DEGRADE_SNR2, score_reduction_max=5):
    """
    Return True if this trigger should be rejected for having a secondary peak.
    """
    if len(timeseries) == 0:
        return False
    snr2max = np.max(timeseries[:, 1]**2 + timeseries[:, 2]**2)
    return processed_clist[1] < (snr2max * (1 - max_friend_degrade_snr2) - score_reduction_max)


def flag_duplicates_per_group_pair(bg_events, group_duration, c0_pos, origin=0,
                                 veto_mask=None, extra_arrays=None, remove=False,
                                 score_func=utils.incoherent_score, **kwargs):
    """
    Flag duplicate candidates (placeholder implementation).
    This function needs to be implemented based on the original logic.
    """
    # Placeholder implementation - return unchanged arrays
    num_events = len(bg_events)
    num_detectors = bg_events.shape[1] if len(bg_events.shape) > 1 else 1
    
    if veto_mask is None:
        veto_mask = np.ones(num_events, dtype=bool)
    
    mask_retain = np.ones((num_events, num_detectors), dtype=bool)
    
    if extra_arrays is None:
        extra_arrays = []
    
    return bg_events, veto_mask, extra_arrays, mask_retain