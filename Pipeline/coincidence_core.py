"""
Core coincidence detection logic for N-detector gravitational wave searches.

This module contains the fundamental algorithms for finding coincident gravitational 
wave triggers across multiple detectors (generalized from 2 to N detectors).
"""

import numpy as np
import itertools
import os
import glob
import utils
import triggers_single_detector_HM as trig
from numba import njit
import params


def load_trigger_file(config_fname):
    """Convenience function to load triggers from a file
    :param config_fname: json file
    :return: processedclist for file
    """
    trig_fname = config_fname.split("_config.json")[0] + ".trig.npy"
    clist = np.load(open(trig_fname, "rb"))

    if utils.checkempty(clist):
        print(f"No triggers in {trig_fname}", flush=True)

    return clist


def get_files_by_epoch(dir_name, detectors, enumerated_epochs=None, n_epochs=None, run=None):
    """
    Generalized file listing for N detectors.
    
    :param dir_name: Directory with json and trig files
    :param detectors: List of detector names ['H1', 'L1', 'V1']
    :param enumerated_epochs: List of epochs, if needed
    :param n_epochs: Number of epochs, if needed
    :param run: String identifying the run
    :return: Dictionary mapping epoch -> list of file paths for each detector
    """
    files_by_detector = {}
    
    # Get files for each detector
    for det in detectors:
        if enumerated_epochs is None:
            files = glob.glob(os.path.join(dir_name, f"*{det}*.json"))
        else:
            enumerated_epochs = [int(x) for x in enumerated_epochs]
            if run is None:
                run = utils.get_run(enumerated_epochs[0])
            
            files = []
            for epoch in enumerated_epochs:
                eplist = [epoch,
                         int(epoch + params.DEF_FILELENGTH),
                         int(epoch - params.DEF_FILELENGTH)]
                files += [utils.get_json_fname(dir_name, ep, det, run)
                         for ep in eplist]
        
        # Remove repetitions and keep only existing files
        files = list(set(files))
        files = [f for f in files if os.path.isfile(f)]
        files_by_detector[det] = files
    
    # Group files by epoch
    files_by_epoch = {}
    
    # Extract epochs from first detector as reference
    if detectors:
        ref_detector = detectors[0]
        ref_files = files_by_detector[ref_detector]
        
        for ref_file in ref_files:
            ref_epoch = utils.get_epoch_from_filename(ref_file)
            epoch_files = [ref_file]
            
            # Find corresponding files for other detectors
            for det in detectors[1:]:
                matching_file = None
                for file in files_by_detector[det]:
                    if utils.get_epoch_from_filename(file) == ref_epoch:
                        matching_file = file
                        break
                
                if matching_file:
                    epoch_files.append(matching_file)
                else:
                    # Skip this epoch if any detector is missing
                    epoch_files = None
                    break
            
            if epoch_files and len(epoch_files) == len(detectors):
                files_by_epoch[ref_epoch] = epoch_files
    
    return files_by_epoch


def group_by_template_id(friends_list, template_id_pos):
    """Group triggers by template ID"""
    if len(friends_list) == 0:
        return [], []
    
    # Sort by template ID
    sorted_indices = np.argsort(friends_list[:, template_id_pos])
    sorted_friends = friends_list[sorted_indices]
    
    # Find unique template IDs and split positions
    unique_template_ids, split_indices = np.unique(
        sorted_friends[:, template_id_pos], return_index=True)
    
    # Split into groups
    friends_split = np.split(sorted_friends, split_indices[1:])
    
    return unique_template_ids, friends_split


def get_best_candidate_tuple(list_of_friends_arrs, template_id_pos, score_pos, 
                           time_pos, detector_names):
    """
    Find the best candidate tuple from N detector friend arrays.
    
    :param list_of_friends_arrs: List of friend arrays, one per detector
    :param template_id_pos: Position of template ID in trigger array
    :param score_pos: Position of score in trigger array  
    :param time_pos: Position of time in trigger array
    :param detector_names: List of detector names
    :return: Best candidate tuple (N triggers, one per detector)
    """
    if not list_of_friends_arrs or any(len(arr) == 0 for arr in list_of_friends_arrs):
        return None
    
    # Group each detector's triggers by template ID
    template_groups_by_detector = []
    for friends_arr in list_of_friends_arrs:
        unique_ids, groups = group_by_template_id(friends_arr, template_id_pos)
        template_groups_by_detector.append(dict(zip(unique_ids, groups)))
    
    # Find template IDs common to all detectors
    common_template_ids = set(template_groups_by_detector[0].keys())
    for template_dict in template_groups_by_detector[1:]:
        common_template_ids &= set(template_dict.keys())
    
    if not common_template_ids:
        return None
    
    best_candidate = None
    best_score = -np.inf
    
    # For each common template, find the best trigger from each detector
    for template_id in common_template_ids:
        candidate_triggers = []
        total_score = 0
        
        for i, template_dict in enumerate(template_groups_by_detector):
            triggers = template_dict[template_id]
            # Find best trigger for this template in this detector
            best_idx = np.argmax(triggers[:, score_pos])
            best_trigger = triggers[best_idx]
            candidate_triggers.append(best_trigger)
            total_score += best_trigger[score_pos]
        
        # Check time consistency (simple check - can be made more sophisticated)
        times = [trigger[time_pos] for trigger in candidate_triggers]
        if max(times) - min(times) < params.COINCIDENCE_WINDOW:
            if total_score > best_score:
                best_score = total_score
                best_candidate = np.array(candidate_triggers)
    
    return best_candidate


def collect_background_candidates(list_of_clists, num_detectors, 
                                score_pos=1, time_pos=3, template_id_pos=0,
                                detector_names=None):
    """
    Generalized N-detector coincidence finding.
    
    :param list_of_clists: List of trigger arrays, one per detector
    :param num_detectors: Number of detectors
    :param score_pos: Position of score in trigger array
    :param time_pos: Position of time in trigger array
    :param template_id_pos: Position of template ID in trigger array
    :param detector_names: List of detector names
    :return: Array of coincident candidates shape (n_candidates, n_detectors, n_params)
    """
    if detector_names is None:
        detector_names = [f"Det{i}" for i in range(num_detectors)]
    
    # Check for empty trigger lists
    if any(utils.checkempty(clist) for clist in list_of_clists):
        return np.array([]).reshape(0, num_detectors, list_of_clists[0].shape[1] if list_of_clists[0].size > 0 else 0)
    
    # Get friends arrays for all detectors
    friends_by_detector = []
    for clist in list_of_clists:
        try:
            friends_arr_list = utils.get_friends_arr(clist)  # This returns list of time buckets
            friends_by_detector.append(friends_arr_list)
        except AttributeError:
            # If get_friends_arr doesn't exist, create simple bucket structure
            friends_by_detector.append([clist])
    
    # Build master template dictionary
    template_info = {}
    
    for det_idx, friends_arr_list in enumerate(friends_by_detector):
        for bucket_idx, friends_arr in enumerate(friends_arr_list):
            if len(friends_arr) == 0:
                continue
                
            unique_templates, friends_split = group_by_template_id(
                friends_arr, template_id_pos)
            
            for temp_id, trigger_group in zip(unique_templates, friends_split):
                if temp_id not in template_info:
                    template_info[temp_id] = {}
                if det_idx not in template_info[temp_id]:
                    template_info[temp_id][det_idx] = []
                
                # Store bucket information
                template_info[temp_id][det_idx].append({
                    'bucket_idx': bucket_idx,
                    'triggers': trigger_group,
                    'best_score': np.max(trigger_group[:, score_pos]),
                    'time_range': (np.min(trigger_group[:, time_pos]), 
                                  np.max(trigger_group[:, time_pos]))
                })
    
    # Find templates present in all detectors
    background_candidates = []
    
    for temp_id, det_buckets in template_info.items():
        if len(det_buckets) == num_detectors:
            # This template exists in all detectors
            # Get all combinations of bucket groups
            bucket_lists = [det_buckets[i] for i in range(num_detectors)]
            
            for bucket_combination in itertools.product(*bucket_lists):
                # Check time consistency across detectors
                time_ranges = [bucket['time_range'] for bucket in bucket_combination]
                earliest_end = min(time_range[1] for time_range in time_ranges)
                latest_start = max(time_range[0] for time_range in time_ranges)
                
                if earliest_end > latest_start:  # Overlapping time ranges
                    # Create list of trigger arrays for this combination
                    friends_arrs = [bucket['triggers'] for bucket in bucket_combination]
                    
                    # Find best candidate tuple
                    candidate = get_best_candidate_tuple(
                        friends_arrs, template_id_pos, score_pos, 
                        time_pos, detector_names)
                    
                    if candidate is not None:
                        background_candidates.append(candidate)
    
    if not background_candidates:
        return np.array([]).reshape(0, num_detectors, 
                                   list_of_clists[0].shape[1] if list_of_clists[0].size > 0 else 0)
    
    return np.array(background_candidates)


def find_interesting_dir(dir_name, detectors=['H1', 'L1'], **kwargs):
    """
    Main driver function for N-detector coincidence analysis.
    
    :param dir_name: Directory containing trigger files
    :param detectors: List of detector names (default: ['H1', 'L1'])
    :param kwargs: Additional parameters passed to downstream functions
    :return: Output filename or processing status
    """
    print(f"Processing directory: {dir_name} with detectors: {detectors}")
    
    # Get files grouped by epoch
    files_by_epoch = get_files_by_epoch(dir_name, detectors, **kwargs)
    
    if not files_by_epoch:
        print("No coincident files found")
        return None
    
    print(f"Found {len(files_by_epoch)} epochs to process")
    
    all_background_candidates = []
    
    # Process each epoch
    for epoch, file_list in files_by_epoch.items():
        print(f"Processing epoch {epoch}")
        
        # Load trigger objects and data
        list_of_trig_objects = []
        list_of_clists = []
        
        for file_path in file_list:
            try:
                trig_obj = trig.TriggerList.from_json(file_path)
                clist = load_trigger_file(file_path)
                list_of_trig_objects.append(trig_obj)
                list_of_clists.append(clist)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                break
        
        if len(list_of_trig_objects) != len(detectors):
            print(f"Skipping epoch {epoch} - incomplete detector data")
            continue
        
        # Find background candidates
        bg_events = collect_background_candidates(
            list_of_clists, len(detectors), detector_names=detectors)
        
        if bg_events.size > 0:
            print(f"Found {len(bg_events)} background candidates in epoch {epoch}")
            all_background_candidates.extend(bg_events)
            
            # TODO: Add call to veto_and_optimize_coincidence_list
            # This will be implemented in coincidence_veto.py
    
    print(f"Total background candidates found: {len(all_background_candidates)}")
    return all_background_candidates


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        dir_name = sys.argv[1]
        detectors = sys.argv[2:] if len(sys.argv) > 2 else ['H1', 'L1']
        result = find_interesting_dir(dir_name, detectors)
        print(f"Processing complete. Result: {result}")