import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from argparse import ArgumentParser
import getpass
import json
import triggers_single_detector_HM as trig
import glob
import params
import utils
import time
import copy
import multiprocess as mp
import sys
import itertools
from numba import njit

# Default paths for running coincidence using helios
DEFAULT_PROGRAM_NAME = 'coincidence_HM.py'
DEFAULT_PROGPATH = os.path.join(utils.CODE_DIR, DEFAULT_PROGRAM_NAME)
TMP_FILENAME = "tmp_submit_script_coincidence.sh"
DEFAULT_TMP_PATH = os.path.join(utils.DATA_ROOT, TMP_FILENAME)

# CS_INSTANCE = cs.CoherentScore()

#TODO: Write loading functions for the new format

# def coherent_score_func(triggers, time_slide_jump=0.1, n_dims=3):
#     """
#     :param triggers:
#         n_cand x 2 x row of processedclists (can be a 2d array if n_cand = 1)
#     :param time_slide_jump: Unit for timeslides (s)
#     :param n_dims: Number of dimensions in template bank
#     :return: Vector of incoherent scores (scalar if n_cand = 1)
#     """
#     coh_scores = np.sum(CS_INSTANCE.get_all_prior_terms(
#         triggers, time_slide_jump=time_slide_jump), axis=-1)
#     rho_sq = utils.incoherent_score(triggers)
#     if triggers.ndim > 2:
#         return coh_scores + rho_sq - (n_dims + 4) * np.log(rho_sq)
#     else:
#         return coh_scores[0] + rho_sq - (n_dims + 4) * np.log(rho_sq)

@njit
def secondary_peak_reject(
        processed_clist, timeseries,
        max_friend_degrade_snr2=params.MAX_FRIEND_DEGRADE_SNR2, score_reduction_max=5):
    """
    Return True if this trigger should be rejected for having a secondary peak that
        ruins the coherent score
    """
    if len(timeseries) == 0:
        return False
    snr2max = np.max(timeseries[:, 1]**2+timeseries[:, 2]**2)
    return processed_clist[1] < (snr2max*(1 - max_friend_degrade_snr2) - score_reduction_max)

def track_job(job, jobname, n_tasks, update_interval=10):
    """Track a multiprocessing job submitted in chunks"""
    old_tasks_remaining = n_tasks + job._chunksize
    while job._number_left > 0:
        tasks_remaining = job._number_left * job._chunksize
        if tasks_remaining < old_tasks_remaining:
            old_tasks_remaining = tasks_remaining
            print(f"Fraction of {jobname} left: {tasks_remaining / n_tasks}",
                  flush=True)
        time.sleep(update_interval)


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


# Functions to send runs to cluster
# ---------------------------------
def create_candidate_output_dir_name(run_path, cver, run="O2"):
    run_subdir = os.path.basename(run_path)
    cand_dir_name = os.path.join(
        utils.CAND_DIR[run.lower()], run_subdir + f"cand{cver}")
    return cand_dir_name


def restart_run_from_json(
        jsonfile, keys_to_pop=(
                'auto_apply_veto',
                'optimize_calpha',
                'n_opt',
                'time_slide_shifts',
                'default_cand_root',
                'old_cand_dir_name'),
        **kwargs):
    with open(jsonfile, "r") as f:
        function_inputs = json.load(f)

    # Pop some of the keys, useful when rerunning files in old runs that were
    # generated using previous versions of the code
    if keys_to_pop is not None:
        for key in keys_to_pop:
            _ = function_inputs.pop(key, None)

    # Fix changed defaults to work with O1/O2 runs
    function_inputs['opt_format'] = function_inputs.get('opt_format', "old")
    function_inputs['outfile_format'] = \
        function_inputs.get('outfile_format', "old")
    function_inputs['max_zero_lag_delay'] = function_inputs.get(
        'max_zero_lag_delay', function_inputs['time_shift_tol'])
    function_inputs['detectors'] = \
        function_inputs.get('detectors', ('H1', 'L1'))
    # TODO: Modify the existing json files after fixing
    function_inputs['weaker_detectors'] = \
        function_inputs.get('weaker_detectors', ())

    # Load O1 run
    function_inputs['output_timeseries'] = \
        function_inputs.get('output_timeseries', False)
    function_inputs['output_coherent_score'] = \
        function_inputs.get('output_coherent_score', False)

    # Load O2 runs
    if 'out_format' in function_inputs.keys():
        out_format = function_inputs.pop('out_format')
        if out_format.lower() == "processedclist":
            function_inputs['output_timeseries'] = False
            function_inputs['output_coherent_score'] = False
        elif out_format.lower() == "timeseries":
            function_inputs['output_timeseries'] = True
            function_inputs['output_coherent_score'] = False
        else:
            raise RuntimeError(
                f"Encountered unknown format {out_format}")

    function_inputs['recompute_psd_drift'] = \
        function_inputs.get('recompute_psd_drift', True)

    # Override any of the kwargs, or add new arguments
    for item in kwargs.items():
        function_inputs[item[0]] = item[1]

    epochs_to_run = find_interesting_dir_cluster(**function_inputs)
    return epochs_to_run


def find_interesting_dir_cluster(
        dir_path, output_dir=None, threshold_chi2=60., min_veto_chi2=30,
        max_time_slide_shift=100, score_reduction_max=5, time_shift_tol=0.01,
        minimal_time_slide_jump=0.1, max_zero_lag_delay=0.015, opt_format="new",
        output_timeseries=True, output_coherent_score=True,
        score_reduction_timeseries=10, detectors=('H1', 'L1'),
        weaker_detectors=(), recompute_psd_drift=False,
        outfile_format="new", job_name="coincidence", n_cores=1,
        n_hours_limit=12, epoch_list=None, n_jobs=256, mem_limit=None,
        debug=False, submit=False, overwrite=False, cluster='typhon',
        exclusive=False, run='O3a', cver=6, rerun=False, old_cver=None,
        exclude_nodes=False):
    """
    :param dir_path: Path with output files of the triggering
    :param output_dir: Where to place the coincident_files, if known
    :param threshold_chi2:
    :param min_veto_chi2:
    :param max_time_slide_shift:
    :param score_reduction_max:
    :param time_shift_tol:
    :param minimal_time_slide_jump:
    :param max_zero_lag_delay:
        Maximum delay between detectors within the same timeslide
    :param opt_format:
        How we choose the finer grid, changed between O1 and O2 analyses
        Exposed here to replicate old runs if needed
    :param output_timeseries: Flag to output timeseries for the candidates
    :param output_coherent_score:
        Flag to compute the coherent score integral for the candidates
    :param score_reduction_timeseries:
        Restrict triggers in timeseries to the ones with
        single_detector_SNR^2 > (base trigger SNR^2) - this parameter
    :param detectors:
        Tuple with names of the two detectors we will be running coincidence
        with ('H1', 'L1', 'V1' supported)
    :param weaker_detectors:
        If needed, tuple with names of weaker detectors that we will compute
        timeseries for as well ('H1', 'L1', 'V1' supported)
        Note: Only works with outfile_format == "new"
    :param recompute_psd_drift:
        Flag to recompute PSD drift correction. We needed it in O2 since the
        trigger files didn't use safemean. Redundant in O3a and forwards
    :param outfile_format:
        Flag whether to save the old style (separate npy files for different
        arrays), or in the new style with a consolidated file per job
    :param job_name:
    :param n_cores: Number of cores to use for splitting the veto computations
    :param n_hours_limit:
    :param epoch_list: If known, list of epochs to do coincidence for
    :param n_jobs:
    :param mem_limit: Memory limit in GB per core requested on cluster
    :param debug:
    :param submit:
    :param overwrite:
        Flag whether to overwrite files (if the base file exists, we preserve
        it with the suffix "_old"). If int(overwrite) == :
        0: Does not touch existing files
        1: Redoes the computations and overwrites the existing file
        2: Avoids redoing computations and continues the weaker detectors if
           missing (Note: Only works with outfile_format == "new")
    :param cluster:
    :param exclusive:
    :param run: String identifying the run
    :param cver: Cand version
    :param rerun:
        Set this flag if rerunning, in which case we append the suffix "_new"
        to filenames
    :param old_cver: Old cand run to use to avoid redoing vetoes, if known
    :param exclude_nodes: Flag to exclude a few nodes from the job
                        (currently only implemented for typhon cluster)
    :return:
    """
    # First create the output directory if needed
    if output_dir is None:
        output_dir = create_candidate_output_dir_name(dir_path, cver, run=run)

    # If we are using an old run to save on vetoes
    old_cand_dir_name = None
    if old_cver is not None:
        old_cand_dir_name = \
            create_candidate_output_dir_name(dir_path, old_cver, run=run)

    # Bring into the right order for the convention (H1, L1, V1)
    detectors = tuple(sorted(detectors))

    if weaker_detectors is None:
        weaker_detectors = ()

    if submit:
        # Sometimes even the default candidate root does not exist
        # In these cases, create it
        default_cand_root = utils.CAND_DIR[run.lower()]
        if not os.path.isdir(default_cand_root):
            os.makedirs(default_cand_root)
            os.system(f"chmod 777 {default_cand_root}")

        # Create the output directory if it doesn't exist
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            os.system(f"chmod 777 {output_dir}")

        # Create a json file with parameters if it doesn't already exist
        # Create dictionary of inputs to function
        function_inputs = locals()
        jsonfile = os.path.join(
            output_dir,
            "coincidence_parameters_" + "_".join(detectors) + ".json")
        if not os.path.isfile(jsonfile):
            with open(jsonfile, 'w') as f:
                json.dump(function_inputs, f, indent=2)

        # Copy the params file with veto parameters
        dest = os.path.join(output_dir, 'params.py')
        os.system(f"cp params.py {dest}")

        # If we are reusing old run to save on vetoes, check it exists
        if ((old_cand_dir_name is not None) and
                (not os.path.isdir(old_cand_dir_name))):
            raise RuntimeError(f"cand {old_cver} doesn't exist")

    # List of files in the first detector
    files1 = glob.glob(os.path.join(dir_path, f"*{detectors[0]}*.json"))

    # Sort epochs by start time and split into jobs
    epochs = sorted([int(f.split("-")[-2]) for f in files1])
    jump = len(epochs) // n_jobs + 1
    start_epochs = epochs[::jump]
    end_epochs = epochs[jump-1::jump] + [epochs[-1]]

    if epoch_list is not None:
        start_epochs = epoch_list
        end_epochs = epoch_list
    
    # Function to run a job, works for both compute servers and clusters
    def run_job(job_index):
        """Returns epoch if it hasn't been run earlier, else returns None
        If running on a compute server, it returns after finishing job
        If running on cluster, queues job and returns"""
        start_epoch = start_epochs[job_index]
        end_epoch = end_epochs[job_index]

        output_root = os.path.join(
            output_dir,
            "coincident_events__" + "_".join(detectors) + "_" +
            str(start_epoch) + "_" + str(end_epoch))
        stdout_output_fname = output_root + ".out"
        stdout_error_fname = output_root + ".err"
        if outfile_format.lower() == "old":
            base_output_fname = output_root + ".npy"
            backup_output_fname = output_root + "_old.npy"
            if rerun:
                output_fname = output_root + "_new.npy"
            else:
                output_fname = output_root + ".npy"
        else:
            base_output_fname = output_root + ".npz"
            backup_output_fname = output_root + "_old.npz"
            if rerun:
                output_fname = output_root + "_new.npz"
            else:
                output_fname = output_root + ".npz"

        # If we aren't overwriting, check if the most recent run finished
        if int(overwrite) == 0 and os.path.isfile(stdout_output_fname):
            with open(stdout_output_fname, "r") as fobj:
                contents = fobj.readlines()
                if len(contents) > 0:
                    if np.any(['finished' in x for x in contents[-100:]]):
                        # We already finished with this set of epochs
                        return None

        if submit:
            # Preserve the base file
            if (os.path.isfile(base_output_fname) and
                    (os.path.getsize(base_output_fname) > 0)):
                # If we're rewriting, save the old version
                if int(overwrite) == 1:
                    os.system(f"mv {base_output_fname} {backup_output_fname}")
                    os.system(f"chmod 666 {backup_output_fname}")
                # os.remove(output_fname)

            # Create empty output and result files
            os.system(f"touch {stdout_output_fname}")
            os.system(f"chmod 666 {stdout_output_fname}")
            os.system(f"touch {output_fname}")
            os.system(f"chmod 666 {output_fname}")
        
        if cluster is None:
            # We're running on a compute server
            import time
            if submit:
                logfile = open(stdout_output_fname, "a")
                sys.stdout = logfile
                sys.stderr = logfile
                enumerated_epochs = [
                    ep for ep in epochs if start_epoch <= ep <= end_epoch]
                print(f"Beginning candidate extraction for " +
                      f"{len(enumerated_epochs)} epochs starting from " +
                      f"{start_epoch} at:", time.ctime(), flush=True)
                candidates = find_interesting_dir(
                    dir_path, enumerated_epochs=enumerated_epochs,
                    time_shift_tol=time_shift_tol,
                    score_reduction_max=score_reduction_max,
                    threshold_chi2=threshold_chi2,
                    max_time_slide_shift=max_time_slide_shift,
                    minimal_time_slide_jump=minimal_time_slide_jump,
                    max_zero_lag_delay=max_zero_lag_delay,
                    min_veto_chi2=min_veto_chi2,
                    out_fname=output_fname,
                    outfile_format=outfile_format,
                    n_cores=n_cores,
                    run=run,
                    opt_format=opt_format,
                    old_cand_dir_name=old_cand_dir_name,
                    output_timeseries=output_timeseries,
                    output_coherent_score=output_coherent_score,
                    score_reduction_timeseries=score_reduction_timeseries,
                    detectors=detectors,
                    weaker_detectors=weaker_detectors,
                    recompute_psd_drift=recompute_psd_drift)
                print("Process finished successfully at:", time.ctime(),
                      flush=True)
                logfile.close()

            return start_epochs[job_index]

        # We're running on the cluster
        if cluster.lower() == 'typhon':
            # We're running on Typhon
            text = f"#!/bin/bash \n#SBATCH --job-name={job_name}\n"
            text += f"#SBATCH --output={stdout_output_fname}\n"
            text += f"#SBATCH --open-mode=append\n"
            # text += f"#SBATCH --nodes=1\n"
            text += f"#SBATCH --ntasks=1\n"
            # Add one to account for the memory used by the parent process
            # if spawning threads using multiprocessing
            if n_cores > 1:
                n_cores_per_task = n_cores + 1
            else:
                n_cores_per_task = n_cores
            if mem_limit is not None:
                # mem_submit = int(mem_limit * 1000)
                # text += f'#SBATCH --mem-per-cpu={mem_submit}\n'
                # Typhon has 4 GB per core, ensure that jobs do not fight
                n_cores_to_request = \
                    max(n_cores_per_task,
                        int(np.ceil(n_cores_per_task * int(mem_limit) / 4)))
            else:
                n_cores_to_request = n_cores_per_task
            text += f"#SBATCH --cpus-per-task={n_cores_to_request}\n"
            text += f"#SBATCH --time={int(n_hours_limit)}:00:00\n"
            if exclude_nodes:
                text += f'#SBATCH --exclude=typhon-node[1-10]\n'
        elif cluster.lower() == 'helios':
            # We're running on Helios
            text = f"#!/bin/bash \n#SBATCH --job-name={job_name}\n"
            text += f"#SBATCH --output={stdout_output_fname}\n"
            text += f"#SBATCH --open-mode=append\n"
            # text += f"#SBATCH --nodes=1\n"
            text += f"#SBATCH --ntasks=1\n"
            # Add one to account for the memory used by the parent process
            # if spawning threads using multiprocessing
            if n_cores > 1:
                n_cores_per_task = n_cores + 1
            else:
                n_cores_per_task = n_cores
            if mem_limit is not None:
                # mem_submit = int(mem_limit * 1000)
                # text += f'#SBATCH --mem-per-cpu={mem_submit}\n'
                # Helios has 4 GB per core, ensure that jobs do not fight
                n_cores_to_request = \
                    max(n_cores_per_task,
                        int(np.ceil(n_cores_per_task * int(mem_limit) / 4)))
            else:
                n_cores_to_request = n_cores_per_task
            text += f"#SBATCH --cpus-per-task={n_cores_to_request}\n"
            text += f"#SBATCH --time={int(n_hours_limit)}:00:00\n"
        elif cluster.lower() == 'hyperion':
            # We're running on Hyperion
            text = f'#!/bin/bash\n'
            text += f'#$ -cwd\n'
            text += f'#$ -w n\n'
            text += f'#$ -N {job_name}\n'
            text += f'#$ -o {stdout_output_fname}\n'
            text += f'#$ -e {stdout_error_fname}\n'
            text += f'#$ -V\n'
            # Add one to account for the memory used by the parent process
            # if spawning threads using multiprocessing
            if n_cores > 1:
                n_cores_per_task = n_cores + 1
            else:
                n_cores_per_task = n_cores
            if mem_limit is not None:
                # text += f'#$ -l h_vmem={int(mem_limit)}G\n'
                # Hyperion has 2 GB per core, ensure that jobs do not fight
                n_cores_to_request = \
                    max(n_cores_per_task,
                        int(np.ceil(n_cores_per_task * int(mem_limit) / 2)))
            else:
                n_cores_to_request = n_cores_per_task
            text += f'#$ -pe smp {n_cores_to_request}\n'
            text += f'#$ -l h_rt={int(n_hours_limit)}:00:00\n'
        else:
            raise ValueError(
                f"I don't know what to do with cluster = {cluster}!")

        text += utils.env_init_lines(cluster=cluster)

        text += f" {DEFAULT_PROGPATH} {dir_path} {output_fname}"
        text += f" {start_epoch} {end_epoch}"
        text += f" --threshold_incoherent_chi2={threshold_chi2}"
        text += f" --time_shift_tol={time_shift_tol}"
        text += f" --min_veto_chi2={min_veto_chi2}"
        text += f" --max_time_slide_shift={max_time_slide_shift}"
        text += f" --minimal_time_slide_jump={minimal_time_slide_jump}"
        text += f" --max_zero_lag_delay={max_zero_lag_delay}"
        text += f" --score_reduction_max={score_reduction_max}"
        text += f" --n_cores={n_cores}"
        text += f" --run={run}"
        text += f" --opt_format={opt_format}"
        text += f" --outfile_format={outfile_format}"
        text += f" --old_cand_dir_name={old_cand_dir_name}"
        text += f" --detectors " + " ".join(detectors)
        if not utils.checkempty(weaker_detectors):
            text += f" --weaker_detectors " + " ".join(weaker_detectors)
        if output_timeseries:
            text += f" --output_timeseries"
        if output_coherent_score:
            text += f" --output_coherent_score"
        if output_timeseries or output_coherent_score:
            text += f" --score_reduction_timeseries={score_reduction_timeseries}"
        if recompute_psd_drift:
            text += f" --recompute_psd_drift"

        text += "\n"

        if debug:
            print(text)
        # else:
        #     print(start_epochs[job_index])

        if submit:
            import time
            current_tmp_filename = \
                DEFAULT_TMP_PATH + str(np.random.randint(0,2**40)) + '.tmp'
            with open(current_tmp_filename, "w") as file:
                file.write(text)
            print("changing its permissions")
            # Add user run permissions
            os.system(f"chmod 777 {current_tmp_filename}")
            print("sending jobs")
            if cluster.lower() in ['typhon','helios']:
                os.system(f"sbatch {current_tmp_filename}")
            elif cluster.lower() == 'hyperion':
                if exclusive:
                    os.system(f"qsub -l excl=true {current_tmp_filename}")
                else:
                    os.system(f"qsub {current_tmp_filename}")
            time.sleep(1)
            print(f"removing config file {current_tmp_filename}")
            os.remove(current_tmp_filename)

        return start_epochs[job_index]

    # if (cluster is None) and (n_cores > 1):
    #     # We're running jobs locally, or on a compute server, using
    #     # multiprocessing to split epochs over cores
    #     if n_cores_for_veto > 1:
    #         raise RuntimeError("Cannot do nested multiprocessing!")
    #     p = mp.Pool(n_cores)
    #     run_start_epochs = p.imap_unordered(run_job, range(len(start_epochs)))
    #     epochs_to_run = [ep for ep in run_start_epochs if ep is not None]
    #     p.close()
    #     p.join()
    # else:
    #     # We're running jobs one by one, or on the cluster
    #     epochs_to_run = []
    #     for job_idx in range(len(start_epochs)):
    #         start_epoch_id = run_job(job_idx)
    #         if start_epoch_id is not None:
    #             epochs_to_run.append(start_epoch_id)

    # We're running jobs one by one, or on the cluster
    epochs_to_run = []
    for job_idx in range(len(start_epochs)):
        start_epoch_id = run_job(job_idx)
        if start_epoch_id is not None:
            epochs_to_run.append(start_epoch_id)

    return epochs_to_run


def main():
    parser = ArgumentParser(description="Find coincident triggers " +
                                        "postprocessing .trig.npy files")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Be verbose")
    parser.add_argument("dir_path", type=str,
                        help="Absolute path to directory with config files")
    parser.add_argument("out_cand_fname", type=str,
                        help="Absolute path to output candidate file")
    parser.add_argument("start_epoch", type=int,
                        help="Starting H1 epoch to analyze with this job")
    parser.add_argument("end_epoch", type=int,
                        help="Ending H1 epoch to analyze with this job")

    # Parameters of single buckets
    parser.add_argument("--time_shift_tol", type=float, default=0.01,
                        help="Maximum delta t allowed for coincident detection")
    parser.add_argument("--score_reduction_max", type=float, default=5,
                        help="Absolute SNR^2 degradation to allow to retain " +
                             "friends of a trigger")

    # Parameters of coincident/bg candidates in pairs of buckets
    parser.add_argument("--threshold_incoherent_chi2", type=float, default=60,
                        help="Threshold of the combined normed overlaps from " +
                             "which to save a candidate")
    parser.add_argument("--max_time_slide_shift", type=float, default=1000,
                        help="Maximum time slide shift to use to compute " +
                             "background distribution")
    parser.add_argument("--minimal_time_slide_jump", type=float, default=0.1,
                        help="Minimum jump in time slide between the " +
                             "detectors when calculating background")

    # Parameters of optimization and veto
    parser.add_argument("--min_veto_chi2", type=float, default=30,
                        help="Minimum single-detector SNR^2 for applying veto")
    parser.add_argument("--n_cores", type=int, default=1,
                        help="Number of cores to parallelize veto over")
    
    parser.add_argument("--run", type=str, help="String marking the run")
    parser.add_argument("--opt_format", type=str, default="new",
                        help="How we choose the finer grid, changed between " +
                             "O1 and O2 analyses. Exposed here to replicate " +
                             "old runs if needed")

    # New parameters, apology students!!!
    parser.add_argument("--detectors", nargs='+', type=str,
                        default=["H1", "L1"],
                        help="List of detector names in alphabetical order, " +
                             "separated by spaces")
    parser.add_argument("--weaker_detectors", nargs='+', type=str,
                        default=[],
                        help="List of weaker detector names in alphabetical " +
                             "order, separated by spaces")
    parser.add_argument("--max_zero_lag_delay", type=float, default=0.015,
                        help="Maximum delay between detectors within the " +
                             "same timeslide")
    parser.add_argument("--outfile_format", type=str, default="new",
                        help="FLag whether to save the old style (separate " +
                             "npy files for different arrays), or in the " +
                             "new style with a consolidated file per job")
    parser.add_argument("--old_cand_dir_name", type=str, default="None",
                        help="Pass old directory to avoid vetoes")
    parser.add_argument("--output_timeseries", action="store_true",
                        help="Flag to output timeseries for the candidates")
    parser.add_argument("--output_coherent_score", action="store_true",
                        help="Flag to compute the coherent score integral " +
                             "for the candidates")
    parser.add_argument("--score_reduction_timeseries", type=float, default=10,
                        help="Restrict triggers in timeseries to the ones " +
                             "with single_detector_SNR^2 > (base trigger " +
                             "SNR^2) - this parameter")
    parser.add_argument('--recompute_psd_drift', action='store_true',
                        help="Flag to recompute the PSD drift correction")

    args = parser.parse_args()

    # List of H1 files
    #files1 = glob.glob(os.path.join(args.dir_path, "*H1*.json"))

    # Epochs sorted by start time
    #epochs = sorted([int(f.split("-")[-2]) for f in files1])
    out_fname = args.out_cand_fname

    # Pick out the epochs that we are searching in
    #enumerated_epochs = [ep for ep in epochs if
    #                     args.start_epoch <= ep <= args.end_epoch]

    # including the end epoch in the list!
    enumerated_epochs = [str(x) for x in range(args.start_epoch, args.end_epoch+1, 4096)]

    print(f"Beginnning candidate extraction for {len(enumerated_epochs)} " +
          f"epochs starting from {args.start_epoch} at:", time.ctime())

    old_cand_dir_name = args.old_cand_dir_name
    if old_cand_dir_name == "None":
        old_cand_dir_name = None

    candidates = find_interesting_dir(
        args.dir_path, enumerated_epochs=enumerated_epochs,
        time_shift_tol=args.time_shift_tol,
        score_reduction_max=args.score_reduction_max,
        threshold_chi2=args.threshold_incoherent_chi2,
        max_time_slide_shift=args.max_time_slide_shift,
        minimal_time_slide_jump=args.minimal_time_slide_jump,
        max_zero_lag_delay=args.max_zero_lag_delay,
        min_veto_chi2=args.min_veto_chi2,
        out_fname=out_fname,
        outfile_format=args.outfile_format,
        n_cores=args.n_cores,
        run=args.run,
        opt_format=args.opt_format,
        old_cand_dir_name=old_cand_dir_name,
        output_timeseries=args.output_timeseries,
        output_coherent_score=args.output_coherent_score,
        score_reduction_timeseries=args.score_reduction_timeseries,
        detectors=tuple(args.detectors),
        weaker_detectors=tuple(args.weaker_detectors),
        recompute_psd_drift=args.recompute_psd_drift)
    print("Process finished successfully at:", time.ctime())

    return


# Driver functions
# ----------------
def find_interesting_dir(
        dir_name, enumerated_epochs=None, n_epochs=None, time_shift_tol=0.01,
        score_reduction_max=5, threshold_chi2=60., max_time_slide_shift=100,
        minimal_time_slide_jump=0.1, min_veto_chi2=30, max_zero_lag_delay=0.015,
        out_fname=None, n_cores=1, run='O3a', opt_format="new",
        outfile_format="new", old_cand_dir_name=None, bad_times=None,
        veto_triggers=True, output_timeseries=True, output_coherent_score=True,
        score_reduction_timeseries=10, detectors=('H1', 'L1'),
        weaker_detectors=(), recompute_psd_drift=False):
    """
    Goes through trigger files for H1 and L1 in a directory and performs
    coincidence analysis
    :param dir_name: Path to a directory with json files for H1 and L1
    :param enumerated_epochs: If desired, list of integer epochs to analyze
    :param n_epochs: If desired, restrict to this number of H1 epochs
    :param time_shift_tol:
        Width (s) of buckets to collect triggers into, the `friends" of a
        trigger with the same calpha live within the same bucket
    :param score_reduction_max:
        Absolute reduction in SNR^2 from the peak value in each bucket to
        retain (we also have a hardcoded relative reduction)
    :param threshold_chi2:
        Threshold in sum(SNR^2) above which we consider triggers for the
        background list (or signal)
    :param max_time_slide_shift: Max delay allowed for background triggers
    :param minimal_time_slide_jump: Jumps in timeslides
    :param min_veto_chi2:
        Apply vetos to candidates above this SNR^2 in a single detector
    :param max_zero_lag_delay:
        Maximum delay between detectors within the same timeslide
    :param out_fname: Path to npy file to save the candidates to
    :param n_cores: Number of cores to use for splitting the veto computations
    :param run: String identifying the run
    :param opt_format:
        How we choose the finer calpha grid, changed between O1 and O2 analyses
        Exposed here to replicate old runs if needed
    :param outfile_format:
        FLag whether to save the old style (separate npy files for different
        arrays), or in the new style with a consolidated file per job
    :param old_cand_dir_name:
        Directory with old vetoed candidate files, if we want to save on veto
        computations when redoing
    :param bad_times: List of lists of times to avoid in H1 and L1, if known
    :param veto_triggers: Flag to turn the veto on/off
    :param output_timeseries: Flag to output timeseries for the candidates
    :param output_coherent_score:
        Flag to compute the coherent score integral for the candidates
    :param score_reduction_timeseries:
        Restrict triggers in timeseries to the ones with
        single_detector_SNR^2 > (base trigger SNR^2) - this parameter
    :param detectors:
        Tuple with names of the two detectors we will be running coincidence
        with ('H1', 'L1', 'V1' supported)
    :param weaker_detectors:
        If needed, tuple with names of weaker detectors that we will compute
        timeseries for as well ('H1', 'L1', 'V1' supported)
        Note: Only works with outfile_format == "new"
    :param recompute_psd_drift:
        Flag to recompute PSD drift correction. We needed it in O2 since the
        trigger files didn't use safemean. Redundant in O3a and forwards
    :return:
    """
    function_inputs = locals()
    print(f"Running find_interesting_dir with the following arguments:")
    print(", ".join([f"{key}={val}" for key, val in function_inputs.items()]))
    
    # Get list of files and epochs
    detectors = tuple(sorted(detectors))
    files_1, files_2, epochs_1, epochs_2 = utils.get_coincident_json_filelist(
        dir_name, enumerated_epochs=enumerated_epochs, n_epochs=n_epochs,
        run=run, det1=detectors[0], det2=detectors[1])

    # List containing all background trigger pairs that are above the
    # incoherent snr^2 threshold and satisfy the various maximum requirements
    all_candidates = []
    # Masks that identify triggers that successfully passed the vetoes
    all_mask_vetoed = []
    # Metadata for the triggers (CBC_CAT2, CBC_CAT3 and glitch tests)
    all_metadata = []
    # Metadata keys
    metadata_keys = []

    # Optional items
    # SNR timeseries for all triggers
    all_timeseries = []
    # 2-detector coherent scores for all triggers
    all_coherent_scores = []
    # TODO: Initialize a structure to hold the weaker detector's output

    perform_coincidence = True
    look_at_weaker_detectors = not utils.checkempty(weaker_detectors)
    if look_at_weaker_detectors:
        raise NotImplementedError

    # Look at coincident files and generate background + candidates if needed
    # -----------------------------------------------------------------------
    # First check if we're only doing one part
    if ((out_fname is not None) and (outfile_format.lower() == "new") and
            os.path.isfile(out_fname) and (os.path.getsize(out_fname) > 0)):
        # We already analyzed these segments earlier without the weaker detector
        # Read what we saved
        data = np.load(out_fname, allow_pickle=True)

        all_candidates = data["candidates"]
        all_mask_vetoed = data["mask_vetoed"]
        all_metadata = data["metadata"]
        metadata_keys = data["metadata_keys"]

        perform_coincidence = False
        look_at_weaker_detectors = False

        if output_coherent_score:
            if "coherent_scores" in data.files:
                all_coherent_scores = data["coherent_scores"]
            else:
                perform_coincidence = True

        if output_timeseries:
            if "timeseries" in data.files:
                all_timeseries = data["timeseries"]
            else:
                perform_coincidence = True

        if not utils.checkempty(weaker_detectors):
            raise NotImplementedError
            # if "timeseries_weaker_detetectors" in data.files:
            #     # TODO: Read the weaker detector output into the structure
            #     pass
            # else:
            #     look_at_weaker_detectors = True

    if perform_coincidence:
        # We haven't performed coincidence previously
        timeseries_obj = []
        for cur_file1, ep1 in zip(files_1, epochs_1):
            # Load the whitened strain data for the H file, and define some
            # parameters
            trig1 = trig.TriggerList.from_json(cur_file1, load_trigs=False)
            c0_pos = trig1.c0_pos

            if recompute_psd_drift:
                # Recompute the H1 PSD drift correction using the `safemean'
                psd_drift_safemean = trig1.gen_psd_drift_correction(
                    calphas=np.zeros(1), avg='safemean', verbose=False)
                trig1.psd_drift_correction[:] = psd_drift_safemean[:]

            # If needed, we read vetoed and un-vetoed candidates from an old run
            # to save time
            if old_cand_dir_name is not None:
                # Assume you are kind and didn't change format between the runs
                if outfile_format.lower() == "old":
                    # Read in vetoed and non-vetoed candidate files
                    vetoed_cands = np.load(
                        os.path.join(
                            old_cand_dir_name,
                            f"coincident_events__{ep1}_{ep1}.npy"))
                    nonvetoed_cands = np.load(os.path.join(
                        old_cand_dir_name,
                        f"coincident_events__{ep1}_{ep1}before_veto.npy"))
                    old_cand_pars = json.load(open(os.path.join(
                        old_cand_dir_name, "coincidence_params.json"), "r"))
                else:
                    # Read in vetoed and non-vetoed candidate files
                    candfile = np.load(os.path.join(
                        old_cand_dir_name,
                        "coincident_events__" + "_".join(detectors) +
                        "_{ep1}_{ep1}.npz"), allow_pickle=True)
                    old_cand_pars = json.load(open(
                        os.path.join(
                            old_cand_dir_name,
                            "coincidence_parameters_" +
                            "_".join(detectors) + ".json"), "r"))

                    nonvetoed_cands = candfile["candidates"]
                    old_mask_vetoed = candfile["mask_vetoed"]
                    vetoed_cands = nonvetoed_cands[old_mask_vetoed]

                # If there was nothing in the old run, skip this file
                if utils.checkempty(vetoed_cands):
                    continue

                # Read in parameters used to generate the old candidate files
                old_min_veto_chi2 = old_cand_pars["min_veto_chi2"]

                # Define list of bad times, as when there is something loud in
                # only one detector, and only in the unvetoed list
                # ------------------------------------------------------------
                # List of bucket IDs of vetoed triggers in detectors 1 and 2
                bucket_ids_vetoed = \
                    (vetoed_cands[:, :, 0] / time_shift_tol).astype(int)
                # List of bucket IDs of non-vetoed triggers in H and L
                bucket_ids_nonvetoed = \
                    (nonvetoed_cands[:, :, 0] / time_shift_tol).astype(int)

                # Mark buckets with non-vetoed triggers that don't appear in
                # vetoed lists
                absent_1 = np.logical_not(
                    np.in1d(bucket_ids_nonvetoed[:, 0],
                            bucket_ids_vetoed[:, 0]))
                absent_2 = np.logical_not(
                    np.in1d(bucket_ids_nonvetoed[:, 1],
                            bucket_ids_vetoed[:, 1]))

                # Mark times in each detector where we know the veto failed
                if bad_times is None:
                    bad_times = [[], []]
                else:
                    bad_times[0] = list(bad_times[0])
                    bad_times[1] = list(bad_times[1])

                # Bad times in detector 1/2 are those with
                # 1. loud non-vetoed triggers in 1/2 with SNR2 > veto threshold
                # 2. These times don't appear in the list of vetoed triggers
                # 3. corresponding faint triggers in 1/2 with
                #    SNR2 < veto threshold (so failed due to 1/2)
                if np.any(absent_1):
                    bad_times[0] += list(nonvetoed_cands[absent_1][
                        np.logical_and(
                            nonvetoed_cands[absent_1, 0, 1] > old_min_veto_chi2,
                            nonvetoed_cands[absent_1, 1, 1] <= old_min_veto_chi2),
                        0, 0])
                if np.any(absent_2):
                    bad_times[1] += list(nonvetoed_cands[absent_2][
                        np.logical_and(
                            nonvetoed_cands[absent_2, 1, 1] > old_min_veto_chi2,
                            nonvetoed_cands[absent_2, 0, 1] <= old_min_veto_chi2),
                        1, 0])

                print("Finished reading in list of vetoed times", flush=True)
            
            # Look to the right and left since we pull in data from adjacent files
            for ep2 in [ep1,
                        int(ep1 + params.DEF_FILELENGTH),
                        int(ep1 - params.DEF_FILELENGTH)]:

                # If the file doesn't exist, move on to the next one
                if ep2 not in epochs_2:
                    continue

                cur_file2 = files_2[np.searchsorted(epochs_2, ep2)]

                # Load processedclists from the h1 and l1 files
                # do clist1 here since we delete it shortly after to save memory
                try:
                    clist1 = load_trigger_file(cur_file1)
                    clist2 = load_trigger_file(cur_file2)
                except FileNotFoundError as no_file_err:
                    print('NO TRIGGER FILE, catching error:')
                    print(no_file_err)
                    continue

                # Remove bad times if known
                if bad_times is not None:
                    if (len(bad_times[0]) > 0) and (len(clist1) > 0):
                        clist1 = utils.remove_bad_times(
                            bad_times[0], clist1[:, 0], time_shift_tol, clist1)
                    if (len(bad_times[1]) > 0) and (len(clist2) > 0):
                        clist2 = utils.remove_bad_times(
                            bad_times[1], clist2[:, 0], time_shift_tol, clist2)

                # Collect background (+ real signals) for this pair of files,
                # with timeslides enforced
                # Set up parameters for function to compare triggers
                        
                # Load and set the l1 trig object up
                trig2 = trig.TriggerList.from_json(cur_file2, load_trigs=False)

                # score_func = utils.incoherent_score
                # kwargs = {}
                if trig1.normfac >= trig2.normfac:
                    score_func = \
                        trig1.templatebank.marginalized_HM_scores_incl_temp_prior
                else:
                    score_func = \
                        trig2.templatebank.marginalized_HM_scores_incl_temp_prior
                kwargs = {'N_det_effective':2}

                bg_events = collect_background_candidates(
                    clist1, clist2, time_shift_tol, score_reduction_max,
                    threshold_chi2, c0_pos=c0_pos,
                    max_time_slide_shift=max_time_slide_shift,
                    minimal_time_slide_jump=minimal_time_slide_jump,
                    max_zero_lag_delay=max_zero_lag_delay,
                    score_func=score_func, **kwargs)

                # Clear up some memory if that is an issue
                del clist1, clist2

                if recompute_psd_drift:
                    # Recompute the l1 PSD drift correction using the `safemean'
                    psd_drift_safemean = trig2.gen_psd_drift_correction(
                        calphas=np.zeros(1), avg='safemean', verbose=False)
                    trig2.psd_drift_correction[:] = psd_drift_safemean[:]

                # Optimize and veto triggers, if needed
                if old_cand_dir_name is not None:
                    # We're saving on computations by using old vetoed lists
                    fn_out = veto_and_optimize_coincidence_list(
                        bg_events, trig1, trig2, time_shift_tol, threshold_chi2,
                        minimal_time_slide_jump, veto_triggers=False,
                        n_cores=n_cores, opt_format=opt_format,
                        output_timeseries=output_timeseries,
                        output_coherent_score=output_coherent_score,
                        score_reduction_timeseries=score_reduction_timeseries,
                        score_reduction_max=score_reduction_max,
                        detectors=detectors, score_func=score_func,
                        **kwargs)
                else:
                    # We're applying vetoes ourself, if needed
                    fn_out = veto_and_optimize_coincidence_list(
                        bg_events, trig1, trig2, time_shift_tol, threshold_chi2,
                        minimal_time_slide_jump, veto_triggers=veto_triggers,
                        min_veto_chi2=min_veto_chi2, n_cores=n_cores,
                        opt_format=opt_format,
                        output_timeseries=output_timeseries,
                        output_coherent_score=output_coherent_score,
                        score_reduction_timeseries=score_reduction_timeseries,
                        score_reduction_max=score_reduction_max,
                        detectors=detectors, score_func=score_func,
                        **kwargs)

                # Clear up some memory
                del trig2

                # Metadata keys are always defined
                metadata_keys = fn_out[-1]
                fn_out = fn_out[:-1]

                bg_events = fn_out[0]
                if utils.checkempty(bg_events):
                    continue

                # Things that are always defined
                if bad_times is not None:
                    # Optimization can cause things to spread out of the
                    # interval, so again remove bad times
                    fn_out = utils.remove_bad_times(
                        bad_times[0], bg_events[:, 0, 0], time_shift_tol,
                        *fn_out)

                    bg_events = fn_out[0]
                    if utils.checkempty(bg_events):
                        continue

                    fn_out = utils.remove_bad_times(
                        bad_times[1], bg_events[:, 1, 0], time_shift_tol,
                        *fn_out)

                    bg_events = fn_out[0]
                    if utils.checkempty(bg_events):
                        continue

                # Other quantities that are always defined
                mask_vetoed = fn_out[1]
                metadata_arr = fn_out[2]

                # Append to list of background and significant triggers
                all_candidates.append(bg_events)
                all_mask_vetoed.append(mask_vetoed)
                all_metadata.append(metadata_arr)

                if output_timeseries or output_coherent_score:
                    if output_timeseries:
                        timeseries = fn_out[3]
                        timeseries_obj.append(timeseries)
                        if output_coherent_score:
                            coherent_scores = fn_out[4]
                            all_coherent_scores.append(coherent_scores)
                    else:
                        coherent_scores = fn_out[3]
                        all_coherent_scores.append(coherent_scores)

        if utils.checkempty(all_candidates):
            # Nothing in these files
            return all_candidates

        all_candidates = np.concatenate(all_candidates, axis=0)
        all_mask_vetoed = np.concatenate(all_mask_vetoed, axis=0)
        all_metadata = np.concatenate(all_metadata, axis=0)

        if not utils.checkempty(timeseries_obj):
            # Do it this way to avoid a subtle bug in np.array when the
            # dimensions of the candidates in the two detectors is equal
            # all_timeseries = np.concatenate(timeseries_obj, axis=0)
            timeseries_obj = sum(timeseries_obj, [])
            all_timeseries = np.zeros((len(timeseries_obj), 2), dtype=object)
            for ind in range(len(all_timeseries)):
                all_timeseries[ind, 0] = timeseries_obj[ind][0]
                all_timeseries[ind, 1] = timeseries_obj[ind][1]

        if not utils.checkempty(all_coherent_scores):
            all_coherent_scores = np.concatenate(all_coherent_scores, axis=0)

    if look_at_weaker_detectors:
        raise NotImplementedError
        # Save timeseries results in the weaker detector(s) for fishing

        # if utils.checkempty(all_candidates):
        #     # Nothing in these files
        #     return all_candidates
        #
        # # TODO: Do something here to incorporate weaker detectors
        # # Epochs to look in the weaker detectors for each candidate
        # epochs_weaker = [
        #     [int(ep - params.DEF_FILELENGTH),
        #      ep,
        #      int(ep + params.DEF_FILELENGTH)] for ep in epochs_1]
        # epochs_weaker = sorted(list(set(sum(epochs_weaker, []))))
        #
        # # n_candidates x n_detectors array that will hold timeseries in the
        # # weaker detectors
        # weaker_candidates = np.array(
        #     [[None] * len(weaker_detectors)] * len(all_candidates))
        #
        # # # Sort the candidates in order of gps time in that detector
        # # isort = np.argsort(all_candidates[:, 0, 0])
        # # times_1 = all_candidates[isort, 0, 0]
        #
        # # Go through each epoch in each weajer detector, create the
        # # timeseries for all applicable candidates, and overwrite if
        # # it doesn't exist/has a higher SNR
        # for epoch in epochs_weaker:
        #     for detector in weaker_detectors:
        #         config_fname = utils.get_json_fname(
        #             dir_name, epoch, detector, run=run)
        #
        #         if not os.path.isfile(config_fname):
        #             # Either the detector was off or we failed to make a trigger file
        #             continue
        #
        #         trig_obj_weaker = trig.TriggerList.from_json(
        #             config_fname, load_trigs=False)
        #
        #         if recompute_psd_drift:
        #             # Recompute the H1 PSD drift correction using the `safemean'
        #             psd_drift_safemean = \
        #                 trig_obj_weaker.gen_psd_drift_correction(
        #                     calphas=np.zeros(1), avg='safemean', verbose=False)
        #             trig_obj_weaker.psd_drift_correction[:] = \
        #                 psd_drift_safemean[:]
        #
        #         # Define intervals around trigger time that contain the waveform
        #         bank = trig_obj_weaker.templatebank
        #         dt_right = - bank.shift_whitened_wf * bank.dt
        #         dt_left = (bank.support_whitened_wf +
        #                    bank.shift_whitened_wf) * bank.dt
        #
        #         for ind, candidate in enumerate(all_candidates):
        #             # We will use the first detector as the reference detector
        #             # See if we have the data required to compute any of the triggers
        #             left_lim = candidate[0, 0] - DT_BOUND_TIMESERIES - dt_left
        #             right_lim = candidate[0, 0] + DT_BOUND_TIMESERIES + dt_right
        #
        #             trigs_calpha = trig_obj_weaker.gen_triggers_local(
        #                 trigger=candidate[0], dt_left=DT_BOUND_TIMESERIES,
        #                 dt_right=DT_BOUND_TIMESERIES,
        #                 compute_calphas=[candidate[0, trig_obj_weaker.c0_pos:]],
        #                 apply_threshold=False, relative_binning=False,
        #                 zero_pad=False)
        #             # TODO: What do we set as the `zero - lag trigger to pass to the coherent score'?

    if out_fname is not None:
        # I didn't bother making the old format compatible with saving the metadata
        # and coherent scores, from now on we won't use it
        if outfile_format.lower() == "old":
            significant_triggers = all_candidates[all_mask_vetoed]
            np.save(out_fname, significant_triggers)
            os.system(f"chmod 777 {out_fname}")

            if old_cand_dir_name is None:
                # Save even non-vetoed background
                all_events_fname = out_fname.split(".npy")[0] + "before_veto.npy"
                np.save(all_events_fname, all_candidates)
                os.system(f"chmod 777 {all_events_fname}")

            if output_timeseries:
                # Save timeseries
                timeseries_fname = out_fname.split(".npy")[0] + \
                    "timeseries.npy"
                np.save(timeseries_fname, all_timeseries)
                os.system(f"chmod 777 {timeseries_fname}")

        else:
            # TODO: Save the weaker detector

            save_dic = dict(
                candidates=all_candidates,
                mask_vetoed=all_mask_vetoed,
                metadata=all_metadata,
                metadata_keys=metadata_keys,
                detectors=detectors)
            if output_timeseries:
                save_dic["timeseries"] = all_timeseries
            if output_coherent_score:
                save_dic["coherent_scores"] = all_coherent_scores
            np.savez(out_fname, **save_dic)

    return all_candidates


def fish_in_weaker_detector(
        trig_file_h1, trig_file_l1, trigger=None, time_lf=None, det_ind=1,
        max_time_slide_shift=100, time_shift_tol=0.01, score_reduction_max=5,
        minimal_time_slide_jump=0.1, score_func=utils.incoherent_score,
        verbose=False):
    """
    Warning: This function has not been updated by Jay for HM case
    :param trig_file_h1: Trigger file with H1 triggers
    :param trig_file_l1: Trigger file with L1 triggers
    :param trigger:
        Row of processedclist with strong trigger, only used to fix the time
    :param time_lf: Linear-free time, used if trigger wasn't given
    :param det_ind: Index of strong detector (0 for H1, 1 for L1)
    :param max_time_slide_shift: Max delay allowed for background triggers
    :param time_shift_tol: Maximum physical delay for zero-lag triggers
    :param score_reduction_max:
        Allow a reduction in SNR^2 of max * params.MAX_FRIEND_DEGRADE_SNR2 +
        score_reduction_max when keeping friends
    :param minimal_time_slide_jump: Jumps in timeslides
    :param score_func:
        Function to take two triggers and return the coherent score
    :param verbose: Flag to print progress
    :return: Background and zero-lag events as in coincidence, without cut
    """
    if det_ind == 1:
        trig_file_strong = trig_file_l1
        trig_file_weak = trig_file_h1
    else:
        trig_file_strong = trig_file_h1
        trig_file_weak = trig_file_l1

    # Do not operate with the trigger itself, since it depends on
    # subbank and candidate collection, and has already been optimized
    # Pick triggers near the trigger on the original calpha grid
    if trigger is not None:
        time_lf = trigger[0]
    elif time_lf is None:
        raise RuntimeError("I need a location to look in!")

    trig_file_strong.filter_triggers(
        filters={
            'time': (time_lf - trig_file_strong.t0 - time_shift_tol,
                     time_lf - trig_file_strong.t0 + time_shift_tol)})

    if len(trig_file_strong.filteredclist) == 0:
        # We made a hole in this subbank, return nothing
        bg_events = np.zeros((0, 2, len(trig_file_strong.processedclist[0])))
        return bg_events

    # Place best trigger in the middle of a group to avoid edge effects
    best_time = trig_file_strong.filteredclist[
        np.argmax(trig_file_strong.filteredclist[:, 1]), 0]
    gp_origin = best_time - (minimal_time_slide_jump / 2)

    # Keep friends of the best local triggers, similar to coincidence
    friends_arr = get_friends_arr(
        trig_file_strong.filteredclist, time_shift_tol, score_reduction_max,
        origin=gp_origin)
    strong_trigs = np.vstack(friends_arr)
    calphas = strong_trigs[:, trig_file_strong.c0_pos:]

    if verbose:
        print(f"Generating triggers in weaker detector", flush=True)

    # Generate triggers in the weaker detector without cuts on snr^2
    # Relative binning becomes unreliable at large time shifts
    best_strong_trig = strong_trigs[np.argmax(strong_trigs[:, 1])]
    weak_trigs = trig_file_weak.gen_triggers_local(
        trigger=best_strong_trig, dt_left=max_time_slide_shift,
        dt_right=max_time_slide_shift, compute_calphas=calphas,
        apply_threshold=False, relative_binning=False)

    if verbose:
        print(f"Collecting coincident and background triggers " +
              "in weaker detector", flush=True)

    # Now apply coincidence logic as earlier to collect bg + zero-lag,
    # with incoherent cut arranged to not discard any fished triggers
    threshold_chi2 = np.min(strong_trigs[:, 1])
    c0_pos = trig_file_strong.c0_pos
    kwargs = {}

    # Apply timeslides
    bg_events = collect_background_candidates(
        strong_trigs, weak_trigs, time_shift_tol,
        score_reduction_max, threshold_chi2, c0_pos=c0_pos,
        max_time_slide_shift=max_time_slide_shift,
        minimal_time_slide_jump=minimal_time_slide_jump,
        origin=gp_origin, score_func=score_func, **kwargs)

    # Veto and optimize
    bg_events_all, mask_vetoed, *_ = veto_and_optimize_coincidence_list(
        bg_events, trig_file_strong, trig_file_weak, time_shift_tol,
        threshold_chi2, minimal_time_slide_jump,  apply_threshold=False,
        score_reduction_max=score_reduction_max,
        origin=gp_origin, score_func=score_func, **kwargs)
    bg_events = bg_events_all[mask_vetoed]

    if verbose:
        print(f"{len(bg_events)} coincident and background triggers survive",
              flush=True)

    # Swap order to default if needed
    if det_ind == 1 and len(bg_events) > 0:
        bg_events = np.stack([bg_events[:, 1, :], bg_events[:, 0, :]], axis=1)

    return bg_events


# Functions to collect candidates
# -------------------------------
def collect_background_candidates(
        clist1, clist2, time_shift_tol, score_reduction_max, threshold_chi2,
        c0_pos, restricted_times=None, max_time_slide_shift=None,
        minimal_time_slide_jump=None, max_zero_lag_delay=None, origin=0,
        score_func=utils.incoherent_score, **kwargs):
    """
    Function to collect background candidates (including real events)
    :param clist1: Processedclist 1
    :param clist2: Processedclist 2
    :param time_shift_tol:
        Width (s) of buckets to collect triggers into, the `friends" of a
        trigger with the same calpha live within the same bucket
    :param score_reduction_max:
        Absolute reduction in SNR^2 from the peak value in each bucket to
        retain (we also have a hardcoded relative reduction)
    :param threshold_chi2:
        Threshold in sum(SNR^2) above which we consider triggers for the
        background list (or signal)
    :param c0_pos: Index of c0 in the processedclists
    :param restricted_times:
        If needed, list of times to restrict analysis to save time during
        injection campaign
    :param max_time_slide_shift: Max delay allowed for background triggers
    :param minimal_time_slide_jump: Jumps in timeslides
    :param max_zero_lag_delay:
        Maximum delay between detectors within the same timeslide.
        If not given, it defaults to time_shift_tol
    :param origin: Origin for splitting the times relative to
    :param score_func:
        Function that accepts coincident trigger(s) and returns score(s)
    :param kwargs: Dictionary with extra parameters for score_func
    :return: n_candidate x 2 x len(Processedclist[0]) array with candidates
    """
    if utils.checkempty(clist1) or utils.checkempty(clist2):
        return np.array([])

    if restricted_times is not None:
        clist1 = \
            clist1[utils.is_close_to(restricted_times, clist1[:, 0], eps=1)]
        clist2 = \
            clist2[utils.is_close_to(restricted_times, clist2[:, 0], eps=1)]

    if utils.checkempty(clist1) or utils.checkempty(clist2):
        return np.array([])

    # Group triggers into buckets every time_shift_tol seconds, retaining only
    # those that are closest to the highest in their bucket
    friends_arr_list_1 = get_friends_arr(
        clist1, time_shift_tol, score_reduction_max, origin=origin)
    friends_arr_list_2 = get_friends_arr(
        clist2, time_shift_tol, score_reduction_max, origin=origin)

    # Set of pairs of indices of buckets with background triggers that
    # 1) Are close to the highest in their own bucket (over all calphas)
    # 2) Have the same calphas
    # 3) have sum of SNR^2 > bound_chi2
    # Note that this is a set because multiple calphas can fire in a coincident
    # manner in the same pair of buckets, and we want to avoid repeating the
    # same buckets
    relevant_bucket_pairs = set([])

    # Dictionary that records a list of
    # (bucket, best score in bucket, tmin in bucket, tmax in bucket) for
    # each calpha
    template_id_to_segment_id_score_dic = {}
    for bucket_id, friends_list in enumerate(friends_arr_list_1):
        # Group templates inside this bucket by template calphas
        unique_templates, friends_split_by_template = group_by_id(
            friends_list, c0_pos)

        # Go over all templates that jumped, and add information to dictionary
        # indexed by template id (hash of the calphas)
        for temp_id, trigger_sublist in \
                zip(unique_templates, friends_split_by_template):
            # Get previous list if it exists in the dictionary
            jumplist = template_id_to_segment_id_score_dic.get(temp_id, [])

            # Compute best score in bucket, bounds on time, and append to list
            best_template_score = np.max(trigger_sublist[:, 1])
            min_time_1, max_time_1 = np.min(trigger_sublist[:, 0]), np.max(trigger_sublist[:, 0])
            jumplist.append((bucket_id, best_template_score, min_time_1, max_time_1))

            # Update dictionary
            template_id_to_segment_id_score_dic[temp_id] = jumplist

    print("Created dictionary indexed by calphas", flush=True)

    # Now go over buckets in the second detector, look where triggers jumped,
    # and use the above dictionary to identify relevant times in the first
    # detector
    for bucket_id, friends_list in enumerate(friends_arr_list_2):
        # Group templates by template calphas as earlier
        unique_templates, friends_split_by_template = group_by_id(
            friends_list, c0_pos)

        # Go over templates that jumped, and look in the above dictionary
        for temp_id, trigger_sublist in \
                zip(unique_templates, friends_split_by_template):
            # Compute best score in bucket, and bounds on time
            best_template_score = np.max(trigger_sublist[:, 1])
            min_time_2, max_time_2 = np.min(trigger_sublist[:, 0]), np.max(trigger_sublist[:, 0])

            # Find all the relevant buckets in the other detector in which
            # trigger(s) with the same calpha was(were) retained
            other_times = template_id_to_segment_id_score_dic.get(temp_id, [])
            if (len(other_times) > 0) and \
                    ((best_template_score +
                      np.max([x[1] for x in other_times])) > threshold_chi2):
                for first_bucket_id, first_detector_score, min_time_1, max_time_1 in other_times:
                    if ((best_template_score + first_detector_score) >
                            threshold_chi2):
                        if max_time_slide_shift is not None:
                            # Check ahead of time to save computations
                            if min(np.abs(min_time_2 - max_time_1),
                                   np.abs(max_time_2 - min_time_1)) > max_time_slide_shift:
                                continue
                        relevant_bucket_pairs.add((first_bucket_id, bucket_id))

    # List of pairs of triggers that are in the relevant buckets, and are
    # the highest coincident pair in their buckets (note this returns only
    # one candidate per pair of buckets)
    background_candidates = []
    for first_bucket_id, second_bucket_id in relevant_bucket_pairs:
        friends_arr_1 = friends_arr_list_1[first_bucket_id]
        friends_arr_2 = friends_arr_list_2[second_bucket_id]
        # if max_time_slide_shift is not None:
        #     # Check ahead of time to save computations
        #     min_time_1 = np.min(friends_arr_1[:, 0])
        #     max_time_1 = np.max(friends_arr_1[:, 0])
        #     min_time_2 = np.min(friends_arr_2[:, 0])
        #     max_time_2 = np.max(friends_arr_2[:, 0])
        #     if min(np.abs(min_time_2 - max_time_1),
        #            np.abs(max_time_2 - min_time_1)) > max_time_slide_shift:
        #         continue
        background_candidates.append(
            get_best_candidate_segments(
                friends_arr_1, friends_arr_2, c0_pos,
                score_func=score_func, **kwargs))

    background_candidates = np.asarray(background_candidates)

    # Create background and zero-lag with proper selection
    if (len(background_candidates) > 0) and \
            (minimal_time_slide_jump is not None):
        if max_zero_lag_delay is None:
            max_zero_lag_delay = time_shift_tol

        background_candidates = create_shifted_observations(
            background_candidates, max_time_slide_shift,
            minimal_time_slide_jump, max_zero_lag_delay)

        # Maximize per minimal_time_slide_jump pair
        background_candidates = flag_duplicates_per_group_pair(
            background_candidates, minimal_time_slide_jump, c0_pos,
            origin=origin, remove=True, score_func=score_func, **kwargs)

    print("Identified relevant timeslides", flush=True)

    return background_candidates


def veto_and_optimize_coincidence_list(
        bg_events, trig1, trig2, time_shift_tol, threshold_chi2,
        minimal_time_slide_jump, veto_triggers=True, min_veto_chi2=32,
        apply_threshold=True, origin=0, n_cores=1, opt_format="new",
        output_timeseries=True, output_coherent_score=True,
        score_reduction_timeseries=10, score_reduction_max=5,
        detectors=('H1', 'L1'), score_func=utils.incoherent_score, **kwargs):
    """
    Returns list of vetoed and optimized coincident candidates (w/ timeslides)
    and any extra information
    :param bg_events:
        n_candidate x 2 x len(Processedclist[0]) array with candidates
    :param trig1: Trigger object 1
    :param trig2: Trigger object 2
    :param time_shift_tol:
        Width (s) of buckets to collect triggers into, the `friends" of a
        trigger with the same calpha live within the same bucket
    :param threshold_chi2:
        Threshold in sum(SNR^2) above which we consider triggers for the
        background list (or signal)
    :param minimal_time_slide_jump: Jumps in timeslides
    :param veto_triggers: Flag to veto triggers
    :param min_veto_chi2:
        Apply vetos to candidates above this SNR^2 in a single detector
    :param apply_threshold:
        Flag to apply threshold on single-detector chi2 after optimizing
    :param origin: Origin to split the trigger times relative to
    :param n_cores: Number of cores to use for splitting the veto computations
    :param opt_format:
        How we choose the finer calpha grid, changed between O1 and O2 analyses
        Exposed here to replicate old runs if needed
    :param output_timeseries: Flag to output timeseries for the candidates
    :param output_coherent_score:
        Flag to compute the coherent score integral for the candidates
    :param score_reduction_timeseries:
        Restrict triggers in timeseries to the ones with
        single_detector_SNR^2 > (base trigger SNR^2) - this parameter
    :param score_reduction_max:
        Absolute reduction in SNR^2 from the peak value to be allowed
        for secondary peak in the function secondary_peak_reject()
    :param detectors:
        Tuple with names of the two detectors we will be running coincidence
        with
    :param score_func:
        Function to use to decide on the representative trigger
        (once we use the coherent score integral, this choice becomes
        unimportant)
    :param kwargs: Extra arguments to score_func, if needed
    :return:
        1. n_candidates x 2 x len(processedclist[0]) array with optimized
           coincident triggers
        2. Mask into coincident triggers that identifies triggers that survived
        3. Boolean array of shape n_candidate x 2 x
            (len(self.outlier_reasons) + 11 + 2 * len(split_chunks))
            with metadata. Indices represent
            0: CBC_CAT2 flag ("and" of the values for the cloud)
            1: CBC_CAT3 flag ("and" of the values for the cloud)
            The 2:len(self.outlier_reasons) + 10 + 2*len(split_chunks) elements
            have zeros marking glitch tests that fired
            2: len(self.outlier_reasons) + 2: index into outlier reasons
                for excess-power-like tests
            len(self.outlier_reasons) + 2: Finer PSD drift killed it
            len(self.outlier_reasons) + 3: No chunks present for phase tests
            len(self.outlier_reasons) + 4: Overall chi-2 test
            len(self.outlier_reasons) + 5:
                len(self.outlier_reasons) + 5 + len(split_chunks): Split tests
            len(outlier_reasons) + 5 + len(split_chunks):
                Finer sinc-interpolation
            len(outlier_reasons) + 6 + len(split_chunks):
                No chunks present for stringent phase test
            len(outlier_reasons) + 7 + len(split_chunks): Stringent chi-2 test
            len(outlier_reasons) + 8 + len(split_chunks):
                len(outlier_reasons) + 8 + 2*len(split_chunks):
                    Stringent split tests
            len(outlier_reasons) + 8 + 2*len(split_chunks):
                Not enough chunks present for chi2 test with higher nchunk
            len(outlier_reasons) + 9 + 2*len(split_chunks):
                chi2 test with higher nchunk
            len(outlier_reasons) + 10 + 2*len(split_chunks):
                Found another louder trigger in the same time-shift-tol window
        4. If output_timeseries, list of 2-tuples with H1 and L1 timeseries for 
           each candidate
        5. If output_coherent_score, array with coherent scores for each
            candidiate
        6. Text keys for Boolean array for quickly reading off which test failed
    """
    # Create key to interpret the metadata
    npower = len(trig1.outlier_reasons)
    nsplit = len(params.SPLIT_CHUNKS)
    metadata_keys = np.zeros(npower + 11 + 2 * nsplit, dtype='<U64')
    metadata_keys[0] = "CBC_CAT2"
    metadata_keys[1] = "CBC_CAT3"
    metadata_keys[2:2 + npower] = trig1.outlier_reasons[:]
    metadata_keys[2 + npower] = "Finer PSD drift"
    metadata_keys[3 + npower] = f"No chunks for phase tests with {params.N_CHUNK} chunks"
    metadata_keys[4 + npower] = f"chi2 test with {params.N_CHUNK} chunks"
    for ind_chunk, chunk_split in enumerate(params.SPLIT_CHUNKS):
        metadata_keys[5 + npower + ind_chunk] = \
            f"Split test with chunk sets {chunk_split[0]} and {chunk_split[1]}"
    metadata_keys[5 + npower + nsplit] = "Finer sinc-interpolation"
    metadata_keys[6 + npower + nsplit] = \
        f"No chunks for stringent phase tests with {params.N_CHUNK} chunks"
    metadata_keys[7 + npower + nsplit] = \
        f"Stringent chi2 test with {params.N_CHUNK} chunks"
    for ind_chunk, chunk_split in enumerate(params.SPLIT_CHUNKS):
        metadata_keys[8 + npower + nsplit + ind_chunk] = \
            f"Stringent split test with chunk sets {chunk_split[0]} and {chunk_split[1]}"
    metadata_keys[8 + npower + 2 * nsplit] = \
        f"No chunks for chi2 test with {params.N_CHUNK_2} chunks"
    metadata_keys[9 + npower + 2 * nsplit] = \
        f"chi2 test with {params.N_CHUNK_2} chunks"
    metadata_keys[10 + npower + 2 * nsplit] = \
        f"Louder trigger within the {time_shift_tol} s bucket"
    if output_timeseries or output_coherent_score:
        metadata_keys = np.append(
            metadata_keys, 'Secondary_peak_timeseries')

    if utils.checkempty(bg_events):
        mask_vetoed = np.ones(0, dtype=bool)
        timeseries = []
        coherent_scores = np.zeros(0)
        if output_timeseries or output_coherent_score:
            metadata_arr = np.ones((0, 2, npower + 11 + 2 * nsplit+1), dtype=bool)
            if not output_coherent_score:
                return bg_events, mask_vetoed, metadata_arr, timeseries, \
                    metadata_keys
            elif not output_timeseries:
                return bg_events, mask_vetoed, metadata_arr, coherent_scores, \
                    metadata_keys
            else:
                return bg_events, mask_vetoed, metadata_arr, timeseries, \
                    coherent_scores, metadata_keys
        else:
            metadata_arr = np.ones((0, 2, npower + 11 + 2 * nsplit), dtype=bool)
            return bg_events, mask_vetoed, metadata_arr, metadata_keys

    print(f"Going to veto and optimize {len(bg_events)} triggers", flush=True)

    # Optimize and veto the background candidates
    # -------------------------------------------
    # First veto and optimize triggers in H1
    veto_dict_H1 = veto_and_optimize_single_detector(
        bg_events[:, 0, :], trig1, time_shift_tol,
        group_duration=minimal_time_slide_jump, veto_triggers=veto_triggers,
        min_veto_chi2=min_veto_chi2, apply_threshold=apply_threshold,
        origin=origin, n_cores=n_cores, opt_format=opt_format)

    print(f"Vetoed detector 1, going to veto detector 2", flush=True)

    # Then veto and optimize triggers in L1
    veto_dict_L1 = veto_and_optimize_single_detector(
        bg_events[:, 1, :], trig2, time_shift_tol,
        group_duration=minimal_time_slide_jump, veto_triggers=veto_triggers,
        min_veto_chi2=min_veto_chi2, apply_threshold=apply_threshold,
        origin=origin, n_cores=n_cores, opt_format=opt_format)

    print("Vetoed detector 2, going to pick coincident " +
          "optimized candidates", flush=True)

    # Pick best triggers from two clouds, also applies finer PSD drift veto
    bg_events_all, mask_vetoed, metadata_arr = \
        select_optimal_trigger_pairs(
            bg_events, [veto_dict_H1, veto_dict_L1], time_shift_tol,
            threshold_chi2, trig1, origin=origin, score_func=score_func,
            **kwargs)

    print(f"Picked {len(bg_events_all)} coincident optimized candidates",
          flush=True)

    # Clear up some memory if that is an issue
    del veto_dict_H1, veto_dict_L1

    # Repeated candidates are possible due to edge effects of the optimization
    # (looking out of the bucket), flag such repeats, prefering to pick vetoed
    # triggers
    bg_events_all, mask_vetoed, metadata_arr, mask_retain = \
        flag_duplicates_per_group_pair(
            bg_events_all, minimal_time_slide_jump, trig1.c0_pos, origin=origin,
            veto_mask=mask_vetoed, extra_arrays=[metadata_arr],
            remove=False, score_func=score_func, **kwargs)
    metadata_arr = metadata_arr[0]
    mask_vetoed *= np.all(mask_retain, axis=-1)

    # Add entries for the stringent veto in the metadata array and the
    # "adjacent loud trigger" veto
    metadata_arr = np.pad(
        metadata_arr, ((0, 0), (0, 0), (0, 5 + nsplit)),
        mode="constant", constant_values=True)
    # Update the results of the "adjacent loud trigger" veto
    metadata_arr[:, :, -1] = mask_retain[:]

    if veto_triggers and np.any(mask_vetoed):
        # Changing the code to compute the stringent vetoes for all candidates
        # to maintain consistency in case we retroactively remove one of the
        # previous tests

        # Apply stringent vetoes for the edge case in which the best
        # single-detector triggers in the buckets survive, but the
        # coincident ones do not
        print(f"Applying stringent veto to " +
              f"{len(bg_events_all)} candidates", flush=True)

        # Veto Hanford
        trigger_fates_H1, metadata_stringent_H1 = stringent_veto(
            bg_events_all, 0, trig1, min_veto_snr2=min_veto_chi2,
            group_duration=minimal_time_slide_jump, origin=origin,
            n_cores=n_cores)

        # Add the stringent veto to the metadata
        metadata_arr[:, 0, -(5 + nsplit):-1] = metadata_stringent_H1[:]
        # mask_vetoed[mask_vetoed] = trigger_fates_H1[:]

        # Veto Livingston
        trigger_fates_L1, metadata_stringent_L1 = stringent_veto(
            bg_events_all, 1, trig2, min_veto_snr2=min_veto_chi2,
            group_duration=minimal_time_slide_jump, origin=origin,
            n_cores=n_cores)

        # Add the stringent veto to the metadata
        metadata_arr[:, 1, -(5 + nsplit):-1] = metadata_stringent_L1[:]

        # Apply all masks
        mask_vetoed *= np.logical_and(trigger_fates_H1, trigger_fates_L1)

        print(f"Saving {len(bg_events_all)} and {np.sum(mask_vetoed)} " +
              "non-vetoed and vetoed candidates", flush=True)

    if output_timeseries or output_coherent_score:
        print(f"Computing SNR time series for {len(bg_events_all)} candidates",
              flush=True)

        if output_coherent_score:
            # Initializing the coherent score here below applying it below
            print(f"Also computing coherent score integral for {len(bg_events_all)}" +
                  " candidates", flush=True)
            
            import coherent_score_hm_search as cs 
            cs_instance = cs.initialize_cs_instance(trig1, trig2, detectors=detectors)
                  
            coherent_scores, timeseries = cs.compute_coherent_scores(
                            cs_instance, bg_events_all, trig1, trig2,
                           minimal_time_slide_jump=minimal_time_slide_jump,
                            score_reduction_timeseries=score_reduction_timeseries,
                            output_timeseries=output_timeseries,
                            output_coherent_score=output_coherent_score)
            print(f"Computed coherent score integral and SNR timeseries\
                   for {len(bg_events_all)} candidates", flush=True)
        
        else:
            # following code is copy pasted from cs.compute_coherent_scores()
            timeseries = []
            for ind, (trig_h1, trig_l1) in enumerate(bg_events_all):
                tdiff = (trig_l1[0] - trig_h1[0]) % minimal_time_slide_jump
                if tdiff > (minimal_time_slide_jump/2):
                    tdiff -= minimal_time_slide_jump
                tdiff_l1 = tdiff / (1 + np.exp((trig_l1[1]-trig_h1[1])/2))
                tdiff_l1 = np.round((tdiff_l1 - (tdiff/2))/trig1.dt)*trig1.dt\
                                        + (tdiff/2)
                tdiff_h1 = tdiff - tdiff_l1
                tdiff_l1 = np.round(tdiff_l1, 13)
                tdiff_h1 = np.round(tdiff_h1, 13)
                if (trig1.dt != trig2.dt):
                    raise NotImplementedError('Currently only works when dt of'+\
                                                'both detectors are the same')
                
                # Compute SNR time series near the peak
                trigs_calpha_h1= trig1.gen_triggers_local(
                    trigger=trig_h1,
                    dt_left= params.DT_BOUND_TIMESERIES - tdiff_h1,
                    dt_right= params.DT_BOUND_TIMESERIES + tdiff_h1,
                    compute_calphas=[trig_h1[trig1.c0_pos:]],
                    apply_threshold=False, relative_binning=False,
                    zero_pad=False, orthogonalize_modes=False,
                    return_mode_covariance=False)
                trigs_calpha_l1 = trig2.gen_triggers_local(
                    trigger=trig_l1,
                    dt_left= params.DT_BOUND_TIMESERIES + tdiff_l1,
                    dt_right= params.DT_BOUND_TIMESERIES - tdiff_l1,
                    compute_calphas=[trig_l1[trig2.c0_pos:]],
                    apply_threshold=False, relative_binning=False,
                    zero_pad=False, orthogonalize_modes=False,
                    return_mode_covariance=False)
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

            print(f"Computed only SNR time series for {len(bg_events_all)} candidates",
                flush=True)
        
        veto_timeseries = np.array([
            [[not (secondary_peak_reject(
                bg_events_all[i][0], timeseries[i][0], 
                score_reduction_max=score_reduction_max))],
            [not (secondary_peak_reject(
                bg_events_all[i][1], timeseries[i][1], 
                score_reduction_max=score_reduction_max))]]
            for i in range(len(bg_events_all))])
        metadata_arr = np.append(metadata_arr, veto_timeseries, axis=-1)
        mask_vetoed = mask_vetoed * veto_timeseries[:,0,0] * veto_timeseries[:,1,0]

        if output_timeseries and output_coherent_score:
            return bg_events_all, mask_vetoed, metadata_arr, timeseries, \
                    np.array(coherent_scores), metadata_keys
        elif not output_timeseries:
            return bg_events_all, mask_vetoed, metadata_arr, \
                    np.array(coherent_scores), metadata_keys
        else:
            return bg_events_all, mask_vetoed, metadata_arr, timeseries, \
                metadata_keys

    return bg_events_all, mask_vetoed, metadata_arr, metadata_keys


def get_friends_arr(clist, time_tol, score_reduction_max, origin=0):
    """Groups triggers in clist into buckets every time_tol, and retains only
    those high enough relative to the maximum in each bucket
    :param clist: Processedclist
    :param time_tol: Width of each bucket in time (s)
    :param score_reduction_max:
        Allow a reduction in SNR^2 of max * params.MAX_FRIEND_DEGRADE_SNR2 +
        score_reduction_max when keeping friends
    :param origin: Origin for splitting the times relative to
    :return: List of processedclists of friends, one for each bucket
    """
    # Split according to trigger time
    split_triggers = utils.splitarray(
        clist, clist[:, 0], time_tol, axis=0, origin=origin)

    # Find maximum score in each bucket, and keep triggers within
    # score_reduction_max, allowing for loudness
    segment_max_arr = np.array([np.max(a[:, 1]) for a in split_triggers])
    score_reduction_arr = segment_max_arr * params.MAX_FRIEND_DEGRADE_SNR2 + \
        score_reduction_max
    friends_arrs = [a[a[:, 1] > (smax - sred)] for a, smax, sred in
                    zip(split_triggers, segment_max_arr, score_reduction_arr)]

    return friends_arrs


def group_by_id(clist, c0_pos, ncalpha=None):
    """
    Groups the triggers in a clist into sublists with common calphas
    :param clist: Processedclist
    :param c0_pos: Index of c0
    :param ncalpha:
        Use only up to ncalpha coefficients if needed, useful for making
        heatmaps
    :return: 1. List of unique hashes (template ids) of templates in clist
             2. List of sublists of the clist with each sublist having triggers
                with a common template id
    """
    # Compute hashes that index the calphas
    if ncalpha is not None:
        template_ids = np.asarray(
            utils.make_template_ids(clist[:, c0_pos:c0_pos + ncalpha]),
            dtype=np.int64)
    else:
        template_ids = np.asarray(
            utils.make_template_ids(clist[:, c0_pos:]), dtype=np.int64)

    return utils.splitarray(
        clist, template_ids, 1, axis=0, return_split_keys=True)


def get_best_candidate_segments(
        friends_arr1, friends_arr2, c0_pos, score_func=utils.incoherent_score,
        **kwargs):
    """Returns best trigger pair given two lists of friends. Doesn't impose a
    time-delay constraint, assumes that time-delay is either irrelevant (bg),
    or taken care of by using the appropriate time_shift_tol
    :param friends_arr1: processedclist of triggers
    :param friends_arr2: processedclist of triggers
    :param c0_pos: Index of c0 in processedclists
    :param score_func:
        Function that accepts coincident trigger(s) and returns score(s)
        (in the future maybe split into incoherent and coherent score funcs)
    :param kwargs: Dictionary with extra parameters for score_func
    :return: tuple of len(2) with H1 trigger and L1 trigger"""

    # Check if None, will create error otherwise
    if utils.checkempty(friends_arr1) or utils.checkempty(friends_arr2):
        return None, None

    # First group both sets of triggers by template ids (i.e., calphas)
    template_ids1, triggers1 = group_by_id(friends_arr1, c0_pos)
    template_ids2, triggers2 = group_by_id(friends_arr2, c0_pos)
    template_ids2_list = list(template_ids2)

    # List of indices of common template ids
    common_temp_id_indices = [
        (id_index_1, template_ids2_list.index(temp_id)) for
        id_index_1, temp_id in enumerate(template_ids1)
        if temp_id in template_ids2_list]

    best_score = None
    best_trigger_pair = [None, None]
    for id_index_1, id_index_2 in common_temp_id_indices:
        # O(n^2), is there something better we can do?
        trigs_h1 = triggers1[id_index_1]
        trigs_l1 = triggers2[id_index_2]
        # inds_h1, inds_l1 = np.meshgrid(
        #     np.arange(len(trigs_h1)), np.arange(len(trigs_l1)))
        # paired_triggers = np.stack(
        #     [trigs_h1[inds_h1.flatten()], trigs_l1[inds_l1.flatten()]], axis=1)
        # trig_scores = score_func(paired_triggers, **kwargs)
        # best_score_ind = np.argmax(trig_scores)
        # if best_score is None or (trig_scores[best_score_ind] > best_score):
        #     best_score = trig_scores[best_score_ind]
        #     best_trigger_pair = paired_triggers[best_score_ind]

        # Warning: OK only for incoherent score!
        t1 = trigs_h1[np.argmax(score_func(trigs_h1, no_sum=True, **kwargs))]
        t2 = trigs_l1[np.argmax(score_func(trigs_l1, no_sum=True, **kwargs))]
        candidate_best_score = score_func(
            np.stack([t1, t2], axis=0), **kwargs | {'single_det': False})
        if best_score is None or (candidate_best_score > best_score):
            best_score = candidate_best_score
            best_trigger_pair = [t1, t2]

    return best_trigger_pair


def create_shifted_observations(
        candidates, max_time_slide_shift, minimal_time_slide_jump,
        max_delay, *assoc_arrays):
    # Restrict to maximum time slide shift and demands coincidence after shifts
    # --------------------------------------------------------------------------
    if len(candidates) == 0:
        return candidates

    if max_time_slide_shift is not None:
        candidates = candidates[
            np.abs(candidates[:, 0, 0] - candidates[:, 1, 0]) <=
            max_time_slide_shift]

        if len(candidates) == 0:
            return candidates

    # Enforce minimum jump in time slide shifts, within which we enforce "real"
    # coincidence
    # -------------------------------------------------------------------------
    # Ensures that upto shifts of minimal_time_slide_jump, only triggers with
    # physical delays contribute to the background with a single bucket in the
    # other detector. Allows for real events to trigger in different buckets
    # (off-split)
    mask_1 = ((candidates[:, 0, 0] - candidates[:, 1, 0]) %
              minimal_time_slide_jump) <= max_delay
    mask_2 = ((- candidates[:, 0, 0] + candidates[:, 1, 0]) %
              minimal_time_slide_jump) <= max_delay
    mask = np.logical_or(mask_1, mask_2)
    candidates = candidates[mask]

    if len(assoc_arrays) > 0:
        if len(assoc_arrays) == 1:
            arr_to_return = assoc_arrays[0][mask] \
                if not utils.checkempty(assoc_arrays[0]) else assoc_arrays[0]
            return candidates, arr_to_return
        else:
            assoc_arrays = \
                [arr[mask] if not utils.checkempty(arr) else arr
                 for arr in assoc_arrays]
            return candidates, assoc_arrays
    else:
        return candidates


def flag_duplicates_per_group_pair(
        pair_trigger_array, group_size, c0_pos, origin=0, veto_mask=None,
        extra_arrays=None, remove=False,
        score_func=utils.incoherent_score, **kwargs):
    """
    Picks a subset of pair_trigger_array such that each bucket of group_size
    in H1 and L1 has a unique `coincident' candidate. If a veto_mask is
    provided, defaults to preferentially keep in vetoed elements
    Note: The order of the input and output pair trigger array isn't identical
    in the general case
    :param pair_trigger_array:
        n_trigger x 2 x len(processedclist) array with H1 and L1 triggers
    :param group_size: Bucket size (s)
    :param origin: Origin for bucketing (s)
    :param veto_mask:
        If known, boolean mask of len(pair_trigger_array) with zeros at triggers
        that failed vetoes
    :param extra_arrays:
        If required, a list of arrays, each of size n_trigger, to dice up
        along with pair_trigger_array
    :param remove:
        If True, we remove the duplicates. Else we return a boolean mask that
        marks the duplicates with zeros
    :param c0_pos: Index of c0 in the processedclists, used if remove == False
    :param score_func:
        Function that takes in a pair of triggers and returns a scalar
    :param kwargs: Extra arguments to score_func
    :return: 1. n_trigger_new x 2 x len(processedclist) array with
                subset of maximized H1 and L1 triggers
             2. If veto_mask is provided, subset of veto_mask of
                size n_trigger_new (if provided in the first place)
             3. If extra_arrays are provided, list with their subsets
             4. if remove is False, n_trigger_new = n_trigger, and it returns
                a n_trigger x 2 boolean mask with zeros at the eliminated
                triggers in each detector
    """
    # --------------------------------------------
    # TODO: check that real guys pass this!
    if utils.checkempty(pair_trigger_array):
        if veto_mask is not None:
            if extra_arrays is not None:
                if remove:
                    return pair_trigger_array, veto_mask, extra_arrays
                else:
                    return pair_trigger_array, veto_mask, extra_arrays, \
                        np.ones((0, 2), dtype=bool)
            else:
                if remove:
                    return pair_trigger_array, veto_mask
                else:
                    return pair_trigger_array, veto_mask, \
                        np.ones((0, 2), dtype=bool)
        else:
            if extra_arrays is not None:
                if remove:
                    return pair_trigger_array, extra_arrays
                else:
                    return pair_trigger_array, extra_arrays, \
                        np.ones((0, 2), dtype=bool)
            else:
                if remove:
                    return pair_trigger_array
                else:
                    return pair_trigger_array, np.ones((0, 2), dtype=bool)

    return_mask = True
    if veto_mask is None:
        return_mask = False
        veto_mask = np.ones(len(pair_trigger_array), dtype=bool)

    if extra_arrays is None:
        extra_arrays = []

    for arr in extra_arrays:
        if len(arr) != len(pair_trigger_array):
            raise RuntimeError(
                "Need the subsidiary arrays to have the same length " +
                "as the original")

    # Group candidates according to the H1 time every
    # minimal_time_slide_jump
    groups_h1 = utils.splitarray(
        pair_trigger_array, pair_trigger_array[:, 0, 0], group_size, axis=0,
        origin=origin)

    # Indices of the elements in these groups in the parent array
    inds_groups_h1 = utils.splitarray(
        np.arange(len(pair_trigger_array)), pair_trigger_array[:, 0, 0],
        group_size, axis=0, origin=origin)

    # Trig_ids in each detector
    mask_retain = []
    trig_ids_h1 = []
    trig_ids_l1 = []
    if not remove:
        trig_ids_h1 = utils.make_trigger_ids(
            pair_trigger_array[:, 0, :], c0_pos)
        trig_ids_l1 = utils.make_trigger_ids(
            pair_trigger_array[:, 1, :], c0_pos)

    best_trigs = []
    best_veto_mask = []
    best_extra_arrays = [[] for _ in extra_arrays]

    for group_h1, inds_group_h1 in zip(groups_h1, inds_groups_h1):
        # Create subgroups according to L1 trigger time
        subgroups_l1 = utils.splitarray(
            group_h1, group_h1[:, 1, 0], group_size, axis=0, origin=origin)
        subinds_groups_l1 = utils.splitarray(
            inds_group_h1, group_h1[:, 1, 0], group_size, axis=0, origin=origin)

        # Pick best `coincident' pair per pair of groups. Safe to candidates
        # on edge, because they just enter a different group
        for subgroup_l1, subinds_group_l1 in zip(
                subgroups_l1, subinds_groups_l1):
            # Try to pick vetoed candidates if available
            veto_mask_subgroup_l1 = veto_mask[subinds_group_l1]
            if np.any(veto_mask_subgroup_l1):
                mask_pick = veto_mask_subgroup_l1
            else:
                mask_pick = np.ones(len(subgroup_l1), dtype=bool)

            ibest_mask = np.argmax([score_func(cand, 
                    **kwargs|{'single_det': False}) 
                    for cand in subgroup_l1[mask_pick]])
            ibest = np.where(mask_pick)[0][ibest_mask]
            best_trig = subgroup_l1[ibest]

            if remove:
                # Append only the surviving elements
                best_trigs.append([best_trig.copy()])
                best_veto_mask.append([veto_mask_subgroup_l1[ibest]])
                [child.append([parent[subinds_group_l1][ibest]]) for
                 child, parent in zip(best_extra_arrays, extra_arrays)]

            else:
                # Send the split structures through
                best_trigs.append(subgroup_l1)
                best_veto_mask.append(veto_mask_subgroup_l1)
                [child.append(parent[subinds_group_l1]) for child, parent in
                 zip(best_extra_arrays, extra_arrays)]

                # Make a mask with zeros at the failed triggers in the
                # appropriate detector
                mask_retain_gp = np.zeros((len(subgroup_l1), 2), dtype=bool)
                if len(subgroup_l1) == 1:
                    mask_retain_gp[0, :] = True
                else:
                    # We have multiple triggers in this set of windows in
                    # H1 and L1
                    best_trig_ids = utils.make_trigger_ids(best_trig, c0_pos)
                    # Mark as successful in a detector only if the trigger ID
                    # matches that of the best trigger in the bucket in that
                    # detector
                    mask_retain_gp[:, 0] = \
                        (trig_ids_h1[subinds_group_l1] == best_trig_ids[0])
                    mask_retain_gp[:, 1] = \
                        (trig_ids_l1[subinds_group_l1] == best_trig_ids[1])

                mask_retain.append(mask_retain_gp)

    # If bg_events is not empty, vstack is safe
    pair_trigger_array = np.concatenate(best_trigs, axis=0)
    best_veto_mask = np.concatenate(best_veto_mask, axis=0)
    best_extra_arrays = [np.concatenate(a, axis=0) for a in best_extra_arrays]

    if not remove:
        mask_retain = np.concatenate(mask_retain, axis=0)

    if return_mask:
        if len(extra_arrays) > 0:
            if remove:
                return pair_trigger_array, best_veto_mask, best_extra_arrays
            else:
                return pair_trigger_array, best_veto_mask, best_extra_arrays, \
                    mask_retain
        else:
            if remove:
                return pair_trigger_array, best_veto_mask
            else:
                return pair_trigger_array, best_veto_mask, mask_retain
    else:
        if len(extra_arrays) > 0:
            if remove:
                return pair_trigger_array, best_extra_arrays
            else:
                return pair_trigger_array, best_extra_arrays, mask_retain
        else:
            if remove:
                return pair_trigger_array
            else:
                return pair_trigger_array, mask_retain


def remove_bad_times(bg_events, bad_times, rejection_interval=25):
    # Throw out bad times in H1
    for bad_time_H1 in bad_times[0]:
        bg_events = bg_events[
            np.logical_or(
                bg_events[:, 0, 0] <= bad_time_H1,
                bg_events[:, 0, 0] >=
                (bad_time_H1 + rejection_interval))]

    # Throw out bad times in L1
    for bad_time_L1 in bad_times[1]:
        bg_events = bg_events[
            np.logical_or(
                bg_events[:, 1, 0] <= bad_time_L1,
                bg_events[:, 1, 0] >=
                (bad_time_L1 + rejection_interval))]

    return bg_events


# Functions involved in vetoing
# -----------------------------
def select_optimal_trigger_pairs(
        candidates, veto_dicts, time_shift_tol, threshold_chi2, trig_obj,
        origin=0, score_func=utils.incoherent_score, **kwargs):
    """Function to return optimized background, with extra veto based on
    significant psd drift
    :param candidates: n_candidates x 2 x len(processedclist[0]) array
    :param veto_dicts:
        Dictionaries as output by veto_and_optimize_single_detector
    :param time_shift_tol: Tolerance to sort candidates into buckets
    :param threshold_chi2:
        Incoherent threshold (only for veto based on significant PSD drift)
    :param trig_obj:
        Trigger object, used to read off an index offset, position of c0, and 
        dimensions in the finer grid
    :param origin: Origin for splitting the times relative to
    :param score_func: Function that accepts two triggers and returns a score
    :param kwargs: Dictionary with extra parameters for score_func
    :return: 1. List of optimized background candidates
             2. Mask into candidates that passed all the vetoes
             3. len(optimized_candidates) x 2 x
                (len(self.outlier_reasons) + 6 + len(split_chunks))
                boolean array with metadata about the trigger in each detector
                (see veto_and_optimize_group for the meaning of the entries)
    """
    all_candidates = []
    mask_vetoed = []
    metadata_arr = []
    
    # Warning: only consistent with what we did in >= O2, should be irrelevant 
    # as we only use it to ensure completeness 
    _, spacings_opt = trig_obj.define_finer_grid_func()

    for candidate in candidates:
        # Find keys into veto dicts
        dict_keys = np.floor(
            (candidate[:, 0] - origin) / time_shift_tol).astype(int)

        # Find optimized friends of the triggers
        passed_veto = True
        skip = False

        # Loop over detectors
        friends_arrs = []
        snr2_corr_factors = []
        metadatas = []
        for trig_time, dict_key, veto_dict in zip(
                candidate[:, 0], dict_keys, veto_dicts):
            # Get friends and finer PSD drift correction factors for triggers
            opt_results = veto_dict.get(dict_key)

            if opt_results is None:
                print("Weird missing key (is SNR close to bar?):",
                      candidate, flush=True)
                pass_flag = False
                friends_to_optimize = np.array([])
                snr2_corr_factor = 1
                # Create metadata array and record the failure flag as failed
                # after finer sinc interpolation
                metadata = np.ones(
                    len(trig_obj.outlier_reasons) + 6 + len(params.SPLIT_CHUNKS),
                    dtype=bool)
                metadata[-1] = False
                skip = True

            else:
                pass_flag, cloud, snr2_corr_factor, metadata = opt_results
                friends_to_optimize = \
                    trig.TriggerList.filter_processed_clist(
                        np.asarray(cloud),
                        filters={'time': (trig_time - params.DT_OPT,
                                          trig_time + params.DT_OPT)})

                if utils.checkempty(friends_to_optimize):
                    print("Finer sinc-interpolation killed the candidate:",
                          candidate, flush=True)
                    pass_flag = False
                    metadata[-1] = False
                    skip = True

            passed_veto *= pass_flag
            friends_arrs.append(friends_to_optimize)
            snr2_corr_factors.append(snr2_corr_factor)
            # We might overwrite this below, so copy
            metadatas.append(copy.deepcopy(metadata))

        if skip:
            # Skip this candidate, mark it as killed by finer sinc interpolation
            # Keep the original candidate in the list to be complete
            if len(candidate[0][trig_obj.c0_pos:]) < len(spacings_opt):
                npad = len(spacings_opt) - len(candidate[0][trig_obj.c0_pos:])
                candidate = np.pad(
                    candidate, (0, npad), 'constant', constant_values=0.)
            best_trig_h1 = candidate[0]
            best_trig_l1 = candidate[1]

        else:
            if np.any(np.asarray(snr2_corr_factors)>1.1):
                # Check if the candidate is killed by significant finer PSD drift
                finer_incoherent_score = np.dot(
                    candidate[:, 1], np.asarray(snr2_corr_factors))

                if finer_incoherent_score < threshold_chi2:
                    passed_veto = False

                    # Index representing PSD drift failure
                    for metadata in metadatas:
                        metadata[len(trig_obj.outlier_reasons) + 2] = False

            # Pick the best coincident pair from optimized clouds
            best_trig_h1, best_trig_l1 = get_best_candidate_segments(
                friends_arrs[0], friends_arrs[1], trig_obj.c0_pos,
                score_func=score_func, **kwargs)

            if best_trig_h1 is None or best_trig_l1 is None:
                # In an edge case, the sinc interpolation can create friends
                # that do not intersect, apology to the grad student
                print("Bug or Feature?: No common element found in clouds!",
                      flush=True)
                continue

        # Everything worked, update the return structure
        optimized_candidate = np.array([best_trig_h1, best_trig_l1])
        all_candidates.append(optimized_candidate)
        mask_vetoed.append(passed_veto)
        metadata_arr.append(metadatas)

    if len(all_candidates) > 0:
        try:
            all_candidates = np.stack(all_candidates)
        except ValueError:
            # Fix in case the finer grid was different in O1, to make O1 run work
            trig_dims = np.array([x.shape[-1] for x in all_candidates])
            target_dim = np.max(trig_dims)
            for ind in range(len(all_candidates)):
                if trig_dims[ind] < target_dim:
                    npad = target_dim - trig_dims[ind]
                    all_candidates[ind] = np.pad(
                        all_candidates[ind], (0, npad), 'constant', 
                        constant_values=0.)
            all_candidates = np.stack(all_candidates)
            
        mask_vetoed = np.array(mask_vetoed, dtype=bool)
        metadata_arr = np.array(metadata_arr, dtype=bool)
    else:
        # Return arrays that formally have the right shape
        cand_shape = candidates.shape
        all_candidates = np.zeros((0,) + cand_shape[1:])
        mask_vetoed = np.ones(0, dtype=bool)
        metadata_arr = np.zeros(
            (0,
             cand_shape[1],
             len(trig_obj.outlier_reasons) + 6 + len(params.SPLIT_CHUNKS)),
            dtype=bool)

    return all_candidates, mask_vetoed, metadata_arr


def read_channel_dict(trig_obj, times, chan_name):
    """
    Reads off boolean channel dict entries for times
    :param trig_obj: Instance of trig.TriggerList
    :param times: Time or list of times to read the channel dict for
    :param chan_name: Key into trig_obj.channel_dict
    :return: channel dict entry or entries corresponding to times
    """
    # Channel dict is at 1 Hz
    inds = np.floor(np.asarray(times) - trig_obj.time[0]).astype(int)
    return trig_obj.channel_dict[chan_name][inds]


def veto_and_optimize_group(
        trigs_gp, trig_obj, time_shift_tol, veto_triggers=True,
        min_veto_chi2=None, apply_threshold=True, relative_binning=True,
        origin=0, opt_format='new'):
    """
    :param trigs_gp: Processedclist of triggers
    :param trig_obj:
        Instance of trig.TriggerList with processed data in the detector
    :param time_shift_tol:
    :param veto_triggers: Flag indicating whether to veto triggers
    :param min_veto_chi2: If given, veto only the candidates above this bar
    :param apply_threshold:
        Flag to apply threshold on single-detector chi2 when optimizing
    :param relative_binning: Flag to turn relative binning on/off
    :param origin: Origin in time (s) to split triggers relative to
    :param opt_format:
        How we choose the finer grid, changed between O1 and O2 analyses
        Exposed here to replicate old runs if needed
    :return: Dictionary indexed by time / 0.01 s with
            1. Pass/fail flag
            2. Cloud of optimized triggers
            3. Finer PSD drift correction factor if significant
            4. Boolean array of size
                len(self.outlier_reasons) + 6 + len(split_chunks)
                with metadata
                0: CBC_CAT2 flag ("and" of the values for the cloud)
                1: CBC_CAT3 flag ("and" of the values for the cloud)
                The 2:len(self.outlier_reasons) + 6 + len(split_chunks) elements
                have zeros marking glitch tests that fired
                The indices correspond to:
                2: len(self.outlier_reasons) + 2: index into outlier reasons
                    for excess-power-like tests
                len(self.outlier_reasons) + 2: Finer PSD drift killed it
                len(self.outlier_reasons) + 3: No chunks present
                len(self.outlier_reasons) + 4: Overall chi-2 test
                len(self.outlier_reasons) + 5:
                    len(self.outlier_reasons) + 5 + len(split_chunks):
                        Split tests
                len(outlier_reasons) + 5 + len(split_chunks):
                    Finer sinc-interpolation
    """
    if utils.checkempty(trigs_gp):
        return {}

    # Define subset of data to veto any trigger within this group
    subset_details = trig_obj.prepare_subset_for_vetoes(trigs_gp)

    # Extreme times in the group
    min_time_gp, max_time_gp = np.min(trigs_gp[:, 0]), np.max(trigs_gp[:, 0])

    # Define parameters useful for defining safety margins for vetoes
    # Half of grid spacing of finer grid
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
            # Fix for bank (4, 3)
            # Should have been /2, but we keep the bug
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
        # Define function that returns finer grid points given a location,
        # and spacings of finer grid
        finer_grid_func, spacings_opt = trig_obj.define_finer_grid_func()

    # Define bin edges for relative binning
    if relative_binning:
        dt_rb = max_time_gp - min_time_gp + 2 * params.DT_OPT
        relative_freq_bins = trig_obj.templatebank.def_relative_bins(
            spacings_opt, dt=dt_rb, delta=0.1)
    else:
        relative_freq_bins = None

    # Optimize all triggers
    # ---------------------
    # Get set of unique calphas on coarse grid to compute triggers for
    calphas_gp = trigs_gp[:, trig_obj.c0_pos:]
    calphas_gp = {tuple(row) for row in calphas_gp}

    # Get unique list of finer calphas around the coarse calphas
    finer_calphas = []
    for calpha in calphas_gp:
        finer_calphas.append(finer_grid_func(calpha))
    #finer_calphas = np.vstack({tuple(row) for row in np.vstack(finer_calphas)})
    finer_calphas = np.unique(np.vstack(finer_calphas), axis=0)

    # Generate triggers for finer calphas throughout the group, using the
    # loudest trigger as reference
    trigger = trigs_gp[np.argmax(trigs_gp[:, 1])]
    dt_left = trigger[0] - min_time_gp + params.DT_OPT
    dt_right = max_time_gp - trigger[0] + params.DT_OPT
    opt_triggers = trig_obj.gen_triggers_local(
        trigger=trigger, dt_left=dt_left, dt_right=dt_right,
        apply_threshold=apply_threshold, relative_binning=relative_binning,
        relative_freq_bins=relative_freq_bins, subset_defined=True,
        compute_calphas=finer_calphas, orthogonalize_modes=True)

    # Add new triggers to dictionary indexed by template IDs
    cloud_scratch_dict = {}
    if not utils.checkempty(opt_triggers):
        temp_ids, opt_trigger_groups = group_by_id(
            opt_triggers, trig_obj.c0_pos)
        for temp_id, opt_trigger_group in zip(temp_ids, opt_trigger_groups):
            cloud_scratch_dict[temp_id] = opt_trigger_group

    # globals()["cloud_scratch_dict"] = cloud_scratch_dict

    # Divide triggers into buckets, populate dictionary, and record veto results
    # --------------------------------------------------------------------------
    keys, indices_buckets = utils.splitarray(
        np.arange(len(trigs_gp)), trigs_gp[:, 0], time_shift_tol,
        origin=origin, return_split_keys=True)
    opt_dic_gp = {}

    for key, indices_bucket in zip(keys, indices_buckets):
        # Boolean array with metadata
        metadata = np.ones(
            len(trig_obj.outlier_reasons) + 6 + len(params.SPLIT_CHUNKS),
            dtype=bool)

        # Triggers in this bucket
        trigs_bucket = trigs_gp[indices_bucket]

        # Collect friends of unique triggers (loud triggers tend to repeat)
        #unique_trigs = np.vstack({tuple(row) for row in trigs_bucket})
        unique_trigs = np.unique(trigs_bucket, axis=0)
        lists_of_friends = []

        for unique_trig in unique_trigs:
            finer_calphas = finer_grid_func(unique_trig[trig_obj.c0_pos:])
            previous_opt_triggers = find_friends_in_cloud(
                cloud_scratch_dict, finer_calphas)
            previous_opt_triggers = trig_obj.filter_processed_clist(
                previous_opt_triggers,
                filters={'time': (unique_trig[0] - params.DT_OPT,
                                  unique_trig[0] + params.DT_OPT)})
            if len(previous_opt_triggers) > 0:
                lists_of_friends.append(previous_opt_triggers)

        if len(lists_of_friends) == 0:
            # Finer sinc-interpolation killed all the triggers
            passed_veto = False
            list_of_friends = np.array([])
            snr2_corr_factor = 1

            # Read off CBC_CAT2 and CBC_CAT3 for the parent triggers
            metadata[0] = np.all(read_channel_dict(
                trig_obj, unique_trigs[:, 0], 'CBC_CAT2'))
            metadata[1] = np.all(read_channel_dict(
                trig_obj, unique_trigs[:, 0], 'CBC_CAT3'))

            # Record the failure flag
            metadata[-1] = False
        else:
            # Prune duplicates
            # list_of_friends = np.vstack(
            #    {tuple(row) for row in np.vstack(lists_of_friends)})
            list_of_friends = np.unique(np.vstack(lists_of_friends), axis=0)

            # Read off CBC_CAT2 and CBC_CAT3 for the set of friends
            metadata[0] = np.all(read_channel_dict(
                trig_obj, list_of_friends[:, 0], 'CBC_CAT2'))
            metadata[1] = np.all(read_channel_dict(
                trig_obj, list_of_friends[:, 0], 'CBC_CAT3'))

            # Pick the best trigger
            trigger_to_veto = list_of_friends[np.argmax(list_of_friends[:, 1])]

            # Veto the trigger at the single detector level
            if (veto_triggers and ((min_veto_chi2 is None) or
                                   (trigger_to_veto[1] > min_veto_chi2))):
                # Fix safety margins for vetoes
                veto_spacing = dcalphas_veto.copy()

                # Noise can lead to larger uncertainties
                snr_trig = np.sqrt(trigger_to_veto[1])
                veto_spacing[
                    np.logical_and(extent > 1/snr_trig,
                                   1/snr_trig > veto_spacing)] = 1/snr_trig

                # globals()["trigger_to_veto"] = trigger_to_veto
                # globals()["veto_spacing"] = veto_spacing
                # globals()["trig_obj"] = trig_obj
                # globals()["subset_details"] = subset_details
                                 
                passed_veto, snr2_corr_factor, glitch_mask = \
                    trig_obj.veto_trigger_all(
                        trigger_to_veto, dcalphas=veto_spacing,
                        subset_details=subset_details, lazy=False)

                # Record the veto results
                metadata[2:-1] = glitch_mask[:]
            else:
                passed_veto = True
                # If needed, record the finer psd drift correction
                if veto_triggers:
                    # Only do the veto for trigger close to a hole as that is not
                    # very expensive
                    passed_veto, _, glitch_mask = trig_obj.veto_trigger_all(
                        trigger_to_veto, do_costly_vetos=False,
                        subset_details=subset_details, lazy=False)
                    # Note: By the time we reach this point, we should have
                    # computed the safemean
                    snr2_corr_factor = trig_obj.finer_psd_drift(
                        trigger_to_veto, average='safemean')
                else:
                    snr2_corr_factor = 1

        opt_dic_gp[key] = (passed_veto, list_of_friends,
                           snr2_corr_factor, metadata)

    return opt_dic_gp


def veto_and_optimize_single_detector(
        triggers, trig_obj, time_shift_tol, group_duration=0.1,
        veto_triggers=True, min_veto_chi2=None, apply_threshold=True,
        origin=0, n_cores=1, opt_format="new"):
    """
    Veto and optimize coincident candidates in a single detector
    :param triggers: Processedclist with candidates in a detector
    :param trig_obj:
        Instance of trig.TriggerList with processed data in the detector
    :param time_shift_tol:
        Veto only one trigger every bucket of time_shift_tol seconds
    :param group_duration:
        Divide triggers into groups every group_duration seconds and
        define subset for this group at once, saves on FFTs
    :param veto_triggers: Flag indicating whether to veto triggers
    :param min_veto_chi2: If given, veto only the candidates above this bar
    :param apply_threshold:
        Flag to apply threshold on single-detector chi2 when optimizing
    :param origin: Origin to split the trigger times relative to
    :param n_cores: Number of cores to use for splitting the computation
    :param opt_format:
        How we choose the finer grid, changed between O1 and O2 analyses
        Exposed here to replicate old runs if needed
    :return:
        Dictionary indexed by bucket_ids with passed/failed veto,
        a processedclist of friends, a snr^2 correction factor, and metadata
        about data quality and glitch tests
        (see veto_and_optimize_group for an explanation)
    """
    if utils.checkempty(triggers):
        return {}

    # Group triggers according to time in the detector to define subsets
    index_groups = utils.splitarray(
        np.arange(len(triggers)), triggers[:, 0], group_duration,
        axis=0, origin=origin)

    # Function to go over groups and veto candidates within
    def veto_opt_gp(inds_gp):
        """
        :param inds_gp: Indices into triggers of a group
        :return: Dictionary indexed by time / 0.01 s with with pass/fail flag,
            cloud of optimized triggers, and finer PSD drift correction factor
            if significant
        """
        # TODO: Can we make the core not see triggers to save memory?
        trigs_gp = triggers[inds_gp]
        opt_dic_gp = veto_and_optimize_group(
            trigs_gp, trig_obj, time_shift_tol, veto_triggers=veto_triggers,
            min_veto_chi2=min_veto_chi2, apply_threshold=apply_threshold,
            origin=origin, opt_format=opt_format)

        return opt_dic_gp

    opt_dic_file = {}
    if n_cores == 1:
        for i, inds_group in enumerate(index_groups):
            print(i/len(index_groups), len(inds_group), flush=True)
            opt_dic_group = veto_opt_gp(inds_group)
            opt_dic_file.update(copy.deepcopy(opt_dic_group))
    else:
        n_groups = len(index_groups)
        print(f"Going to veto and optimize triggers in {n_groups} groups " +
              f"every {group_duration} seconds", flush=True)
        # Do in chunks of n_cores with chunksize of 1 to treat weird 
        # keyboardinterrupt
        p = mp.Pool(n_cores)
        index_group_chunks = \
            [index_groups[i:i + n_cores] for i in
             range(len(index_groups))[::n_cores]]
        n_groups_done = 0

        for index_group_chunk in index_group_chunks:
            opt_details = p.map_async(
                veto_opt_gp, index_group_chunk, chunksize=1)
            track_job(opt_details, "veto and optimization", n_groups)
            opt_details = opt_details.get()
            for opt_dic_group in opt_details:
                opt_dic_file.update(copy.deepcopy(opt_dic_group))
            n_groups_done += len(index_group_chunk)
            print(f"Finished with {n_groups_done} groups", flush=True)

        p.close()
        p.join()

    return opt_dic_file


def stringent_veto(
        triggers, det_ind, trig_obj, min_veto_snr2=None, group_duration=0.1,
        origin=0, n_cores=1):
    """
    Apply stringent vetoes to optimized candidates for the edge case in which
    the best one in the bucket survives, but the coincident ones do not
    :param triggers:
        n_triggers x 2 x row(processedclist) with coincident candidates
    :param det_ind: Index of detector to veto (into 2nd dimension of trigger)
    :param trig_obj:
        Instance of trig.TriggerList with processed data in the detector
    :param min_veto_snr2: Veto triggers above this threshold
    :param group_duration:
        Group triggers every group_duration to save on setup FFTs
    :param origin: Origin to split the times relative to
    :param n_cores:
    :return:
        1. Boolean array marking whether the triggers pass stringent vetoes
        2. Boolean array of shape n_triggers x (4 + len(split_chunks))
            with zeros where glitch tests fired
            The indices correspond to
            0: No chunks present
            1: Overall chi-2 test
            2: 2 + len(split_chunks): Split test
            2 + len(split_chunks): No chunks with greater nchunk
            3 + len(split_chunks): Chi^2 with greater nchunk
    """
    if utils.checkempty(triggers):
        return np.zeros(0, dtype=bool), \
               np.ones((0, 4 + len(params.SPLIT_CHUNKS)), dtype=bool)

    metadata_arr = np.ones(
        (len(triggers), 4 + len(params.SPLIT_CHUNKS)), dtype=bool)

    # Define parameters useful for defining safety margins for vetoes
    # Define veto safety
    _, dcalphas_veto = trig_obj.define_finer_grid_func(
        dcalpha_coarse=trig_obj.delta_calpha / 2, trim_dims=False)

    # Half-width of bank in dimensions
    extent = (trig_obj.templatebank.bounds[:len(dcalphas_veto), 2] -
              trig_obj.templatebank.bounds[:len(dcalphas_veto), 0]) / 2
    extent *= trig_obj.template_safety

    # Group triggers according to time in the detector to define subsets
    indices_groups = utils.splitarray(
        np.arange(len(triggers)), triggers[:, det_ind, 0], group_duration,
        axis=0, origin=origin)

    def stringent_veto_gp(trigs_gp):
        trigs_gp_det = trigs_gp[:, det_ind, :]

        # Check if we have to do any work at all
        if ((min_veto_snr2 is not None) and
                (np.max(trigs_gp_det[:, 1]) <= min_veto_snr2)):
            return np.ones(len(trigs_gp), dtype=bool), \
                np.ones((len(trigs_gp), 4 + len(params.SPLIT_CHUNKS)),
                        dtype=bool)

        # Define subset of data for vetoes
        subset_details = trig_obj.prepare_subset_for_vetoes(trigs_gp_det)

        # Pick unique triggers, since they tend to repeat
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

            # globals()["trigger_to_veto"] = trigger_to_veto
            # globals()["subset_details"] = subset_details
            # print(f"Applying stringent vetoes to {trigger_to_veto}")

            # Veto at the single detector level
            if (min_veto_snr2 is None) or (trigger_to_veto[1] > min_veto_snr2):
                # Fix safety margins for vetoes
                veto_spacing = dcalphas_veto.copy()

                # Noise can lead to larger uncertainties
                trigs_chi2 = trigs_gp[unique_ind_gp[0], :, 1]
                snr_trig = np.sqrt(np.sum(trigs_chi2))
                veto_spacing[
                    np.logical_and(
                        extent > 1 / snr_trig,
                        1 / snr_trig > veto_spacing)] = 1 / snr_trig

                # globals()["veto_spacing"] = veto_spacing

                # Apply more stringent vetoes to the optimized trigger
                passed_veto, phase_veto_mask = \
                    trig_obj.veto_trigger_phase(
                        trigger_to_veto, dcalphas=veto_spacing,
                        subset_details=subset_details, verbose=True, lazy=False)
                glitch_mask[:len(phase_veto_mask)] = phase_veto_mask[:]
                # print("Passed veto 1: ", passed_veto)

                # Perform chi-squared test with a large number of chunks
                passed_veto_2, large_chunk_veto_mask = \
                    trig_obj.veto_trigger_phase(
                        trigger_to_veto, n_chunk=params.N_CHUNK_2,
                        split_chunks=[], dcalphas=veto_spacing,
                        subset_details=subset_details, verbose=True, lazy=False)
                glitch_mask[len(phase_veto_mask):] = large_chunk_veto_mask[:]
                # print("Passed veto 2: ", passed_veto_2)

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

    # def stringent_veto_trigger(trigger, subset_details=None):
    #     if (min_veto_snr2 is None) or (trigger[1] > min_veto_snr2):
    #         # Apply more stringent veto on the optimized trigger
    #         if subset_details is None:
    #             subset_details = trig_obj.prepare_subset_for_vetoes(trigger)
    #         passed_veto = trig_obj.veto_trigger_phase(
    #             trigger, dcalphas=dcalphas_veto, subset_details=subset_details,
    #             verbose=False)
    #         # Perform chi-squared test with a large number of chunks
    #         passed_veto &= trig_obj.veto_trigger_phase(
    #             trigger, n_chunk=params.N_CHUNK_2, split_chunks=[],
    #             dcalphas=dcalphas_veto, subset_details=subset_details,
    #             verbose=False)
    #     else:
    #         passed_veto = True
    #
    #     return passed_veto

    trigger_fates_file = []
    metadata_file = []
    if n_cores == 1:
        for i, indices_group in enumerate(indices_groups):
            print(i/len(indices_groups), len(indices_group), flush=True)
            trigs_group = triggers[indices_group]
            trigger_fates_group, metadata_group = stringent_veto_gp(trigs_group)
            trigger_fates_file.append(np.c_[indices_group, trigger_fates_group])
            metadata_file.append(np.c_[indices_group, metadata_group])

        # for ind, trigger in enumerate(triggers):
        #     if ind % 10 == 0:
        #         print(ind, flush=True)
        #     passed_stringent_veto = stringent_veto_trigger(
        #         trigger, subset_details)
        #     trigger_fates.append(passed_stringent_veto)
    else:
        # Do in chunks of n_cores with chunksize of 1 to treat weird 
        # keyboardinterrupt
        p = mp.Pool(n_cores)
        indices_groups_chunks = [indices_groups[i:i + n_cores] for i in
                                 range(len(indices_groups))[::n_cores]]
        print(f"Applying stringent veto to triggers in {len(indices_groups)}" +
              f" groups every {group_duration} seconds", flush=True)

        n_groups_done = 0
        for index_groups_chunk in indices_groups_chunks:
            trigger_groups_chunk = [triggers[ids] for ids in index_groups_chunk]
            opt_details = p.map_async(
                stringent_veto_gp, trigger_groups_chunk, chunksize=1)
            track_job(opt_details, "stringent veto", len(indices_groups))
            opt_details = opt_details.get()
            # Check if the behavior is inconsistent in old versions of
            # multiprocess!
            for indices_group, (trigger_fates_group, metadata_group) in zip(
                    index_groups_chunk, opt_details):
                trigger_fates_file.append(
                    np.c_[indices_group, trigger_fates_group])
                metadata_file.append(
                    np.c_[indices_group, metadata_group])
            n_groups_done += len(index_groups_chunk)
            print(f"Finished with {n_groups_done} groups", flush=True)

        # triggers_chunks = [triggers[i:i + n_cores] for i in range(len(triggers))[::n_cores]]
        # n_triggers_done = 0

        # for iter_id in range(len(triggers_chunks)):
        #     passed_stringent_veto = p.map_async(
        #         stringent_veto_trigger, triggers_chunks[iter_id], chunksize=1)
        #     # track_job(passed_stringent_veto, "stringent veto", len(triggers))
        #     trigger_fates += passed_stringent_veto.get()
        #     n_triggers_done += len(triggers_chunks[iter_id])
        #     print(f"Finished with {n_triggers_done} triggers", flush=True)
       
        p.close()
        p.join()

    # Now sort back into original order and return the fates of the triggers
    trigger_fates_file = np.vstack(trigger_fates_file)
    trigger_fates_file = trigger_fates_file[
        trigger_fates_file[:, 0].argsort(), 1]
    metadata_file = np.vstack(metadata_file)
    metadata_file = metadata_file[metadata_file[:, 0].argsort(), 1:]

    # return np.array(trigger_fates, dtype=bool)

    return np.array(trigger_fates_file, dtype=bool), \
        np.array(metadata_file, dtype=bool)


# TODO: Add sinc interpolation to the step fit veto
def veto_orthogonal_step(f_grid, strain_whitened_fd, whitening_filter_fd, best_waveform_fd,
                         double_step=False, use_cosine_steps=True, delta_t_max=0.015,
                         N_margin=8192, dt=1/1024., index_of_interest=None, f_power=1):
    """

    :param f_grid:
    :param strain_whitened_fd:
    :param whitening_filter_fd:
    :param best_waveform_fd:
    :param double_step: if True, two step functions would be used.
    :param use_cosine_steps: if True, would fit also the Hilbert transform of a step.
    :param delta_t_max:
    :param N_margin:
    :param dt: dt of whitened strain samples.
    :param index_of_interest: index around which to search for the highest overlap (Important to give, as can auto tune to an unrelated location!)
    :param f_power: will use a waveform that is 1/f**f_power. default is 1
    :return:
    """
    # Assume that the trigger_center is in shift len(strain_whitened_td)/2 (approx, tol=64 ms)
    # Assume that the waveform is such that the linear free point is at ZERO

    wf_glitch_fd = utils.gen_step_fd(f_grid, f_power=f_power)

    wf_glitch_td = (np.fft.irfft(wf_glitch_fd * whitening_filter_fd) + 1j*np.fft.irfft(1j*wf_glitch_fd * whitening_filter_fd))
    wf_glitch_td /= np.sum(np.abs(wf_glitch_td)**2)**0.5

    wf_glitch_cos = wf_glitch_td.real * 2**0.5
    wf_glitch_sin = wf_glitch_td.imag * 2**0.5

    # I am sure you can get here by slightly less FFT operations, but let it be...
    cos_scores = np.fft.irfft(strain_whitened_fd * np.fft.rfft(wf_glitch_cos))
    sin_scores = np.fft.irfft(strain_whitened_fd * np.fft.rfft(wf_glitch_sin))

    best_wf_cos = np.fft.irfft(best_waveform_fd)
    best_wf_sin = np.fft.irfft(1j*best_waveform_fd)

    overlaps_wf_cos = np.fft.irfft(strain_whitened_fd * np.conj(np.fft.rfft(best_wf_cos)))/dt / 2**0.5
    overlaps_wf_sin = np.fft.irfft(strain_whitened_fd * np.conj(np.fft.rfft(best_wf_sin)))/dt / 2**0.5

    if index_of_interest is None:
        index_of_interest = int(np.argmax(overlaps_wf_cos[index_of_interest-20:index_of_interest+20]**2 +
                                          overlaps_wf_sin[index_of_interest-20:index_of_interest+20]**2))+20
        print(index_of_interest)
    sl = slice(index_of_interest - N_margin, index_of_interest + N_margin)

    best_score = 0
    best_ts = None
    scores_record = []
    best_scores = []

    ind_max = int(delta_t_max/dt)

    for n_inds1 in np.arange(-ind_max, ind_max+1, 1):

        cos_scores_rot1 = np.roll(cos_scores, n_inds1)
        sin_scores_rot1 = np.roll(sin_scores, n_inds1)

        if double_step:
            n_inds2_list = list(range(n_inds1, ind_max+1))
        else:
            n_inds2_list = [n_inds1]
        for n_inds2 in n_inds2_list:
            if n_inds2 != n_inds1:
                cos_scores_rot2 = np.roll(cos_scores, n_inds2)
                if use_cosine_steps:
                    sin_scores_rot2 = np.roll(sin_scores, n_inds2)
                    vecs = np.array([overlaps_wf_cos[sl], overlaps_wf_sin[sl],
                                 cos_scores_rot1[sl], sin_scores_rot1[sl], cos_scores_rot2[sl], sin_scores_rot2[sl]])
                else:
                    # Note: confusingly, the step function is the cosine waveform.
                    vecs = np.array([overlaps_wf_cos[sl], overlaps_wf_sin[sl],
                                     cos_scores_rot1[sl], cos_scores_rot2[sl]])
            else:
                if use_cosine_steps:
                    vecs = np.array([overlaps_wf_cos[sl], overlaps_wf_sin[sl],
                                     cos_scores_rot1[sl], sin_scores_rot1[sl]])
                else:
                    vecs = np.array(
                        [overlaps_wf_cos[sl], overlaps_wf_sin[sl], cos_scores_rot1[sl]])

            cov = np.zeros([len(vecs), len(vecs)])

            for i,v1 in enumerate(vecs):
                for j,v2 in enumerate(vecs):
                    cov[i,j] = np.mean(v1*v2) - np.mean(v1)*np.mean(v2)

            inv_cov_12 = np.linalg.inv(cov[:2,:2])
            inv_cov = np.linalg.inv(cov)

            scores = np.sum(vecs * np.dot(inv_cov, vecs),0) - np.sum(vecs[:2] * np.dot(inv_cov_12, vecs[:2]),0)
            scores_record.append(scores)

            score = scores[N_margin]
            if score > best_score:
                best_score = score
                best_ts = (n_inds1,n_inds2)
                best_scores = scores

    return best_score, best_ts, best_scores, np.max(scores_record,0)


def fit_step_veto(trigger, det_ind, f_power=1):
    """

    :param trigger: A (trigger, bank_id) pair.
    :param det_ind: 0 for Hanford, 1 for Livingston.
    :return: 1. veto_score - higher is worse.
             2. best_delta_ts - the relative shift between the GW and the best fit step functions
             3. The best scores as function of position withing 16 seconds of the event time
             4. Empirical score distribution of the max (fit_step chi2) over position, for 16 seconds near event time
    """

    fname = utils.get_detector_fnames(trigger[0][0],trigger[1][0],trigger[1][1])[det_ind]

    T = trig.TriggerList.from_json(fname, load_trigs=False)

    ind_start = max(np.searchsorted(T.time, trigger[0][0]) - T.fftsize//2,64*1024)
    ind_end = ind_start + T.fftsize
    if ind_end > (len(T.time) - 64*1024):
        ind_end = len(T.time) - 64*1024
        ind_start = ind_end - T.fftsize
        # TODO fit step veto require less spares. Possibly by using a much shorter FFT.
        if ind_start < 0:
            print("Cannot find a 2^20 segment. Dying!")
            return [True] + [None]*5

    index_of_interest = np.searchsorted(T.time, trigger[0][0]) - ind_start
    relevant_whitened_strain = T.strain[ind_start:ind_end]
    f_grid = np.fft.rfftfreq(T.fftsize, T.dt)

    best_waveform_fd = T.templatebank.gen_wfs_fd_from_calpha(trigger[0][T.c0_pos:], f_grid) * T.templatebank.wt_filter_fd
    best_waveform_fd = best_waveform_fd / np.sum(np.abs(best_waveform_fd)**2)**0.5

    whitened_strain_fd = np.fft.rfft(relevant_whitened_strain)

    best_score, best_ts, best_scores, scores_record = veto_orthogonal_step(f_grid, whitened_strain_fd,
                                                                           T.templatebank.wt_filter_fd,
                                                                           best_waveform_fd,
                                                                           index_of_interest=index_of_interest,
                                                                           f_power=f_power)
    return best_score, best_ts, best_scores, scores_record


def find_friends_in_cloud(cloud_dict, finer_calphas):
    """
    :param cloud_dict: Dictionary with processedclists indexed by template id
    :param finer_calphas: List of finer calphas to look for in cloud_dict
    :return:
        processedclist of all triggers in the cloud with calphas in
        finer_calphas
    """
    if len(cloud_dict.keys()) == 0:
        # return [], []
        return []

    # Make ids of triggers on finer grid
    tg_ids_finer = np.array(
        utils.make_template_ids(finer_calphas), dtype=np.int64)
    
    members_of_cloud = []
    # member_ids = []
    for tg_id in tg_ids_finer:
        cloud_with_id = cloud_dict.get(tg_id)
        if cloud_with_id is not None:
            members_of_cloud.append(cloud_with_id)
            # member_ids.append(tg_id)
            
    if len(members_of_cloud) > 0:
        members_of_cloud = np.vstack(members_of_cloud)

    # return members_of_cloud, member_ids
    return members_of_cloud

    # triggers = [tg for tg_id, tg in zip(tg_ids_in_cloud, cloud)
    #             if tg_id in tg_ids_finer]
    # return np.array(triggers)


def get_files_with_glitches(dir_name, chi2_bound=80):
    files1 = glob.glob(os.path.join(dir_name, "*H1*.json"))
    files2 = glob.glob(os.path.join(dir_name, "*L1*.json"))

    known_times = utils.get_lsc_event_times() + utils.get_injection_details()[1]

    bad_files = []

    try:
        for ind, file in enumerate(files1 + files2):

            T = trig.TriggerList.from_json(file, do_ffts=False, load_data=False)
            best_100_ind = np.argsort(T.processedclist[:,1])[-100:]
            best_100_trigs = T.processedclist[best_100_ind,:]
            remaining = [t for t in best_100_trigs
                         if t[1] > chi2_bound and
                         not utils.is_close_to(t[0], known_times, eps=30)]

            if len(remaining) >0:
                bad_files.append(file)
                print("Found bad file. Bad file fraction:", len(bad_files)/(ind+1))
    except KeyboardInterrupt:
        return bad_files
    return bad_files


def get_histogram_stats(dir_name, chi2_bound=36):
    files1 = glob.glob(os.path.join(dir_name, "*H1*.json"))
    files2 = glob.glob(os.path.join(dir_name, "*L1*.json"))

    known_times = utils.get_lsc_event_times() + utils.get_injection_details()[1]

    trigger_nums=[]

    all_files = [f for f in (files1 + files2) if os.path.exists(f.split("_config")[0] + ".trig.npy")]
    print(len(all_files))

    try:
        for ind, file in enumerate(files1 + files2):

            print(ind, file)
            T = trig.TriggerList.from_json(file, do_ffts=False, load_data=False)

            trigger_nums.append(np.sum(T.processedclist[:,1] > chi2_bound) / np.sum(T.mask * T.valid_mask))

    except KeyboardInterrupt:
        return trigger_nums
    return trigger_nums


def collect_files(filelist, ncores=1):
    """Returns list of length nfiles with loaded arrays,
    an element is None if the byte-size < 10"""
    nfiles = len(filelist)
    candidates = []
    print("total number of epochs to collect:", nfiles)

    def collect_file(npy_file):
        if os.path.getsize(npy_file) > 10:
            return np.load(npy_file, allow_pickle=True)
        else:
            return None

    def add_to_struct(output, container):
        container.append(output)
        return

    if ncores > 1:
        # Divide files into chunks, keep printing to update in between
        chunksize = max(100 // ncores, 1) * ncores
        nchunk = int(np.ceil(nfiles / chunksize))
        fnamechunks = \
            [filelist[i*chunksize:(i+1)*chunksize] for i in range(nchunk)]
        # Create pool and send the jobs in
        p = mp.Pool(ncores)
        for ichunk, fnamechunk in enumerate(fnamechunks):
            print(f"Dealing with files {ichunk * chunksize}:" +
                  f"{min(ichunk * chunksize + chunksize, nfiles)} of {nfiles}")
            results = p.map(collect_file, fnamechunk)
            for result in results:
                add_to_struct(result, candidates)
        p.close()
        p.join()
    else:
        for i, fname in enumerate(filelist):
            if i % 100 == 0:
                print("Collecting candidates from file index:",
                      round(i / nfiles, 2) * 100, "% complete")
            result = collect_file(fname)
            add_to_struct(result, candidates)

    return candidates


def collect_files_npz(filelist, ncores=1, collect_timeseries=True):
    """Returns list of length nfiles with each element being a tuple with
    events, mask_vetoed, veto_metadata, coherent_scores
    (+timeseries if collect_timeseries)
    an element is None if the byte-size < 10"""
    nfiles = len(filelist)
    candidate_info = []
    print("total number of epochs to collect:", nfiles)

    def collect_file(npz_file):
        if os.path.getsize(npz_file) > 10:
            data = np.load(npz_file, allow_pickle=True)
            candidates = data["candidates"]
            mask_vetoed = data["mask_vetoed"]
            veto_metadata = data["metadata"]
            coherent_scores = data["coherent_scores"]
            if collect_timeseries:
                timeseries = data["timeseries"]
                return candidates, mask_vetoed, veto_metadata, \
                    coherent_scores, timeseries
            else:
                return candidates, mask_vetoed, veto_metadata, coherent_scores
        else:
            return None

    def add_to_struct(output, container):
        container.append(output)
        return

    if ncores > 1:
        # Divide files into chunks, keep printing to update in between
        chunksize = max(100 // ncores, 1) * ncores
        nchunk = int(np.ceil(nfiles / chunksize))
        fnamechunks = \
            [filelist[i*chunksize:(i+1)*chunksize] for i in range(nchunk)]
        # Create pool and send the jobs in
        p = mp.Pool(ncores)
        for ichunk, fnamechunk in enumerate(fnamechunks):
            print(f"Dealing with files {ichunk * chunksize}:" +
                  f"{min(ichunk * chunksize + chunksize, nfiles)} of {nfiles}")
            results = p.map(collect_file, fnamechunk)
            for result in results:
                add_to_struct(result, candidate_info)
        p.close()
        p.join()
    else:
        for i, fname in enumerate(filelist):
            if i % 100 == 0:
                print("Collecting candidates from file index:",
                      round(i / nfiles, 2) * 100, "% complete")
            result = collect_file(fname)
            add_to_struct(result, candidate_info)

    return candidate_info


def pick_lists(list1, list2):
    """Returns elements from list1 whose corresponding elements in list2
    aren't empty"""
    outlist = list1
    if not utils.checkempty(list1):
        if not utils.checkempty(list2):
            outlist = [x for x, y in zip(list1, list2)
                       if not utils.checkempty(y)]
    return outlist


@njit
def inds_into_before(
        cand_after_veto, cand_before_veto, isort_after, isort_before):
    """
    Gives indices into cand_before_veto for triggers in cand_after_veto
    We can use tolist(), but this is almost 10 times faster
    :param cand_after_veto:
        n1 x n_det x (row of processedclist) array with candidates
        after vetoes
    :param cand_before_veto:
        (n2 > n1) x n_det x (row of processedclist) array with candidates
        before vetoes
    :param isort_after:
        n1 array with arrays for lexicographic sort of cand_after_veto
    :param isort_before:
        n2 array with arrays for lexicographic sort of cand_before_veto
    :return:
    """
    inds_map_sorted = np.zeros(len(cand_after_veto), dtype=np.int32)
    i2 = 0
    for i1 in range(len(cand_after_veto)):
        while not np.all(
                cand_after_veto[isort_after[i1], :, 0] ==
                cand_before_veto[isort_before[i2], :, 0]):
            i2 += 1
        inds_map_sorted[isort_after[i1]] = isort_before[i2]
    return inds_map_sorted


@njit
def mypad(arr, n):
    ashape = list(arr.shape)
    ashape[-1] += n
    arrnew = np.zeros((ashape[0], ashape[1], ashape[2]))
    for i0 in range(arr.shape[0]):
        for i1 in range(arr.shape[1]):
            for i2 in range(arr.shape[2]):
                arrnew[i0, i1, i2] = arr[i0, i1, i2]
    return arrnew


def combine_files(cands_after_veto, cands_before_veto, timeseries):
    """
    Combines files that were read in using collect_files
    :param cands_after_veto:
        (possibly empty) List of (possibly empty) n_files read in
    :param cands_before_veto:
        (possibly empty) list of (possibly empty) n_files read in
    :param timeseries:
        (possibly empty) list of (possibly empty) n_files read in
    :return:
    """
    timeseries_after = []

    # First retain only the nonempty candidate lists before vetoes if passed
    if not utils.checkempty(cands_before_veto):
        cands_after_veto = pick_lists(cands_after_veto, cands_before_veto)
        timeseries = pick_lists(timeseries, cands_before_veto)
        cands_before_veto = pick_lists(cands_before_veto, cands_before_veto)

        # If we also have timeseries objects, their indices will be into
        # cands_before_veto, so assign indices for cands_after_veto
        if not (utils.checkempty(timeseries) or
                utils.checkempty(cands_after_veto)):
            for cand_after_veto, cand_before_veto, timeseries_before in zip(
                    cands_after_veto, cands_before_veto, timeseries):
                if utils.checkempty(cand_after_veto):
                    timeseries_after.append(None)
                else:
                    isort_after = np.lexsort(cand_after_veto[:, :, 0].T)
                    isort_before = np.lexsort(cand_before_veto[:, :, 0].T)
                    i_before = inds_into_before(
                        cand_after_veto, cand_before_veto,
                        isort_after, isort_before)
                    timeseries_after.append(timeseries_before[i_before])

        # Combine the arrays
        # Combine cands_before_veto
        # If some of them were rerun, we can have fewer c_alphas
        # because the new version trims zeros at the end
        if not utils.checkempty(cands_before_veto):
            candlen = np.max([x.shape[-1] for x in cands_before_veto])
            cands_before_veto = np.vstack(
                [x if x.shape[-1] == candlen else
                 mypad(x, candlen - x.shape[-1]) for x in cands_before_veto])

        # Combine timeseries
        if not utils.checkempty(timeseries):
            timeseries = np.vstack(timeseries)

    if not utils.checkempty(cands_after_veto):
        # Fix to make cand8/9 work for O2
        if utils.checkempty(timeseries_after):
            timeseries_after = timeseries

        # Clean up the candidates and timeseries
        timeseries_after = pick_lists(timeseries_after, cands_after_veto)
        cands_after_veto = pick_lists(cands_after_veto, cands_after_veto)

        # Combine the arrays
        # Combine cands_after_veto
        # If some of them were rerun, we can have fewer c_alphas
        # because the new version trims zeros at the end
        if not utils.checkempty(cands_after_veto):
            candlen = np.max([x.shape[-1] for x in cands_after_veto])
            cands_after_veto = np.vstack(
                [x if x.shape[-1] == candlen else
                 mypad(x, candlen - x.shape[-1]) for x in cands_after_veto])

        # Combine timeseries
        if not utils.checkempty(timeseries_after):
            timeseries_after = np.vstack(timeseries_after)

    return cands_after_veto, cands_before_veto, timeseries_after, timeseries


def combine_files_npz(candidate_info, apply_vetoes=False):
    """
    Combines files that were read in using collect_files_npz
    :param candidate_info:
        (possibly empty) List of (possibly empty) n_files read in
    :param apply_vetoes: Flag to enforece vetoes when saving candidates
    :return:
        Combined candidate info, in the form of a list with the elements being
        candidates, veto_metadata, coherent score (+timeseries) if asked for,
        with vetoes applied if needed
    """
    # Skip empty files
    candidate_info = [x for x in candidate_info if not utils.checkempty(x)]

    if utils.checkempty(candidate_info):
        return candidate_info

    # Apply vetoes if asked
    if apply_vetoes:
        candidate_info = \
            [[x[i][x[1]] for i in range(len(x)) if i != 1]
             for x in candidate_info]
        # Skip files where everything was vetoed
        candidate_info = \
            [x for x in candidate_info if not utils.checkempty(x[0])]
    else:
        # Ignore the veto flag, we will apply the metadata in plots_publication
        candidate_info = \
            [[x[i] for i in range(len(x)) if i != 1] for x in candidate_info]

    if utils.checkempty(candidate_info):
        return candidate_info

    # Transpose the structure
    out_candidate_info = [[] for _ in candidate_info[0]]
    for arr in candidate_info:
        for ind, entry in enumerate(arr):
            out_candidate_info[ind].append(entry)

    # Concatenate and return
    out_candidate_info = [np.concatenate(x, axis=0) for x in out_candidate_info]

    return out_candidate_info


def collect_all_candidate_files_npy(
        dir_path, collect_after_veto=True, collect_before_veto=False,
        collect_rerun=False, collect_timeseries=False, ncores=1):
    """
    :param dir_path:
    :param collect_after_veto:
    :param collect_before_veto:
    :param collect_rerun: Flag indicating whether to collect rerun files
    :param collect_timeseries:
        Flag whether to collect timeseries (assumes we collected both
        before/after in the run)
    :param ncores: Number of cores to use for collection
    :return:
    """
    candidates_before = []
    candidates_after = []
    timeseries_before = []

    # Get list of filenames
    if collect_before_veto or collect_timeseries:
        # In O2 run, cand8/9 has only vetoed timeseries, but this
        # isn't true moving forward, when we will save vetoed-non-vetoed
        # timeseries so we will make it more general
        files_before_veto = glob.glob(os.path.join(dir_path, "*before*.npy"))
        if len(files_before_veto) > 0:
            print("Reading in candidates before vetoes were applied")
            candidates_before = collect_files(files_before_veto, ncores=ncores)

    if collect_after_veto:
        # Get list of filenames
        if collect_rerun:
            files_after_veto = \
                [x for x in glob.glob(os.path.join(dir_path, "*.npy"))
                 if "before" not in x and "old" not in x
                 and "timeseries" not in x]
        else:
            files_after_veto = \
                [x for x in glob.glob(os.path.join(dir_path, "*.npy"))
                 if "before" not in x and "new" not in x
                 and 'timeseries' not in x]

        if len(files_after_veto) > 0:
            print("Reading in candidates after vetoes were applied")
            candidates_after = collect_files(files_after_veto, ncores=ncores)

    if collect_timeseries:
        timeseries_files = glob.glob(os.path.join(dir_path, "*timeseries.npy"))
        if len(timeseries_files) > 0:
            print("Reading in timeseries files")
            timeseries_before = collect_files(timeseries_files, ncores=ncores)

    candidates_after, candidates_before, timeseries_after, timeseries_before = \
        combine_files(candidates_after, candidates_before, timeseries_before)

    if collect_after_veto and collect_before_veto:
        if collect_timeseries:
            return candidates_after, candidates_before, timeseries_after, \
                timeseries_before
        else:
            return candidates_after, candidates_before
    elif collect_after_veto:
        if collect_timeseries:
            return candidates_after, timeseries_after
        else:
            return candidates_after
    elif collect_before_veto:
        if collect_timeseries:
            return candidates_before, timeseries_before
        else:
            return candidates_before
    else:
        return


def collect_all_candidate_files_npz(
        dir_path, collect_before_veto=True,
        collect_rerun=True, collect_timeseries=False, detectors=("H1", "L1"),
        ncores=1):
    """
    :param dir_path:
    :param collect_before_veto:
        Flag whether to return results before vetoes. If False, we will only
        return the vetoed candidates
    :param collect_rerun: Flag indicating whether to collect rerun files
    :param collect_timeseries: Flag whether to collect timeseries
    :param detectors:
        Tuple with names of the two detectors we will be loading results for
    :param ncores: Number of cores to use for collection
    :return:
        events, veto_metadata, coherent_scores, veto_metadata_keys
        (+ timeseries if collect_timeseries)
    """
    # Get list of filenames
    master_filelist = glob.glob(
        os.path.join(dir_path, "*" + "_".join(detectors) + "*.npz"))
    if collect_rerun:
        files = [x for x in master_filelist
                 if "before" not in x and "old" not in x]
    else:
        files = [x for x in master_filelist
                 if "before" not in x and "new" not in x]

    assert len(files) > 0, "What are you doing reading an empty directory?"

    print("Reading in candidates")
    candidate_info = collect_files_npz(
        files, ncores=ncores, collect_timeseries=collect_timeseries)
    
    out_list = combine_files_npz(
        candidate_info, apply_vetoes=np.logical_not(collect_before_veto))

    assert len(out_list) > 0, "Not a single candidate survived your cuts!"

    # Read in names of glitch tests
    i = 0
    while os.path.getsize(files[i]) < 10:
        i += 1
    data = np.load(files[i], allow_pickle=True)
    veto_metadata_keys = data["metadata_keys"]

    out_list.insert(3, veto_metadata_keys)

    return out_list


if __name__ == "__main__":
    main()
    exit()




#tmp file names: #chosen to have perfect analysis
#/data/bzackay/GW/OutputDir/Output_Tue_Aug_28_04_37/L-L1_LOSC_4_V1-1135689728-4096.trig.npy
#/data/bzackay/GW/OutputDir/Output_Tue_Aug_28_04_37/L-L1_LOSC_4_V1-1136861184-4096.trig.npy
#/data/bzackay/GW/OutputDir/Output_Tue_Aug_28_04_37/L-L1_LOSC_4_V1-1136459776-4096.trig.npy
#/data/bzackay/GW/OutputDir/Output_Tue_Aug_28_04_37/L-L1_LOSC_4_V1-1135882240-4096.trig.npy
#/data/bzackay/GW/OutputDir/Output_Tue_Aug_28_04_37/L-L1_LOSC_4_V1-1134632960-4096.trig.npy
#/data/bzackay/GW/OutputDir/Output_Tue_Aug_28_04_37/L-L1_LOSC_4_V1-1134301184-4096.trig.npy
