# For triggering on the data, this code calls gw_detect_file.py
# For a tutorial on running this code, refer to the notebook
# 4.Trig_Coin_on_cluster.ipynb

import os
import getpass
import glob
import numpy as np
import utils
import json
from json import JSONDecodeError
import sys
import time
import warnings

import params

TMP_FILENAME = 'tmp_submit_script.sh'
DEFAULT_JOBNAME = ''
DEFAULT_PROG = 'gw_detect_file.py'
DEFAULT_PROG_CHECK = 'triggering_on_cluster.py'

DEFAULT_TMP_PATH = os.path.join(utils.DATA_ROOT, TMP_FILENAME)
DEFAULT_PROGPATH = os.path.join(utils.CODE_DIR, DEFAULT_PROG)
DEFAULT_PROGPATH_CHECK = os.path.join(utils.CODE_DIR, DEFAULT_PROG_CHECK)

BBH_KEYS = tuple(f'BBH_{i}' for i in range(5)[::-1])
ALL_BBH_KEYS = tuple(f'BBH_{i}' for i in range(7)[::-1])
DEFAULT_FFTLOG2SIZE = int(np.round(np.log2(params.DEF_FFTSIZE)))


def get_strain_filelist(run, detector="H1", trim_empty=True, filelim=None):
    root = utils.STRAIN_ROOT[run.lower()]
    if run.lower() == "o1":
        if filelim is None:
            if detector == "H1":
                filelim = 1.299e8
            elif detector == "L1":
                filelim = 1.24e8
    elif run.lower() in ["o2", "o2new"]:
        if filelim is None:
            filelim = 1000

    detfiles = glob.glob(os.path.join(root, detector, "*"))

    if trim_empty and (filelim is not None):
        detfiles = [f for f in detfiles if os.path.getsize(f) > filelim]

    return detfiles


def get_problematic(dir):
    '''
    Checks what files in the directory are excessively large
    '''
    # Note: Seems hardcoded for the O1 run
    fnames = glob.glob(os.path.join(dir, "*.trig*"))
    sizes = [os.path.getsize(f) for f in fnames]
    med_size = np.median(sizes)
    bad_files = [f.split('.trig')[0].split('/')[-1] + '.hdf5' for f in fnames if os.path.getsize(f)>3*med_size]
    output_list = []
    for f in bad_files:
        if "H1" in f:
            f_dir = "/data/bzackay/GW/H1/"
        if "L1" in f:
            f_dir = "/data/bzackay/GW/L1/"
        output_list.append(f_dir + f)
    return output_list


# Functions to deal with interrupted runs 
# ---------------------------------------
def inspect_completion(fnames, output_dir):
    """
    Inspects the current status of files in the output directory. 
    Useful for checking which files failed, ran partially, etc.
    
    :param fnames: List of filenames for strain data
                    (e.g., '../H-H1_GWOSC_O3a_4KHZ_R1-1238355968-4096.hdf5')
    :param output_dir: Directory where the output files were generated
    :return: List with entries
        0:unprocessed_files, 1:untriggered_files, 2:partial_files
        3:failed_files, 4:indeterminate_files, 5:completed_files
        
        unprocessed_files: List of files that haven't been preprocessed
        untriggered_files: List of files that have been preprocessed 
                            but not triggered
        partial_files: Dictionary of files that have been partially triggered
                        (key: filename, value: number of bankchunks done)
        failed_files: List of files that have failed
        indeterminate_files: List of files that have an unknown status
        completed_files: List of files that have been completely triggered
    """
    unprocessed_files = []
    untriggered_files = []
    partial_files = {}
    failed_files = []
    indeterminate_files = []
    completed_files = []

    max_nbankchunks = 0
    for fname in fnames:
        output_fname, error_fname, error_fname_bak, trig_fname, \
            config_fname, preprocessing_fname = filelist(fname, output_dir)

        if not os.path.isfile(preprocessing_fname):
            # Check if error_fname exists (helios doesn't make one)
            if os.path.isfile(error_fname):
                file_to_check = error_fname
            elif os.path.isfile(output_fname):
                file_to_check = output_fname
            else:
                file_to_check = None
                
            if file_to_check is None:
                # We just haven't reached here for preprocessing
                unprocessed_files.append(fname)
            else:
                with open(file_to_check, 'r') as f:
                    if "error" in f.read().lower():
                        failed_files.append(fname)
                    else:
                        indeterminate_files.append(fname)
                
        elif not os.path.isfile(trig_fname):
            # We have preprocessed, but haven't generated triggers
            untriggered_files.append(fname)

        elif not os.path.isfile(config_fname):
            # There was likely an error while writing the trig file
            indeterminate_files.append(fname)

        else:
            # We have preprocessed, and generated triggers, and
            # written a config file
            with open(config_fname, 'r') as fp:
                try:
                    dic = json.load(
                        fp, object_hook=utils.TupleEncoder.tuple_in_hook)
                    nbankchunks = dic.get('nbankchunks', 1)
                    nbankchunks_done = dic.get('nbankchunks_done', 1)

                    max_nbankchunks = max(nbankchunks, max_nbankchunks)

                    if nbankchunks_done < nbankchunks:
                        partial_files.update({fname: nbankchunks_done})
                    else:
                        completed_files.append(fname)

                except JSONDecodeError:
                    # Unreadable json file, count this as indeterminate
                    indeterminate_files.append(fname)

    # Print results and return list of unstarted and partial files to restart from
    print(f"Reporting run statistics for {len(fnames)} files")
    print(f"Divided bank into {max_nbankchunks} chunks for generating triggers")
    print(f"Didn't start preprocessing for {len(unprocessed_files)} files")
    print(f"Only finished preprocessing for {len(untriggered_files)} files")
    print(f"Partially generated triggers for {len(partial_files)} files")
    print(f"Failed on {len(failed_files)} files")
    print(f"Couldn't determine fate for {len(indeterminate_files)} files")
    print(f"Finished generating triggers for {len(completed_files)} files")

    return unprocessed_files, untriggered_files, partial_files, failed_files, \
        indeterminate_files, completed_files


def filelist(fname, output_dir):
    output_fname = os.path.join(
        output_dir, fname.split('/')[-1].split('.h')[0] + ".out")
    error_fname = os.path.join(
        output_dir, fname.split('/')[-1].split('.h')[0] + ".err")
    error_fname_bak = os.path.join(
        output_dir, fname.split('/')[-1].split('.h')[0] + ".err.bak")
    trig_fname = os.path.join(
        output_dir, fname.split('/')[-1].split('.h')[0] + ".trig.npy")
    config_fname = os.path.join(
        output_dir, fname.split('/')[-1].split('.h')[0] + "_config.json")
    preprocessing_fname = os.path.join(
        output_dir, fname.split('/')[-1].split('.h')[0] + ".npz")
    
    return output_fname, error_fname, error_fname_bak, trig_fname, \
        config_fname, preprocessing_fname


def clean_file(fname, output_dir, clean_preproc=False):
    fs = filelist(fname, output_dir)
    if not clean_preproc:
        fs = fs[:-1]
    for f in fs:
        if os.path.isfile(f):
            os.system(f"rm {f}")
    return


def check_file(fname, output_dir, duplicate_check=True, verbose=False,
               delete_file=True):
    # Warning, destructive to incomplete files, caution if we want to
    # preserve old runs
    # Checks for six kinds of errors:
    # 1. Files with an unloadable preprocessing file
    # 2. Config files without a trig file
    # 3. Unreadable config files
    # 4. Trig files without a config file
    # 5. Unreadable trig files
    # 6. Trig files with duplicate triggers (memory intensive)
    if verbose:
        print("Checking file:", fname)
    
    output_fname, error_fname, error_fname_bak, trig_fname, \
        config_fname, preproc_fname = filelist(fname, output_dir)
    
    # Check preprocessing file
    if os.path.isfile(preproc_fname):
        try:
            preprocfile = np.load(preproc_fname)
        except:
            # Unreadable preprocessing file
            print(f"Unreadable preprocessing file for {fname}")
            clean_file(fname, output_dir, clean_preproc=True)
            return
        
    # Check valid json files
    if os.path.isfile(config_fname):
        # This is destructive to incomplete files, reconsider?
        if not os.path.isfile(trig_fname):
            # json without accompanying trig file
            print(f"Missing trig file for {fname}")
            clean_file(fname, output_dir)
            return

        try:
            with open(config_fname, 'r') as fp:
                dic = json.load(
                    fp, object_hook=utils.TupleEncoder.tuple_in_hook)
        except:
            # Unreadable json file
            print(f"Unreadable json file for {fname}")
            clean_file(fname, output_dir)
            return

    # Check valid trig files
    if os.path.isfile(trig_fname):
        if not os.path.isfile(config_fname):
            # trig file without accompanying json file
            print(f"Missing config file for {fname}")
            clean_file(fname, output_dir)
            return

        try:
            triglist = np.load(trig_fname)
            if duplicate_check:
                if (triglist.dtype != np.dtype('O')) and (len(triglist) > 0):
                    myset = {tuple(row) for row in triglist}
                    if len(myset) < len(triglist):
                        triglist_nodupl = np.vstack(myset)
                        print(f"Repeated triggers found. Removing duplicates!")
                        np.save(trig_fname, triglist_nodupl)
                        os.system(f"chmod 666 {trig_fname}")
        except:
            # Unreadable trig file
            print(f"Unreadable trig file for {fname}")
            clean_file(fname, output_dir)
            return

    return


def check_files_server(files, output_dir, helios=True, duplicate_check=True, mem_limit=8):
    """Warning: doesn't work on Hyperion due to lack to bash commands!"""
    if not os.path.isdir(output_dir):
        print("Directory not found!")
        return
    
    for fname in files:
        output_check = os.path.join(
            output_dir, "check_" + fname.split('/')[-1].split('.h')[0] + ".out")
        os.system(f"touch {output_check}")
        os.system(f"chmod 666 {output_check}")
        
        if helios:
            mem_submit = int(mem_limit) * 1000
            text = f'#!/bin/bash \n#SBATCH --job-name=check_run\n'
            text += f'#SBATCH --output={output_check}\n'
            text += f'#SBATCH --open-mode=append\n'
            text += f'#SBATCH --nodes=1\n'
            text += f'#SBATCH --mem={mem_submit}\n'
            text += f'#SBATCH --time=01:00:00\n'

            text += utils.env_init_lines(cluster="helios")
        
            text += f" {DEFAULT_PROGPATH_CHECK} {fname} {output_dir} {duplicate_check}"

            cur_tmp_file_name = DEFAULT_TMP_PATH[:-2] + str(np.random.randint(0,2**31)) + '_helios.sh'
            with open(cur_tmp_file_name, 'w') as file:
                file.write(text)
            os.system(f"chmod 777 {cur_tmp_file_name}")  # add user run permissions
            os.system(f'sbatch {cur_tmp_file_name}')
            time.sleep(0.1)
            os.remove(cur_tmp_file_name)
        else:
            # Running on hyperion
            error_check = os.path.join(
                output_dir, "check_" + fname.split('/')[-1].split('.h')[0] + ".err")
            os.system(f"touch {error_check}")
            os.system(f"chmod 666 {error_check}")
            
            text = f'#!/bin/bash\n'
            text += f'#$ -cwd\n'
            text += f'#$ -w n\n'
            text += f'#$ -N check_run\n'
            text += f'#$ -o {output_check}\n'
            text += f'#$ -e {error_check}\n'
            text += f'#$ -V\n'
            text += f'#$ -pe smp 1\n'
            text += f'#$ -l h_rt=01:00:00\n'
            text += f'#$ -l h_vmem={int(mem_limit)}G\n'
            
            text += 'source activate conda-lal\n'
            
            text += f"python {DEFAULT_PROGPATH_CHECK} {fname} {output_dir} {duplicate_check}"
            
            with open(DEFAULT_TMP_PATH, 'w') as file:
                file.write(text)
            os.system(f"chmod 777 {DEFAULT_TMP_PATH}")  # add user run permissions
            os.system(f'qsub {DEFAULT_TMP_PATH}')
            os.remove(DEFAULT_TMP_PATH)
        
    return


def finish_checks(files, output_dir, helios=True, resubmit_server=False, 
                  resubmit_local=False, duplicate_check=False, mem_limit=8):
    """Warning: doesn't work on Hyperion due to lack of bash commands!"""
    # Some of the files in check_files_server can be missed due to queueing errors
    error_files = []
    for fname in files:
        output_check = os.path.join(
            output_dir, "check_" + fname.split('/')[-1].split('.h')[0] + ".out")
        error_check = os.path.join(
                output_dir, "check_" + fname.split('/')[-1].split('.h')[0] + ".err")
        if helios:
            file_to_check = output_check
        else:
            # Check was run on hyperion
            file_to_check = error_check
        if os.path.isfile(file_to_check):
            with open(file_to_check, "r") as f:
                if "error" in f.read():
                    error_files.append(fname)
                    
    print(f"Need to go over {len(error_files)} files due to queueing errors")
    if resubmit_server:
        # Clean up old check files
        for fname in files:
            output_check = os.path.join(
                output_dir, "check_" + fname.split('/')[-1].split('.h')[0] + ".out")
            error_check = os.path.join(
                output_dir, "check_" + fname.split('/')[-1].split('.h')[0] + ".err")
            if os.path.isfile(output_check):
                os.remove(output_check)
            if os.path.isfile(error_check):
                os.remove(error_check)
        if len(error_files) > 0:
            check_files_server(error_files, output_dir, helios=helios, 
                               duplicate_check=duplicate_check, mem_limit=mem_limit)
    elif resubmit_local:
        for ind, fname in enumerate(error_files):
            print(f"Checking file {ind}: {fname}")
            check_file(fname, output_dir, duplicate_check=duplicate_check)
    else:
        pass
    return error_files


# Functions to submit runs on clusters
# ------------------------------------
def create_output_dir_name(template_conf, base_path=None, run_str='O2'):
    """
    Function to create an output directory name for a subbank in a run.
    The name carries information about when the analysis was launched.
    :param template_conf: Path to the metadata.json file for the subbank
    :param base_path:
        Path to the root directory within which the output directories for all
        subbanks will be located. If None, we will use the default for the run
        specified in utils.py
    :param run_str: String identifying the run (e.g., "O2", "O3a")
    :return: Path to the output directory
    """
    if base_path is None:
        base_path = utils.OUTPUT_DIR[run_str.lower()]
    str_time = "_".join("_".join(time.ctime().split()[:4]).split(":")[:2])
    output_path = os.path.join(
        base_path,
        run_str + "_" +
        str_time + '_'.join(template_conf.split('templates/')[-1].split('/')[:-1]))
    return output_path


def submit_multibanks(
        tbp_to_use, files_to_submit=None, bank_ids=BBH_KEYS, cluster="typhon",
        observing_run='O3a', test_few=None, output_dirs=None, submit=True,
        save_hole_correction=True,
        preserve_max_snr=params.DEF_PRESERVE_MAX_SNR,
        fmax=params.FMAX_OVERLAP, n_cores=None, n_hours_limit=24,
        fftlog2size=DEFAULT_FFTLOG2SIZE, njobchunks=1,
        trim_empty=False, queue_name=None, env_command=None,
        use_HM=False, mem_per_cpu=4, exclude_nodes=False):
    """
    Submits multibanks to the cluster
    :param tbp_to_use: Template bank parameter object to use
    :param files_to_submit: If desired, list of files to run
    :param bank_ids:
        List of bank_ids, keys to dictionaries in template_bank_params....py
    :param cluster: Name of cluster we are submitting jobs to
    :param observing_run: If files_to_submit is None,
        we use the name of observing run (can be "O1", "O1new", "O2",
                          "O3a", or "O3b")
    :param test_few: If desired, number of files to submit as a test
    :param output_dirs:
        If we are rerunning files, list of lists of output directories
        (each entry a directory for a subbank) with [n_bank_ids x n_subbanks],
        return value of utils.get_dirs.
        One can also specify individual sub-banks
        has recently been added (dirnames should end with subbank #)
    :param submit: Flag whether to submit to the cluster
    :param save_hole_correction:
        Flag whether to save hole correction, exposed here to rerun files
        in run "O1"
    :param preserve_max_snr:
        Exposed here since it changed between "O1" and subsequent runs
    :param fmax:
        Exposed here since it changes between BBH-0 and the rest of the banks
    :param n_cores:
        Number of cores to use for each file, we use sensible values if
        not provided
    :param n_hours_limit:
        Number of hours to limit the job to.
        If n_hours_limit<=24, we may get into a faster queue in the cluster
    :param fftlog2size: log_2(fftsize) to use for submission
    :param njobchunks:
        Number of chunks to split the job into, useful if we want to save
        trigger files in between and restart jobs
        (useful when n_hours_limit<=24)
    :param trim_empty: Trim apparently empty files
    :param queue_name: name of queue (used only in WEXAC submission)
    :param env_command:
            If required, pass the command to activate the conda environment
            If None, will infer from the username if it's in utils
    :param use_HM:
            Boolean flag indicating whether we want to use 33 and 44 modes alongside 22
    :param mem_per_cpu: Memory per CPU in GB. 4GB/core is the default in Typhon
    :param exclude_nodes: Flag to exclude nodes from the job submission
                        (if other people start complaining about cluster use)
    :return:
    """
    if use_HM:
        import template_bank_generator_HM as tg_HM
    else:
        import template_bank_generator as tg

    if cluster.lower() == "typhon":
        submit_func = submit_files_typhon
    elif cluster.lower() == "hyperion":
        submit_func = submit_files_hyperion
    elif cluster.lower() == "helios":
        submit_func = submit_files_helios
    elif cluster.lower() == 'wexac':
        submit_func = submit_files_wexac
    else:
        raise RuntimeError(f"Cluster {cluster} unknown!")

    overall_running_files = [[] for _ in bank_ids]
    subbanks_to_use = [[] for _ in bank_ids]

    for multi_bank_idx, multi_bank_id in enumerate(bank_ids):
        #multi_bank_dir =  template_root + f"{multi_bank_id}_multibank/"
        multi_bank_dir = tbp_to_use.mb_dirs[multi_bank_id]
        delta_calpha = tbp_to_use.delta_calpha[multi_bank_id]
        fudge = tbp_to_use.fudge[multi_bank_id]
        force_zero = tbp_to_use.force_zero

        subbanks = glob.glob(multi_bank_dir + "bank_*")
        n_subbanks = len(subbanks)

        # # We made some crazy choices in the O1 run
        # if observing_run == 'O1':
        #     base_thresholds_chi2 = tbp_to_use.base_threshold_chi2[multi_bank_id]
        #     thresholds_chi2 = tbp_to_use.threshold_chi2[multi_bank_id]

        # These thresholds can be modified based on the number of templates in the bank
        # TODO: In the future, try to write continuous function for the thresholds
        # based on the number of templates (rather than discrete jumps)
        if use_HM: 
            total_n_templates = 0
            for isb in range(tbp_to_use.nsubbanks[multi_bank_id]):
                bank =  tg_HM.TemplateBank.from_json(
                            os.path.join(tbp_to_use.DIR,multi_bank_id,
                            'bank_'+str(isb),'metadata.json'))
                total_n_templates += bank.ntemplates(
                    delta_calpha, fudge=fudge, force_zero=force_zero)
                
            base_threshold_chi2 = 23  
            threshold_chi2 = 26
            # HM values increased by 4 to compensate for
            # the larger number of triggers
            # HM thresholds could be lowered even further in future analyses
        else: 
            multi_bank = tg.MultiBank.from_json(multi_bank_dir + 'metadata.json')
            total_n_templates = multi_bank.ntemplates(
                delta_calpha, fudge=fudge, force_zero=force_zero)
            
            base_threshold_chi2 = 20
            threshold_chi2 = 23
        
        if total_n_templates < 100:
            if use_HM: base_threshold_chi2 = 17; threshold_chi2 = 20
            else: base_threshold_chi2 = 13; threshold_chi2 = 16        
        
        elif total_n_templates < 500:
            if use_HM: base_threshold_chi2 = 19; threshold_chi2 = 22
            else: base_threshold_chi2 = 15; threshold_chi2 = 18
                
        elif total_n_templates < 4000:
            if use_HM: base_threshold_chi2 = 21; threshold_chi2 = 24
            else: base_threshold_chi2 = 17; threshold_chi2 = 20

        print(f'''We have {total_n_templates} templates in the bank,
        so will use threshold_chi2 before and after sinc-interp 
        as {base_threshold_chi2} and {threshold_chi2} respectively.''')
            
        base_thresholds_chi2 = [base_threshold_chi2] * n_subbanks
        thresholds_chi2 = [threshold_chi2] * n_subbanks

        subbanks_to_use[multi_bank_idx] = list(np.arange(n_subbanks))
        if output_dirs is not None:
            if len(output_dirs[multi_bank_idx]) < n_subbanks:
                subbanks_to_use[multi_bank_idx] = [int(i.split('_')[-1])for i in \
                                     output_dirs[multi_bank_idx]]
        print(f'''We will use the following subbanks: {subbanks_to_use[multi_bank_idx]}''')
                   
        for subbank_idx, subbank_id in enumerate(subbanks_to_use[multi_bank_idx]):
            subbank_path = multi_bank_dir + f"bank_{subbank_id}/metadata.json"

            if n_cores is None:
                if use_HM: subbank = tg_HM.TemplateBank.from_json(subbank_path)
                else: subbank = tg.TemplateBank.from_json(subbank_path)
                n_cores = 2
                ntemp = subbank.ntemplates(
                    delta_calpha, fudge, force_zero=force_zero)
                if use_HM: ntemp *= 3 # accounting for additional time due to HM templates
                if ntemp > 4000:
                    if cluster.lower() == "typhon":
                        n_cores = np.min([24, int(np.ceil((ntemp-2000)/1000))])
                        # set minimum hours to get into the queue
                        # in the cluster with best priority
                    elif cluster.lower() == "hyperion":
                        n_cores = np.min([16, int(np.ceil((ntemp-2000)/1000))])
                    elif cluster.lower() == "helios":
                        n_cores = np.min([28, int(np.ceil((ntemp-2000)/1000))])
                    elif cluster.lower() == 'wexac':
                        n_cores = np.min([8, int(np.ceil((ntemp - 2000) / 1000))])
                    else:
                        raise RuntimeError(f"Cluster {cluster} unknown!")

            if files_to_submit is None:
                if observing_run.lower() == "o1":
                    detectors = ["H1", "L1"]
                else:
                    detectors = ["H1", "L1", "V1"]
                files_to_submit = []
                for detector in detectors:
                    files_to_submit += get_strain_filelist(
                        observing_run, detector=detector, trim_empty=trim_empty)

            if output_dirs is None:
                output_dir = None
            else:
                output_dirs_multibank = output_dirs[multi_bank_idx]
                # We might not have run a bank
                if len(output_dirs_multibank) > 0:
                    output_dir = output_dirs_multibank[subbank_idx]
                else:
                    output_dir = None

            if test_few is None:
                finished_files, running_files, text = submit_func(
                    files_to_submit, output_dir=output_dir,
                    template_conf=subbank_path, delta_calpha=delta_calpha,
                    template_safety=fudge, force_zero=force_zero,
                    base_threshold_chi2=base_thresholds_chi2[subbank_id],
                    threshold_chi2=thresholds_chi2[subbank_id],
                    save_hole_correction=save_hole_correction,
                    preserve_max_snr=preserve_max_snr,
                    fmax=fmax, job_name=DEFAULT_JOBNAME + multi_bank_id,
                    ncores=n_cores, n_hours_limit=n_hours_limit,
                    fftlog2size=fftlog2size, njobchunks=njobchunks,
                    run_str=observing_run, submit=submit,
                    queue_name=queue_name, env_command=env_command, 
                    use_HM=use_HM, mem_per_cpu=mem_per_cpu)
            else:
                subset_of_files = np.random.choice(files_to_submit, test_few)
                finished_files, running_files, text = submit_func(
                    subset_of_files, output_dir=output_dir,
                    template_conf=subbank_path, delta_calpha=delta_calpha,
                    template_safety=fudge, force_zero=force_zero,
                    base_threshold_chi2=base_thresholds_chi2[subbank_id],
                    threshold_chi2=thresholds_chi2[subbank_id],
                    save_hole_correction=save_hole_correction,
                    preserve_max_snr=preserve_max_snr,
                    fmax=fmax, job_name=DEFAULT_JOBNAME + multi_bank_id,
                    ncores=n_cores, n_hours_limit=n_hours_limit,
                    fftlog2size=fftlog2size, njobchunks=njobchunks,
                    run_str=observing_run, submit=submit,
                    queue_name=queue_name, env_command=env_command,
                    exclude_nodes=exclude_nodes,
                    use_HM=use_HM, mem_per_cpu=mem_per_cpu)

            overall_running_files[multi_bank_idx].append(running_files.copy())

    for multi_bank_idx, multi_bank_id in enumerate(bank_ids):
        for subbank_idx, running_files_subbank in enumerate(
                overall_running_files[multi_bank_idx]):
            print(f"Running for {len(running_files_subbank)} files in " +
                  f"subbank {subbanks_to_use[multi_bank_idx][subbank_idx]} "+
                  f"of bank {multi_bank_id}")

    return overall_running_files


def submit_files_typhon(
        fnames, mem_per_cpu=4, n_hours_limit=24,
        output_dir=None, template_conf=None, delta_calpha=0.5,
        template_safety=1.05, remove_nonphysical=True, force_zero=True,
        base_threshold_chi2=16, threshold_chi2=25, save_hole_correction=True,
        preserve_max_snr=params.DEF_PRESERVE_MAX_SNR, fmax=params.FMAX_OVERLAP,
        prog_path=DEFAULT_PROGPATH, job_name=DEFAULT_JOBNAME, ncores=24,
        fftlog2size=20, n_waveforms_per_core=-1,
        njobchunks=1, run_str='O2', check_completion=True, submit=False,
        env_command=None, use_HM=False, exclude_nodes=False,
         **kwargs):

    if template_conf is None:
        raise Exception("Template Configuration must be given!")

    if output_dir is None:
        # Create output directory if it doesn't already exist
        output_dir = create_output_dir_name(
            template_conf=template_conf, run_str=run_str)

    if submit:

        # Create the output_dir if needed
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            os.system(f"chmod 777 {output_dir}")
        
        # Before doing anything, save params.py in output_dir to keep a record
        dest = os.path.join(output_dir, 'params.py')
        os.system(f"cp params.py {dest}")
        os.system(f"chmod 777 {dest}")

    finished_files = []
    running_files = []
    text = ""
    for fname in fnames:
        # Create the associated filenames
        output_fname, error_fname, error_fname_bak, trig_fname, config_fname, \
            preprocessing_fname = filelist(fname, output_dir)

        # Check if we already completed the file, in which case, skip it
        # Written so that it works with old runs
        if check_completion and os.path.isfile(config_fname):
            with open(config_fname, 'r') as fp:
                dic = json.load(
                    fp, object_hook=utils.TupleEncoder.tuple_in_hook)
                # Works if we didn't save in old runs
                nbankchunks = dic.get('nbankchunks', 1)
                nbankchunks_done = dic.get('nbankchunks_done', 1)
            if nbankchunks <= nbankchunks_done:
                finished_files.append(fname)
                continue

        # We're sending this file in
        running_files.append(fname)

        if submit:
            os.system(f"touch {output_fname}")
            os.system(f"chmod 666 {output_fname}")
            # Backup old error file if present, useful to save grepping
            if os.path.isfile(error_fname):
                os.system(f"mv {error_fname} {error_fname_bak}")
                os.system(f"chmod 666 {error_fname_bak}")

        text = f'#!/bin/bash \n#SBATCH --job-name={job_name}\n'
        text += f'#SBATCH --output={output_fname}\n'
        text += f'#SBATCH --open-mode=append\n'
        text += f'#SBATCH --nodes=1\n'
        text += f'#SBATCH --cpus-per-task={ncores}\n'
        text += f'#SBATCH --time={int(n_hours_limit)}:00:00\n'
        text += f'#SBATCH --mem-per-cpu={mem_per_cpu}G\n'
        if exclude_nodes:
            text += f'#SBATCH --exclude=typhon-node[1-10]\n'

        text += utils.env_init_lines(env_command=env_command, cluster="typhon")

        text += f" {prog_path} {fname} {preprocessing_fname}" + \
                f" {template_conf} {trig_fname} {config_fname}"
        text += f" --delta_calpha={delta_calpha}" + \
                f" --template_safety={template_safety}" + \
                f" --threshold_chi2={threshold_chi2}" + \
                f" --base_threshold_chi2={base_threshold_chi2}" + \
                f" --preserve_max_snr={preserve_max_snr}" + \
                f" --fmax={fmax}" + \
                f" --ncores={ncores}" + \
                f" --njobchunks={njobchunks}"

        if remove_nonphysical:
            text += " --remove_nonphysical"
        if force_zero:
            text += " --force_zero"
        if save_hole_correction:
            text += " --save_hole_correction"
        if use_HM:
            text += " --use_HM"

        text += f" --fftlog2size={fftlog2size}"
        text += f" --n_waveforms_per_core={n_waveforms_per_core}\n"

        print(text)

        if submit:
            cur_tmp_file_name = DEFAULT_TMP_PATH[:-2] + \
                                str(np.random.randint(0,2**31)) + '_typhon.sh'
            with open(cur_tmp_file_name, 'w') as file:
                file.write(text)
            print("changing its permissions")
            os.system(f"chmod 777 {cur_tmp_file_name}")  # add user run permissions
            print("sending jobs")
            os.system(f'sbatch {cur_tmp_file_name}')
            time.sleep(0.1)
            print(f"removing config file {cur_tmp_file_name}")
            os.remove(cur_tmp_file_name)

    return finished_files, running_files, text

def submit_files_helios(
        fnames, output_dir=None, template_conf=None, delta_calpha=0.7,
        template_safety=1.1, remove_nonphysical=True, force_zero=True,
        base_threshold_chi2=16, threshold_chi2=25, save_hole_correction=True,
        preserve_max_snr=params.DEF_PRESERVE_MAX_SNR, fmax=params.FMAX_OVERLAP,
        prog_path=DEFAULT_PROGPATH, job_name=DEFAULT_JOBNAME, ncores=28,
        fftlog2size=20, n_hours_limit=24, n_waveforms_per_core=-1,
        njobchunks=1, run_str='O2', check_completion=True, submit=False,
        env_command=None, use_HM=False, mem_per_cpu=4, **kwargs):

    if template_conf is None:
        raise Exception("Template Configuration must be given!")

    if output_dir is None:
        # Create output directory if it doesn't already exist
        output_dir = create_output_dir_name(
            template_conf=template_conf, run_str=run_str)

    if submit:
        # # In the beginning, even the root output directory doesn't exist
        # # In this case, create it
        # default_outputdir = utils.OUTPUT_DIR[run_str.lower()]
        # if not os.path.isdir(default_outputdir):
        #     os.makedirs(default_outputdir)
        #     os.system(f"chmod 777 {default_outputdir}")

        # Create the output_dir if needed
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            os.system(f"chmod 777 {output_dir}")
        
        # Before doing anything, save params.py in output_dir to keep a record
        dest = os.path.join(output_dir, 'params.py')
        os.system(f"cp params.py {dest}")
        os.system(f"chmod 777 {dest}")

    finished_files = []
    running_files = []
    text = ""
    for fname in fnames:
        # Create the associated filenames
        output_fname, error_fname, error_fname_bak, trig_fname, config_fname, \
            preprocessing_fname = filelist(fname, output_dir)

        # Check if we already completed the file, in which case, skip it
        # Written so that it works with old runs
        if check_completion and os.path.isfile(config_fname):
            with open(config_fname, 'r') as fp:
                dic = json.load(
                    fp, object_hook=utils.TupleEncoder.tuple_in_hook)
                # Works if we didn't save in old runs
                nbankchunks = dic.get('nbankchunks', 1)
                nbankchunks_done = dic.get('nbankchunks_done', 1)
            if nbankchunks <= nbankchunks_done:
                finished_files.append(fname)
                continue

        # We're sending this file in
        running_files.append(fname)

        if submit:
            os.system(f"touch {output_fname}")
            os.system(f"chmod 666 {output_fname}")
            # Backup old error file if present, useful to save grepping
            if os.path.isfile(error_fname):
                os.system(f"mv {error_fname} {error_fname_bak}")
                os.system(f"chmod 666 {error_fname_bak}")

        text = f'#!/bin/bash \n#SBATCH --job-name={job_name}\n'
        text += f'#SBATCH --output={output_fname}\n'
        text += f'#SBATCH --open-mode=append\n'
        text += f'#SBATCH --nodes=1\n'
        text += f'#SBATCH --cpus-per-task={ncores}\n'
        text += f'#SBATCH --time={int(n_hours_limit)}:00:00\n'
        text += f'#SBATCH --mem-per-cpu={mem_per_cpu}G\n'

        text += utils.env_init_lines(env_command=env_command, cluster="helios")

        text += f" {prog_path} {fname} {preprocessing_fname}" + \
                f" {template_conf} {trig_fname} {config_fname}"
        text += f" --delta_calpha={delta_calpha}" + \
                f" --template_safety={template_safety}" + \
                f" --threshold_chi2={threshold_chi2}" + \
                f" --base_threshold_chi2={base_threshold_chi2}" + \
                f" --preserve_max_snr={preserve_max_snr}" + \
                f" --fmax={fmax}" + \
                f" --ncores={ncores}" + \
                f" --njobchunks={njobchunks}"

        if remove_nonphysical:
            text += " --remove_nonphysical"
        if force_zero:
            text += " --force_zero"
        if save_hole_correction:
            text += " --save_hole_correction"
        if use_HM:
            text += " --use_HM"

        text += f" --fftlog2size={fftlog2size}"
        text += f" --n_waveforms_per_core={n_waveforms_per_core}\n"

        print(text)

        if submit:
            cur_tmp_file_name = DEFAULT_TMP_PATH[:-2] \
                                + str(np.random.randint(0,2**31)) + '_helios.sh'
            with open(cur_tmp_file_name, 'w') as file:
                file.write(text)
            print("changing its permissions")
            os.system(f"chmod 777 {cur_tmp_file_name}")  # add user run permissions
            print("sending jobs")
            os.system(f'sbatch {cur_tmp_file_name}')
            time.sleep(0.1)
            print(f"removing config file {cur_tmp_file_name}")
            os.remove(cur_tmp_file_name)

    return finished_files, running_files, text
    

def submit_files_hyperion(
        fnames, output_dir=None, template_conf=None, delta_calpha=0.7,
        template_safety=1.1, remove_nonphysical=True, force_zero=True,
        base_threshold_chi2=16, threshold_chi2=25, save_hole_correction=True,
        preserve_max_snr=params.DEF_PRESERVE_MAX_SNR, fmax=params.FMAX_OVERLAP,
        prog_path=DEFAULT_PROGPATH, job_name=DEFAULT_JOBNAME, ncores=8,
        fftlog2size=20, n_hours_limit=8, n_waveforms_per_core=-1,
        njobchunks=1, run_str='O2', check_completion=True, submit=False,
        exclusive=False, **kwargs):
    """Hyperion doesn't have as much memory, so suggest using exclusive"""

    if template_conf is None:
        raise Exception("Template Configuration must be given!")

    if output_dir is None:
        # Create output directory if it doesn't already exist
        output_dir = create_output_dir_name(template_conf=template_conf, run_str=run_str)

        # Sometimes even the default output dir does not exist. In these cases, create it
        default_outputdir = utils.OUTPUT_DIR[run_str.lower()]
        if not os.path.isdir(default_outputdir):
            os.makedirs(default_outputdir)
            os.system(f"chmod 777 {default_outputdir}")

    # If user has passed in an output_dir, create it if needed
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.system(f"chmod 777 {output_dir}")
        
    finished_files = []
    running_files = []
    for fname in fnames:
        output_fname = os.path.join(
            output_dir, fname.split('/')[-1].split('.h')[0] + ".out")
        error_fname = os.path.join(
            output_dir, fname.split('/')[-1].split('.h')[0] + ".err")
        preprocessing_fname = os.path.join(
            output_dir, fname.split('/')[-1].split('.h')[0] + ".npz")
        trig_fname = os.path.join(
            output_dir, fname.split('/')[-1].split('.h')[0] + ".trig")
        config_fname = os.path.join(
            output_dir, fname.split('/')[-1].split('.h')[0] + "_config.json")

        # Check if we already completed the file, in which case, skip it
        # Written so that it works with old runs
        if check_completion and os.path.isfile(config_fname):
            with open(config_fname, 'r') as fp:
                dic = json.load(
                    fp, object_hook=utils.TupleEncoder.tuple_in_hook)
                nbankchunks = dic.get('nbankchunks', 1)
                nbankchunks_done = dic.get('nbankchunks_done', 1)
            if nbankchunks <= nbankchunks_done:
                finished_files.append(fname)
                continue

        # We're sending this file in
        running_files.append(fname)
        if submit:
            os.system(f"touch {output_fname}")
            os.system(f"chmod 666 {output_fname}")
            # Backup old error file if present, useful to save grepping
            if os.path.isfile(error_fname):
                error_fname_bak = error_fname + ".bak"
                os.system(f"mv {error_fname} {error_fname_bak}")
                os.system(f"chmod 666 {error_fname_bak}")
            os.system(f"touch {error_fname}")
            os.system(f"chmod 666 {error_fname}")

        text = f'#!/bin/bash\n'
        text += f'#$ -cwd\n'
        text += f'#$ -w n\n'
        text += f'#$ -N {job_name}\n'
        text += f'#$ -o {output_fname}\n'
        text += f'#$ -e {error_fname}\n'
        text += f'#$ -V\n'
        text += f'#$ -pe smp {ncores}\n'
        text += f'#$ -l h_rt={int(n_hours_limit)}:00:00\n'
        # TODO: Add something about the memory cap...
        
        python_name = 'python'
        if getpass.getuser() == 'tejaswi':
            text += f"source activate conda-lal\n"
        if getpass.getuser() == 'bzackay':
            #text += f"module load anaconda3\n"
            python_name = '/home/bzackay/anaconda3/bin/python'            
            
        text += f"{python_name} {prog_path} {fname} {preprocessing_fname}" + \
                f" {template_conf} {trig_fname} {config_fname}"

        text += f" --delta_calpha={delta_calpha}" + \
                f" --template_safety={template_safety}" + \
                f" --threshold_chi2={threshold_chi2}" + \
                f" --base_threshold_chi2={base_threshold_chi2}" + \
                f" --preserve_max_snr={preserve_max_snr}" + \
                f" --fmax={fmax}" + \
                f" --ncores={ncores}" + \
                f" --njobchunks={njobchunks}"

        if remove_nonphysical:
            text += " --remove_nonphysical"
        if force_zero:
            text += " --force_zero"
        if save_hole_correction:
            text += " --save_hole_correction"

        text += f" --fftlog2size={fftlog2size}"
        text += f" --n_waveforms_per_core={n_waveforms_per_core}\n"

        print(text)

        if submit:
            cur_tmp_file_name = DEFAULT_TMP_PATH[:-2] + str(np.random.randint(0,2**31)) + '_hyperion.sh'
            with open(cur_tmp_file_name, 'w') as file:
                file.write(text)
            print("changing its permissions")    
            os.system(f"chmod 777 {cur_tmp_file_name}")  # add user run permissions
            print("sending jobs")
            if exclusive:
                os.system(f'qsub -l excl=true {cur_tmp_file_name}')
            else:
                os.system(f'qsub {cur_tmp_file_name}')
            time.sleep(0.1)
            print(f"removing config file {cur_tmp_file_name}")
            os.remove(cur_tmp_file_name)

    return finished_files, running_files, text


def submit_files_wexac(
        fnames, output_dir=None, template_conf=None, delta_calpha=0.7,
        template_safety=1.1, remove_nonphysical=True, force_zero=True,
        base_threshold_chi2=16, threshold_chi2=25, save_hole_correction=True,
        preserve_max_snr=params.DEF_PRESERVE_MAX_SNR, fmax=params.FMAX_OVERLAP,
        prog_path=DEFAULT_PROGPATH, job_name=DEFAULT_JOBNAME, ncores=28,
        fftlog2size=20, n_hours_limit=23, n_waveforms_per_core=-1,
        njobchunks=1, run_str='O2', check_completion=True, submit=False,
        queue_name=None, verbose=False, **kwargs):
    if template_conf is None:
        raise Exception("Template Configuration must be given!")

    if output_dir is None:
        # Create output directory if it doesn't already exist
        output_dir = create_output_dir_name(
            template_conf=template_conf, run_str=run_str)

    if queue_name is None:
        queue_name = 'new-short'

    if submit:
        # Sometimes even the default output dir does not exist
        # In these cases, create it
        default_outputdir = utils.OUTPUT_DIR[run_str.lower()]
        if not os.path.isdir(default_outputdir):
            os.makedirs(default_outputdir)
            os.system(f"chmod 777 {default_outputdir}")

        # Create the output_dir if needed
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            os.system(f"chmod 777 {output_dir}")

        # Before doing anything, save params in output_dir to keep record
        dest = os.path.join(output_dir, 'params.py')
        os.system(f"cp params.py {dest}")
        os.system(f"chmod 777 {dest}")

    finished_files = []
    running_files = []
    text = ""
    for fname in fnames:
        try:
            # Create filenames
            output_fname = os.path.join(
                output_dir, fname.split('/')[-1].split('.h')[0] + ".out")
            error_fname = os.path.join(
                output_dir, fname.split('/')[-1].split('.h')[0] + ".err")
            preprocessing_fname = os.path.join(
                output_dir, fname.split('/')[-1].split('.h')[0] + ".npz")
            trig_fname = os.path.join(
                output_dir, fname.split('/')[-1].split('.h')[0] + ".trig")
            config_fname = os.path.join(
                output_dir, fname.split('/')[-1].split('.h')[0] + "_config.json")

            # Check if we already completed the file, in which case, skip it
            # Written so that it works with old runs
            if check_completion and os.path.isfile(config_fname):
                with open(config_fname, 'r') as fp:
                    dic = json.load(
                        fp, object_hook=utils.TupleEncoder.tuple_in_hook)
                    nbankchunks = dic.get('nbankchunks', 1)
                    nbankchunks_done = dic.get('nbankchunks_done', 1)
                if nbankchunks <= nbankchunks_done:
                    finished_files.append(fname)
                    continue

            # We're sending this file in
            running_files.append(fname)

            if submit:
                os.system(f"touch {output_fname}")
                os.system(f"chmod 660 {output_fname}")
                # Backup old error file if present, useful to save grepping
                if os.path.isfile(error_fname):
                    error_fname_bak = error_fname + ".bak"
                    os.system(f"mv {error_fname} {error_fname_bak}")
                    os.system(f"chmod 660 {error_fname_bak}")

            text =  f' -q {queue_name} '
            text += f'-J {job_name} '
            text += f'-o {output_fname} '
            text += f'-e {error_fname} '
            #text += f'#SBATCH --open-mode=append\n'
            text += f'-n {ncores} '
            mem = 4000
            if mem * ncores < 12000:
                mem = int(12000/ncores)
            text += f'-R "span[hosts=1] rusage[mem={mem}MB]" '
            #text += f'#BSUB parallelJob\n'
            text += f'-W {int(n_hours_limit)}:00 '

            python_name = '/home/labs/barakz/barakz//anaconda3/envs/gwias/bin/python'

            text += f"{python_name} {prog_path} {fname} {preprocessing_fname}" + \
                    f" {template_conf} {trig_fname} {config_fname}"
            text += f" --delta_calpha={delta_calpha}" + \
                    f" --template_safety={template_safety}" + \
                    f" --threshold_chi2={threshold_chi2}" + \
                    f" --base_threshold_chi2={base_threshold_chi2}" + \
                    f" --preserve_max_snr={preserve_max_snr}" + \
                    f" --fmax={fmax}" + \
                    f" --ncores={ncores}" + \
                    f" --njobchunks={njobchunks}"

            if remove_nonphysical:
                text += " --remove_nonphysical"
            if force_zero:
                text += " --force_zero"
            if save_hole_correction:
                text += " --save_hole_correction"

            text += f" --fftlog2size={fftlog2size}"
            text += f" --n_waveforms_per_core={n_waveforms_per_core}"

            if verbose:
                print(text)

            if submit:
                #cur_tmp_file_name = TMP_FILENAME[:-2] + str(np.random.randint(0,
                # 2 ** 31)) + '_helios.sh'
                #with open(cur_tmp_file_name, 'w') as file:
                #    file.write(text)
                #print("changing its permissions")
                #os.system(f"chmod 777 {cur_tmp_file_name}")  # add user run
                # permissions
                print("sending jobs")
                os.system(f'bsub {text}')
                time.sleep(0.1)
                #print(f"removing config file {cur_tmp_file_name}")
                #os.remove(cur_tmp_file_name)
        except:
            print("Failed on file: ", fname)
    return finished_files, running_files, text


if __name__ == "__main__":
    # Only called by processes spawned by check_files_helios
    nargs = len(sys.argv)
    if nargs < 4:
        sys.exit("I need a filename, a directory, and a flag to check duplicates")
    filename = sys.argv[1]
    dirname = sys.argv[2]
    dupl_check = (sys.argv[3] == 'True')
    # Check this file
    check_file(filename, dirname, duplicate_check=dupl_check)
    print(f"Finished checking {filename}")
    sys.exit(0)
