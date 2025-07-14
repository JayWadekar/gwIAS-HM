#!/usr/bin/env python3
"""
Updated coincidence_HM.py that uses the new modular, N-detector architecture.

This is a drop-in replacement for the original coincidence_HM.py that:
1. Uses the new modular structure (coincidence_core, coincidence_veto)
2. Supports N detectors instead of being hardcoded for 2
3. Maintains backward compatibility with existing command-line interface
4. Provides the same outputs as the original version
"""

import os
import sys
import numpy as np
from argparse import ArgumentParser
import getpass
import json
import time

# Import the new modular components
from coincidence_core import find_interesting_dir as find_interesting_dir_new
from coincidence_veto import veto_and_optimize_coincidence_list
import triggers_single_detector_HM as trig
import utils
import params

# Default paths (same as original)
DEFAULT_PROGRAM_NAME = 'coincidence_HM_new.py'
DEFAULT_PROGPATH = os.path.join(utils.CODE_DIR, DEFAULT_PROGRAM_NAME)
TMP_FILENAME = "tmp_submit_script_coincidence_new.sh"
DEFAULT_TMP_PATH = os.path.join(utils.DATA_ROOT, TMP_FILENAME)


def create_candidate_output_dir_name(run_path, cver, run="O2"):
    """Create output directory name for candidates."""
    run_subdir = os.path.basename(run_path)
    cand_dir_name = os.path.join(
        utils.CAND_DIR[run.lower()], run_subdir + f"cand{cver}")
    return cand_dir_name


def find_interesting_dir_cluster(
        dir_name, detectors=('H1', 'L1'), n_cores=1, cluster_n_cores=1,
        cluster_run_time="06:00:00", cluster_memory="16GB", 
        cluster_queue="normal", cluster_system="slurm", **kwargs):
    """
    Submit coincidence analysis jobs to cluster for N detectors.
    
    :param dir_name: Directory containing trigger files
    :param detectors: List of detector names (e.g., ['H1', 'L1', 'V1'])
    :param n_cores: Number of cores per job
    :param cluster_n_cores: Number of cores to request from cluster
    :param cluster_run_time: Wall time limit
    :param cluster_memory: Memory limit
    :param cluster_queue: Queue name
    :param cluster_system: Cluster system ("slurm" or "sge")
    :param kwargs: Additional parameters
    :return: None (submits jobs)
    """
    
    print(f"Submitting coincidence analysis for {len(detectors)} detectors: {detectors}")
    
    # Create job script
    script_content = f"""#!/bin/bash
#SBATCH --job-name=coincidence_{len(detectors)}det
#SBATCH --output=coincidence_{len(detectors)}det_%j.out
#SBATCH --error=coincidence_{len(detectors)}det_%j.err
#SBATCH --time={cluster_run_time}
#SBATCH --mem={cluster_memory}
#SBATCH --cpus-per-task={cluster_n_cores}
#SBATCH --partition={cluster_queue}

# Load environment
module load python/3.8
source /path/to/venv/bin/activate

# Run coincidence analysis
python {DEFAULT_PROGPATH} "{dir_name}" --detectors {' '.join(detectors)} --n_cores {n_cores} {' '.join([f'--{k} {v}' for k, v in kwargs.items()])}
"""
    
    # Write script to file
    with open(DEFAULT_TMP_PATH, 'w') as f:
        f.write(script_content)
    
    # Submit job
    if cluster_system.lower() == "slurm":
        os.system(f"sbatch {DEFAULT_TMP_PATH}")
    elif cluster_system.lower() == "sge":
        # Convert to SGE format
        sge_script = script_content.replace("#SBATCH", "#$")
        sge_script = sge_script.replace("--job-name=", "-N ")
        sge_script = sge_script.replace("--output=", "-o ")
        sge_script = sge_script.replace("--error=", "-e ")
        sge_script = sge_script.replace("--time=", "-l h_rt=")
        sge_script = sge_script.replace("--mem=", "-l h_vmem=")
        sge_script = sge_script.replace("--cpus-per-task=", "-pe smp ")
        sge_script = sge_script.replace("--partition=", "-q ")
        
        with open(DEFAULT_TMP_PATH, 'w') as f:
            f.write(sge_script)
        
        os.system(f"qsub {DEFAULT_TMP_PATH}")
    else:
        raise ValueError(f"Unsupported cluster system: {cluster_system}")
    
    print(f"Job submitted successfully for {len(detectors)}-detector analysis")


def find_interesting_dir(
        dir_name, detectors=('H1', 'L1'), 
        time_shift_tol=None, threshold_chi2=None, minimal_time_slide_jump=None,
        veto_triggers=True, min_veto_chi2=32, apply_threshold=True,
        origin=0, n_cores=1, opt_format="new", output_timeseries=True,
        output_coherent_score=True, score_reduction_timeseries=10,
        score_reduction_max=5, score_func=utils.incoherent_score,
        save_output=True, output_dir=None, **kwargs):
    """
    Main function for N-detector coincidence analysis.
    
    This function serves as a wrapper around the new modular implementation,
    providing the same interface as the original coincidence_HM.py.
    
    :param dir_name: Directory containing trigger files
    :param detectors: List of detector names (e.g., ['H1', 'L1', 'V1'])
    :param time_shift_tol: Time tolerance for bucketing
    :param threshold_chi2: Chi2 threshold for filtering
    :param minimal_time_slide_jump: Minimum time slide jump
    :param veto_triggers: Whether to apply veto tests
    :param min_veto_chi2: Minimum chi2 for veto application
    :param apply_threshold: Whether to apply threshold filtering
    :param origin: Time origin for bucketing
    :param n_cores: Number of cores for parallel processing
    :param opt_format: Optimization format ("new" or "old")
    :param output_timeseries: Whether to output timeseries
    :param output_coherent_score: Whether to output coherent scores
    :param score_reduction_timeseries: Score reduction for timeseries
    :param score_reduction_max: Maximum score reduction
    :param score_func: Scoring function to use
    :param save_output: Whether to save output files
    :param output_dir: Directory to save output files
    :param kwargs: Additional parameters
    :return: Output filename or processing status
    """
    
    print(f"Starting {len(detectors)}-detector coincidence analysis")
    print(f"Detectors: {detectors}")
    print(f"Directory: {dir_name}")
    
    # Set default parameters if not provided
    if time_shift_tol is None:
        time_shift_tol = params.TIME_SHIFT_TOL
    if threshold_chi2 is None:
        threshold_chi2 = params.THRESHOLD_CHI2
    if minimal_time_slide_jump is None:
        minimal_time_slide_jump = params.MINIMAL_TIME_SLIDE_JUMP
    
    # Use the new modular implementation
    result = find_interesting_dir_new(
        dir_name=dir_name,
        detectors=list(detectors),
        time_shift_tol=time_shift_tol,
        threshold_chi2=threshold_chi2,
        minimal_time_slide_jump=minimal_time_slide_jump,
        veto_triggers=veto_triggers,
        min_veto_chi2=min_veto_chi2,
        apply_threshold=apply_threshold,
        origin=origin,
        n_cores=n_cores,
        opt_format=opt_format,
        output_timeseries=output_timeseries,
        output_coherent_score=output_coherent_score,
        score_reduction_timeseries=score_reduction_timeseries,
        score_reduction_max=score_reduction_max,
        score_func=score_func,
        **kwargs
    )
    
    # Save output if requested
    if save_output and result is not None:
        if output_dir is None:
            output_dir = dir_name
        
        # Generate output filename
        detector_string = "_".join(detectors)
        timestamp = int(time.time())
        output_filename = os.path.join(
            output_dir, f"coincident_candidates_{detector_string}_{timestamp}.npy")
        
        try:
            np.save(output_filename, result)
            print(f"Results saved to: {output_filename}")
            return output_filename
        except Exception as e:
            print(f"Error saving results: {e}")
            return None
    
    return result


def main():
    """Main function for command-line interface."""
    
    parser = ArgumentParser(
        description="N-detector gravitational wave coincidence analysis",
        epilog="This is the updated version supporting arbitrary numbers of detectors.")
    
    parser.add_argument("dir_name", help="Directory containing trigger files")
    parser.add_argument("--detectors", nargs='+', default=['H1', 'L1'],
                       help="List of detector names (e.g., H1 L1 V1)")
    parser.add_argument("--cluster", action='store_true',
                       help="Submit to cluster instead of running locally")
    parser.add_argument("--n_cores", type=int, default=1,
                       help="Number of cores for parallel processing")
    parser.add_argument("--cluster_n_cores", type=int, default=1,
                       help="Number of cores to request from cluster")
    parser.add_argument("--cluster_run_time", default="06:00:00",
                       help="Wall time limit for cluster job")
    parser.add_argument("--cluster_memory", default="16GB",
                       help="Memory limit for cluster job")
    parser.add_argument("--cluster_queue", default="normal",
                       help="Queue name for cluster submission")
    parser.add_argument("--cluster_system", default="slurm",
                       choices=["slurm", "sge"],
                       help="Cluster system type")
    parser.add_argument("--time_shift_tol", type=float,
                       help="Time tolerance for bucketing")
    parser.add_argument("--threshold_chi2", type=float,
                       help="Chi2 threshold for filtering")
    parser.add_argument("--minimal_time_slide_jump", type=float,
                       help="Minimum time slide jump")
    parser.add_argument("--no_veto", action='store_true',
                       help="Disable veto tests")
    parser.add_argument("--min_veto_chi2", type=float, default=32,
                       help="Minimum chi2 for veto application")
    parser.add_argument("--no_threshold", action='store_true',
                       help="Disable threshold filtering")
    parser.add_argument("--origin", type=float, default=0,
                       help="Time origin for bucketing")
    parser.add_argument("--opt_format", default="new",
                       choices=["new", "old"],
                       help="Optimization format")
    parser.add_argument("--no_timeseries", action='store_true',
                       help="Disable timeseries output")
    parser.add_argument("--no_coherent_score", action='store_true',
                       help="Disable coherent score output")
    parser.add_argument("--score_reduction_timeseries", type=float, default=10,
                       help="Score reduction for timeseries")
    parser.add_argument("--score_reduction_max", type=float, default=5,
                       help="Maximum score reduction")
    parser.add_argument("--output_dir", 
                       help="Directory to save output files")
    parser.add_argument("--no_save", action='store_true',
                       help="Don't save output files")
    
    args = parser.parse_args()
    
    # Prepare arguments
    kwargs = {
        'detectors': tuple(args.detectors),
        'n_cores': args.n_cores,
        'veto_triggers': not args.no_veto,
        'min_veto_chi2': args.min_veto_chi2,
        'apply_threshold': not args.no_threshold,
        'origin': args.origin,
        'opt_format': args.opt_format,
        'output_timeseries': not args.no_timeseries,
        'output_coherent_score': not args.no_coherent_score,
        'score_reduction_timeseries': args.score_reduction_timeseries,
        'score_reduction_max': args.score_reduction_max,
        'save_output': not args.no_save,
        'output_dir': args.output_dir
    }
    
    # Add optional parameters if provided
    if args.time_shift_tol is not None:
        kwargs['time_shift_tol'] = args.time_shift_tol
    if args.threshold_chi2 is not None:
        kwargs['threshold_chi2'] = args.threshold_chi2
    if args.minimal_time_slide_jump is not None:
        kwargs['minimal_time_slide_jump'] = args.minimal_time_slide_jump
    
    # Run analysis
    if args.cluster:
        find_interesting_dir_cluster(
            args.dir_name,
            cluster_n_cores=args.cluster_n_cores,
            cluster_run_time=args.cluster_run_time,
            cluster_memory=args.cluster_memory,
            cluster_queue=args.cluster_queue,
            cluster_system=args.cluster_system,
            **kwargs
        )
    else:
        result = find_interesting_dir(args.dir_name, **kwargs)
        if result is None:
            print("Analysis completed with no output")
        else:
            print(f"Analysis completed successfully")


if __name__ == "__main__":
    main()