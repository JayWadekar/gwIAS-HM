#!/usr/bin/env python
from argparse import ArgumentParser
import os
import json
import params
import utils


def main():
    parser = ArgumentParser(description='Find triggers in strain files ' +
                                        'in a directory')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Be verbose')

    # File names
    parser.add_argument('data_fname', type=str,
                        help='Absolute path to strain data (.hdf5 format)')
    parser.add_argument('preprocessing_fname', type=str,
                        help='Absolute path to file to save preprocessing ' +
                             'data to, or load from')
    parser.add_argument('template_conf', type=str,
                        help='Absolute path to template metadata')
    parser.add_argument('trig_fname', type=str,
                        help='Absolute path to file to save triggers to. ' +
                             'Will override if already exists')
    parser.add_argument('config_fname', type=str,
                        help='Absolute path to save the configuration ' +
                             'parameters to reproduce the trigger file')

    # Parameters of analysis
    parser.add_argument('--delta_calpha', type=float, default=0.7,
                        help='Spacing between adjacent bins in each basis ' +
                             'dimension of the templates')
    parser.add_argument('--template_safety', type=float, default=1.1,
                        help='Factor to inflate calpha ranges by')
    parser.add_argument('--remove_nonphysical', action='store_true',
                        help='Flag indicating whether we want to keep only ' +
                             'the gridpoints that are close to a physical ' +
                             'waveform')
    parser.add_argument('--force_zero', action='store_true',
                        help='Flag indicating whether we want to center the ' +
                             'template grid to get as much overlap as ' +
                             'possible with the central region')
    parser.add_argument('--save_hole_correction', action='store_true',
                        help='Flag indicating whether we want to save hole ' +
                             'corrections, not used in old O1 run')
    parser.add_argument('--use_HM', action='store_true',
                        help='Flag indicating whether we want to use' +
                             '33 and 44 modes alongside 22'   )

    parser.add_argument('--threshold_chi2', type=float, default=25,
                        help='Threshold of the normed overlaps from which to ' +
                             'save the triggers ')
    parser.add_argument('--base_threshold_chi2', type=float, default=16,
                        help='Threshold of the normed overlaps above which ' +
                             'to sinc interpolate the triggers ')
    parser.add_argument('--preserve_max_snr', type=float,
                        default=params.DEF_PRESERVE_MAX_SNR,
                        help='Used to set thresholds for glitch tests, ' +
                             'exposed here to rerun old O1 analysis')
    parser.add_argument('--fmax', type=float,
                        default=params.FMAX_OVERLAP,
                        help='Used to set FMAX for calculating overlaps ' +
                             'of template with data and (2x value used for PSD)')

    parser.add_argument('--ncores', type=int, default=24,
                        help='Number of cores to be used in the process')
    parser.add_argument('--njobchunks', type=int, default=1,
                        help='Number of chunks to split job into for saving ' +
                             'intermediate results')
    parser.add_argument('--fftlog2size', type=int, default=20,
                        help='log2 of the fftsize to be used with ' +
                             'overlap and save')
    parser.add_argument('--n_waveforms_per_core', type=int, default=-1,
                        help="option to limit the number of templates to " +
                             "test all the preprocessing " +
                             "default (-1) would exhaust the template bank")

    # parser.add_argument('--log_fname', type=str,
    # default ='/tmp/default_log_file_run_gw_detect.log',
    # help = "Log file to print progress")

    args = parser.parse_args()
    
    if args.use_HM:
        import triggers_single_detector_HM as tgd_HM
        tgd_in_use = tgd_HM
    else:
        import triggers_single_detector as tgd
        tgd_in_use = tgd

    # If we're restarting, don't redo preprocessing
    if os.path.isfile(args.preprocessing_fname):
        print(f"Restarting analysis of {args.data_fname} from saved " +
              "preprocessing")
        load_data = save_preprocessing = False
    else:
        print(f"DATA:{args.data_fname}, TEMPLATE: {args.template_conf}, " +
              f"FFTLOG2SIZE:{args.fftlog2size}")
        load_data = save_preprocessing = True

    left_path, right_path = utils.get_left_right_fnames(args.data_fname)

    if os.path.isfile(args.config_fname):
        # We already generated some triggers, start from where we left off
        # Don't load the previous triggers during computation to save on memory
        # config json file contains all information needed to load the whitened 
        # data and prepare for triggers
        triglist = tgd_in_use.TriggerList.from_json(
            args.config_fname, load_trigs=False, load_data=False,
            save_hole_correction=args.save_hole_correction)
    else:
        # If the file with preprocessed data exists, start generating triggers,
        # else start from scratch and preprocess the data
        # Print the arguments before running, useful when debugging
        print(f"Running TriggerList with the following arguments:")
        print(f"fname={args.data_fname}, "
              f"left_fname={left_path}, "
              f"right_fname={right_path}, "
              f"fname_preprocessing={args.preprocessing_fname}, "
              f"load_data={load_data}, "
              f"save_preprocessing={save_preprocessing}, "
              f"template_conf={args.template_conf}, "
              f"fftsize={2 ** args.fftlog2size}, "
              f"preserve_max_snr={args.preserve_max_snr}, "
              f"fmax={args.fmax}, "
              f"save_hole_correction={args.save_hole_correction}")
        triglist = tgd_in_use.TriggerList(
            fname=args.data_fname, left_fname=left_path, right_fname=right_path,
            fname_preprocessing=args.preprocessing_fname, load_data=load_data,
            save_preprocessing=save_preprocessing,
            template_conf=args.template_conf, fftsize=2 ** args.fftlog2size,
            preserve_max_snr=args.preserve_max_snr, fmax=args.fmax,
            save_hole_correction=args.save_hole_correction)

    if load_data:
        print("Finished initialization")

    print(f"Generating triggers: delta_calpha={args.delta_calpha}, " +
          f"threshold_chi2={args.threshold_chi2}, " +
          f"base_threshold_chi2={args.base_threshold_chi2}, " +
          f"ncores={args.ncores}, " +
          f"trig_fname={args.trig_fname}, config_fname={args.config_fname}")

    if args.n_waveforms_per_core < 0:
        triglist.gen_triggers(delta_calpha=args.delta_calpha,
                              template_safety=args.template_safety,
                              remove_nonphysical=args.remove_nonphysical,
                              force_zero=args.force_zero,
                              threshold_chi2=args.threshold_chi2,
                              base_threshold_chi2=args.base_threshold_chi2,
                              trig_fname=args.trig_fname,
                              config_fname=args.config_fname,
                              ncores=args.ncores,
                              njobchunks=args.njobchunks)
    else:
        print("enumerating over a limited number of waveforms")
        triglist.gen_triggers(delta_calpha=args.delta_calpha,
                              template_safety=args.template_safety,
                              remove_nonphysical=args.remove_nonphysical,
                              force_zero=args.force_zero,
                              threshold_chi2=args.threshold_chi2,
                              base_threshold_chi2=args.base_threshold_chi2,
                              trig_fname=args.trig_fname,
                              config_fname=args.config_fname,
                              ncores=args.ncores,
                              njobchunks=args.njobchunks,
                              n_wf_to_digest=args.n_waveforms_per_core)

    print("Process finished successfully")
    return


if __name__ == "__main__":
    main()
    exit()


pass
