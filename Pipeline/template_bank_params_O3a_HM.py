"""
Choices of template bank parameters for the search with higher modes
The name of this file has O3a but these parameters were used in O3a and O3b
The hyperparameters given below were generated using
the notebook: 1.Template_banks.ipynb
"""
import numpy as np
import utils
import os
# import template_bank_generator_HM as tg

run = 'O3_hm'

# ------------------------------------------------------------------------------
# Directory where the template bank files are located
DIR = os.path.join(utils.TEMPLATE_DIR, run, 'Multibanks')

# Path to the ASD file
# (currently using the O3a file)
asd_filepath = os.path.join(utils.SMALL_DATA_DIR, 'asd_o3a.npy')

# directory where generated waveforms will be stored
wf_DIR = os.path.join(utils.TEMPLATE_DIR, run, 'wf_reservoir')

# directory where the template prior will be stored
template_prior_DIR = os.path.join(utils.TEMPLATE_DIR, run, 'Template_prior')

# ------------------------------------------------------------------------------
# Hyperparameters associated with template banks

# Total number of banks
nbanks = 17

# Number of sub-banks in each bank
nsubbanks = {
'BBH_0': 5,
 'BBH_1': 3,
 'BBH_2': 3,
 'BBH_3': 3,
 'BBH_4': 3,
 'BBH_5': 3,
 'BBH_6': 3,
 'BBH_7': 2,
 'BBH_8': 2,
 'BBH_9': 1,
 'BBH_10': 1,
 'BBH_11': 1,
 'BBH_12': 1,
 'BBH_13': 1,
 'BBH_14': 1,
 'BBH_15': 1,
 'BBH_16': 1}

# Flag used in grid_range function in
# template_bank_generator_hm.py to center the calpha grid
force_zero = True

# Following will be changed later
mb_keys = [
    'BBH_0', 
    'BBH_1', 
    'BBH_2', 
    'BBH_3',
    'BBH_4',
    'BBH_5', 
    'BBH_6', 
    'BBH_7', 
    'BBH_8',
    'BBH_9',
    'BBH_10', 
    'BBH_11', 
    'BBH_12', 
    'BBH_13',
    'BBH_14',
    'BBH_15',
    'BBH_16',
    ]

all_mb_keys = mb_keys

delta_calpha = {
 'BBH_0': 0.55,
 'BBH_1': 0.5,
 'BBH_2': 0.45,
 'BBH_3': 0.45,
 'BBH_4': 0.4,
 'BBH_5': 0.35,
 'BBH_6': 0.3,
 'BBH_7': 0.25,
 'BBH_8': 0.25,
 'BBH_9': 0.35,
 'BBH_10': 0.3,
 'BBH_11': 0.3,
 'BBH_12': 0.3,
 'BBH_13': 0.3,
 'BBH_14': 0.3,
 'BBH_15': 0.25,
 'BBH_16': 0.2}

mb_dirs = {x: os.path.join(DIR, x+'/') for x in all_mb_keys}

# Fudge is just kept for backward compatibility
# and is not currently being used in our search
fudge = {
 'BBH_0': 1.05,
 'BBH_1': 1.05,
 'BBH_2': 1.05,
 'BBH_3': 1.05,
 'BBH_4': 1.05,
 'BBH_5': 1.05,
 'BBH_6': 1.05,
 'BBH_7': 1.05,
 'BBH_8': 1.05,
 'BBH_9': 1.05,
 'BBH_10': 1.05,
 'BBH_11': 1.05,
 'BBH_12': 1.05,
 'BBH_13': 1.05,
 'BBH_14': 1.05,
 'BBH_15': 1.05,
 'BBH_16': 1.05}

# Mchirp bins for each subbank
# Convention: use subbank_mask = (mchirp_bank>bin[0])*(mchirp_bank<=bin[1])
# Now these are being saved in the template banks so do not need to be included here

# SubBank_Mchirp_bins = {'BBH_0': [[2.63, 5.27],
#   [5.27, 6.76],
#   [6.76, 8.38],
#   [8.38, 10.65],
#   [10.65, 19.85]],
#  'BBH_1': [[5.81, 11.86], [11.86, 15.79], [15.79, 27.29]],
#  'BBH_2': [[8.92, 18.52], [18.52, 23.04], [23.04, 38.58]],
#  'BBH_3': [[11.97, 22.42], [22.42, 27.35], [27.35, 49.63]],
#  'BBH_4': [[5.82, 10.77], [10.77, 13.47], [13.47, 19.5]],
#  'BBH_5': [[14.7, 23.91], [23.91, 30.28], [30.28, 64.18]],
#  'BBH_6': [[10.34, 21.03], [21.03, 29.47], [29.47, 75.94]],
#  'BBH_7': [[12.87, 30.59], [30.59, 92.36]],
#  'BBH_8': [[15.09, 39.88], [39.88, 108.26]],
#  'BBH_9': [[18.5, 127.33]],
#  'BBH_10': [[21.06, 149.11]],
#  'BBH_11': [[24.65, 168.6]],
#  'BBH_12': [[28.37, 173.53]],
#  'BBH_13': [[32.57, 173.76]],
#  'BBH_14': [[37.85, 173.56]],
#  'BBH_15': [[43.17, 173.69]],
#  'BBH_16': [[51.81, 165.98]]}

# input_wf_dirs = {x: os.path.join(DIR, x + '_input_wfs/')
#                  for x in all_mb_keys}
# test_wf_dirs = {x: os.path.join(DIR, x + '_test_wfs/')
#                 for x in all_mb_keys}
# coverage_wf_dirs = {x: os.path.join(DIR, x + '_coverage_wfs/')
#                     for x in all_mb_keys}
# 
# # for effectualness testing with precession and HM
# approximants_aligned = {x: 'IMRPhenomD' for x in mb_keys_BBH}
# approximants_aligned_HM = {x: 'IMRPhenomHM' for x in mb_keys_BBH}
# approximants_precessing = {x: 'IMRPhenomPv2' for x in mb_keys_BBH}
# approximants_precessing_HM = {x: 'IMRPhenomXPHM' for x in mb_keys_BBH}
# 
# test_wf_dirs_aligned = {x: os.path.join(DIR, x + '_test_wfs_aligned/')
#                         for x in mb_keys_BBH}
# test_wf_dirs_aligned_HM = {x: os.path.join(DIR, x + '_test_wfs_aligned_HM/')
#                            for x in mb_keys_BBH}
# test_wf_dirs_precessing = {x: os.path.join(DIR, x + '_test_wfs_precessing/')
#                            for x in mb_keys_BBH}
# test_wf_dirs_precessing_HM = {x: os.path.join(DIR, x + '_test_wfs_precessing_HM/')
#                               for x in mb_keys_BBH}
# test_wf_dirs_aligned_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_aligned_angles/') for x in mb_keys_BBH}
# test_wf_dirs_aligned_HM_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_aligned_HM_angles/') for x in mb_keys_BBH}
# test_wf_dirs_precessing_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_precessing_angles/') for x in mb_keys_BBH}
# test_wf_dirs_precessing_HM_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_precessing_HM_angles/') for x in mb_keys_BBH}


# mcrng = {
#     'BNS_0': (0, 1.1),
#     'BNS_1': (1.1, 1.3),
#     'BNS_2': (1.3, np.inf),
#     'NSBH_0': (0, 3),
#     'NSBH_1': (3, 6),
#     'NSBH_2': (6, np.inf),
#     'BBH_0': (0, 5),
#     'BBH_1': (5, 10),
#     'BBH_2': (10, 20),
#     'BBH_3': (20, 40),
#     'BBH_4': (40, np.inf),
#     'BBH_5': (20, 200), 'BBH_6': (20, 200)
# }