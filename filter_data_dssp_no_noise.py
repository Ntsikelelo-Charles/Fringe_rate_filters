import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pyuvdata import UVData
import hera_cal as hc
import uvtools as uvt
import hera_pspec as hp
import copy

Model_complete_file = '/net/ike/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Model_data_complete.uvh5'
Model_complete = hc.frf.FRFilter(Model_complete_file)
freqs = Model_complete.freqs/1e6
times = (Model_complete.times-Model_complete .times.min()) * 24 * 3600  # seconds

uvd = UVData()
uvd.read(Model_complete_file)
uvd.conjugate_bls()
uvd.inflate_by_redundancy()
F_model = hc.frf.FRFilter(uvd)
filte_width_text=np.array(['0.25e-3','0.40e-3','0.60e-3'])
for fil in range(3):
    filter_max=0
    if filte_width_text[fil]=='0.25e-3':
        filter_max=25
    if filte_width_text[fil]=='0.40e-3':
        filter_max=40
    if filte_width_text[fil]=='0.60e-3':
        filter_max=60        
    # units are fringe-rates [Hz]
    filter_center     = [0.0]         
    filter_half_width = [float(filte_width_text[fil])]
    print(filter_max)

    # unitless
    filter_factor     = [1e-8]

    # make covariance
    C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)


    # take inverse to get filter matrix
    R = np.linalg.pinv(C, rcond=1e-10)




    raw_file = '/net/jake/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_data_no_noise.uvh5'
    raw = hc.frf.FRFilter(raw_file)
    freqs = raw.freqs/1e6
    times = (raw.times-raw .times.min()) * 24 * 3600  # seconds

    uvd = UVData()
    uvd.read(raw_file)
    uvd.conjugate_bls()
    F_model = hc.frf.FRFilter(uvd)



    # filter the data!
    raw_filt_data = copy.deepcopy(F_model.data)
    for k in raw_filt_data:
        raw_filt_data[k] = R @ raw_filt_data[k]

    print("raw data filter with half_width = "+str(filter_half_width)+" and filter center = "+str(filter_center))    

    F_model.write_data(raw_filt_data,'/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Raw_filtered_data_no_noise_'+str(filter_max)+'.uvh5',overwrite=True)
