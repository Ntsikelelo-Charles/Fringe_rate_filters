
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pyuvdata import UVData
import hera_cal as hc
import uvtools as uvt
import hera_pspec as hp
import copy

m=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/m_slope_filter.npy")
# filte_width_text=np.array(['0.25e-3'])

filte_width_text=np.array(['0.40e-3','0.60e-3'])
path="/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/"
print("Notch filtering the data ")

# mode_array=np.array(["high"])
mode_array=np.array(["low","high"])
filter_type=""
for i in range(len(mode_array)):
    mode=mode_array[i]



    for fil in range(len(filte_width_text)):
        filter_max=0
        if filte_width_text[fil]=='0.25e-3':
            filter_max=25
            filter_type="dpss_25"
        if filte_width_text[fil]=='0.40e-3':
            filter_max=40
            filter_type="dpss_40"
        if filte_width_text[fil]=='0.60e-3':
            filter_max=60        
            filter_type="dpss_60"


        # units are fringe-rates [Hz]

        Model_complete_file = path+"Corrected_redundant_array_Model_zeroth_order_visibilities_2h_"+mode+"_baseline_selected.uvh5"
        Model_complete = hc.frf.FRFilter(Model_complete_file)
        freqs = Model_complete.freqs/1e6
        times = (Model_complete.times-Model_complete .times.min()) * 24 * 3600  # seconds

        uvd = UVData()
        uvd.read(Model_complete_file)
        F_model = hc.frf.FRFilter(uvd)
        filter_center     = [0.0]         
        filter_half_width = [float(filte_width_text[fil])]
        print(filter_max)

        # unitless
        filter_factor     = [1e-8]

        # make covariance
        C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)

        antpos = F_model.antpos
        # take inverse to get filter matrix
        R = np.linalg.pinv(C, rcond=1e-10)

        # filter the data!
        Model_complete_filt_data = copy.deepcopy(F_model.data)
        for k in Model_complete_filt_data:
            blvec = (antpos[k[1]] - antpos[k[0]])
            bl_len_EW = blvec[0]
            fringe_value=m*bl_len_EW

    #         if np.abs(fringe_value) > 2:
            Model_complete_filt_data[k] = R @ Model_complete_filt_data[k]

        print("Model complete data filter with half_width = "+str(filter_half_width)+" and filter center = "+str(filter_center))    

        F_model.write_data(Model_complete_filt_data,path+"Model_complete_filtered_data_"+filter_type+"_non_redundancy_"+mode+"_2h.uvh5",overwrite=True)




        raw_file = path+'Raw_data_non_redundant_with_noise_'+mode+'_2h.uvh5'



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
            blvec = (antpos[k[1]] - antpos[k[0]])
            bl_len_EW = blvec[0]
            fringe_value=m*bl_len_EW

    #         if np.abs(fringe_value) > 2:

            raw_filt_data[k] = R @ raw_filt_data[k]

        print("raw data filter with half_width = "+str(filter_half_width)+" and filter center = "+str(filter_center))    

        F_model.write_data(raw_filt_data,path+"Raw_filtered_data_non_redundant_deep_"+filter_type+"_"+mode+"_2h.uvh5",overwrite=True)

