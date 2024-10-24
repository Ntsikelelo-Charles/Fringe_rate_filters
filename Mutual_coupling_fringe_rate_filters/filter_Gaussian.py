import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pyuvdata import UVData
import hera_cal as hc
import uvtools as uvt
import hera_pspec as hp
import copy
from scipy import integrate
from scipy import optimize

def gauss(x, amp, loc, scale):
    return amp * np.exp(-0.5 * (x-loc)**2 / scale**2)

def chisq(x0, x, y):
    yfit = gauss(x, *x0)
    return np.sum(np.abs(yfit - y)**2)

def gauss_fit(x0, x, y, method='powell'):
    fit = optimize.minimize(chisq, x0, args=(x, y), method=method)
    ypred = gauss(x, *fit.x)
    return fit, ypred

path="/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/"
print("Main lobe filtering the data")



# mode_array=np.array(["high"])
mode_array=np.array(["low","high"])

for i in range(len(mode_array)):
    mode=mode_array[i]
    
    Model_complete_file = path+"Corrected_redundant_array_Model_zeroth_order_visibilities_2h_"+mode+"_baseline_selected.uvh5"
    Model_complete = hc.frf.FRFilter(Model_complete_file)
    uvd = UVData()
    uvd.read(Model_complete_file)
    freqs = Model_complete.freqs/1e6
    times = (Model_complete.times-Model_complete .times.min()) * 24 * 3600  # seconds
    uvd.conjugate_bls()
    F = hc.frf.FRFilter(uvd)



    raw_file = path+"Raw_data_non_redundant_with_noise_"+mode+"_2h.uvh5"
    raw = hc.frf.FRFilter(raw_file)
    uvd = UVData()
    uvd.read(raw_file)
    uvd.conjugate_bls()
    F_raw = hc.frf.FRFilter(uvd)



    filter_factor     = [1e-8]
    print("filter factor = "+str(filter_factor))
    F.fft_data(ax='time', window='blackman', overwrite=True, ifft=True)

    fr_select = (0< F.frates) & (F.frates < 5)

    fr_select_negative=(-5 < F.frates) & (F.frates<0)

    m=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/m_slope_filter.npy")
    c=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/c_intercept_filter.npy")

    x0 = np.array([1e-3, 2.0, 0.3])


    x_negative = F.frates[fr_select_negative]
    x = F.frates[fr_select]

    antpos = F.antpos
    filter_fun=np.zeros(F.frates.shape)
    filt_data_complete = copy.deepcopy(F.data)
    filt_data_raw=copy.deepcopy(F_raw.data)

    E_W_bls_list={}
    fringe_value_list={}
    sigma_list={}

    for k in filt_data_raw:
        blvec = (antpos[k[1]] - antpos[k[0]])
        bl_len_EW = blvec[0]
        fringe_value=m*bl_len_EW



        if fringe_value > 0.5:


            y = np.abs(F.dfft[k]).mean(1)
            x0[1]=fringe_value
            fit, ypred = gauss_fit(x0, x, y[fr_select],  method='powell')
             # make the filter
            gmean, gsigma = fit.x[1:]    
            filter_center = -gmean * 1e-3
            filter_half_width = np.abs(gsigma) * 2 * 1e-3
    #         print(filter_center,filter_half_width)
            fringe_value_list[k]=filter_center
            sigma_list[k]=gsigma
            E_W_bls_list[k]=bl_len_EW

            C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
            R = np.linalg.pinv(C, rcond=1e-10)

            filt_data_complete[k] = filt_data_complete[k] - R @ filt_data_complete[k]
            filt_data_raw[k] = filt_data_raw[k] - R @ filt_data_raw[k]




        if fringe_value < -0.5:


            y = np.abs(F.dfft[k]).mean(1)    
            x0[1]=fringe_value
            fit, ypred = gauss_fit(x0, x_negative, y[fr_select_negative],  method='powell')
             # make the filter
            gmean, gsigma = fit.x[1:]    
            filter_center = -gmean * 1e-3

            filter_half_width = np.abs(gsigma) * 2 * 1e-3
            
            fringe_value_list[k]=filter_center
            sigma_list[k]=gsigma
            E_W_bls_list[k]=bl_len_EW

            C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
            R = np.linalg.pinv(C, rcond=1e-10)

            filt_data_complete[k] = filt_data_complete[k] - R @ filt_data_complete[k]
            filt_data_raw[k] = filt_data_raw[k] - R @ filt_data_raw[k]

#         else:


#             filter_center =0

#             filter_half_width = 0.25 * 1e-3

#             C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
#             R = np.linalg.pinv(C, rcond=1e-10)

#             filt_data_complete[k] = R @ filt_data_complete[k]
#             filt_data_raw[k] = R @ filt_data_raw[k]


        
    np.save("/home/ntsikelelo/non_redundancy_sim/sigma_gaussian_"+mode+".npy",np.array(sigma_list))
    np.save("/home/ntsikelelo/non_redundancy_sim/fringe_value_gaussian_"+mode+".npy",np.array(fringe_value_list))
    np.save("/home/ntsikelelo/non_redundancy_sim/E_W_baseline_gaussian_"+mode+".npy",np.array(E_W_bls_list))





    F.write_data(filt_data_complete,path+"Model_complete_filtered_Gaussian_non_redundant_"+mode+"_2h.uvh5",overwrite=True)
    F_raw.write_data(filt_data_raw,path+"Raw_data_filtered_Gaussian_non_redundant_with_noise_"+mode+"_2h.uvh5",overwrite=True)













