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
path2="/net/jake/home/ntsikelelo/Simulated_data_files/UVH5_files/"
print("new")
Model_complete_file = path+"Model_data_complete.uvh5"
Model_complete = hc.frf.FRFilter(Model_complete_file)
uvd = UVData()
uvd.read(Model_complete_file)
freqs = Model_complete.freqs/1e6
times = (Model_complete.times-Model_complete .times.min()) * 24 * 3600  # seconds
uvd.inflate_by_redundancy()
uvd.conjugate_bls()
F = hc.frf.FRFilter(uvd)

Model_incomplete_file = path+"Model_data_incomplete.uvh5"
Model_incomplete = hc.frf.FRFilter(Model_incomplete_file)
uvd = UVData()
uvd.read(Model_incomplete_file)
uvd.inflate_by_redundancy()
uvd.conjugate_bls()
F_incomplete = hc.frf.FRFilter(uvd)

# raw_file = path2+"Raw_data_with_noise.uvh5"

raw_file = path2+"Raw_data_no_noise.uvh5"
raw = hc.frf.FRFilter(raw_file)
uvd = UVData()
uvd.read(raw_file)
uvd.conjugate_bls()
F_raw = hc.frf.FRFilter(uvd)



filter_factor     = [1e-8]
print("filter factor = "+str(filter_factor))
F.fft_data(ax='time', window='blackman', overwrite=True, ifft=True)

fr_select = (0< F.frates) & (F.frates < 7)

fr_select_negative=(-7 < F.frates) & (F.frates<0)

m=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/m_slope_filter.npy")
c=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/c_intercept_filter.npy")

x0 = np.array([1e-3, 2.0, 0.3])


x_negative = F.frates[fr_select_negative]
x = F.frates[fr_select]

antpos = F.antpos
filter_fun=np.zeros(F.frates.shape)
sigma_frac={}
filt_data_complete = copy.deepcopy(F.data)
filt_data_incomplete = copy.deepcopy(F_incomplete.data)
filt_data_raw=copy.deepcopy(F_raw.data)



for k in F.data:
    blvec = (antpos[k[1]] - antpos[k[0]])
    bl_len_EW = blvec[0]
    fringe_value=m*bl_len_EW
   


    if fringe_value > 0:
      
        
        y = np.abs(F.dfft[k]).mean(1)
        x0[1]=fringe_value
        fit, ypred = gauss_fit(x0, x, y[fr_select],  method='powell')
         # make the filter
        gmean, gsigma = fit.x[1:]    
        filter_center = -gmean * 1e-3
        filter_half_width = np.abs(gsigma) * 2 * 1e-3
      

        C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
        R = np.linalg.pinv(C, rcond=1e-10)

        filt_data_complete[k] = filt_data_complete[k] - R @ filt_data_complete[k]
        filt_data_incomplete[k] = filt_data_incomplete[k] - R @ filt_data_incomplete[k]
        filt_data_raw[k] = filt_data_raw[k] - R @ filt_data_raw[k]
        



    elif fringe_value < 0:
        
        
        y = np.abs(F.dfft[k]).mean(1)    
        x0[1]=fringe_value
        fit, ypred = gauss_fit(x0, x_negative, y[fr_select_negative],  method='powell')
         # make the filter
        gmean, gsigma = fit.x[1:]    
        filter_center = -gmean * 1e-3
    
        filter_half_width = np.abs(gsigma) * 2 * 1e-3
            
        C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
        R = np.linalg.pinv(C, rcond=1e-10)

        filt_data_complete[k] = filt_data_complete[k] - R @ filt_data_complete[k]
        filt_data_incomplete[k] = filt_data_incomplete[k] - R @ filt_data_incomplete[k]
        filt_data_raw[k] = filt_data_raw[k] - R @ filt_data_raw[k]
        

        






F.write_data(filt_data_complete,path+"Model_complete_filtered_Gaussian.uvh5",overwrite=True)
F.write_data(filt_data_incomplete,path+"Model_incomplete_filtered_Gaussian.uvh5",overwrite=True)
F.write_data(filt_data_raw,path+"Raw_data_filtered_no_noise_Gaussian.uvh5",overwrite=True)
# F.write_data(filt_data_raw,path+"Raw_data_filtered_Gaussian.uvh5",overwrite=True)












