# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal
import glob
import os
import copy
import hera_sim as hs
from scipy import stats
# from memory_profiler import memory_usage
# import multiprocess
from pyuvdata import UVData, UVBeam, utils as uvutils
import hera_pspec as hp
import hera_cal as hc
import healpy
plt.rcParams.update({'font.size': 14})
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)

N=120
path2='/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/'

path='/net/jake/home/ntsikelelo/sims/'






glm_file_deep ='/net/sinatra/vault-ike/nkern/conv_cal/sims/gleam_ATeam_25mJy_nside128_hera90.uvh5'
glm_deep = hc.io.HERAData(glm_file_deep)
glm_deep.read(read_data=False)
times = np.unique(glm_deep.time_array)
glm_deep.read(times=times[N:N+200],polarizations=[-5])

 
# print(times[25:50].shape)

glm_file ='/net/sinatra/vault-ike/nkern/conv_cal/sims/gleam_ATeam_100mJy_nside128_hera90.uvh5'
glm = hc.io.HERAData(glm_file)
glm.read(read_data=False)
times = np.unique(glm.time_array)
glm.read(times=times[N:N+200],polarizations=[-5])



gsm_file ='/net/sinatra/vault-ike/nkern/conv_cal/sims/gsm2016_nside128_hera90.uvh5'
gsm = hc.io.HERAData(gsm_file)
gsm.read(read_data=False)
gsm.time_array=gsm.time_array
gsm.read(times=times[N:N+200],polarizations=[-5])
gsm.telescope_location = glm.telescope_location

# path='/net/ike/vault-ike/nkern/conv_cal/sims/'

# '','',''
# glm_file_deep =path+'gleamdeep_nside128_hera90.uvh5'
# glm_deep = hc.io.HERAData(glm_file_deep)
# glm_deep.read(read_data=False)
# times = np.unique(glm_deep.time_array)
# glm_deep.read(times=times[N:N+200],polarizations=[-5])

 
# # print(times[25:50].shape)

# glm_file =path+'gleam_nside128_hera90.uvh5'
# glm = hc.io.HERAData(glm_file)
# glm.read(read_data=False)
# times = np.unique(glm.time_array)
# glm.read(times=times[N:N+200],polarizations=[-5])



# gsm_file =path+'gsm2016_nside128_hera90.uvh5'
# gsm = hc.io.HERAData(gsm_file)
# gsm.read(read_data=False)
# gsm.time_array=gsm.time_array
# gsm.read(times=times[N:N+200],polarizations=[-5])
# gsm.telescope_location = glm.telescope_location




raw_model = copy.deepcopy(glm_deep)
for key in raw_model.get_antpairs():
    raw_ind = raw_model.antpair2ind(*key)
    dif_ind = gsm.antpair2ind(*key)
    raw_model.data_array[raw_ind]=raw_model.data_array[raw_ind]+ gsm.data_array[dif_ind]
    
print("DATA = GLEAM_DEEP + GSM")   
raw_model.write_uvh5(path2+"Model_data_complete.uvh5",clobber=True)

### Models
# build datacontainers
print("Model incomplete= GLEAM")
model_incomplete = copy.deepcopy(glm)  
model_incomplete.write_uvh5(path2+"Model_data_incomplete.uvh5",clobber=True)