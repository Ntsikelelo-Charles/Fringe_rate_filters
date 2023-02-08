#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import stats

from pyuvdata import UVData, UVBeam
import linsolve
import hera_cal as hc

from collections import OrderedDict as odict
from hera_cal.abscal import fill_dict_nans

## run this in Jake! gains

# load the metadata
path='/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/'
model = hc.io.HERAData(path+'Model_data_complete.uvh5')
times = np.unique(model.time_array)
model.read()


model.inflate_by_redundancy()
model.conjugate_bls()
model._determine_blt_slicing()
model._determine_pol_indexing()

antpos, ants = model.get_ENU_antpos()
antpos_d = dict(zip(ants, antpos))

model_data, _, _ = model.build_datacontainers()

freqs = model.freq_array[0]/1e6
# get a list of antennas
ants = sorted(set(np.ravel([k[:2] for k in model_data.keys()])))

np.random.seed(0)
Nants = len(ants)
amps = np.random.normal(0.03, 0.001, Nants) # amp
#phs = np.random.normal(0, np.pi/4, Nants) # radians
dly =0*np.random.normal(0, 200, Nants) * 1e-9 # in seconds
amp_plaw = np.random.normal(-2.6, 0.2, Nants)
gains = amps * (freqs[:, None] / 150)**amp_plaw * np.exp(1j * 2 * np.pi * dly * freqs[:, None]*1e6)
phase_gains=np.zeros(gains.shape,dtype=complex)
for ant in range(Nants):
    k_a,k_b = np.random.normal(0.005, 0.0005, 2)
    phs=np.cos(k_a*freqs)+np.sin(k_b*freqs)
    phase_gains[:,ant]=np.exp(1j * phs)
gains*=phase_gains 

## need to run in Jake
np.save("/home/ntsikelelo/Simulated_data_files/simulated_gains.npy",gains)
gains_init = {(ant, 'Jee'): gains[:, i][None, :] for i, ant in enumerate(ants)}
full_gains = hc.abscal.merge_gains([gains_init])#, gains_resid])

# set reference antenna
ref_ant = (0, 'Jee')
hc.abscal.rephase_to_refant(full_gains, refant=ref_ant)


from pyuvdata import UVData, UVBeam, utils as uvutils
glm=model
#load gains
path2='/net/sinatra/vault-ike/nkern/hera_phaseI_validation/'
uvc = hc.io.HERACal(path2+"gains/2458101.sum.true_gains.singletime.calfits")
uvc.read()
uvc.select(freq_chans=np.arange(1024)[512:512+len(model.freqs)],jones=[-5])  # select out high band



## setting up the gains to have ant=ant_model
## np.random to keep the same numbers in
g_terms=np.zeros((len(ants),1,308,1,1),dtype=complex)
for ant in range (91):
    g_terms[ant,0,:,0,0]=full_gains[(ant, 'Jee')]

bool_arr=np.array(g_terms,dtype=bool)
bool_arr[:,:,:,:,:]=False

uvc.gain_array=np.array(g_terms)
uvc.Nants_data=len(ants)
uvc.ant_array=np.arange(len(ants))

uvc.antenna_positions=glm.antenna_positions
uvc.antenna_names=glm.antenna_names
uvc.antenna_numbers=glm.antenna_numbers
uvc.flag_array=bool_arr

uvc.quality_array=np.ones(g_terms.shape)
uvc.freq_array[0,:]=np.array(model.freqs)
uvc.times=np.array(glm.times)
uvc.time_array=np.array(glm.time_array)
uvc.x_orientation='east'

model_corrupt=copy.deepcopy(model)
raw = uvutils.uvcalibrate(model_corrupt, uvc, inplace=False, undo=True, time_check=False)

# Nrms =1e-6

# print("noise rms = "+str(Nrms))
# noise=(np.random.normal(0, 1, raw.data_array.size)+ 1j * np.random.normal(0, 1, raw.data_array.size)).reshape(raw.data_array.shape)*(Nrms/np.sqrt(2))
# raw.data_array += noise


raw.write_uvh5("/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_data_no_noise.uvh5",clobber=True)