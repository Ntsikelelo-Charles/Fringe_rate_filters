import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import stats

from pyuvdata import UVData, UVBeam
import linsolve
import hera_cal as hc

from collections import OrderedDict as odict
from hera_cal.abscal import fill_dict_nans
from scipy import signal

## load data redandant and non-redundant 
Nrms = 1e-5
add_noise=True
mode_array=np.array(["low","high"])

for i in range(len(mode_array)):
    mode=mode_array[i]

    print("running simulate first order raw data")
    # load the metadata
    # hd = hc.io.HERAData('/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Model_first_order_visibilities_0h_boosted_'+mode+'.uvh5')

    hd = hc.io.HERAData("/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Model_first_order_visibilities_2h_boosted_"+mode+".uvh5")
    lsts = np.unwrap(hd.lsts) * 12/np.pi - 24
    freqs = hd.freq_array
    antpos, ants = hd.get_ENU_antpos(pick_data_ants=True)
    antpos_d = dict(zip(ants, antpos))

    time=np.unique(hd.time_array)
    # only load a couple of times and frequencies
    pol=hd.polarization_array
    hd.read(polarizations=pol[0],frequencies=freqs,times=time)

    # inflate by full redundancy
    reduncy_xtalk=False
    if reduncy_xtalk:
        hd.inflate_by_redundancy()
    hd.conjugate_bls()
    hd._determine_blt_slicing()
    hd._determine_pol_indexing()
    hd.x_orientation = 'east'


    # load datacontainer
    model, _, _ = hd.build_datacontainers() # x-talk model

    Ntimes, Nfreqs = hd.Ntimes, hd.Nfreqs
    freqs = hd.freq_array[0]/1e6


    #get a list of antennas

    np.random.seed(0)
    Nants = len(ants)
    amps = np.random.normal(0.03, 0.001, Nants) # amp
    #phs = np.random.normal(0, np.pi/4, Nants) # radians
    dly =0*np.random.normal(0, 200, Nants) * 1e-9 # in seconds
    amp_plaw = np.random.normal(-2.6, 0.2, Nants)
    gains = amps * (freqs[:, None] / 150)**amp_plaw * np.exp(1j * 2 * np.pi * dly * freqs[:, None]*1e6)
    phase_gains=np.zeros(gains.shape,dtype=complex)
    for ant in range(Nants):
        k_a,k_b = np.random.normal(0.0005, 0.0005, 2)
        phs=np.cos(k_a*freqs)+np.sin(k_b*freqs)
        phase_gains[:,ant]=np.exp(1j * phs)
    gains*=phase_gains 

    # gains=np.load("/home/ntsikelelo/non_redundancy_sim/Applied_antenna_gains.npy")

    gains_init = {(ant, 'Jee'): gains[:, i][None, :] for i, ant in enumerate(ants)}
    full_gains = hc.abscal.merge_gains([gains_init])#, gains_resid])

    # set reference antenna
    ref_ant = (127, 'Jee')
    hc.abscal.rephase_to_refant(full_gains, refant=ref_ant)

    # copy over to raw_data
    data = copy.deepcopy(model)

    # apply gains to raw_data
    hc.abscal.calibrate_in_place(data, full_gains, gain_convention='multiply')

    # insert noise

    if add_noise:
        np.random.seed(0)

        for k in data:
            n=(np.random.normal(0, 1, data[k].size) \
                        + 1j * np.random.normal(0, 1, data[k].size)).reshape(data[k].shape) * Nrms / np.sqrt(2) 
            data[k] +=n  

    else:
        Nrms = 1   

    raw=copy.deepcopy(hd)

    for key in hd.get_antpairs():
    #     print(key)
        bls=(key[0],key[1],'ee')
        raw_ind = raw.antpair2ind(*key)
        raw_data=np.zeros((raw.data_array[raw_ind].shape),dtype=complex)
        raw_data[:,0,:,0]=data[bls]
        raw.data_array[raw_ind]=raw_data    

    if add_noise:
        raw.write_uvh5("/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Raw_data_non_redundant_with_noise_"+mode+"_2h.uvh5", clobber=True)

