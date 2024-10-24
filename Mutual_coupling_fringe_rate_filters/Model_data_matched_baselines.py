import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import stats

from pyuvdata import UVData, UVBeam
import linsolve
import hera_cal as hc

from collections import OrderedDict as odict
from hera_cal.abscal import fill_dict_nans

mode_array=np.array(["low","high"])

for i in range(len(mode_array)):
    mode=mode_array[i]

    print(mode)

    hd = hc.io.HERAData('/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Model_first_order_visibilities_2h_boosted_'+mode+'.uvh5')
    pol=hd.polarization_array
    lsts = np.unwrap(hd.lsts) * 12/np.pi - 24
    freqs = hd.freq_array
    antpos, ants = hd.get_ENU_antpos()
    antpos_d = dict(zip(ants, antpos))
    time=np.unique(hd.time_array)

    hd2 = hc.io.HERAData('/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Corrected_redundant_array_Model_zeroth_order_visibilities_2h_'+mode+'.uvh5')
    hd2.read(polarizations=pol[0],frequencies=freqs,times=time)

    hd2.inflate_by_redundancy()
    hd2.conjugate_bls()
    hd2._determine_blt_slicing()
    hd2._determine_pol_indexing()

    baseline_match_array=np.load("/home/ntsikelelo/Simulated_data_files/matched_baselines.npy")
    bls_match=[]
    for i in range (len(baseline_match_array)):
        key=baseline_match_array[i,:]
        bls_match.append((key[0],key[1]))

    hd2.select(bls=bls_match)
    hd2.x_orientation = 'east'
    model2, _, _ = hd2.build_datacontainers() #sky model 

    model=copy.deepcopy(hd2)

    for key in hd2.get_antpairs():
    #     print(key)
        bls=(key[0],key[1],'ee')
        model_ind = model.antpair2ind(*key)
        model_data=np.zeros((model.data_array[model_ind].shape),dtype=complex)
        model_data[:,0,:,0]=model2[bls]
        model.data_array[model_ind]=model_data 

    model.write_uvh5("/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Corrected_redundant_array_Model_zeroth_order_visibilities_2h_"+mode+"_baseline_selected.uvh5", clobber=True)