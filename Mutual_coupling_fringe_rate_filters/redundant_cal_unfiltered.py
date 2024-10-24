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
Nrms = 1e-5
mode_array=np.array(["low","high"])
# mode_array=np.array(["high"])

for i in range(len(mode_array)):
    mode=mode_array[i]

    print(mode)  
    hd = hc.io.HERAData('/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Model_first_order_visibilities_2h_boosted_'+mode+'.uvh5')
    hd2 = hc.io.HERAData('/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Corrected_redundant_array_Model_zeroth_order_visibilities_2h_'+mode+'_baseline_selected.uvh5')
    hd_unfil = hc.io.HERAData('/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/Raw_data_non_redundant_with_noise_'+mode+'_2h.uvh5')
    lsts = np.unwrap(hd.lsts) * 12/np.pi - 24
    freqs = hd.freq_array
    antpos, ants = hd.get_ENU_antpos(pick_data_ants=True)
    antpos_d = dict(zip(ants, antpos))



    time=np.unique(hd.time_array)
    # only load a couple of times and frequencies
    pol=hd.polarization_array
    hd.read(polarizations=pol[0],frequencies=freqs,times=time)
    hd2.read(polarizations=pol[0],frequencies=freqs,times=time)
    hd_unfil.read(polarizations=pol[0],frequencies=freqs,times=time)

    # inflate by full redundancy
    reduncy_xtalk=False
    if reduncy_xtalk:
        hd.inflate_by_redundancy()
    hd.conjugate_bls()
    hd._determine_blt_slicing()
    hd._determine_pol_indexing()

#     hd2.inflate_by_redundancy()
    hd2.conjugate_bls()
    hd2._determine_blt_slicing()
    hd2._determine_pol_indexing()

    baseline_match_array=np.load("/home/ntsikelelo/Simulated_data_files/matched_baselines.npy")
    bls_match=[]
    for i in range (len(baseline_match_array)):
        key=baseline_match_array[i,:]
        bls_match.append((key[0],key[1]))

    hd.select(bls=bls_match)
    hd2.select(bls=bls_match)
    hd_unfil.select(bls=bls_match)

    hd.x_orientation = 'east'
    hd2.x_orientation = 'east'
    hd_unfil.x_orientation='east'

    # load datacontainer
    data_un, _, _ = hd_unfil.build_datacontainers() #xtalk raw data
    model, _, _ = hd.build_datacontainers() # x-talk model
    model2, _, _ = hd2.build_datacontainers() #sky model zeroth
    Ntimes, Nfreqs = hd.Ntimes, hd.Nfreqs
    freqs = hd.freq_array[0]/1e6


    # get a list of antennas
#     ants = sorted(set(np.ravel([k[:2] for k in model.keys()])))

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

    gains_init = {(ant, 'Jee'): gains[:, i][None, :] for i, ant in enumerate(ants)}
    full_gains = hc.abscal.merge_gains([gains_init])#, gains_resid])

    # set reference antenna
    ref_ant = (127, 'Jee')
    hc.abscal.rephase_to_refant(full_gains, refant=ref_ant)

    # copy over to raw_data
    data = copy.deepcopy(model)
    data2 = copy.deepcopy(model2)
    # apply gains to raw_data
    hc.abscal.calibrate_in_place(data, full_gains, gain_convention='multiply')
    hc.abscal.calibrate_in_place(data2, full_gains, gain_convention='multiply')
    # insert noise
    add_noise = True
    if add_noise:
        np.random.seed(0)
        for k in data:
            n=(np.random.normal(0, 1, data[k].size) \
                        + 1j * np.random.normal(0, 1, data[k].size)).reshape(data[k].shape) * Nrms / np.sqrt(2) 
            data[k] +=n  
            data2[k] +=n
    else:
        Nrms = 1   


    # get redundant baseline groups in array
    reds = hc.redcal.get_reds(antpos_d, pols=['ee'])

    # get redundant baseline groups in data
    reds = hc.redcal.get_reds(antpos_d, pols=['ee'])
    red_data=[]
    for red_g in reds:
        bl_group=[]
        for bl in red_g:
            if bl in list(model.keys()):
                bl_group.append(bl)    
        if len(bl_group)>0:   
            red_data.append(bl_group)    

    # rc = hc.redcal.RedundantCalibrator(filtered_reds)
    rc = hc.redcal.RedundantCalibrator(red_data)

    # get noise wgts (use whatever noise RMS you put into the data)
    noise_wgts = {k: np.ones_like(data[k], dtype=float) / Nrms**2 for k in data}


    # perform logcal
    logcal_meta, logcal_sol = rc.logcal(data)
    hc.redcal.make_sol_finite(logcal_sol)
    # remove redcal degeneracies from solution
    logcal_sol = rc.remove_degen(logcal_sol)
    # get gains and model visibilities
    logcal_gains, logcal_vis = hc.redcal.get_gains_and_vis_from_sol(logcal_sol)

    # perform logcal
    logcal_meta2, logcal_sol2 = rc.logcal(data2)
    hc.redcal.make_sol_finite(logcal_sol2)
    # remove redcal degeneracies from solution
    logcal_sol2 = rc.remove_degen(logcal_sol2)
    # get gains and model visibilities
    logcal_gains2, logcal_vis2 = hc.redcal.get_gains_and_vis_from_sol(logcal_sol2)

    # perform omnical (lincal)
    conv_crit = 1e-10
    maxiter = 500
    gain = 0.4
    lincal_meta, lincal_sol = rc.omnical(data, logcal_sol, wgts=noise_wgts, conv_crit=conv_crit,
                                     maxiter=maxiter, gain=gain)
    hc.redcal.make_sol_finite(lincal_sol)
    # remove redcal degeneracies from solution
    lincal_sol = rc.remove_degen(lincal_sol)
    # get gains and model visibilities
    lincal_gains, lincal_vis = hc.redcal.get_gains_and_vis_from_sol(lincal_sol)

    # perform omnical (lincal)
    conv_crit = 1e-10
    maxiter = 500
    gain = 0.4
    lincal_meta2, lincal_sol2 = rc.omnical(data2, logcal_sol2, wgts=noise_wgts, conv_crit=conv_crit,
                                     maxiter=maxiter, gain=gain)
    hc.redcal.make_sol_finite(lincal_sol2)
    # remove redcal degeneracies from solution
    lincal_sol2 = rc.remove_degen(lincal_sol2)
    # get gains and model visibilities
    lincal_gains2, lincal_vis2 = hc.redcal.get_gains_and_vis_from_sol(lincal_sol2)

    N_t=len(np.unique(hd.time_array))
    gains_all=np.zeros(shape=(len(ants),N_t,len(freqs)),dtype=complex)
    ant=0
    for key in lincal_gains:
        gains_per_ant=lincal_gains[key]
        for t in range (N_t):
            gains_all[ant,t,:]=gains_per_ant[t,:]
        ant=ant+1   
    gains_all2=np.zeros(shape=(len(ants),N_t,len(freqs)),dtype=complex)

    ant=0
    for key in lincal_gains:
        gains_per_ant=lincal_gains2[key]
        for t in range (N_t):
            gains_all2[ant,t,:]=gains_per_ant[t,:]
        ant=ant+1  



    np.save("/home/ntsikelelo/non_redundancy_sim/lincal_gains_first_order_vis_no_filter_"+mode+"_2h.npy",gains_all)
    np.save("/home/ntsikelelo/non_redundancy_sim/lincal_gains_zeroth_vis_no_filter_"+mode+"_2h.npy",gains_all2)



    # compute chisq
    rc_red_chisq, chisq_per_ant = hc.redcal.normalized_chisq(data, noise_wgts, red_data, lincal_vis, lincal_gains)
    rc_red_chisq_dof = len(data) - len(lincal_gains)
    rc_red_chisq = rc_red_chisq['Jee']
    np.save("/home/ntsikelelo/non_redundancy_sim/redcal_chisq_first_order_vis_no_filter_"+mode+"_2h.npy",rc_red_chisq)

    rc_red_chisq2, chisq_per_ant2 = hc.redcal.normalized_chisq(data2, noise_wgts, red_data, lincal_vis2, lincal_gains2)
    rc_red_chisq_dof = len(data) - len(lincal_gains)
    rc_red_chisq2 = rc_red_chisq2['Jee']
    np.save("/home/ntsikelelo/non_redundancy_sim/redcal_chisq_zeroth_vis_no_filter_"+mode+"_2h.npy",rc_red_chisq2)


    # calibration with lincal gains
    redcal_data = copy.deepcopy(data)
    hc.apply_cal.calibrate_in_place(redcal_data, lincal_gains)
    rc_flags = {k: np.zeros_like(full_gains[k], dtype=bool) for k in full_gains}

    # calibration with lincal gains
    redcal_data2 = copy.deepcopy(data2)
    hc.apply_cal.calibrate_in_place(redcal_data2, lincal_gains2)

    ##perform abscal
    abscal_gains = hc.abscal.post_redcal_abscal(model2, redcal_data, noise_wgts, rc_flags, verbose=False)
    abscal_gains2 = hc.abscal.post_redcal_abscal(model2, redcal_data2, noise_wgts, rc_flags, verbose=False)

    # combine redcal and abscal gains
    total_gains = hc.abscal.merge_gains([lincal_gains, abscal_gains])
    total_gains2 = hc.abscal.merge_gains([lincal_gains2, abscal_gains2])
    # make sure it has the same reference antenna
    # set reference antenna
    hc.abscal.rephase_to_refant(total_gains, ref_ant)
    hc.abscal.rephase_to_refant(total_gains2, ref_ant)

    gains_all=np.zeros(shape=(len(ants),N_t,len(freqs)),dtype=complex)

    ant=0
    for key in total_gains:
        gains_per_ant=total_gains[key]
        for t in range (N_t):
            gains_all[ant,t,:]=gains_per_ant[t,:]
        ant=ant+1     

    gains_all2=np.zeros(shape=(len(ants),N_t,len(freqs)),dtype=complex)

    ant=0
    for key in total_gains:
        gains_per_ant=total_gains2[key]
        for t in range (N_t):
            gains_all2[ant,t,:]=gains_per_ant[t,:]
        ant=ant+1 


    # get chisq after redcal and abscal
    abs_chisq, nObs, _, _ = hc.utils.chisq(data, model2, gains=total_gains, data_wgts=noise_wgts)
    chisq_dof = nObs.mean() - len(abscal_gains)
    red_chisq = abs_chisq / chisq_dof
    np.save("/home/ntsikelelo/non_redundancy_sim/abscal_chisq_first_order_vis_no_filter_"+mode+"_2h.npy",red_chisq)



    # get chisq after redcal and abscal
    abs_chisq, nObs, _, _ = hc.utils.chisq(data2, model2, gains=total_gains2, data_wgts=noise_wgts)
    chisq_dof = nObs.mean() - len(abscal_gains)
    red_chisq = abs_chisq / chisq_dof

    np.save("/home/ntsikelelo/non_redundancy_sim/abscal_chisq_zeroth_vis_no_filter_"+mode+"_2h.npy",red_chisq)


    # calibrate data
    cal_data = copy.deepcopy(data_un)
    hc.apply_cal.calibrate_in_place(cal_data, total_gains)

    antpos, ants = hd.get_ENU_antpos()
    antpos_dict = dict(zip(ants, antpos))
    bl_lens, bl_groups = [], []
    for red in red_data:
        # get the baseline length of this redundant group
        bl = red[0]
        bl_len = np.linalg.norm(antpos_dict[bl[1]] - antpos_dict[bl[0]])
        # check if this bl_len exists
        if np.isclose(bl_len, bl_lens, atol=1).any():
            bl_groups[-1].extend(red)
        else:
            bl_groups.append(red)
            bl_lens.append(bl_len)


    # now average all baselines within each group
    N_t=len(np.unique(hd.time_array))
    cal_wedge = np.zeros((N_t,len(bl_groups), hd.Nfreqs, hd.Npols), dtype=np.complex128)
    mdl_wedge = np.zeros((N_t,len(bl_groups), hd.Nfreqs, hd.Npols), dtype=np.complex128)

    for i, bl_group in enumerate(bl_groups):
        for j, pol in enumerate(hd.get_pols()):
            cal_wedge[:,i, :, j] = np.mean([cal_data[bl] for bl in bl_group], axis=0)
            mdl_wedge[:,i, :, j] = np.mean([model[bl] for bl in bl_group], axis=0)


    # now take the FFT across frequency: cut the edge channels
    cal_wedge_fft, delays = hc.vis_clean.fft_data(cal_wedge, np.diff(freqs)[0], axis=2,
                                                  edgecut_low=5, edgecut_hi=5, window='bh')
    mdl_wedge_fft, delays = hc.vis_clean.fft_data(mdl_wedge, np.diff(freqs)[0], axis=2,
                                                  edgecut_low=5, edgecut_hi=5, window='bh')

    print(cal_wedge_fft.shape)

    np.save("/home/ntsikelelo/non_redundancy_sim/mdl_wedge_fft_first_order_vis_no_filter_"+mode+"_2h.npy",mdl_wedge_fft)
    np.save("/home/ntsikelelo/non_redundancy_sim/cal_wedge_fft_first_order_vis_no_filter_"+mode+"_2h.npy",cal_wedge_fft) 

    np.save("/home/ntsikelelo/non_redundancy_sim/gains_first_order_vis_no_filter_"+mode+"_2h.npy",gains_all)
    np.save("/home/ntsikelelo/non_redundancy_sim/gains_zeroth_vis_no_filter_"+mode+"_2h.npy",gains_all2)


    # calibrate data
    cal_data = copy.deepcopy(data_un)
    hc.apply_cal.calibrate_in_place(cal_data, total_gains2)


    # now average all baselines within each group
    N_t=len(np.unique(hd.time_array))
    cal_wedge = np.zeros((N_t,len(bl_groups), hd.Nfreqs, hd.Npols), dtype=np.complex128)
    mdl_wedge = np.zeros((N_t,len(bl_groups), hd.Nfreqs, hd.Npols), dtype=np.complex128)

    for i, bl_group in enumerate(bl_groups):
        for j, pol in enumerate(hd.get_pols()):
            cal_wedge[:,i, :, j] = np.mean([cal_data[bl] for bl in bl_group], axis=0)
            mdl_wedge[:,i, :, j] = np.mean([model[bl] for bl in bl_group], axis=0)


    # now take the FFT across frequency: cut the edge channels
    cal_wedge_fft, delays = hc.vis_clean.fft_data(cal_wedge, np.diff(freqs)[0], axis=2,
                                                  edgecut_low=5, edgecut_hi=5, window='bh')
    mdl_wedge_fft, delays = hc.vis_clean.fft_data(mdl_wedge, np.diff(freqs)[0], axis=2,
                                                  edgecut_low=5, edgecut_hi=5, window='bh')

    print(cal_wedge_fft.shape)
    np.save("/home/ntsikelelo/non_redundancy_sim/mdl_wedge_fft_zeroth_vis_no_filter_"+mode+"_2h.npy",mdl_wedge_fft)
    np.save("/home/ntsikelelo/non_redundancy_sim/cal_wedge_fft_zeroth_vis_no_filter_"+mode+"_2h.npy",cal_wedge_fft) 
