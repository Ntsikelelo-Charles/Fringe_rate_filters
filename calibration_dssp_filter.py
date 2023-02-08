import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal
import glob
import os
import copy
import hera_sim as hs
from pyuvdata import UVData, UVBeam, utils as uvutils
import hera_pspec as hp
import hera_cal as hc
import healpy

path='/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/'
path_raw_data='/home/ntsikelelo/Simulated_data_files/UVH5_files/'

## need for gain keys

all_filter_type=np.array(["notch_filter_0.25_dssp","notch_filter_0.40_dssp","notch_filter_0.60_dssp"])

for filt in range (len(all_filter_type)):
    filter_type=all_filter_type[filt]


    filter_max=0
    if filter_type=="notch_filter_0.40_dssp":
        filter_max=40

    if filter_type=="notch_filter_0.25_dssp":
        filter_max=25

    if filter_type=="notch_filter_0.60_dssp":
        filter_max=60    
    print(filter_type)



    mode_array=np.array(["incomplete_with_filter_baseline_cut","incomplete_with_filter","complete_with_filter"])



    sigma_frac=np.load('/net/sinatra/vault-ike/ntsikelelo/Simulated_data_files/sigma_frac_'+str(filter_max)+'.npy')                   


#     t=np.array([118,126])
    t=np.zeros((8),dtype=int)
    t[0:8]=np.linspace(0,200,8)
    chisq_perfect=np.zeros(shape=(200,308))
    chisq_perfect_redcal=np.zeros(shape=(200,308))
    gains_all_times=np.zeros(shape=(91,200,308),dtype=complex)
    gains_all_times_redcal=np.zeros(shape=(91,200,308),dtype=complex)


    cal_wedge_all=np.zeros(shape=(200,32,308),dtype=complex)
    mdl_wedge_all=np.zeros(shape=(200,32,308),dtype=complex)

    model_file=''
    raw_file=path+"Raw_filtered_data_deep_dpss_"+str(filter_max)+".uvh5"
    raw_file_no_filter=path_raw_data+"Raw_data_with_noise.uvh5"

    for step in range (len(mode_array)):
        mode=mode_array[step]
        print(mode)               
        if mode=="incomplete_with_filter_baseline_cut":                    
            model_file=path+"Model_incomplete_filtered_data_dpss_"+str(filter_max)+".uvh5"


        if mode=="complete_with_filter" :                   
            model_file=path+"Model_complete_filtered_data_dpss_"+str(filter_max)+".uvh5"


        if mode=="complete_with_filter_baseline_cut" :                   
            model_file=path+"Model_complete_filtered_data_dpss_"+str(filter_max)+".uvh5"




        if mode=="incomplete_with_filter" :                   
            model_file=path+"Model_incomplete_filtered_data_dpss_"+str(filter_max)+".uvh5"




        for N_time in range (len(t)-1):
            print(t[N_time],t[N_time+1])
            model = hc.io.HERAData(model_file)
            antpos, ants = model.get_ENU_antpos()
            antpos_d = dict(zip(ants, antpos))
            N=0
            times = np.unique(model.time_array)
            model.read(times=times[t[N_time]:t[N_time+1]])

            model.inflate_by_redundancy()
            model.conjugate_bls()
            model._determine_blt_slicing()
            model._determine_pol_indexing()
            model.x_orientation = 'east'

            bl_lens = {bl: np.linalg.norm(antpos_d[bl[1]]-antpos_d[bl[0]]) for bl in model.get_antpairs()}   
            # downselect to maximum baseline length (to reduce computational load in calibration)
            bls = []
            for bl in list(model.get_antpairs()):
                # max 40 m baseline cut (also cut autos)
                bl_len = np.linalg.norm(antpos_d[bl[1]] - antpos_d[bl[0]])
                if bl_len > 0:
                    bls.append(bl[:2])   
            model.select(bls=bls)

            # load the metadata
            raw = hc.io.HERAData(raw_file)
            times = np.unique(raw.time_array)
            raw.read(times=times[t[N_time]:t[N_time+1]])
            raw.select(bls=bls)
            raw.x_orientation = 'east'
            
            ## no filter data
            raw_no_filter = hc.io.HERAData(raw_file_no_filter)
            times = np.unique(raw_no_filter.time_array)
            raw_no_filter.read(times=times[t[N_time]:t[N_time+1]])
            raw_no_filter.select(bls=bls)
            raw_no_filter.x_orientation = 'east'

            model_data, _, _ = model.build_datacontainers()
            raw_data, _, _ = raw.build_datacontainers()
            raw_data_no_filter, _, _ = raw_no_filter.build_datacontainers()


            Nrms =1e-3
            #Choosing baseline cut

            # get chisq after redcal and abscal
            noise_wgts = {k: np.ones_like(raw_data[k], dtype=float) / (sigma_frac*Nrms)**2 for k in raw_data}


            print("data loaded")                
            # get redundant baseline groups
            reds = hc.redcal.get_reds(antpos_d, pols=['ee'])
            # filter redundant baselines
            filtered_reds = hc.redcal.filter_reds(reds, max_dims=2, bls=model_data.keys())

            # get RC
            rc = hc.redcal.RedundantCalibrator(filtered_reds)


            print("performing logcal")
            # perform logcal
            freqs=model.freqs
            logcal_meta, logcal_sol = rc.logcal(raw_data)
            hc.redcal.make_sol_finite(logcal_sol)
            # remove redcal degeneracies from solution
            logcal_sol = rc.remove_degen(logcal_sol)
            # get gains and model visibilities
            logcal_gains, logcal_vis = hc.redcal.get_gains_and_vis_from_sol(logcal_sol)


            print("performing lincal")
            # perform omnical (lincal)
            conv_crit = 1e-10
            maxiter = 500
            gain = 0.4
            lincal_meta, lincal_sol = rc.omnical(raw_data, logcal_sol, wgts=noise_wgts, conv_crit=conv_crit,
                                             maxiter=maxiter, gain=gain)
            hc.redcal.make_sol_finite(lincal_sol)
            # remove redcal degeneracies from solution
            lincal_sol = rc.remove_degen(lincal_sol)
            # get gains and model visibilities
            lincal_gains, lincal_vis = hc.redcal.get_gains_and_vis_from_sol(lincal_sol) 


            # compute chisq
            rc_red_chisq, chisq_per_ant = hc.redcal.normalized_chisq(raw_data, noise_wgts, filtered_reds, lincal_vis, lincal_gains)
            rc_red_chisq_dof = len(raw_data) - len(lincal_gains)
            rc_red_chisq = rc_red_chisq['Jee']

            rc_flags = {k: np.zeros_like(lincal_gains[k], dtype=bool) for k in lincal_gains}
            # calibration with lincal gains
            redcal_data = copy.deepcopy(raw_data)
            hc.apply_cal.calibrate_in_place(redcal_data, lincal_gains)
            # run post-redundant calibration abscal
            print("performing abs_cal post redcal")
            
            if mode=="incomplete_with_filter_baseline_cut" or mode=="complete_with_filter_baseline_cut" :
                print("base line cut")
                for k in raw_data:
                    blvec = (antpos[k[0]] - antpos[k[1]])
                    bl_len_EW = np.abs(blvec[0])

                    if bl_len_EW < 30:    
                        noise_wgts[k][:] = 1e-40


            abscal_gains = hc.abscal.post_redcal_abscal(model_data, redcal_data, noise_wgts, rc_flags, verbose=False, phs_max_iter=100, phs_conv_crit=1e-6)

            print("done callibration")
            ref_ant = (0, 'Jee')
            # combine redcal and abscal gains
            total_gains = hc.abscal.merge_gains([lincal_gains, abscal_gains])

            # make sure it has the same reference antenna
            hc.abscal.rephase_to_refant(total_gains, ref_ant)
            #overwrite weights for chi-sqaure
            noise_wgts = {k: np.ones_like(raw_data[k], dtype=float) / (sigma_frac*Nrms)**2 for k in raw_data}
            abs_chisq, nObs, _, _ = hc.utils.chisq(raw_data, model_data, gains=total_gains, data_wgts=noise_wgts)
            chisq_dof = nObs.mean() - len(abscal_gains)
            red_chisq = abs_chisq / (chisq_dof) 



            # calibrate data
            cal_data = copy.deepcopy(raw_data_no_filter)
            hc.apply_cal.calibrate_in_place(cal_data, total_gains)

            # get redundant groups
            antpos, ants = model.get_ENU_antpos()
            antpos_dict = dict(zip(ants, antpos))
            reds = hc.redcal.get_pos_reds(antpos_dict)

            # get all baselines of the same length
            bl_lens, bl_groups = [], []
            for red in reds:
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
            N_t=len(np.unique(model.time_array))
            cal_wedge = np.zeros((N_t,len(bl_groups), model.Nfreqs, model.Npols), dtype=np.complex128)
            mdl_wedge = np.zeros((N_t,len(bl_groups), model.Nfreqs, model.Npols), dtype=np.complex128)

            for i, bl_group in enumerate(bl_groups):
                for j, pol in enumerate(model.get_pols()):
                    cal_wedge[:,i, :, j] = np.mean([cal_data[bl + (pol,)] for bl in bl_group], axis=0)
                    mdl_wedge[:,i, :, j] = np.mean([model_data[bl + (pol,)] for bl in bl_group], axis=0)


            # now take the FFT across frequency: cut the edge channels
            cal_wedge_fft, delays = hc.vis_clean.fft_data(cal_wedge, np.diff(freqs)[0], axis=2,
                                                          edgecut_low=5, edgecut_hi=5, window='bh')
            mdl_wedge_fft, delays = hc.vis_clean.fft_data(mdl_wedge, np.diff(freqs)[0], axis=2,
                                                          edgecut_low=5, edgecut_hi=5, window='bh')

            print(cal_wedge_fft.shape)

            cal_wedge_all[t[N_time]:t[N_time+1],:,:]=cal_wedge_fft[:,:, :, 0]
            mdl_wedge_all[t[N_time]:t[N_time+1],:,:]=mdl_wedge_fft[:,:, :, 0]
            
 
            for ant in range (91):
            
                gains_all_times_redcal[ant,t[N_time]:t[N_time+1],:]=lincal_gains[(ant, 'Jee')]
                gains_all_times[ant,t[N_time]:t[N_time+1],:]=total_gains[(ant, 'Jee')]
                
                
            chisq_perfect_redcal[t[N_time]:t[N_time+1],:]=rc_red_chisq
            chisq_perfect[t[N_time]:t[N_time+1],:]=red_chisq
            

        path2="/home/ntsikelelo/Simulated_data_files/"
        np.save(path2+"gains_redcal_"+filter_type+"_"+mode,gains_all_times_redcal)
        np.save(path2+"gains_redcal_and_abscal_"+filter_type+"_"+mode,gains_all_times)
        np.save(path2+"correct_sigma_chisq_redcal_sky_model_"+filter_type+"_"+mode+".npy",chisq_perfect_redcal)
        np.save(path2+"correct_sigma_chisq_redcal_and_abscal_sky_model_"+filter_type+"_"+mode+".npy",chisq_perfect)
        np.save(path2+"cal_wedge_redcal_"+filter_type+"_"+mode+".npy",cal_wedge_all) 
        np.save(path2+"mdl_wedge_redcal_"+filter_type+"_"+mode+".npy",mdl_wedge_all)  
