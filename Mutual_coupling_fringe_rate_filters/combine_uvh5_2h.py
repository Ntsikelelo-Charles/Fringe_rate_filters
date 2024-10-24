from pyuvdata import UVData
import numpy as np
import copy

from os import listdir
from os.path import isfile, join


f=open('first_order_names.txt',"r")
lines=f.readlines()
result=[]
for x in lines:
	result.append(x)
f.close()


uvh5name_LST_0h=[]

for i in range (len(result)):
	uvh5name_LST_0h.append(result[i][0:21])
	
		
uvh5name_LST_0h=np.array(uvh5name_LST_0h)
print(len(uvh5name_LST_0h))		

N=len(uvh5name_LST_0h)
uv1=UVData()
filename = '/lustre/aoc/projects/hera/Validation/fringe_rate_filtering/xtalk/coupled_sims_more_bandwidth/zen.LST.01.00275.uvh5'
uv1.read(filename)
antpos, ants = uv1.get_ENU_antpos()
freqs=uv1.freq_array[0,:]
N_times=308
print("max = "+str(N_times*2))


for g in range (len(freqs)-1):

	k=333+g 
	print(k,freqs[k]) 
	freqs_select=freqs[k:k+1]

	flex_array=np.zeros(freqs_select.shape,dtype=int)
	N=len(uvh5name_LST_0h)
	print("new script "+str(len(uvh5name_LST_0h)))
	uv3=copy.deepcopy(uv1)
	times = np.unique(uv3.time_array)
	uv3.select(times=times[0])
	for n in range (N_times):
	    i=2*n
	    uv2=UVData()
	    filename2 = '/lustre/aoc/projects/hera/Validation/fringe_rate_filtering/xtalk/coupled_sims_more_bandwidth/'+uvh5name_LST_0h[i]
	    uv2.read(filename2)
	    
	    uv2.flex_spw_id_array=flex_array
	    uv3.flex_spw_id_array=flex_array
	    
	       

	    uv2.select(frequencies=freqs_select)
	    uv3.select(frequencies=freqs_select)
	    
	    times2 = np.unique(uv2.time_array)  
	      
	    uv2.select(times=times2[0])
	    
	    
	    uv3 = uv3 + uv2
	    

	    
	    
	uv3.write_uvh5("/lustre/aoc/projects/hera/ncharles/Model_first_order_visibilities_2h_updated_high_band_"+str(k)+".uvh5",clobber=True)
	    