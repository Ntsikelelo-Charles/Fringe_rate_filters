from pyuvdata import UVData
import numpy as np
import copy

from os import listdir
from os.path import isfile, join


f=open('zeroth_order_names.txt',"r")
lines=f.readlines()
result=[]
for x in lines:
	result.append(x)
f.close()


uvh5name_LST_0h=[]

for i in range (len(result)):
	uvh5name_LST_0h.append(result[i][0:22])
	
		
uvh5name_LST_0h=np.array(uvh5name_LST_0h)
print(len(uvh5name_LST_0h))
		

uv1=UVData()
uv1_freqs=UVData()
filename = '/lustre/aoc/projects/hera/zmartino/hera_calib_model/H4C_1/abscal_files_unique_baselines/zen.2458894.97945.uvh5'
uv1.read(filename)


uv1_freqs=UVData()
filename_freqs = '/lustre/aoc/projects/hera/Validation/fringe_rate_filtering/xtalk/coupled_sims_more_bandwidth/zen.LST.01.00275.uvh5'
uv1_freqs.read(filename_freqs)
freqs=uv1_freqs.freq_array[0,:]

print(freqs.shape)
N_time=150
print("new file")
for k in range (len(freqs)-1):

	uv3=copy.deepcopy(uv1)
	print(freqs[k]) 
#	freqs_select=freqs[np.where((freq_all[k]<freqs) & (freqs<freq_all[k+1]))]  
	freqs_select=freqs[k:k+1]
#	print(freqs_select[-1],freqs_select.shape)
	
	for i in range (N_time):
	    print(N_time-i)
	    uv2=UVData()
	    filename2 = '/lustre/aoc/projects/hera/zmartino/hera_calib_model/H4C_1/abscal_files_unique_baselines/'+uvh5name_LST_0h[i]
	    uv2.read(filename2)
	    
	    
	    uv2.select(frequencies=freqs_select)
	    uv3.select(frequencies=freqs_select)
	    uv3 = uv3 + uv2
	    
	    
	uv3.write_uvh5("/lustre/aoc/projects/hera/ncharles/Model_zeroth_order_visibilities"+str(k)+".uvh5",clobber=True)