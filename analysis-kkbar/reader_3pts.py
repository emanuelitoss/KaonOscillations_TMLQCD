import numpy as np
import matplotlib.pyplot as plt

home_dir = '/Users/emanuelerosi/Thesis_MSc/kaons-oscillations/tm-mesons-obc/mesons-master/dat/'
file_name = 'crrltrs_run_name.correlators.dat'
path_to_file = home_dir+file_name

######################## FUNCTIONS, TOOLS ########################

def printRed(skk):
    print("\033[91m{}\033[00m".format(skk))
def prCyan(skk):
    print("\033[96m{}\033[00m".format(skk))

def error_check(condition,error_message,optional_message=''):
    if(condition):
        printRed('ERROR:'+optional_message)
        printRed(error_message)
        quit()

########################## DATA READING ##########################
    # dtypes:
    #   np.int16    --> i2
    #   np.int32    --> i4
    #   np.int64    --> i8
    #   np.double   --> f8 
    # use <f for LITTLE_ENDIAN
    # use >f for BIG_ENDIAN

int_reads = np.fromfile(path_to_file,dtype='<i4',offset=0,count=6)
ncorr=int_reads[0]
nnoise=int_reads[1]
ntimes=int_reads[2]
noisetype=int_reads[3]
x0=int_reads[4]
z0=int_reads[5]
print(  '\nGeneral settings:'
        '\n\tNumber of correlators Ncorr =',ncorr,
        '\n\tNumber of noise spinors Nnoise =',nnoise,
        '\n\tNumber of timeslices N_T =',ntimes,
        '\n\tNoisetype:',noisetype,
        '\n\tSources timeslices: z0 =',z0,' x0 =',x0)

kappas=[[0]*4]*ncorr
mus=[[0]*4]*ncorr
typeA_x=[0]*ncorr
typeC_z=[0]*ncorr
operator_type=[0]*ncorr
isreal=[0]*ncorr

for icorr in np.arange(0,ncorr,1):
    print("\nCorrelator #",icorr)
    offset_byte = 6*4 + icorr*(8*8+4*4)
    tmp=np.fromfile(path_to_file,dtype='<f8',offset=offset_byte,count=8)
    kappas[icorr]=[tmp[0],tmp[1],tmp[2],tmp[3]]
    mus[icorr]=[tmp[4],tmp[5],tmp[6],tmp[7]]
    tmp = np.fromfile(path_to_file,dtype='<i4',offset=offset_byte+8*8,count=4)
    typeA_x[icorr]=tmp[0]
    typeC_z[icorr]=tmp[1]
    operator_type[icorr]=tmp[2]
    isreal[icorr]=tmp[3]
    tmp_string=[0,0,0,0]
    for idx,val in enumerate(kappas[icorr]):
        tmp_string[idx] = '{:.3f}'.format(val)
    print(  '\tHopping parameters for each correlator:',tmp_string,
            '\n\tTwisted masses:',mus[icorr],
            '\n\tOperator in source z0',typeC_z[icorr],
            '\n\tOperator in source x0',typeA_x[icorr],
            '\n\tIsreal:',isreal[icorr])
    
numcount=2*ncorr*nnoise*nnoise*ntimes
offset_byte=6*4 + ncorr*(8*8+4*4)
raw_data = np.fromfile(path_to_file,dtype='<f8',offset=offset_byte)
error_check(numcount!=len(raw_data),'len(data)!=expected lenght')

########################## DATA REORDERING ##########################

data=[[[0]*2]*ntimes]*ncorr
for i2 in np.arange(0,nnoise,1):
    for i1 in np.arange(0,nnoise,1):
        for time in np.arange(0,ntimes,1):
            for corr in np.arange(0,ncorr,1):
                idx=i2 + nnoise*i1 + nnoise*nnoise*(time+ntimes*corr)
                data[corr][time][0] += raw_data[2*idx]      # real part
                data[corr][time][1] += raw_data[2*idx+1]    # immaginary part
                
data = np.array(data)
                
########################## CORRELATOR PLOT ##########################             

# The first correlator is PS, the second SP. Then:
corr_PSpSP = data[0] + data[1]
corr_PSmSP = data[0] - data[1]

tmp = np.array([0]*len(corr_PSpSP))
for idx,val in enumerate(corr_PSpSP):
    tmp[idx]=val[0]    

fig = plt.figure(figsize=(13,5),dpi=300)
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Real value of the correlator")
plt.xlim(0,ntimes)
if(min(tmp)!=max(tmp)):
    plt.ylim(min(tmp)*0.9,max(tmp)*1.12)
plt.grid()
plt.plot(tmp,'1',c='darkred', alpha=0.7,lw=0.9)
corr_name = '$\mathbb{Re}(C_{PS+SP}(x_4,y_4,z_4))$'
plt.title('Correlator: '+corr_name)
fig.savefig('plots/sampleplot_3pts.png',dpi=fig.dpi)
