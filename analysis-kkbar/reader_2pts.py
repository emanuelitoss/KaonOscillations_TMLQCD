import numpy as np
import matplotlib.pyplot as plt

home_dir = '/Users/emanuelerosi/Thesis_MSc/kaons-oscillations/tm-mesons-obc/mesons-master/dat/'
file_name = 'mesons_run_name.mesons.dat'
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

# read file_head
int_reads = np.fromfile(path_to_file,dtype='<i4',offset=0,count=4)
ncorr=int_reads[0]
nnoise=int_reads[1]
ntimes=int_reads[2]
noisetype=int_reads[3]
print(  '\nGeneral settings:'
        '\n\tNumber of correlators Ncorr =',ncorr,
        '\n\tNumber of noise spinors Nnoise =',nnoise,
        '\n\tNumber of timeslices N_T =',ntimes,
        '\n\tNoisetype:',noisetype)

kappas=[[0]*2]*ncorr
mus=[[0]*2]*ncorr
type1=[0]*ncorr
type2=[0]*ncorr
x0=[0]*ncorr
isreal=[0]*ncorr

for icorr in np.arange(0,ncorr,1):
    print("\nCorrelator #",icorr)
    offset_byte = 4*4 + icorr*(4*8+4*4)
    tmp=np.fromfile(path_to_file,dtype='<f8',offset=offset_byte,count=4)
    kappas[icorr]=[tmp[0],tmp[1]]
    mus[icorr]=[tmp[2],tmp[3]]
    tmp = np.fromfile(path_to_file,dtype='<i4',offset=offset_byte+4*8,count=4)
    type1[icorr]=tmp[0]
    type2[icorr]=tmp[1]
    x0[icorr]=tmp[2]
    isreal[icorr]=tmp[3]
    tmp_string=[0,0,0,0]
    for idx,val in enumerate(kappas[icorr]):
        tmp_string[idx] = '{:.3f}'.format(val)
    print(  '\tHopping parameters for each correlator:',tmp_string,
            '\n\tTwisted masses:',mus[icorr],
            '\n\tOperator in source y0',type2[icorr],
            '\n\tOperator in source x0',type1[icorr],
            '\n\tIsreal:',isreal[icorr])

# expected data length
numcount=nnoise*ntimes*sum(np.array([2]*ncorr)-isreal)
# number of bytes already read
offset_byte=4*4 + ncorr*(4*8+4*4)
# read data
raw_data = np.fromfile(path_to_file,dtype='<f8',offset=offset_byte)
error_check(numcount!=len(raw_data),'len(data)!=expected lenght')

########################## DATA REORDERING ##########################

data=[[[0]*ntimes]*ncorr]*2
for i1 in np.arange(0,nnoise,1):
    for time in np.arange(0,ntimes,1):
        for corr in np.arange(0,ncorr,1):
            idx=i1 + nnoise*(time+ntimes*corr)
            data[0][corr][time] += raw_data[idx]        # real part
            if(isreal[corr]==0):                        # immaginary part
                data[1][corr][time] += raw_data[idx] 
                
data = np.array(data)
                
########################## CORRELATOR PLOT ##########################             

fig = plt.figure(figsize=(13,5),dpi=300)
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Real value of the correlator")
plt.xlim(0,ntimes)
if(min(tmp)!=max(tmp)):
    plt.ylim(min(data[0][0])*0.9,max(data[0][0])*1.12)
plt.grid()
plt.plot(data[0][0],'1',c='darkblue', alpha=0.7,lw=0.7)
corr_name = '$\mathbb{Re}(C_{PS+SP}(x_4,y_4,z_4))$'
plt.title('Correlator: '+corr_name)
fig.savefig('plots/sampleplot_2pts.png',dpi=fig.dpi)
