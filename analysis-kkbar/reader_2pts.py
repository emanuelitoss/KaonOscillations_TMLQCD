import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir = '/Users/emanuelerosi/Thesis_MSc/kaons-oscillations/tm-mesons-obc/mesons-master/dat/'
file_name = 'mesons_run_name.mesons.dat'
path_to_file = home_dir+file_name

# this must be set by the user 
N1=8
N2=8
N3=8
volume3D=N1*N2*N3   # number of lattice points for each timeslice

######################## TOOLS ########################

def printRed(message):
    print("\033[91m{}\033[00m".format(message))
def printCyan(message):
    print("\033[96m{}\033[00m".format(message))

def error_check(condition,error_message,optional_message=''):
    if(condition):
        printRed('ERROR:'+optional_message)
        printRed(error_message)
        quit()
        
def noise_to_string(type):
    match type:
        case 0:
            return 'Z2'
        case 1:
            return 'GAUSS'
        case 2:
            return 'U1'
        case _:
            error_check(True,"Invalid noise type")
        
def dirac_to_str(type):
    match type:
        case 0:
            return 'G0'
        case 1:
            return 'G1'
        case 2:
            return 'G2'
        case 3:
            return 'G3'
        case 5:
            return 'G5'
        case 6:
            return '1'
        case 7:
            return 'G0G1'
        case 8:
            return 'G0G2'
        case 9:
            return 'G0G3'
        case 10:
            return 'G0G5'
        case 11:
            return 'G1G2'
        case 12:
            return 'G1G3'
        case 13:
            return 'G1G5'
        case 14:
            return 'G2G3'
        case 15:
            return 'G2G5'
        case 16:
            return 'G3G5'
        case _:
            error_check(True,'Invalid Dirac matrix: '+str(type))

########################## DATA READING ##########################
    # LEGEND:
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
printCyan('\nGeneral settings:')
print(  '\tNumber of correlators =',ncorr,
        '\n\tNumber of noise spinors =',nnoise,
        '\n\tNumber of timeslices =',ntimes,
        '\n\tNoisetype:',noise_to_string(noisetype),'\n')

kappas=[[0]*2]*ncorr
mus=[[0]*2]*ncorr
type1=[0]*ncorr
type2=[0]*ncorr
x0=[0]*ncorr
isreal=[0]*ncorr

for icorr in np.arange(0,ncorr,1):
    printCyan("Correlator #"+str(icorr))
    offset_byte = 4*4 + icorr*(4*8+4*4)
    tmp=np.fromfile(path_to_file,dtype='<f8',offset=offset_byte,count=4)
    kappas[icorr]=[tmp[0],tmp[1]]
    mus[icorr]=[tmp[2],tmp[3]]
    tmp = np.fromfile(path_to_file,dtype='<i4',offset=offset_byte+4*8,count=4)
    type1[icorr]=tmp[0]
    type2[icorr]=tmp[1]
    x0[icorr]=tmp[2]
    isreal[icorr]=tmp[3]
    tmp_string=[0,0]
    for idx,val in enumerate(kappas[icorr]):
        tmp_string[idx] = '{:.3f}'.format(val)
    print(  '\tHopping parameters:',tmp_string,
            '\n\tTwisted masses:',mus[icorr],
            '\n\tOperator in source y0:',dirac_to_str(type2[icorr]),
            '\n\tOperator in source x0:',dirac_to_str(type1[icorr]),
            '\n\tIsreal:',bool(isreal[icorr]),'\n')

# read configuration number
offset_byte=4*4 + ncorr*(4*8+4*4)
configuration_number=np.fromfile(path_to_file,dtype='<i4',offset=offset_byte,count=1)
printCyan('Configuration number:'+str(configuration_number[0])+' --> loaded')
offset_byte+=4

# read raw data
data_count=nnoise*ntimes*sum(np.array([2]*ncorr)-isreal)
raw_data=np.fromfile(path_to_file,dtype='<f8',offset=offset_byte)
error_check(data_count!=len(raw_data),'len(data)!=expected lenght')

########################## DATA REORDERING ##########################

# create an ordered data structure
offset_idx=0
data = np.array([[[0]*ntimes]*ncorr]*2,dtype=np.float64)
    
for icorr in np.arange(0,ncorr,1):
    if(isreal[icorr]==True):
        for time in np.arange(0,ntimes,1):
            for inoise in np.arange(0,nnoise,1):
                idx=offset_idx+inoise+nnoise*time
                data[0][icorr][time] += raw_data[idx]   # only real part
    else:
        for time in np.arange(0,ntimes,1):
            for inoise in np.arange(0,nnoise,1):
                idx=offset_idx+2*(inoise+nnoise*time)
                data[0][icorr][time] += raw_data[idx]   # real part
                data[1][icorr][time] += raw_data[idx+1] # immaginary part
    offset_idx+=nnoise*ntimes*(2-isreal[icorr])

# divide by the norm - simulation doesn't do it
for idx,value in enumerate(data):
    data[idx]=value/(volume3D*nnoise)
    
########################## CORRELATORS PLOT ##########################             

# Real part of correlators
real_plot=plt.figure(1,figsize=(13,5),dpi=300)
plt.title(r'Correlators $C_{M\overline{M}}(x_4,y_4) = \frac{a^3}{N_{sp}}\sum_{\vec{x},\vec{y}} < M(x) \overline{M}(y) >$')
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Real value of the correlators")
plt.xlim(0,ntimes)
plt.grid()
for icorr in np.arange(0,ncorr,1):
    plt.plot(data[0][icorr],str(icorr+1),label='correlator #'+str(icorr),alpha=1,lw=0.75)
plt.legend(loc='upper right')

# Immaginary part of correlators
immaginary_plot=plt.figure(2,figsize=(13,5),dpi=300)
plt.title(r'Correlators $C_{M\overline{M}}(x_4,y_4) = \frac{a^3}{N_{sp}}\sum_{\vec{x},\vec{y}} < M(x) \overline{M}(y) >$')
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Immaginary value of the correlators")
plt.xlim(0,ntimes)
plt.grid()
for icorr in np.arange(0,ncorr,1):
    plt.plot(data[1][icorr],str(icorr+1),label='correlator #'+str(icorr),alpha=1,lw=0.75)
plt.legend(loc='upper right')

# Save plots in a single file
pp = PdfPages("plots/Plots_2pts.pdf")
pp.savefig(real_plot, dpi=real_plot.dpi, transparent = True)
pp.savefig(immaginary_plot, dpi=immaginary_plot.dpi, transparent = True)
pp.close()
