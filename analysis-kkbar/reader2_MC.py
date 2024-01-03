# QUESTO è UGUALE A READER2PTS.PY MA PUò LEGGERE PIù CONFIGURAZIONI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir = '/Users/emanuelerosi/thesis-MSc/kaons-oscillations/tm-mesons-obc/mesons-master/dat/'
file_name = 'YMpureSU3-2pts.mesons.dat'
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
int_tmp = np.fromfile(path_to_file,dtype='<i4',offset=0,count=4)
ncorr=int_tmp[0]
nnoise=int_tmp[1]
ntimes=int_tmp[2]
noisetype=int_tmp[3]
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

# configurations reading
data_count=nnoise*ntimes*sum(np.array([2]*ncorr)-isreal)    # data to read for each configuration
offset_byte=4*4+ncorr*(4*8+4*4)                             # offset in byte due to file_head
cnfgs_nums=[]                                               # numbers of configurations
raw_data=[]                                                 # giga-array of data 
reader_counter=True                                         # it tells whenever to stop the reading procedure

printCyan('Reading configurations...')

while(reader_counter==True):
    
    # read configuration number
    offset=offset_byte+len(cnfgs_nums)*(4+data_count*8)
    configuration_number=np.fromfile(path_to_file,dtype='<i4',offset=offset,count=1)
    cnfgs_nums.append(configuration_number[0])
    offset+=4
    
    # read raw data
    raw_data.append(np.fromfile(path_to_file,dtype='<f8',offset=offset,count=data_count))
    print('\tconfiguration number:'+str(cnfgs_nums[-1])+' --> loaded')
    
    # is there any other configuration?
    configuration_number=np.fromfile(path_to_file,dtype='<i4',offset=offset+data_count*8,count=1)
    reader_counter=(len(configuration_number)==1)
    
printCyan('Data loaded.\n')
    
########################## DATA REORDERING ##########################

# create an ordered data structure
data=np.array([[[0]*2]*ntimes]*ncorr,dtype=np.float128)
data_err=np.array([[[0]*2]*ntimes]*ncorr,dtype=np.float128)
data_tmp=np.array([[[[0]*2]*ntimes]*ncorr]*len(cnfgs_nums),dtype=np.float128)
twos=np.array([[[[2]*2]*ntimes]*ncorr]*len(cnfgs_nums),dtype=np.float128)

printCyan('...reordering data...\n')

for int_tmp,nc in enumerate(cnfgs_nums):
    offset_idx=0
    for corr in np.arange(0,ncorr,1):
        if(isreal[corr]==True):
            for time in np.arange(0,ntimes,1):
                for i1 in np.arange(0,nnoise,1):
                    idx=offset_idx + i1 + time*nnoise
                    data_tmp[int_tmp][corr][time][0] += raw_data[int_tmp][idx]    # only real part
                data_tmp[int_tmp][corr][time][0] = data_tmp[int_tmp][corr][time][0]/(volume3D*nnoise)

        else:
            for time in np.arange(0,ntimes,1):
                for i1 in np.arange(0,nnoise,1):
                    idx=offset_idx + 2*(i1 + time*nnoise)
                    data_tmp[int_tmp][corr][time][0] += raw_data[int_tmp][idx]    # real part
                    data_tmp[int_tmp][corr][time][1] += raw_data[int_tmp][idx+1]  # immaginary part
                data_tmp[int_tmp][corr][time][0]/=(volume3D*nnoise)
                data_tmp[int_tmp][corr][time][1]/=(volume3D*nnoise)
        offset_idx+=nnoise*ntimes*(2-isreal[corr])


np.sum(data_tmp/len(cnfgs_nums),axis=0,dtype=np.float128,out=data)          # mean <x>

data_tmp_err=np.power(data_tmp,twos)
np.sum(data_tmp_err/len(cnfgs_nums),axis=0,dtype=np.float128,out=data_err)  # mean <x^2>
data_err=data_err-(data*data)   # variance <x^2> - <x>^2
data_err=np.sqrt(data_err)

########################## CORRELATORS PLOT ##########################             

printCyan('...plotting data...\n')

# Real part of correlators
real_plot=plt.figure(1,figsize=(13,5),dpi=300)
plt.title(r'Correlators $C_{M\overline{M}}(x_4,y_4) = \frac{a^3}{N_{sp}}\sum_{\vec{x},\vec{y}} < M(x) \overline{M}(y) >$')
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Real value of the correlators")
plt.xlim(0,ntimes-1)
plt.grid(linewidth=0.1)
for icorr in np.arange(0,ncorr,1):
    plt.errorbar(np.arange(0,ntimes,1),data[icorr,:,0],yerr=data_err[icorr,:,0],label='correlator #'+str(icorr),
            marker='o', markersize=2.5, lw=0, elinewidth=0.5, capsize=5, markeredgewidth=0.5)
plt.legend(loc='upper right')

# Immaginary part of correlators
immaginary_plot=plt.figure(2,figsize=(13,5),dpi=300)
plt.title(r'Correlators $C_{M\overline{M}}(x_4,y_4) = \frac{a^3}{N_{sp}}\sum_{\vec{x},\vec{y}} < M(x) \overline{M}(y) >$')
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Immaginary value of the correlators")
plt.xlim(0,ntimes-1)
plt.grid(linewidth=0.1)
for icorr in np.arange(0,ncorr,1):
    plt.errorbar(np.arange(0,ntimes,1),data[icorr,:,1],yerr=data_err[icorr,:,1],label='correlator #'+str(icorr),
            marker='o', markersize=2.5, lw=0, elinewidth=0.5, capsize=5, markeredgewidth=0.5)
plt.legend(loc='upper right')

# Save plots in a single file
pp = PdfPages("plots/pureYM-2pts.pdf")
pp.savefig(real_plot, dpi=real_plot.dpi, transparent = True)
pp.savefig(immaginary_plot, dpi=immaginary_plot.dpi, transparent = True)

if ncorr==1:
    firstPage = plt.figure(1,figsize=(13,5),dpi=300)
    firstPage.clf()
    text = 'GENERAL SETTINGS:\n'+r'$N_{config}$ = '+str(len(cnfgs_nums))+'\n'+r'$N_{corr}$ = '+str(ncorr)+'\n'+'$N_{noise}$ = '+str(nnoise)+'\n$N_T$ = '+str(ntimes)+'\nSource timeslice: $x_0$ = '+str(x0)+'\n'+'\n\nCORRELATOR:\nHopping parameters: '+str(tmp_string)+'\n'+r'Twisted masses $\mu_s$ : '+str(mus[0])+'\n'+r'$\Gamma_X$: '+dirac_to_str(type1[0])+'\n'+r'$\Gamma_Y$: '+dirac_to_str(type2[0])+' \nIsreal: '+str(bool(isreal[0]))+'\n'
    firstPage.text(0.1,0.2,text,transform=firstPage.transFigure,size=12,ha="left",fontstyle='normal')
    pp.savefig(firstPage, dpi=firstPage.dpi, transparent = True)
    plt.close()
else:
    firstPage = plt.figure(1,figsize=(13,5),dpi=300)
    firstPage.clf()
    text = 'GENERAL SETTINGS:\n'+r'$N_{config}$ = '+str(len(cnfgs_nums))+'\n'+r'$N_{corr}$ = '+str(ncorr)+'\n'+'$N_{noise}$ = '+str(nnoise)+'\n$N_T$ = '+str(ntimes)+'\nSource timeslice: $x_0$ = '+str(x0)+'\n'
    firstPage.text(0.1,0.2,text,transform=firstPage.transFigure,size=12,ha="left",fontstyle='normal')
    pp.savefig(firstPage, dpi=firstPage.dpi, transparent = True)
    plt.close()
    
pp.close()