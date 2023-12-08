# QUESTO è UGUALE A READER3PTS.PY MA PUò LEGGERE PIù CONFIGURAZIONI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir = '/Users/emanuelerosi/Thesis_MSc/kaons-oscillations/tm-mesons-obc/mesons-master/dat/'
file_name = 'test3.correlators.dat'
path_to_file = home_dir+file_name

# this must be set by the user 
N1=8
N2=8
N3=8
volume3D=N1*N2*N3   # number of lattice points for each timeslice

######################## FUNCTIONS, TOOLS ########################

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
            error_check(1,'Invalid Dirac matrix: '+str(type))
            
def operator_to_str(type):
    match type:
        case 0:
            return 'SS'
        case 1:
            return 'PP'
        case 2:
            return 'SP'
        case 3:
            return 'PS'
        case 4:
            return 'VV'
        case 5:
            return 'AA'
        case 6:
            return 'VA'
        case 7:
            return 'AV'
        case 8:
            return 'TT'
        case 9:
            return 'TTt'
        case _:
            error_check(1,'Invalid operator matrix: '+str(type))

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
int_reads = np.fromfile(path_to_file,dtype='<i4',offset=0,count=6)
ncorr=int_reads[0]
nnoise=int_reads[1]
ntimes=int_reads[2]
noisetype=int_reads[3]
x0=int_reads[4]
z0=int_reads[5]
printCyan('\nGeneral settings')
print(  '\tNumber of correlators Ncorr =',ncorr,
        '\n\tNumber of noise spinors Nnoise =',nnoise,
        '\n\tNumber of timeslices N_T =',ntimes,
        '\n\tNoisetype:',noise_to_string(noisetype),
        '\n\tSource timeslice: x0 =',x0,
        '\n\tSource timeslice: z0 =',z0,'\n')

kappas=[[0]*4]*ncorr
mus=[[0]*4]*ncorr
typeA_x=[0]*ncorr
typeC_z=[0]*ncorr
operator_type=[0]*ncorr
isreal=[0]*ncorr

for icorr in np.arange(0,ncorr,1):
    printCyan("Correlator # "+str(icorr))
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
    print(  '\tHopping parameters:',tmp_string,
            '\n\tTwisted masses:',mus[icorr],
            '\n\tGammaA:',dirac_to_str(typeC_z[icorr]),
            '\n\tGammaC:',dirac_to_str(typeA_x[icorr]),
            '\n\tMixing operator:',operator_to_str(operator_type[icorr]),
            '\n\tIsreal:',bool(isreal[icorr]),'\n')

# configurations reading
data_count=nnoise*nnoise*ntimes*sum(np.array([2]*ncorr)-isreal)    # data to read for each configuration
offset_byte=6*4+ncorr*(8*8+4*4)                             # offset in byte due to file_head
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

printCyan('...reordering data...\n')

for int_tmp,nc in enumerate(cnfgs_nums):
    offset_idx=0 
    for corr in np.arange(0,ncorr,1):
        if(isreal[corr]==True):
            for time in np.arange(0,ntimes,1):
                for i1 in np.arange(0,nnoise,1):
                    for i2 in np.arange(0,nnoise,1):
                        idx=offset_idx + i2 + i1*nnoise + time*nnoise*nnoise
                        data[corr][time][0] += raw_data[int_tmp][idx]    # only real part
                data[corr][time][0] = data[corr][time][0]/(volume3D*nnoise*nnoise)
        else:
            for time in np.arange(0,ntimes,1):
                for i1 in np.arange(0,nnoise,1):
                    for i2 in np.arange(0,nnoise,1):
                        idx=offset_idx + 2*(i2 + i1*nnoise + time*nnoise*nnoise)
                        data[corr][time][0] += raw_data[int_tmp][idx]    # real part
                        data[corr][time][1] += raw_data[int_tmp][idx+1]  # immaginary part
                data[corr][time][0] = data[corr][time][0]/(volume3D*nnoise*nnoise)
                data[corr][time][1] = data[corr][time][1]/(volume3D*nnoise*nnoise)
        offset_idx+=nnoise*nnoise*ntimes*(2-isreal[corr])
                
########################## CORRELATORS PLOT ##########################             

printCyan('...plotting data...\n')

# The first correlator is Psi1, the second Psi2.
# Then I create [VV+AA] and [VV-AA]
# ... and so on ...
correlators = [-2*(data[0]+data[1]),-2*(data[0]-data[1]),2*(data[2]-data[3]),2*(data[2]+data[3]),2*data[4]]
tmp = np.array([0]*ntimes,dtype=np.float128)
plot_names = [r'$O_{VV+AA}$',r'$O_{VV-AA}$',r'$O_{SS-PP}$',r'$O_{SS+PP}$',r'$O_{TT}$']

pp = PdfPages("plots/3pts-plot.pdf")
        
# Real part of correlators
real_plot=plt.figure(1,figsize=(13,5),dpi=300)
plt.title(r'Correlators: $\mathbb{Re}\left\{C_{i[+]}(x_4,y_4,z_4)\right\}$')
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Real value of the correlator")
plt.xlim(0,ntimes-1)
plt.grid()
for idx,corr in enumerate(correlators):
    for i,val in enumerate(corr):
        tmp[i]=-val[1]  # there is an 'i' in front of the correlator definition!
    if(idx<2):
        plt.plot(tmp,str(idx+1),label=plot_names[idx],alpha=1,lw=0.75)
    else:
        plt.plot(tmp,'*',label=plot_names[idx],alpha=1,lw=0.75)
plt.legend(loc='upper right')
pp.savefig(real_plot, dpi=real_plot.dpi, transparent = True)
plt.close()

# Immaginary part of correlators
immaginary_plot=plt.figure(2,figsize=(13,5),dpi=300)
plt.title(r'Correlators: $\mathbb{Im}\left\{C_{i[+]}(x_4,y_4,z_4)\right\}$')
plt.xlabel("Timeslice $y_4$")
plt.ylabel("Immaginary value of the correlator")
plt.xlim(0,ntimes-1)
plt.grid()
for idx,corr in enumerate(correlators):
    for i,val in enumerate(corr):
        tmp[i]=val[0]   # there is an 'i' in front of the correlator definition!
    if(idx<2):
        plt.plot(tmp,str(idx+1),label=plot_names[idx],alpha=1,lw=0.75)
    else:
        plt.plot(tmp,'*',label=plot_names[idx],alpha=1,lw=0.75)
plt.legend(loc='upper right')    
pp.savefig(immaginary_plot, dpi=immaginary_plot.dpi, transparent = True)
plt.close()

if ncorr==1:
    firstPage = plt.figure(1,figsize=(13,5),dpi=300)
    firstPage.clf()
    text = 'GENERAL SETTINGS:\n'+r'$N_{config}$ = '+str(len(cnfgs_nums))+'\n'+r'$N_{corr}$ = '+str(ncorr)+'\n'+'$N_{noise}$ = '+str(nnoise)+'\n$N_T$ = '+str(ntimes)+'\nSource timeslice: $x_0$ = '+str(x0)+'\n'+'\nSource timeslice: $z_0$ = '+str(z0)+'\n'+'\n\nCORRELATOR:\nHopping parameters: '+str(tmp_string)+'\n'+r'Twisted masses $\mu_s$ : '+str(mus[0])+'\n'+r'$\Gamma_A$: '+dirac_to_str(typeA_x[0])+'\n'+r'$\Gamma_C$: '+dirac_to_str(typeC_z[0])+'\n'+r'Operator $\Psi_i: $'+operator_to_str(operator_type[0])+' \nIsreal: '+str(bool(isreal[0]))+'\n'
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
