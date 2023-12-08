import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir = '/Users/emanuelerosi/Thesis_MSc/kaons-oscillations/tm-mesons-obc/mesons-master/dat/'
file_name = 'check4.mesons.dat'
path_to_file = home_dir+file_name

# this must be set by the user 
N1=8
N2=8
N3=8
volume3D=N1*N2*N3   # number of lattice points for each timeslice

# this number must be set by the user
EPSILON = 1.0e-5   # error on the Gauge tranformed data

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
        '\n\tNumber of timeslices =',ntimes,'\n')

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
data_count=2*ncorr*nnoise*ntimes*sum(np.array([2]*ncorr)-isreal)
raw_data=np.fromfile(path_to_file,dtype='<f8',offset=offset_byte)
error_check(data_count!=len(raw_data),'len(data)!=expected lenght')

########################## DATA REORDERING ##########################

# create an ordered data structure
offset_idx=0
data_std=np.array([[[0]*ntimes]*ncorr]*2,dtype=np.float128)
data_gau=np.array([[[0]*ntimes]*ncorr]*2,dtype=np.float128)
    
for icorr in np.arange(0,ncorr,1):
    if(isreal[icorr]==True):
        for time in np.arange(0,ntimes,1):
            for inoise in np.arange(0,nnoise,1):
                idx=offset_idx+inoise+nnoise*time
                data_std[0][icorr][time] += raw_data[idx]   # only real part
            data_std[0][icorr][time] = data_std[0][icorr][time]/(nnoise*volume3D)
    else:
        for time in np.arange(0,ntimes,1):
            for inoise in np.arange(0,nnoise,1):
                idx=offset_idx+2*(inoise+nnoise*time)
                data_std[0][icorr][time] += raw_data[idx]   # real part
                data_std[1][icorr][time] += raw_data[idx+1] # immaginary part
            data_std[0][icorr][time] = data_std[0][icorr][time]/(nnoise*volume3D)
            data_std[1][icorr][time] = data_std[1][icorr][time]/(nnoise*volume3D)
    offset_idx+=nnoise*ntimes*(2-isreal[icorr])

for icorr in np.arange(0,ncorr,1):
    if(isreal[icorr]==True):
        for time in np.arange(0,ntimes,1):
            for inoise in np.arange(0,nnoise,1):
                idx=offset_idx+inoise+nnoise*time
                data_gau[0][icorr][time] += raw_data[idx]   # only real part
            data_gau[0][icorr][time] = data_gau[0][icorr][time]/(nnoise*volume3D)
    else:
        for time in np.arange(0,ntimes,1):
            for inoise in np.arange(0,nnoise,1):
                idx=offset_idx+2*(inoise+nnoise*time)
                data_gau[0][icorr][time] += raw_data[idx]   # real part
                data_gau[1][icorr][time] += raw_data[idx+1] # immaginary part
            data_gau[0][icorr][time] = data_gau[0][icorr][time]/(nnoise*volume3D)
            data_gau[1][icorr][time] = data_gau[1][icorr][time]/(nnoise*volume3D)
    offset_idx+=nnoise*ntimes*(2-isreal[icorr])
    
############################## CHECK ##############################

counter_good_re=0
counter_bad_re=0
counter_good_im=0
counter_bad_im=0

for corr in np.arange(0,ncorr,1):
    for time in np.arange(0,ntimes,1):
        check_re = abs(data_std[0][corr][time]-data_gau[0][corr][time])<EPSILON
        check_im = abs(data_std[1][corr][time]-data_gau[1][corr][time])<EPSILON
        if(check_re==False):
            counter_bad_re+=1
        else:
            counter_good_re+=1
        if(check_im==False):
            counter_bad_im+=1
        else:
            counter_good_im+=1

printCyan('\nCheck ---> completed')
print('\tEpsilon = ',EPSILON)
print('\tReal part:')
print('\t\tNumber of positively checked values = ',counter_good_re)
print('\t\tNumber of negatively checked values = ',counter_bad_re)
print('\tImmaginary part:')
print('\t\tNumber of positively checked values = ',counter_good_im)
print('\t\tNumber of negatively checked values = ',counter_bad_im)
error_check((counter_good_re+counter_bad_re)!=ncorr*ntimes or (counter_good_im+counter_bad_im)!=ncorr*ntimes,
            'Incorret data check. Rewrite the code.')

########################## CORRELATORS PLOT ##########################

tmp = np.array([0]*ntimes,dtype=np.float128)
pp = PdfPages("plots/check4.pdf")

for corr in np.arange(0,ncorr,1):         
    # Real part of correlators
    real_plot=plt.figure(1,figsize=(13,5),dpi=300)
    plt.title(r'Real part of correlator #{}'.format(corr))
    plt.xlabel("Timeslice $y_4$")
    plt.ylabel("Real value of the correlator")
    plt.xlim(0,ntimes-1)
    plt.grid()
    for idx_trnsfrm in [1,2]:
        if(idx_trnsfrm==1):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_std[0][corr][i]
            plt.plot(tmp,str(idx_trnsfrm),label='Original data',alpha=1,lw=0.75)
        elif(idx_trnsfrm==2):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_gau[0][corr][i]
            plt.plot(tmp,str(idx_trnsfrm),label='Transformed data',alpha=1,lw=0.75)
    plt.legend(loc='upper right')
    pp.savefig(real_plot, dpi=real_plot.dpi, transparent = True)
    plt.close() 

    # Immaginary part of correlators
    immaginary_plot=plt.figure(1,figsize=(13,5),dpi=300)
    plt.title(r'Immaginary part of correlator #{}'.format(corr))
    plt.xlabel("Timeslice $y_4$")
    plt.ylabel("Immaginary value of the correlator")
    plt.xlim(0,ntimes-1)
    plt.grid()
    for idx_trnsfrm in [1,2]:
        if(idx_trnsfrm==1):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_std[1][corr][i]
            plt.plot(tmp,str(idx_trnsfrm),label='Original data',alpha=1,lw=0.75)
        elif(idx_trnsfrm==2):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_gau[1][corr][i]
            plt.plot(tmp,str(idx_trnsfrm),label='Transformed data',alpha=1,lw=0.75)
    plt.legend(loc='upper right')
    pp.savefig(immaginary_plot, dpi=immaginary_plot.dpi, transparent = True)
    plt.close()

# info print    
firstPage = plt.figure(1,figsize=(13,5),dpi=300)
firstPage.clf()
text = 'GENERAL SETTINGS:\n'+r'$N_{corr}$ = '+str(ncorr)+'\n'+'$N_{noise}$ = '+str(nnoise)+'\n$N_T$ = '+str(ntimes)+'\nSource timeslice: $x_0$ = '+str(x0)+'\n'
if ncorr==1:
    text+='\n\nCORRELATOR:\nHopping parameters: '+str(tmp_string)+'\n'+r'Twisted masses $\mu_s$ : '+str(mus[0])+'\n'+r'$\Gamma_X$: '+dirac_to_str(type1[0])+'\n'+r'$\Gamma_Y$: '+dirac_to_str(type2[0])+' \nIsreal: '+str(bool(isreal[0]))+'\n'
firstPage.text(0.1,0.2,text,transform=firstPage.transFigure,size=12,ha="left",fontstyle='normal')
text = 'CHECK OF THE VALUES:\n'+r'$\epsilon$ = '+str(EPSILON)+'\n'+r'$N_{good}(real)$ = '+str(counter_good_re)+'\n'+r'$N_{bad}(real)$ = '+str(counter_bad_re)+'\n'+r'$N_{good}(immag.)$ = '+str(counter_good_im)+'\n'+r'$N_{bad}(immag.)$ = '+str(counter_bad_im)+'\n'
firstPage.text(0.87,0.5,text,transform=firstPage.transFigure,size=12,ha="right",fontstyle='normal')
pp.savefig(firstPage, dpi=firstPage.dpi, transparent = True)
plt.close()

# Save plots in a single file
pp.close()