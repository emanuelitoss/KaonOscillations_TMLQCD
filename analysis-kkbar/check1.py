#########################################################################
#########################################################################
#
#   This is a check test. It aims to validate Gauge invariance
#   of the program.
# 
#   Each noise vector represents a couple of indices (c,alpha)
#   where 'c' is the colour index of SU(3) and 'alpha' the Dirac
#   index. Thus I do not evaluate noise average.
#
#   Correlators must be Gauge invariant. The program analyzes
#   two data sets - one the Gauge tranformed of the other - and 
#   checks the invariance. In other words, the results of the
#   correlators must be identical.
#
#########################################################################
#########################################################################

################################ MODULES ################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir = '/Users/emanuelerosi/thesis-MSc/kaons-oscillations/tm-mesons-obc/mesons-master/dat/'
file_name = 'tests/check1dan-rndm.correlators.dat'
path_to_file = home_dir+file_name

# this must be set by the user 
N1=8
N2=8
N3=8
volume3D=N1*N2*N3   # number of lattice points for each timeslice

# this number must be set by the user
EPSILON = 1.0e-17   # error on the Gauge tranformed data

number_of_colours=3
number_of_dirac_indices=4

########################### FUNCTIONS, TOOLS ############################

def printRed(message,message1='',message2=''):
    print("\033[91m{}\033[00m".format(message+message1+message2))
def printCyan(message,message1='',message2=''):
    print("\033[96m{}\033[00m".format(message+message1+message2))

def error_check(condition,error_message,optional_message=''):
    if(condition):
        printRed('ERROR:'+optional_message)
        printRed(error_message)
        quit()
    
def dirac_idx(idx):
    if(idx<0 or idx>=12):
        error_check(True,'Invalid index [dirac_idx]')
    return int((idx-(idx%number_of_colours))/number_of_colours)
        
def colour_idx(idx):
    if(idx<0 or idx>=12):
        error_check(True,'Invalid index [colour_idx]')
    return int(idx%number_of_colours)
        
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

############################## DATA READING ##############################
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
 
# read configuration number
offset_byte=6*4+ncorr*(8*8+4*4)
configuration_number=np.fromfile(path_to_file,dtype='<i4',offset=offset_byte,count=1)
printCyan('Configuration number: '+str(configuration_number[0])+' ---> loaded')
offset_byte+=4
   
# read data
data_count=2*ncorr*nnoise*nnoise*ntimes
data_count=2*data_count
raw_data=np.fromfile(path_to_file,dtype='<f8',offset=offset_byte)
error_check(data_count!=len(raw_data),'len(data)!=expected lenght')

############################## DATA REORDERING ##############################

# create an ordered data structure
data_std=np.array([[[0]*2]*ntimes]*ncorr,dtype=np.float128)
data_gau=np.array([[[0]*2]*ntimes]*ncorr,dtype=np.float128)
offset_idx=0

# standard data
for corr in np.arange(0,ncorr,1):
    if(isreal[corr]==True):
        for time in np.arange(0,ntimes,1):
            for i1 in np.arange(0,nnoise,1):
                for i2 in np.arange(0,nnoise,1):
                    idx=offset_idx + i2 + i1*nnoise + time*nnoise*nnoise
                    data_std[corr][time][0] += raw_data[idx]    # only real part
            data_std[corr][time][0] /= (0.5*volume3D)

    else:
        for time in np.arange(0,ntimes,1):
            for i1 in np.arange(0,nnoise,1):
                for i2 in np.arange(0,nnoise,1):
                    idx=offset_idx + 2*(i2 + i1*nnoise + time*nnoise*nnoise)
                    data_std[corr][time][0] += raw_data[idx]    # real part
                    data_std[corr][time][1] += raw_data[idx+1]  # immaginary part
            data_std[corr][time][0] /= (0.5*volume3D)
            data_std[corr][time][1] /= (0.5*volume3D)
    offset_idx+=nnoise*nnoise*ntimes*(2-isreal[corr])
    
# gauge transformed data
for corr in np.arange(0,ncorr,1):
    if(isreal[corr]==True):
        for time in np.arange(0,ntimes,1):
            for i1 in np.arange(0,nnoise,1):
                for i2 in np.arange(0,nnoise,1):
                    idx=offset_idx + i2 + i1*nnoise + time*nnoise*nnoise
                    data_gau[corr][time][0] += raw_data[idx]    # only real part
            data_gau[corr][time][0] /= (0.5*volume3D)
    else:
        for time in np.arange(0,ntimes,1):
            for i1 in np.arange(0,nnoise,1):
                for i2 in np.arange(0,nnoise,1):
                    idx=offset_idx + 2*(i2 + i1*nnoise + time*nnoise*nnoise)
                    data_gau[corr][time][0] += raw_data[idx]    # real part
                    data_gau[corr][time][1] += raw_data[idx+1]  # immaginary part
            data_gau[corr][time][0] /= (0.5*volume3D)
            data_gau[corr][time][1] /= (0.5*volume3D)
    offset_idx+=nnoise*nnoise*ntimes*(2-isreal[corr])   
    
############################## CHECK ##############################

counter_good_re=0
counter_bad_re=0
counter_good_im=0
counter_bad_im=0

estimator=0

for corr in np.arange(0,ncorr,1):
    for time in np.arange(0,ntimes,1):
        check2re = abs(data_std[corr][time][0]-data_gau[corr][time][0])<EPSILON
        if(check2re==False):
            counter_bad_re+=1
        else:
            counter_good_re+=1
        check2im = abs(data_std[corr][time][1]-data_gau[corr][time][1])<EPSILON
        if(check2im==False):
            counter_bad_im+=1
        else:
            counter_good_im+=1
        
        if(time!=0 and time!=(ntimes-1)):
            estimator+=(abs(data_std[corr][time][0]-data_gau[corr][time][0])/data_std[corr][time][0])
            print(abs(data_std[corr][time][0]-data_gau[corr][time][0])/data_std[corr][time][0])
estimator/=(ncorr*(ntimes-2))
            
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

stringhetta = '\nEstimator: '+str(estimator)
printCyan(stringhetta)

############################## CORRELATOR PLOT ##############################            
# Example of some correlators

tmp = np.array([0]*ntimes,dtype=np.float128)
pp = PdfPages("plots/check1-dan16-rndm.pdf")

for corr in np.arange(0,ncorr,1):
    #for i1 in np.arange(0,nnoise,1):
    #    for i2 in np.arange(0,nnoise,1):
                
    # Real part of correlators
    real_plot=plt.figure(1,figsize=(13,5),dpi=300)
    plt.title(r'Real part of correlator #{}'.format(corr))
    plt.xlabel("Timeslice $y_4$")
    plt.ylabel("Real value of the correlator")
    plt.xlim(0,ntimes-1)
    plt.grid(linewidth=0.1)
    for idx_trnsfrm in [1,2]:
        if(idx_trnsfrm==1):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_std[corr][i][0]
            plt.plot(tmp,'+',markersize=8,label='Original data',alpha=1,lw=0,color='crimson')
        elif(idx_trnsfrm==2):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_gau[corr][i][0]
            plt.plot(tmp,'x',markersize=6,label='Transformed data',alpha=1,lw=0,color='steelblue')
    plt.legend(loc='upper right')
    pp.savefig(real_plot, dpi=real_plot.dpi, transparent = True)
    plt.close() 

    # Real part of correlators
    real_plot=plt.figure(1,figsize=(13,5),dpi=300)
    plt.title(r'Real part of correlator #{}'.format(corr))
    plt.xlabel("Timeslice $y_4$")
    plt.ylabel("Real value of the correlator")
    plt.xlim(0,ntimes-1)
    plt.ylim(-1e-11,1e-11)
    plt.grid(linewidth=0.1)
    for idx_trnsfrm in [1,2]:
        if(idx_trnsfrm==1):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_std[corr][i][0]
            plt.plot(tmp,'+',markersize=8,label='Original data',alpha=1,lw=0,color='crimson')
        elif(idx_trnsfrm==2):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_gau[corr][i][0]
            plt.plot(tmp,'x',markersize=6,label='Transformed data',alpha=1,lw=0,color='steelblue')
    plt.legend(loc='upper right')
    pp.savefig(real_plot, dpi=real_plot.dpi, transparent = True)
    plt.close() 
    
    # Immaginary part of correlators
    immaginary_plot=plt.figure(1,figsize=(13,5),dpi=300)
    plt.title(r'Immaginary part of correlator #{}'.format(corr))
    plt.xlabel("Timeslice $y_4$")
    plt.ylabel("Immaginary value of the correlator")
    plt.xlim(0,ntimes-1)
    plt.grid(linewidth=0.1)
    for idx_trnsfrm in [1,2]:
        if(idx_trnsfrm==1):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_std[corr][i][1]
            plt.plot(tmp,'+',markersize=8,label='Original data',alpha=1,lw=0,color='crimson')
        elif(idx_trnsfrm==2):
            for i in np.arange(0,ntimes,1):
                tmp[i]=data_gau[corr][i][1]
            plt.plot(tmp,'x',markersize=6,label='Transformed data',alpha=1,lw=0,color='steelblue')
    plt.legend(loc='upper right')
    pp.savefig(immaginary_plot, dpi=immaginary_plot.dpi, transparent = True)
    plt.close()

# info print
firstPage = plt.figure(1,figsize=(13,5),dpi=300)
firstPage.clf()
text='GENERAL SETTINGS:\n'+r'$N_{corr}$ = '+str(ncorr)+'\n'+'$N_{noise}$ = '+str(nnoise)+'\n$N_T$ = '+str(ntimes)+'\nSource timeslice: $x_0$ = '+str(x0)+'\nSource timeslice: $z_0$ = '+str(z0)
if ncorr==1:
    text+='\n'+'\n\nCORRELATOR:\nHopping parameters: '+str(tmp_string)+'\n'+r'Twisted masses $\mu_s$ : '+str(mus[0])+'\n'+r'$\Gamma_A$: '+dirac_to_str(typeC_z[0])+'\n'+r'$\Gamma_C$: '+dirac_to_str(typeA_x[0])+'\nMixing operator: '+operator_to_str(operator_type[0])+' \nIsreal: '+str(bool(isreal[0]))+'\n'
firstPage.text(0.1,0.2,text,transform=firstPage.transFigure,size=12,ha="left",fontstyle='normal')
text = 'CHECK OF THE VALUES:\n'+r'$\epsilon$ = '+str(EPSILON)+'\n'+r'$N_{good}(real)$ = '+str(counter_good_re)+'\n'+r'$N_{bad}(real)$ = '+str(counter_bad_re)+'\n'+r'$N_{good}(imm.)$ = '+str(counter_good_im)+'\n'+r'$N_{bad}(imm.)$ = '+str(counter_bad_im)+'\n'
firstPage.text(0.87,0.5,text,transform=firstPage.transFigure,size=12,ha="right",fontstyle='normal')
pp.savefig(firstPage, dpi=firstPage.dpi, transparent = True)
plt.close()
    
# Save plots in a single file
pp.close()
