
from __future__ import division, print_function
## steps 1. read wav file 2. get spec 3. get peaks of spectrogram 4. get those corresponding row/columns MFCC's


## import numpy because NUMPY

import numpy as np

## wav read imports
import scipy.io.wavfile as wav






##SPEC and MFCC imports
from features import mfcc
from features import logfbank


###plotting imports
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


###  peaks imports

import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
from detect_peaks import detect_peaks


#get wav

(rate,sig) = wav.read("BF4.wav")


#MFCC
mfcc_feat_not_norm = mfcc(sig,rate)
max_mfcc = np.amax(mfcc_feat_not_norm)
mfcc_feat = (1/max_mfcc) * mfcc_feat_not_norm

mfcc_size = len(mfcc_feat[:,1]) # x dimensions MFCC


#Log Spec
fbank_feat_not_norm = logfbank(sig,rate)
max_log = np.amax(fbank_feat_not_norm)
fbank_feat = (1/max_log) * fbank_feat_not_norm
logSizeX = len(fbank_feat[1,:])# y dimensions log spec
logSizeY =len(fbank_feat[:,1])# x dimensions log spec


'''
#plotting Log Spec
fig = plt.figure(1)
ax = fig.add_subplot(2, 1, 1, projection='3d')
X = np.arange(0, logSizeX, 1)
Y = np.arange(0, logSizeY, 1)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = fbank_feat

ax.set_xlabel('Bank')
ax.set_ylabel('Time')


surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


#plt.show()




#PLOTTING mfcc
#fig = plt.figure()
ax = fig.add_subplot(2, 1, 2, projection='3d')
X2 = np.arange(1,14 , 1)
Y2 = np.arange(0, mfcc_size, 1)
X2, Y2 = np.meshgrid(X2, Y2)
Z2 = mfcc_feat

surf = ax.plot_surface(Y2, X2, Z2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Time')
ax.set_ylabel('MFCC')

plt.show()

'''


#getting peaks from spec, this will give us an array with all the time slices we need to get MFCCs for
spec_peaks_array = [];
print(len(fbank_feat[1,:]))
print(len(fbank_feat[:,1]))
for n in range (0,logSizeX):

    ind = detect_peaks(fbank_feat[:,n], mph = 0.8, mpd = 10)
    
  

    spec_peaks_array = np.concatenate((ind, spec_peaks_array), axis = 0)
print(spec_peaks_array)
print(len(spec_peaks_array))

#spec_peaks_array is a list of the time coordinates of the peaks for the call. Theses are the locations we need to get the MFCC's for

##get rid of duplications in spec_peaks array

spec_peaks = list(set(spec_peaks_array))

print(spec_peaks)
print(len(spec_peaks))


#ratio wanted 5peaks/100time = 0.05 peaks/time

ratio = len(spec_peaks)/logSizeY
print(ratio)


#for quiet samples lower the min peak height to 0.6
if ratio < 0.05:

    #getting peaks from spec, this will give us an array with all the time slices we need to get MFCCs for
    spec_peaks_array = [];
    print(len(fbank_feat[1,:]))
    print(len(fbank_feat[:,1]))
    for n in range (0,logSizeX):
    
        ind = detect_peaks(fbank_feat[:,n], mph = 0.6, mpd = 10)
    
    
    
        spec_peaks_array = np.concatenate((ind, spec_peaks_array), axis = 0)
        print(spec_peaks_array)
    print(len(spec_peaks_array))

    #spec_peaks_array is a list of the time coordinates of the peaks for the call. Theses are the locations we need to get the MFCC's for

    ##get rid of duplications in spec_peaks array

    spec_peaks = list(set(spec_peaks_array))

    #print(spec_peaks)
    #print(len(spec_peaks))


MFCC_features = np.empty([len(spec_peaks), 13]);

for counter in range(0, len(spec_peaks)):

    time_slice = spec_peaks[counter]
    
    temp =mfcc_feat[time_slice, :]

    #print(temp.transpose())
    
    MFCC_features[counter] = temp.transpose()

MFCC_features = np.array(MFCC_features)




np.savetxt('new_call_mfcc_noUI.txt', MFCC_features, fmt = '%7.7f')





