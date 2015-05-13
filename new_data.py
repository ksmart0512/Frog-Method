

from __future__ import division, print_function
## steps 1. read wav file 2. get spec 3. get peaks of spectrogram 4. get those corresponding row/columns MFCC's

### import to reast list of files

from __future__ import print_function
import os

## import numpy because NUMPY

import numpy as np
from scipy import stats

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
from matplotlib.mlab import PCA

import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
from detect_peaks import detect_peaks


###For PCA
from matplotlib.mlab import PCA

from numpy.linalg import inv #inverse of matrix


##for KNN
from sklearn.neighbors import KNeighborsClassifier

import pickle

###for RF
from sklearn.ensemble import RandomForestClassifier




### have to have this to correctly concatenate

all_data_MFCC_unshifted_UI = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
ND = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
NDnoUI = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
all_data_MFCC_unshifted = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
number_of_zeros = 0;

###read the new call

(rate,sig) = wav.read('CF03.wav')
    
    
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


#getting peaks from spec, this will give us an array with all the time slices we need to get MFCCs for
spec_peaks_array = [];
    
for n in range (0,logSizeX):
        
    ind = detect_peaks(fbank_feat[:,n], mph = 0.8, mpd = 10)
    
            
            
    spec_peaks_array = np.concatenate((ind, spec_peaks_array), axis = 0)


    
#spec_peaks_array is a list of the time coordinates of the peaks for the call. Theses are the locations we need to get the MFCC's for
        
##get rid of duplications in spec_peaks array
        
spec_peaks = list(set(spec_peaks_array))



        
#ratio wanted 5peaks/100time = 0.05 peaks/time
        
ratio = len(spec_peaks)/logSizeY


      
      
      
        
 #for quiet samples lower the min peak height to 0.6
if ratio < 0.05:
     #print("ratio is less than 0.5 for" + file)
            
            
            #getting peaks from spec, this will give us an array with all the time slices we need to get MFCCs for
    spec_peaks_array = [];

    for n in range (0,logSizeX):
                
        ind = detect_peaks(fbank_feat[:,n], mph = 0.53, mpd = 10)
            
            
        spec_peaks_array = np.concatenate((ind, spec_peaks_array), axis = 0)

                
            
                
 #spec_peaks_array is a list of the time coordinates of the peaks for the call. Theses are the locations we need to get the MFCC's for
                
 ##get rid of duplications in spec_peaks array
                
spec_peaks = list(set(spec_peaks_array))
                    


                    
                    
MFCC_features_UI = np.zeros([len(spec_peaks), 17]);
MFCC_features = np.zeros([len(spec_peaks), 14]);



for counter in range(0, len(spec_peaks)):
    time_slice = spec_peaks[counter]
        
        
        
    temp = mfcc_feat[time_slice, :]
    
    #### add user inputs, season will always be there
   

    with open('user_input.txt', 'r') as t:
        line =  [t.readline().strip() for i in range(1)]
        Uinput = [u for u in line]
        UI1 = [ui for ui in u.split(' ')]
        UI = map(int, UI1)
        Season = map(int, UI1[0])

    UI = np.array(UI)

    if len(UI) == 4:
    
        MFCC_features_UI[counter] = np.concatenate((  UI, temp.transpose()), axis = 0)
        
        

        
    MFCC_features[counter] =np.concatenate((  Season, temp.transpose()), axis = 0)

all_data_MFCC_unshifted_UI = MFCC_features_UI

all_data_MFCC_unshifted =  MFCC_features
    #print(all_data_MFCC_unshifted)



PCA = np.loadtxt('PCA_matrix_UI')
######112

#apply PCA to matix then concatenate with UI if applicable,
if len(UI) == 4:
    projection = np.dot(all_data_MFCC_unshifted_UI[1:,4:], PCA)
    for c in range(1, len(projection[:,1])):
        NDtemp = np.concatenate((all_data_MFCC_unshifted_UI[c,:4],  projection[c,:]), axis = 0)
        ND = np.vstack((NDtemp, ND))
elif len(UI) == 1:
    projection = np.dot(all_data_MFCC_unshifted[1:,1:], PCA)
    for c in range(1, len(projection[:,1])):
        d = np.array(all_data_MFCC_unshifted[c, 0])
        NDtemp = np.append(np.array(all_data_MFCC_unshifted[c, 0]),  projection[c,:])
        
        NDnoUI = np.vstack((NDtemp, NDnoUI))

    ND = NDnoUI

#####Classifiers:
####without userInput

if len(ND[1,:]) == 14:



    filename = 'c:\my_python_object_knn.pkl'
    knn_UI = pickle.load(open(filename,'rb'))
    
    filename = 'c:\my_python_object_RF.pkl'
    RF = pickle.load(open(filename,'rb'))
    
    r_knn = knn_UI.predict(np.float32(ND))
    r_RF = RF.predict(np.float32(ND))
    
    results_KNN = r_knn
    results_RF= r_RF
    
    
    
    
    
    
    
    
    results_RF[0] = 15 ## this is to ensure that there is an accurate bin count
    results_KNN[0] = 15
    
    
    results_KNN = results_KNN.astype(int)
    results_RF = results_RF.astype(int)
    
    mode_KNN = stats.mode(results_KNN)
    
    mode_RF = stats.mode(results_RF )
    
    
    
    
    
    if mode_KNN[0] == mode_RF[0]:
        
        
        results = 100* np.add(np.bincount(results_KNN), np.bincount(results_RF)) / (2* len(ND[:,1]))
        print('[Other, Bull F., Barking T. F., S. Chorus F., E. NarrowMouth T. Green F., Green T. F., Little Grass F., Oak T , Pig F., PineWoods T.F., S. Cricket F., Squirrel T. F., S. Leopard F., Spring Peeper, Southern T.')
        
        
        
        print('no UI', results.astype(int))

    else:
        
        a = np.bincount(results_KNN)
        
        a = a.tolist()
        ans_knn = a.index(max(a))
        
        if ans_knn == 1:
            max_acc_rf = 24
        elif ans_knn ==2:
            max_acc_rf = 0
        elif ans_knn == 3:
            max_acc_rf = 36
        elif ans_knn ==4:
            max_acc_rf = 11
        elif ans_knn == 5:
            max_acc_rf = 7
        elif ans_knn == 6:
            max_acc_rf = 4
        elif ans_knn ==7:
            max_acc_rf = 45
        elif ans_knn == 8:
            max_acc_rf = 72
        elif ans_knn ==9:
            max_acc_rf = 12
        elif ans_knn == 10:
            max_acc_rf = 75
        elif ans_knn == 11:
            max_acc_rf = 10
        elif ans_knn == 12:
            max_acc_rf = 13
        elif ans_knn == 13:
            max_acc_rf = 13
        elif ans_knn == 14:
            max_acc_rf = 0
        elif ans_knn == 15:
            max_acc_rf = 32
        else:
            max_acc_rf = 0



        b = np.bincount(results_RF)
        b = b.tolist()
        ans_rf = b.index(max(b))

        if ans_rf == 1:
            max_acc_knn = 18
        elif ans_rf ==2:
            max_acc_knn = 22
        elif ans_rf == 3:
            max_acc_knn = 27
        elif ans_rf ==4:
            max_acc_knn = 22
        elif ans_rf == 5:
            max_acc_knn = 34
        elif ans_rf == 6:
            max_acc_knn = 16
        elif ans_rf ==7:
            max_acc_knn = 45
        elif ans_rf == 8:
            max_acc_knn = 56
        elif ans_rf ==9:
            max_acc_knn = 19
        elif ans_rf == 10:
            max_acc_knn = 26
        elif ans_rf == 11:
            max_acc_knn = 29
        elif ans_rf == 12:
            max_acc_knn = 13
        elif ans_rf == 13:
            max_acc_knn = 6
        elif ans_rf == 14:
            max_acc_knn = 41
        elif ans_rf == 15:
            max_acc_knn = 32
        else:
            max_acc_knn = 0


        if max_acc_rf <=max_acc_knn:
    
            results = 100*  np.bincount(results_RF) / (len(ND[:,1]))
            print('[Other, Bull F., Barking T. F., S. Chorus F., E. NarrowMouth T. Green F., Green T. F., Little Grass F., Oak T , Pig F., PineWoods T.F., S. Cricket F., Squirrel T. F., S. Leopard F., Spring Peeper, Southern T.')
            
            
            
            print('no UI', results.astype(int))
    
        else:
            
            
            results = 100*  np.bincount(results_KNN) / (len(ND[:,1]))
            print('[Other, Bull F., Barking T. F., S. Chorus F., E. NarrowMouth T. Green F., Green T. F., Little Grass F., Oak T , Pig F., PineWoods T.F., S. Cricket F., Squirrel T. F., S. Leopard F., Spring Peeper, Southern T.')
            
            
            
            print('no UI', results.astype(int))





elif len(ND[1,:]) == 17:


    filename = 'c:\my_python_object_neigh_UI.pkl'
    knn_UI = pickle.load(open(filename,'rb'))
    
    filename = 'c:\my_python_object_RF_UI.pkl'
    RF = pickle.load(open(filename,'rb'))
    
    r_knn = knn_UI.predict(np.float32(ND))
    r_RF = RF.predict(np.float32(ND))
    
    results_KNN = r_knn
    results_RF= r_RF
    
 






    results_RF[0] = 15 ## this is to ensure that there is an accurate bin count
    results_KNN[0] = 15


    results_KNN = results_KNN.astype(int)
    results_RF = results_RF.astype(int)
    
    mode_KNN = stats.mode(results_KNN)

    mode_RF = stats.mode(results_RF )





    if mode_KNN[0] == mode_RF[0]:
    
    
        results = 100* np.add(np.bincount(results_KNN), np.bincount(results_RF)) / (2* len(ND[:,1]))
        print('[Other, Bull F., Barking T. F., S. Chorus F., E. NarrowMouth T. Green F., Green T. F., Little Grass F., Oak T , Pig F., PineWoods T.F., S. Cricket F., Squirrel T. F., S. Leopard F., Spring Peeper, Southern T.')



        print(results.astype(int))

    else:
   
        a = np.bincount(results_KNN)
    
        a = a.tolist()
        ans_knn = a.index(max(a))
   
        if ans_knn == 1:
            max_acc_rf = 24
        elif ans_knn ==2:
            max_acc_rf = 0
        elif ans_knn == 3:
            max_acc_rf = 36
        elif ans_knn ==4:
            max_acc_rf = 11
        elif ans_knn == 5:
            max_acc_rf = 7
        elif ans_knn == 6:
            max_acc_rf = 4
        elif ans_knn ==7:
            max_acc_rf = 45
        elif ans_knn == 8:
            max_acc_rf = 72
        elif ans_knn ==9:
            max_acc_rf = 12
        elif ans_knn == 10:
            max_acc_rf = 75
        elif ans_knn == 11:
            max_acc_rf = 10
        elif ans_knn == 12:
            max_acc_rf = 13
        elif ans_knn == 13:
            max_acc_rf = 13
        elif ans_knn == 14:
            max_acc_rf = 0
        elif ans_knn == 15:
            max_acc_rf = 32
        else:
            max_acc_rf = 0



        b = np.bincount(results_RF)
        b = b.tolist()
        ans_rf = b.index(max(b))

        if ans_rf == 1:
            max_acc_knn = 18
        elif ans_rf ==2:
            max_acc_knn = 22
        elif ans_rf == 3:
            max_acc_knn = 27
        elif ans_rf ==4:
            max_acc_knn = 22
        elif ans_rf == 5:
            max_acc_knn = 34
        elif ans_rf == 6:
            max_acc_knn = 16
        elif ans_rf ==7:
            max_acc_knn = 45
        elif ans_rf == 8:
            max_acc_knn = 56
        elif ans_rf ==9:
            max_acc_knn = 19
        elif ans_rf == 10:
            max_acc_knn = 26
        elif ans_rf == 11:
            max_acc_knn = 29
        elif ans_rf == 12:
            max_acc_knn = 13
        elif ans_rf == 13:
            max_acc_knn = 6
        elif ans_rf == 14:
            max_acc_knn = 41
        elif ans_rf == 15:
            max_acc_knn = 32
        else:
            max_acc_knn = 0


        if max_acc_rf <=max_acc_knn:
        
            results = 100*  np.bincount(results_RF) / (len(ND[:,1]))
            print('[Other, Bull F., Barking T. F., S. Chorus F., E. NarrowMouth T. Green F., Green T. F., Little Grass F., Oak T , Pig F., PineWoods T.F., S. Cricket F., Squirrel T. F., S. Leopard F., Spring Peeper, Southern T.')
    
    
    
            print(results.astype(int))

        else:
        

            results = 100*  np.bincount(results_KNN) / (len(ND[:,1]))
            print('[Other, Bull F., Barking T. F., S. Chorus F., E. NarrowMouth T. Green F., Green T. F., Little Grass F., Oak T , Pig F., PineWoods T.F., S. Cricket F., Squirrel T. F., S. Leopard F., Spring Peeper, Southern T.')
        
        
        
            print(results.astype(int))






