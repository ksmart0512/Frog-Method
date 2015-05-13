


from __future__ import division, print_function
## steps 1. read wav file 2. get spec 3. get peaks of spectrogram 4. get those corresponding row/columns MFCC's

### import to reast list of files

from __future__ import print_function
import os

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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

###garbage collection
import gc


###Begin script







all_data_MFCC_unshifted_label_UI =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
all_data_MFCC_unshifted_label =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
number_of_zeros = 0;

for file in os.listdir("/Users/katrinasmart/Desktop/python_shiz/calls"):
    if file.endswith(".wav"):
        #get wav
        file_name = "calls/" + file

        (rate,sig) = wav.read(file_name)
        print(file)


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
 
 ####add in labels
 
            

            spec_peaks_array = np.concatenate((ind, spec_peaks_array), axis = 0)
    
    

        #spec_peaks_array is a list of the time coordinates of the peaks for the call. Theses are the locations we need to get the MFCC's for

        ##get rid of duplications in spec_peaks array

        spec_peaks = list(set(spec_peaks_array))

        #print( file)
        # print(len(spec_peaks))


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
        
    
        
        #print(len(spec_peaks_array))

        #spec_peaks_array is a list of the time coordinates of the peaks for the call. Theses are the locations we need to get the MFCC's for

        ##get rid of duplications in spec_peaks array

        spec_peaks = list(set(spec_peaks_array))

        #print(spec_peaks)
        #print(len(spec_peaks))


        MFCC_features_UI = np.empty([len(spec_peaks), 18]);
        MFCC_features = np.empty([len(spec_peaks), 15]);

        for counter in range(0, len(spec_peaks)):


        
            if file[:2] == "BF":
                label = [1];
            elif file[:2] == "BT":
                label = [2];
            elif file[:2] == "CF":
                label = [3];
            elif file[:2] == "EN":
                label = [4];
            elif file[:2] == "GF":
                label = [5];
            elif file[:2] == "GT":
                label = [6];
            elif file[:2] == "LG":
                label = [7];
            elif file[:2] == "OT":
                label = [8];
            elif file[:2] == "PF":
                label = [9];
            elif file[:2] == "PW":
                label = [10];
            elif file[:2] == "SC":
                label = [11];
            elif file[:2] == "SF":
                label = [12];
            elif file[:2] == "SL":
                label = [13];
            elif file[:2] == "SP":
                label = [14];
            elif file[:2] == "ST":
                label = [15];
            else:
                print(file)
                print("no label")

                    
            time_slice = spec_peaks[counter]
         


    
            temp = mfcc_feat[time_slice, :]
        
            #print(temp.transpose())
            
            
            ###Find user INPUTS
            
            for texts in os.listdir("/Users/katrinasmart/Desktop/python_shiz/old_data_UI"):
                if file[:5] == texts[:5]:
                    text_name = "old_data_UI/" + texts
                    with open(text_name, 'r') as t:
                        line =  [t.readline().strip() for i in range(1)]
                        Uinput = [u for u in line]
                        UI1 = [ui for ui in u.split(' ')]
                        UI = map(int, UI1[:4])
                        Season = map(int, UI1[1])


            UI = np.array(UI)


                
        
            MFCC_features_UI[counter] = np.concatenate((label,  UI, temp.transpose()), axis = 0)
            MFCC_features[counter] = np.concatenate((label, Season, temp.transpose()), axis = 0)


        MFCC_features_UI = np.array(MFCC_features_UI)
        MFCC_features = np.array(MFCC_features)
        
        all_data_MFCC_unshifted_label_UI = np.vstack((all_data_MFCC_unshifted_label_UI, MFCC_features_UI))
        all_data_MFCC_unshifted_label = np.vstack((all_data_MFCC_unshifted_label, MFCC_features))
       


###his will save all the unshifted data with the label as the first column

np.savetxt('data_MFCC_unshifted_label.txt', all_data_MFCC_unshifted_label_UI, fmt = '%7.7f')




###begin PCA projection UI

x = all_data_MFCC_unshifted_label_UI[1:,5:]
xT = x.T
xTx = np.dot(xT,x)
eig_val_cov, eig_vec_cov = np.linalg.eig(xTx)
eig_vec_cov_T = eig_vec_cov.transpose()

projection = np.dot(all_data_MFCC_unshifted_label_UI[1:,5:], eig_vec_cov)


np.savetxt('PCA_matrix_UI', eig_vec_cov, fmt = '%2.14f')

data_shifted_UI = np.concatenate((all_data_MFCC_unshifted_label_UI[1:,0:5], projection), axis = 1)
#####


np.savetxt('all_data_MFCC_shifted_label_UI.txt',data_shifted_UI, fmt = '%7.7f')




data_shifted = np.concatenate((all_data_MFCC_unshifted_label[1:,0:2], projection), axis = 1)

np.savetxt('all_data_MFCC_shifted_label.txt',data_shifted, fmt = '%7.7f')

###create KNN and RF for data w/UI and data w/o UI

training_data_UI = data_shifted_UI[:,1:]
T_data_UI = []
for i in range(0,65471):
    data = training_data_UI[i,:]
    data = data.tolist()
    T_data_UI.append(data)



data_label_T_UI =data_shifted_UI.T
training_labels_UI = data_label_T_UI[0,:]
training_labels_UI = training_labels_UI.tolist()



neigh_UI = KNeighborsClassifier(n_neighbors=1) #create an object neigh
neigh_UI.fit(T_data_UI, training_labels_UI ) #fill object

clf_UI = RandomForestClassifier(n_estimators=200)
clf_UI = clf_UI.fit( T_data_UI, training_labels_UI)

def save_object(neigh_UI, filename):
    with open(filename, 'wb') as output:
        pickle.dump(neigh_UI, output, pickle.HIGHEST_PROTOCOL)

filename = 'c:\my_python_object_neigh_UI.pkl'
#save_object(neigh_UI, filename)


print('knnUI done')


filename = 'c:\my_python_object_RF_UI.pkl'
#save_object(clf_UI, filename)
print('RF UI done')



training_data = data_shifted[:,1:]
T_data= []
for i in range(0,65471):
    data = training_data[i,:]
    data = data.tolist()
    T_data.append(data)

data_label_T =data_shifted.T

training_labels = data_label_T[0,:]
training_labels = training_labels.tolist()

neigh = KNeighborsClassifier(n_neighbors=1) #create an object neigh
neigh.fit(  T_data, training_labels ) #fill object
print('knn done')
filename = 'c:\my_python_object_knn.pkl'
#save_object(neigh, filename)



clf  = RandomForestClassifier(n_estimators= 200) #create an object neigh
clf.fit(  T_data, training_labels ) #fill object

filename = 'c:\my_python_object_RF.pkl'
#save_object(clf, filename)
print('RF done')




