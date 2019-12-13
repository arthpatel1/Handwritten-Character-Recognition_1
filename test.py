# EEL 5840 Final Project
# Handwritten character recognition 
# alphabets: a,b,c,d,h,i,j,k
# Corrosponding labels: 1,2,3,4,5,6,7,8

# If the test image size is grater then (50,50) we assign that image -1 label 
# Authors: Jaber Aloufi,Arunabho Basu, Cheng-Han Ho, Arth Patel

import numpy as np
import math
import pickle
from sklearn.externals import joblib 

# Function to predict labels of test data on already trained model
# Function has two inputs first is trained model and second is test data
def test_char(classifier,data):
     
    def load_pkl(fname):
        with open(fname,'rb') as f:
            return pickle.load(f)
        
    test_dataset = load_pkl(data)  # Load test data
    
    test_set = np.array([])
    
    for i in range (np.size(test_dataset)): 
        if np.shape(test_dataset[i]) <= (50,50) and np.shape(test_dataset[i]) != () : # Remove images grater then 
                                                                                      # (50,50) size or any empth array
            y1 = np.shape(test_dataset[i])
            bool_arr_test = np.zeros(shape=(50, 50), dtype=np.bool)
            if y1[0] == 50:
                n = 50 - y1[1]
                bool_arr_test[0:y1[0], math.ceil(n/2):y1[1]+math.ceil(n/2)] = test_dataset[i] # Center the images
            else:
                n = 50 - y1[0]
                bool_arr_test[math.ceil(n/2):y1[0]+math.ceil(n/2), 0:y1[1]] = test_dataset[i] # Center the images
        
            d2 = np.reshape(bool_arr_test,2500) # Converting each image into 1d array
            test_set = np.append(test_set,d2)  # Put all images into one array
        else:
            test_set = np.append(test_set,np.zeros(2500)) # if image is grater then (50,50) update array by zeros
            
    array_size = np.shape(test_set)
    set_size = int(array_size[0]/2500) 

    test_set.resize((set_size,2500))  #Resize array to seperate each images

    
    classifier1 = joblib.load(classifier) # Load trained model
    
    test_predict = classifier1.predict(test_set)  # Predict labels of test data
    
    class_prob = classifier1.predict_proba(test_set)
    #print(classifier1.predict_proba(test_set))
    s = np.shape(class_prob)
        
    x = np.array([])
    for i in range (s[0]):
        a = class_prob[i]
        res = all(ele < 0.25 for ele in class_prob[i])  
        
        if a[5] >= 0.3108 and a[6] > 0.4725 and a[1] > 0.1025: 
            x = np.append(x,-1)
        elif res == True:
            x = np.append(x,-1)
        else :
            x= np.append(x,0)
    # in our set if image size is grater then (50,50) we assign that image -1 label  
    y = np.array([])
    for i in range (s[0]):
        if x[i] == -1:
            y = np.append(y,x[i])
        else:
            y = np.append(y,test_predict[i])
    
    
    print("Array of predicted labels: ",y)  # Print results
    return y
#========================================================================================
    
# Enter trained model path and test data path to predict labels of test data (test data is in .pkl format)    
test_char(r'C:\Users\Arth\Desktop\UF\EEL 5840\Project\trained_model.pkl',r'C:\Users\Arth\Downloads\data_b.npy') 