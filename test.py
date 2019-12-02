# EEL 5840 Final Project
# Handwritten character recognition 
# alphabets: a,b,c,d,h,i,j,k and unknown
# Corrosponding labels: 1,2,3,4,5,6,7,8 and -1

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

    array_size = np.shape(test_set)
    set_size = int(array_size[0]/2500) 

    test_set.resize((set_size,2500))  #Resize array to seperate each images

    
    classifier1 = joblib.load(classifier) # Load trained model
    
    test_predict = classifier1.predict(test_set)  # Predict labels of test data
            
    # Extra credit---------------------------------------------------------
    # We are recording probabilty of each class of each images
    # Then assign label = -1 if probabilty of each class of the image is less then 0.25
    # So, if prediction probabilities of the given image for all 8 class will be less then 0.25
    # then we can say that our prediction is not very strong and our algorithm has little confidence 
    # in the predicted result. 
    
    class_prob = classifier1.predict_proba(test_set)
    s = np.shape(class_prob)
        
    x = np.array([])
    for i in range (s[0]):
        res = all(ele < 0.25 for ele in class_prob[i])  # Check probabiltilies of the prediction
        if res == False: 
            x = np.append(x,0)
        if res == True:
            x = np.append(x,-1)
    
    final_pred = np.array([])  # Final prdiction array
    for i in range (s[0]):
        if x[i] == -1:
            final_pred = np.append(final_pred,x[i]) # replace previous prediction with -1
        else:
            final_pred = np.append(final_pred,test_predict[i]) # keep original result
    
    print("Array of predicted labels: ",final_pred)  # Print results
    return test_predict
#========================================================================================
    
# Enter trained model path and test data path to predict labels of test data (test data is in .pkl format)    
test_char(r'C:\Users\Downloads\trained_model.pkl',r'C:\Users\Downloads\testdata.pkl') 