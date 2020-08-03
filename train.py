# Handwritten character classifier 
# alphabets: a,b,c,d,h,i,j,k and unknown
# Corrosponding labels: 1,2,3,4,5,6,7,8 and -1


# Import Packages
import numpy as np
import math
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

        
def train_model(load_train_data, load_train_labels):
    def load_pkl(fname):
        with open(fname,'rb') as f:
            return pickle.load(f)
    

    # Load train data and label , Data is attached in github    
    train_data = load_pkl(load_train_data) 
    train_label = np.load(load_train_labels)

    # resize images to (50,50) and center image
    train_d = np.array([])  
    train_l = np.array([])
   
    for i in range (6400):
        if (np.shape(train_data[i]) <= (50,50) and np.shape(train_data[i]) != () 
        and i not in range(960,1040) and i not in range(1868,1948) 
        and i not in range(4908,4987) and i not in range(5239,5307) 
        and i not in range(3708,3788) and i not in range(5708,5788) 
        and i not in [941,942,943,1529,1540,3013,3198,5592,5593,5602]):# remove images with size grater then
                                                                       # (50,50) and any empty arrays
            y = np.shape(train_data[i])
            train_l = np.append(train_l,train_label[i])
            bool_arr = np.zeros(shape=(50, 50), dtype=np.bool)
            if y[0] == 50:
                n = 50 - y[1]
                bool_arr[0:y[0], math.ceil(n/2):y[1]+math.ceil(n/2)] = train_data[i] # Center the image
            else:
                n = 50 - y[0]
                bool_arr[math.ceil(n/2):y[0]+math.ceil(n/2), 0:y[1]] = train_data[i] # Center the image
        
            d1 = np.reshape(bool_arr,2500) 
            train_d = np.append(train_d,d1) # Put all images into one array

    
    array_size1 = np.shape(train_d)
    set_size1= int(array_size1[0]/2500)
    
    train_d.resize((set_size1,2500)) #Resize array to seperate each images

    
    # Split data for cross validation
    # Train data size = 80%
    # Test data size = 20%
    X_train, X_test, y_train, y_test = train_test_split(train_d, train_l, test_size=0.2)

    # Model

    # using Support vector classifier of svm for classification
    # the parameter taken for svm.SVC are ( C=1.0, kernel='rbf', degree=4, gamma=0.005, 
    #   coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, 
    #   verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None )
    
    classifier = svm.SVC(gamma=0.005,degree=4, probability=True).fit(X_train, y_train)  # fit the model
    joblib.dump(classifier,r'C:\Users\Arth\Documents\GitHub\project-01-mlprojectgroup55\trained_model.pkl')  # save trained model
    
    # Here we are also recording probability so we can know that each result has how much probabilty
    # we can use this probaility to remove predictions with very low probabilty 
    
    # knn and other models we experimeted with
    """
    1. KNN
    
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train, y_train)

    knn_pred = knn_clf.predict(X_test)

    print(accuracy_score(y_test,knn_pred))
    
    2. SVM with kernel='linear'
    svc = svm.SVC(kernel='linear', C=C).fit(X_train,y_train)
    score1 = svc.score(X_test,y_test)
    print(score1)
    
    3. SVM with kernel='poly'
    poly_svc = svm.SVC(kernel='poly', degree=3 , C=C).fit(X_train, y_train)
    score2 = poly_svc.score(X_test,y_test)
    print(score2)
    
    4. LinearSVC 
    lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
    score3 = lin_svc.score(X_test,y_test)
    print(score3)

    """
       
#===========================================================================================
    
train_model(r'C:\Users\Arth\Downloads\train_data (1).pkl', r'C:\Users\Arth\Downloads\finalLabelsTrain (1).npy')
