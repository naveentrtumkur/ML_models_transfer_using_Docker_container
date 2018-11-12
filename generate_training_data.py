# This is the python script to generate training data for our multi-variate Linear regression model

# Background
# multi-variate linear regression would be with multiple features.
# hypothesis = theta0 + x1 * theta1 + x2 * theta2 + x3 * theta3 -----

# Generate suitable 'Y' values and 'X' values.
from random import randint
import numpy as np
import random
import csv
import pandas as pd

#np.empty((8,), float) #[]#np.zeros(1,8)
csvFile = "training_data.csv" #Specify the training data filename

def append_csv_data(csvFile, pArray):
    with open(csvFile, 'a') as file:
        df = pd.DataFrame(pArray)
        df.to_csv(file, header=True)
count = 0
while count < 1:
    X_Y_com = []
    for i in range(0,100):
        val = randint(0,10)
        print("val value =",val) # This is working fine

        X = np.array([2*val, 5*val, 10*val, 100*val, 225*val, 285*val, 350*val])
    
        print("X==",X,"shape=",np.shape(X))
        X_temp = np.copy(X)
        print("x_temp=",X_temp,"x=",X)
        for i in range(len(X)):
            noise = random.uniform(-0.5,0.5)
            #print("noise=",noise)
            X[i] = X[i] * noise
            #print("X[i]=",X[i])

        print("X from individual mult=",X)
        y = np.sum(X)
        print("y value is:",y)
        X_Y_combine = np.concatenate((X,np.array([y])))
        #print("xyxombine=",X_Y_combine)
        #print("1dim=",np.shape(X_Y_combine))
        #print("2dim=",np.shape(X_Y_com))
        X_Y_com = np.append(X_Y_com, X_Y_combine,axis=0)
        #print("inside x-y-com",X_Y_com)

        print("inside x-y-com",X_Y_com)

# Action-1: Reshape the numpy array to rows of length 8.


    persist_array = np.reshape(X_Y_com,(100,8))
    #print the persist array
    print("persist array=",persist_array)

        # Export the content to training.csv

        #create a function that takes csv filename and numpyarray and persists them to csv file.
    append_csv_data(csvFile, persist_array)

    count+=1 #increment the count
        #print("x-temp",X_temp)
        #print("X from dot product=",np.dot(X_temp,0.1))
        #print("outside x-y-com",X_Y_combine)
        # Now let's emit this to training_data.csv and after one batch put a wait of 20 seconds

