#Learned the concepts from teh below tutorial
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

#Define a method to find the accuracy of the multivariate Linear Regression model...
def find_accuracy(samples, reg):
    #Count of number of correct and incorrect predictions maintained
    #num_correct = 0
    #num_incorrect = 0
    sum = 0
    print("The number of samples passed are: ",len(samples))
    # The first column is unnecessary, just drop it.
    samples = np.delete(samples, [0], axis=1)
    print("samples=", samples)
    col_interst = [0, 1, 2, 3, 4, 5, 6]
    test_xvals = samples[:, col_interst]  # Just extract the interested columns from training data
    print("x_vals=", test_xvals)
    # np_yvals = training_data.head(5).values
    test_yvals = samples[:, [7]]
    print ("y_vals=", test_yvals)

    # X = np.array([[1,1],[1,2],[2,2],[2,3]])
    test_X = test_xvals
    # For plotting the points use python's matplotlib..

    '''
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    print(X)
    '''
    # Y-equation calculation
    # y = np.dot(X, np.array([1, 2])) + 3 #w1*x1 + w2*x2 + w0
    test_Y = test_yvals

    y_predict = np.zeros(len(test_X))
    for i in range(0,len(test_X)):
        print("\nY[i]=",test_Y[i])
        y_predict[i] = reg.predict([test_X[i]])
        print("\n prediction=",y_predict)
        '''if reg.predict(X[i]) == Y[i]:
            num_correct += 1
        else:
            num_incorrect += 1
        '''
        #print("val=",y_predict[0][0])
    #test_Y = test_Y.reshape(len(test_X),1)
    test_Y = test_Y.flatten()
    print("predic=",y_predict)
    print("actual=",test_Y)
    sum += sqrt(mean_squared_error(test_Y , y_predict))
    print("Sum=",sum)

    print("Error in prediction=",sum/len(test_X))

    #Instead of ,essing with such calculations, use scikit learn's accuracy score meteric
    print(y_predict)
    print (test_Y)
    print("Accuracy in prediction==",r2_score(y_predict,test_Y))


training_data = pd.read_csv("/Users/naveentr/Desktop/VMM_Docker/Fisrt_baseline_transfer/training_data.csv")

print("First 5 entries")
print(training_data.head(5))

# Get the first 3 examples as the test data instances.
# This will be evaluated against the learned model
test_data = training_data.head(3).values

#print the first and second y value.
print("first value=",training_data.head(5)['7'])
#print("Second value=",training_data.head(5)['7'])

#Extract the X-vals from training data
dRow = training_data.head(5).values
#The first column is unnecessary, just drop it.
dRow = np.delete(dRow,[0],axis=1)
print("drow=",dRow)
col_interst = [0,1,2,3,4,5,6]
np_xvals = dRow[:,col_interst] #Just extract the interested columns from training data
print("x_vals=",np_xvals)
#np_yvals = training_data.head(5).values
np_yvals = dRow[:,[7]]
print ("y_vals=",np_yvals)

#X = np.array([[1,1],[1,2],[2,2],[2,3]])
X = np_xvals
#For plotting the points use python's matplotlib..

'''
plt.scatter(X[:,0], X[:,1])
plt.show()
print(X)
'''
#Y-equation calculation
#y = np.dot(X, np.array([1, 2])) + 3 #w1*x1 + w2*x2 + w0
y = np_yvals
#Create an object of LinearRegression() Class
reg = LinearRegression().fit(X,y)

reg.score(X,y) #Returns the co-efficient of X2

print("score=",reg.score(X,y))

#Print the predicted co-efficients
reg.coef_

print("co-efficients==",reg.coef_)

#Print the intercept value
reg.intercept_

'''
2.0
10.0
34.0
175.0
661.0
969.0
40.0
'''
print("Intercept=",reg.intercept_)

#predVal = reg.predict(np.array([[3,5]]))
predVal = reg.predict(np.array([[2,10,34,175,661,969,40]]))
print("predicted value=",predVal)

#Calling the prediction on test_data
print("Test Data prediction and accuracy...")
find_accuracy(test_data,reg)




