#This is the main file which will have generate records and classification within it.
import os
import simple_LinearRegression_Sklearn
import generate_training_data
import copy
import pandas_manipulate


#Step-1: Generate the data for a loop of 5 times.
def generate_data(loop):
    #Run the generate data for a loop of specified times.
    for i in range(0,loop):
        os.system('python3 generate_training_data.py')


# Step-2 : Classify the data by selecting few chunks.
def classify_data(count):
    #How many chunks of the data to be classified.
    i = 0
    csvFile = "/Users/naveentr/Desktop/VMM_Docker/First_baseline_transfer/training_data.csv"
    end_index = 0
    while i<count:
        index = 0
        prepare_data(csvFile_name, index)

        index += 150
        end_index = copy.deepcopy(index)
    cleanup_training(csvFile, end_index) #starting from beginning till the end_index, delete those data frames.


#Step-3 : Clean up the training data by deleting those records.
def cleanup_training(csvFile_name, end_index_val):
    #Call cleanup method and do the cleanup
    cleanup(csvFile_name, end_index_val)