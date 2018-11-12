import pandas as pd
import numpy as np

csvFile = "/Users/naveentr/Desktop/VMM_Docker/Fisrt_baseline_transfer/pandas_manipulate_learning.csv"
cur_data = pd.read_csv(csvFile)

#Print the current data frame....
print("Cur data=",cur_data)

#Remove the first five records and write to the file again.
cur_data = cur_data.drop(cur_data.index[:5], inplace=True)

with open(csvFile,'w') as file:
    cur_data.to_csv(file,header=True)


print("Write operation complete.. Please check the file")