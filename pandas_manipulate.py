import pandas as pd
import numpy as np


def cleanup(csvfile, endIndex):
    cur_data = pd.read_csv(csvFile)

    #Print the current data frame....
    print("Cur data=",cur_data)

    #Remove the first five records and write to the file again.
    #cur_data = cur_data.drop(cur_data.index[:5], inplace=True)
    cur_data = cur_data.iloc[endIndex:]

    #print("befor open",cur_data)
    with open(csvFile,'w') as file:
        #cur_data = pd.DataFrame(cur_data)
        #print("cur_dat=",cur_data)
        #print(df)
        cur_data.to_csv(file,index=False)


    print("Write operation complete.. Please check the file")

if __name__ == "__main__":
    csvfile = "/Users/naveentr/Desktop/VMM_Docker/First_baseline_transfer/pandas_manipulate_learning.csv"
    cleanup(csvfile, 10)