# Data Analytics Library
# Developed by: Batuhan Atasoy, Ph.D. Mechatronics Engineering

# Importing Libraries
import pandas as pd
import numpy as np

# Developing Data Class
class Data:
    
    def __init__(self, file):
        self.file=file
        self.dataframe=pd.read_csv(self.file)
        pd.options.plotting.backend="plotly"
        
    def inspect_df(self):
        print("\n Dataframe Info: \n")
        print(self.dataframe.info())

# Main
if __name__=="__main__":
    file="ai4i2020.csv"
    d=Data(file)
    d.inspect_df()