# Data Analytics Library
# Developed by: Batuhan Atasoy, Ph.D. Mechatronics Engineering

# Importing Libraries
import pandas as pd
import numpy as np
from dash import Dash,dcc,html
import matplotlib.pyplot as plt

# Developing Data Class
class Data:
    
    def __init__(self, file):
        self.file=file
        self.dataframe=pd.read_csv(self.file)
        self.headings=self.dataframe.columns
        pd.options.plotting.backend="plotly"
        
    def inspect_df(self):
        # Initial Dataframe Information
        print("\n Dataframe Info: \n")
        print(self.dataframe.info())
        
        # Seperating the Dataframes wrt Process Variables and Failure Modes
        # Process Variables Dataframe (pv_df)
        self.pv_df=self.dataframe.iloc[:,3:8]
        self.pv_head=list(self.pv_df.columns)
        print("\n Process Variables Dataframe Info: \n")
        print(self.pv_df.info())
        
        # Failure Modes Dataframe (fm_df)
        self.fm_df=self.dataframe.iloc[:,8:]
        self.fm_head=list(self.fm_df.columns)
        print("\n Failure Modes Dataframe Info: \n")
        print(self.fm_df.info())
        
    # Attribute to Re-combine the dataframes wrt the Failure Modes
    def recombine_df(self):
        self.sub_df=[pd.concat([self.pv_df, self.fm_df[element]],axis=1) for element in self.fm_head]
        print("\n Sub-Dataframe Info: \n")
        print(f"The Number of Sub-Dataframes: {len(self.sub_df)}.")
        
    # Attribute to Analyze the Dataframes wrt Failure Modes
    def splitted_df(self,split_df):
        self.split_df=split_df
        self.split_head=list(self.split_df.columns)
        
        self.sub_split_df=[pd.concat([self.split_df.iloc[:,i],self.split_df.iloc[:,-1]],axis=1) for i in range(len(self.split_head)-1)]
        print(f"\n The Number of Splitted Sub-Dataframes: {len(self.sub_split_df)}.")
        #print(f"\n The First Splitted Sub-Dataframe Info: {self.sub_split_df[0].columns}.") # Unit Test Code to Check the Separated DF
        return self.sub_split_df
        
    # Calling Sub_splitted Dataframes
    def call_df(self):
        self.subgroup_df=self.splitted_df(self.sub_df[0])
        
        #self.figure=self.subgroup_df[4].groupby(self.subgroup_df[4].columns[-1])[self.subgroup_df[4].columns[0]].plot.line()
        #print(self.subgroup_df[4])
        
        """
        plt.legend(set(self.subgroup_df[4].iloc[:,-1]))
        plt.grid(True)
        plt.show()
        """
        return self.subgroup_df[4]
        
# Main
if __name__=="__main__":
    file="ai4i2020.csv"
    d=Data(file)
    
    d.inspect_df()
    d.recombine_df()
    dashboard=d.call_df()
    fig_1=dashboard.groupby(dashboard.columns[-1])[dashboard.columns[0]].plot.line()
    
    app=Dash("__name__")
    app.layout=html.Div([
        html.H1("Example line plot..."),
        dcc.Graph(figure=fig_1)
    ])
    
    app.run_server(debug=True, host="0.0.0.0", port=5000)