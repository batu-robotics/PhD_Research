# Data Analytics Library
# Developed by: Batuhan Atasoy, Ph.D. Mechatronics Engineering

# Importing Libraries
import pandas as pd
import numpy as np
from dash import Dash,dcc,html

# Developing Data Class
class Data:
    
    def __init__(self, file):
        self.file=file
        self.dataframe=pd.read_csv(self.file)
        self.headings=self.dataframe.columns
        pd.options.plotting.backend="plotly"
        
    def inspect_df(self):
        """
        print("\n Dataframe Info: \n")
        print(self.dataframe.info())
        """
        
        self.new_df=self.dataframe.iloc[:,3:]
        print("\n New Dataframe Info: \n")
        print(self.new_df.info())
        
        print(f"\n New Dataframe Columns:{list(self.new_df.columns)}\n")
        self.failure_modes=list(self.new_df.columns[-6:])
        print(f"\n Failure Modes:{self.failure_modes}\n")   
    
        self.sub_df=self.new_df.iloc[:,:5]
        #print(self.sub_df.head())
        
        self.figure=[self.sub_df.iloc[:,i].plot.line() for i in range(5)]
        return self.figure
        
        
# Main
if __name__=="__main__":
    file="ai4i2020.csv"
    d=Data(file)
    figure_1=d.inspect_df()
    
    app=Dash("__name__")
    app.layout=html.Div([
        html.H1("Distro"),
        dcc.Graph(figure=figure_1),
    ])
    
    app.run_server(debug=True,host="0.0.0.0",port=5000)