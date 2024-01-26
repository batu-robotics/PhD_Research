# Classification of the AI4I-2020 Milling Dataset
# Written by: Batuhan Atasoy, Ph.D. Mechatronics Engineering

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.stats import linregress
from scipy.cluster.hierarchy import dendrogram, linkage

from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

# Creating Data Class
class Data:
    
    # Initializers
    def __init__(self,filename):
        self.filename=filename
        self.dataframe=pd.read_csv(filename)
        self.headings=self.dataframe.columns
        self.le=LabelEncoder()

    # Gathering the first Information
    def output(self):
        print(self.dataframe.info())

    # Data Manipulation
    def manipulate(self):
        self.df_v2=self.dataframe.iloc[:,2:]

        # Label Encoding of Type
        self.le.fit(self.df_v2.iloc[:,0])        
        self.df_v2.iloc[:,0]=self.le.transform(self.df_v2.iloc[:,0])
        print(self.df_v2.info())

        # Dependant-Independent Variable Separation
        self.x=self.df_v2.iloc[:,:6]
        self.y=self.df_v2.iloc[:,6:]
        #print(self.x.info())
        #print(self.y.info())

        # Creating the Correlation Heatmap
        self.plot_1="Correlation Heatmap of AI4I-2020 Dataset/ Dimension=>{}".format(self.x.shape[1])
        plt.figure(self.plot_1)
        plt.title(self.plot_1)
        sns.heatmap(self.x.corr(),annot=True,cmap=plt.cm.Reds)
        plt.show()

        # Feature Engineering with Correlation Matrix


    # Unsupervised Learning for Clustering
    def cluster(self):
        self.model_1=AgglomerativeClustering()
        self.model_1=self.model_1.fit(self.x)
        
        self.relation=linkage(self.model_1.children_)
        plt.figure("Dendrogram")
        self.k=dendrogram(self.relation)
        plt.show()

        self.n_class=len(set(self.k['leaves_color_list']))
        print("Number of detected clusters={}".format(self.n_class))

        print('\n Analysis Results \n')

        self.k_means=KMeans(n_clusters=self.n_class, random_state=0)
        self.k_means=self.k_means.fit(self.x)
        self.results_1=self.k_means.labels_
        print('Clustered Data \n',self.results_1)

        self.y_machine=self.y["Machine failure"]
        print('Machine Faults \n',self.y_machine)

        self.plot_3="Confusion Matrix"
        plt.figure(self.plot_3)
        plt.title(self.plot_3)
        sns.heatmap(confusion_matrix(self.y_machine,self.results_1),annot=True,cmap=plt.cm.Blues)
        plt.show()
        
    # Dimensional Reduction with PCA
    def pca(self):

        self.pca_list=[]
        
        for j in range(self.x.shape[1]-1):
            self.pca=PCA(j+1,whiten=True)
            self.pca.fit(self.x)
            self.x_pca=self.pca.transform(self.x)
            self.pca_list.append(float("{:.2f}".format(100*sum(self.pca.explained_variance_ratio_))))

        # Constructing PCA for the Dataset if Variance Ratio is Greater than or Equal to 0.98
            if sum(self.pca.explained_variance_ratio_)>0.97:
                self.embedded_dimension=j+1
                print("\n Optimal Number of Base Axes={} \n".format(j+1))
                self.dataframe_pca=self.pca.transform(self.x)
                break

        self.plot_4="Principal Component Analysis"
        plt.figure(self.plot_4)
        plt.plot(self.pca_list,'-r')
        plt.plot(self.pca_list,'^b')
        plt.title(self.plot_4)
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Variance Ratio")
        plt.grid(True)
        plt.show()

        self.head_pca=[]
        # generating the New Independent Dataframe
        for i in range(self.embedded_dimension):
            self.head_pca.append('PCA-{}'.format(i+1))

        self.x_pca=pd.DataFrame(self.x_pca,columns=self.head_pca)
        print(self.x_pca.info())

        self.plot_5="PCA Graph for Machine Failure Model"
        plt.figure(self.plot_5)
        plt.title(self.plot_5)
        plt.scatter(self.x_pca.iloc[:,0],self.x_pca.iloc[:,1],c=self.y_machine)
        plt.xlabel(self.x_pca.columns[0])
        plt.ylabel(self.x_pca.columns[1])
        plt.legend(list(self.y_machine.unique()))
        plt.grid(True)
        plt.show()

    # Deep Learning with ANN
    def ann(self):
        self.sc=StandardScaler()
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x_pca,self.y_machine,test_size=0.2,shuffle=True)

        self.x_train=self.sc.fit_transform(self.x_train)
        self.x_test=self.sc.transform(self.x_test)

        self.function='relu'

        self.ann_model=Sequential()
        self.ann_model.add(Dense(units=2*self.x_train.shape[1],input_dim=self.x_train.shape[1],kernel_initializer = 'uniform',activation=self.function))
        self.ann_model.add(Dense(units=3*self.x_train.shape[1],input_dim=2*self.x_train.shape[1],kernel_initializer = 'uniform',activation=self.function))
        self.ann_model.add(Dense(units=4*self.x_train.shape[1],input_dim=3*self.x_train.shape[1],kernel_initializer = 'uniform',activation=self.function))
        self.ann_model.add(Dense(units=1,input_dim=4*self.x_train.shape[1],kernel_initializer = 'uniform',activation=self.function))
        self.ann_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

        self.ann_model_2=self.ann_model.fit(self.x_train,self.y_train,batch_size=16,epochs=20)
        self.y_ann=self.ann_model.predict(self.x_test)
        self.slope,self.intercept,self.r_value,self.p_value,self.std_err=linregress(self.y_test,self.y_ann.T)

        self.results_df=pd.DataFrame(np.column_stack((self.y_test,self.y_ann)),columns=['Test Data','ANN Result']).sort_values(by=['Test Data'])

    # Run the Class
    def run(self):
        self.output()
        self.manipulate()
        self.cluster()
        self.pca()
        self.ann()

# Main
if __name__ == '__main__':
    file="ai4i2020.csv"
    d=Data(file)
    d.run()