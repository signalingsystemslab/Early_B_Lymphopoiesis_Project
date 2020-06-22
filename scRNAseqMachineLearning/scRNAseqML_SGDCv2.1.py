# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:33:27 2019
# Please download the original data files from GEO accession provided with the paper, and change the file name accordingly.
@author: Eason
"""
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
import seaborn as sns
import math
import time
# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from joblib import dump, load
# %%
# Enter the name for the dataset which matrix.mtx, feature.tsv, and barcodes.tsv are renamed with (for example, scRNAseq1v2matrix.mtx)
DataSet = 'scRNAseq1v2'
# Enter the file name of the DE gene list exported from Loupe Browser
MarkerGeneOrigin = 'WTY_DE'
# Enter the threshold p-value for the determination of marker genes for the heatmap
Marker_P_Threshold = 0.01
# Enter the list of name for clusters in the DE gene list for marker determination
MarkerCellClusterList = ['WTY_ClusterA', 'WTY_ClusterB', 'WTY_ClusterC', 'WTY_ClusterCp', 'WTY_ClusterD']
# Enter the name of the heatmap
Samplename = 'WTYKO_CR31_HMv1'
# Enter the file name for clusters information input
ClusterInfo = 'scRNAseq1v2_Kmeans5_Cluster_FACS.csv'
# Enter the list of the clusters exported from loupe browser for heatmap x-axis
CellClusterList = ['WTY_FrA_ClusterA', 'WTY_FrB_ClusterB', 'WTY_FrC_ClusterC', 'WTY_FrCp_ClusterCp', 'WTY_FrD_ClusterD', 'WTO_FrA_ClusterA', 'WTO_FrB_ClusterB', 'WTO_FrC_ClusterC', 'WTO_FrCp_ClusterCp', 'WTO_FrD_ClusterD', 'IkBKO_FrA_ClusterA', 'IkBKO_FrB_ClusterB', 'IkBKO_FrB_ClusterC', 'IkBKO_FrB_ClusterCp', 'IkBKO_FrCCp_ClusterC', 'IkBKO_FrCCp_ClusterCp', 'IkBKO_FrD_ClusterD']
# Enter the list of clusters you would like to show in the generated heatmap, by the order from right to left shown in the heatmap
WTY_AllClusters = ['WTY_FrA_ClusterA', 'WTY_FrB_ClusterB', 'WTY_FrC_ClusterC', 'WTY_FrCp_ClusterCp', 'WTY_FrD_ClusterD']
WTO_AllClusters = ['WTO_FrA_ClusterA', 'WTO_FrB_ClusterB', 'WTO_FrC_ClusterC', 'WTO_FrCp_ClusterCp', 'WTO_FrD_ClusterD']
IkBKO_AllClusters = ['IkBKO_FrA_ClusterA', 'IkBKO_FrB_ClusterB', 'IkBKO_FrB_ClusterC', 'IkBKO_FrB_ClusterCp', 'IkBKO_FrCCp_ClusterC', 'IkBKO_FrCCp_ClusterCp', 'IkBKO_FrD_ClusterD']
# Enter the percentage of data splitted for training, the rest of it would be testing dataset
TrainingPercentage = 80
# Enter the colormap for heatmap
ColorMap = ["purple","black","orange"]
divnorm = DivergingNorm(vmin=0, vcenter=7, vmax=90)
# %%
start = time.time()
print('Main Dataframe Matrix Loading Started')
# Enter the name of the scRNAseq data matrix here
df4=pd.read_csv(DataSet + 'matrix.mtx', sep=' ', header=0, skiprows=[1,2], usecols = ['%%MatrixMarket', 'matrix', 'coordinate'], dtype={'%%MatrixMarket':'int16', 'matrix':'int16', 'coordinate':'int16'})
df4.rename(columns={'%%MatrixMarket': 'GeneID', 'matrix': 'CellID', 'coordinate': 'Count'}, inplace=True)
df5=df4.pivot(index='GeneID', columns='CellID', values='Count')
print('Main dataframe matrix Loading Completed')
# %%
# Combination of Marker Genes for heatmap based on the DE genes in all different clusters
df301=pd.read_csv(MarkerGeneOrigin + '_Marker_Genes.csv', sep=',', header=0)
#df301=df301.drop(columns=['FeatureID', 'WTY_ClusterD Average', 'WTY_ClusterCp Average', 'WTY_ClusterC Average', 'WTY_ClusterB Average', 'WTY_ClusterA Average'])

counter = 0
df302=df301.loc[df301[MarkerCellClusterList[counter] + ' P-Value'] < Marker_P_Threshold]
df302=df302.loc[df302[MarkerCellClusterList[counter] + ' Log2 Fold Change'] > 0]
df302=df302.sort_values(MarkerCellClusterList[counter] + ' Log2 Fold Change', ascending = False)
df303=pd.Series(df302['FeatureName']).to_frame()
df303.rename(columns={'FeatureName': 'Markers'}, inplace=True)
df303['Cluster'] = MarkerCellClusterList[counter]
df304 = df303.reset_index(drop=True)

counter = 1
for counter in range(1,len(MarkerCellClusterList)):
    df302=df301.loc[df301[MarkerCellClusterList[counter] + ' P-Value'] < Marker_P_Threshold]
    df302=df302.loc[df302[MarkerCellClusterList[counter] + ' Log2 Fold Change'] > 0]
    df302=df302.sort_values(MarkerCellClusterList[counter] + ' Log2 Fold Change', ascending = False)
    df303=pd.Series(df302['FeatureName']).to_frame()
    df303.rename(columns={'FeatureName': 'Markers'}, inplace=True)
    df303['Cluster'] = MarkerCellClusterList[counter]
    df303 = df303.reset_index(drop=True)
    df304 = pd.concat([df304,df303], join='outer', ignore_index=True)
    counter = counter+1
else:
    df305 = df304.drop_duplicates(subset=['Markers'], keep = 'first')
    df305.to_csv( MarkerGeneOrigin + '_MarkerGeneList_Output.csv', index=False)
    print('Marker Gene List Generation Completed')
# %%
print('Gene of Interest ID Query Started')
df201=pd.read_table(DataSet + 'features.tsv', header=None)
df201.columns = ['Entrez', 'GeneName', 'Type']
df202 = df305
#df202=pd.read_csv('Markers' + Samplename + '.csv', sep=',', header=0, usecols = ['Markers'])
Marker=df202['Markers'].tolist()


SingleGene = (Marker[0])
IndexNumber = (df201[df201['GeneName']==SingleGene].index.values) +1

counter = 1
for counter in range(1,len(df202['Markers'])):
    SingleGene = (Marker[counter])
    IndexNumberQuery = (df201[df201['GeneName']==SingleGene].index.values) +1
    IndexNumber = np.concatenate((IndexNumber,IndexNumberQuery))
    counter = counter+1
else:
    print('Gene of Interest ID Query Completed')

IndexNumberList=IndexNumber.tolist()
df202['GeneID'] = IndexNumberList
df202.to_csv( Samplename + '_MarkerGeneQuery_output.csv', index=False)

# %%
print('Barcode-Cluster information Loading Started')
#Enter the name of the list of barcodes here
df1=pd.read_table(DataSet + 'barcodes.tsv' ,header=None)
df1.columns = ['Barcode']
#df1=pd.read_csv('scRNAseq1AGGRbarcodes.csv', sep=',',header=0)
#Enter the name of the table correlating barcodes and clusters here, the file should consist of Barcode as first column, Clusters as second column, row0 is header.
#Make sure to exclude doublets and unwanted genes(Xist or other ribosomal gene for example), leave the clusters information in the doublets as empty
df2=pd.read_csv(ClusterInfo, sep=',',header=0)
df4 = pd.merge(df1,df2, on='Barcode', how='outer')
df4.rename({list(df4)[1]: 'Clusters'}, axis=1, inplace=True)
df4['Clusters'].fillna('Delete', inplace = True)
df4['CellID'] = range(1, len(df4)+1)
df4.to_csv( Samplename + '_Barcode_Cluster_output.csv', index=False)
print('Barcode-Cluster Information Mapped')
# %%
df3 = df4
df101 = pd.DataFrame(df202['GeneID'])
df100 = pd.DataFrame(df202['Markers'])

print('Barcode-Cluster Information Extraction')
counter = 0
for counter in range(0,len(CellClusterList)):
    vars()[CellClusterList[counter]] = df3[df3.Clusters == CellClusterList[counter]]['CellID'].tolist()
    print(CellClusterList[counter])
else:
    Delete = df3[df3.Clusters == 'Delete']['CellID'].tolist()
    print('Delete')
    
WTYTotal = []
for counter in range(0,len(WTY_AllClusters)):
    WTYCluster = globals()[WTY_AllClusters[counter]]
    WTYTotal = WTYTotal + WTYCluster

WTOTotal = []
for counter in range(0,len(WTO_AllClusters)):
    WTOCluster = globals()[WTO_AllClusters[counter]]
    WTOTotal = WTOTotal + WTOCluster

IkBKOTotal = []
for counter in range(0,len(IkBKO_AllClusters)):
    IkBKOCluster = globals()[IkBKO_AllClusters[counter]]
    IkBKOTotal = IkBKOTotal + IkBKOCluster
print('Barcode-Cluster information Loading Completed')
# %%
print('Marker Gene Extraction Started')
GeneList = df101.loc[:,'GeneID'].tolist()
df6 = df5.fillna(0)
df7 = df6.loc[GeneList]
print('Marker Gene Extraction Completed')
print('Alignment of Cell-ID to Clusters Started')
dfWTYMarkerGene = df7.filter(WTYTotal, axis=1)
dfWTOMarkerGene = df7.filter(WTOTotal, axis=1)
dfIkBKOMarkerGene = df7.filter(IkBKOTotal, axis=1)
print('Alignment of Cell-ID to Clusters Completed')
# %%
# Transformation of the feature matrix from dataset
print('Preparation for feature matrix from dataset Started')
# For MarkerGene feature matrix
dfWTYMarkerGene = dfWTYMarkerGene.transpose()
MarkerGene_WTYFeatures_origin = dfWTYMarkerGene.to_numpy()
dfWTOMarkerGene = dfWTOMarkerGene.transpose()
MarkerGene_WTOFeatures_origin = dfWTOMarkerGene.to_numpy()
dfIkBKOMarkerGene = dfIkBKOMarkerGene.transpose()
MarkerGene_IkBKOFeatures_origin = dfIkBKOMarkerGene.to_numpy()
# For AllGene feature matrix
dfWTYAllGene = df6.filter(WTYTotal, axis=1)
dfWTYAllGene = dfWTYAllGene.transpose()
AllGene_WTYFeatures_origin = dfWTYAllGene.to_numpy()
dfWTOAllGene = df6.filter(WTOTotal, axis=1)
dfWTOAllGene = dfWTOAllGene.transpose()
AllGene_WTOFeatures_origin = dfWTOAllGene.to_numpy()
dfIkBKOAllGene = df6.filter(IkBKOTotal, axis=1)
dfIkBKOAllGene = dfIkBKOAllGene.transpose()
AllGene_IkBKOFeatures_origin = dfIkBKOAllGene.to_numpy()
print('Preparation for feature matrix from dataset Completed')
# %%
# Transformation of label lists from dataset
print('Preparation for label lists from dataset Started')
WTYTotalMinusOne = WTYTotal
WTYTotalMinusOne[:] = [WTYTotalMinusOne - 1 for WTYTotalMinusOne in WTYTotalMinusOne]
WTYLabel = df4.loc[WTYTotalMinusOne]
WTYLabel = WTYLabel.replace('WTY_FrA_ClusterA', 1)
WTYLabel = WTYLabel.replace('WTY_FrB_ClusterB', 2)
WTYLabel = WTYLabel.replace('WTY_FrC_ClusterC', 3)
WTYLabel = WTYLabel.replace('WTY_FrCp_ClusterCp', 4)
WTYLabel = WTYLabel.replace('WTY_FrD_ClusterD', 5)
WTYLabels_origin = WTYLabel['Clusters'].to_numpy()

WTOTotalMinusOne = WTOTotal
WTOTotalMinusOne[:] = [WTOTotalMinusOne - 1 for WTOTotalMinusOne in WTOTotalMinusOne]
WTOLabel = df4.loc[WTOTotalMinusOne]
WTOLabel = WTOLabel.replace('WTO_FrA_ClusterA', 1)
WTOLabel = WTOLabel.replace('WTO_FrB_ClusterB', 2)
WTOLabel = WTOLabel.replace('WTO_FrC_ClusterC', 3)
WTOLabel = WTOLabel.replace('WTO_FrCp_ClusterCp', 4)
WTOLabel = WTOLabel.replace('WTO_FrD_ClusterD', 5)
WTOLabels_origin = WTOLabel['Clusters'].to_numpy()

IkBKOTotalMinusOne = IkBKOTotal
IkBKOTotalMinusOne[:] = [IkBKOTotalMinusOne - 1 for IkBKOTotalMinusOne in IkBKOTotalMinusOne]
IkBKOLabel = df4.loc[IkBKOTotalMinusOne]
IkBKOLabel = IkBKOLabel.replace('IkBKO_FrA_ClusterA', 1)
IkBKOLabel = IkBKOLabel.replace('IkBKO_FrB_ClusterB', 2)
IkBKOLabel = IkBKOLabel.replace('IkBKO_FrB_ClusterC', 3)
IkBKOLabel = IkBKOLabel.replace('IkBKO_FrCCp_ClusterC', 3)
IkBKOLabel = IkBKOLabel.replace('IkBKO_FrB_ClusterCp', 4)
IkBKOLabel = IkBKOLabel.replace('IkBKO_FrCCp_ClusterCp', 4)
IkBKOLabel = IkBKOLabel.replace('IkBKO_FrD_ClusterD', 5)
IkBKOLabels_origin = IkBKOLabel['Clusters'].to_numpy()
print('Preparation for label lists from dataset Completed')
# %%
# Data splitting
print('Data Splitting Started')
## Testing Dataset splitting
WTYFeatures, WTYFeatures_test, WTYLabels, WTYLabels_test = train_test_split(MarkerGene_WTYFeatures_origin, WTYLabels_origin, test_size=(1-(TrainingPercentage/100)), shuffle=True, random_state=1)
WTOFeatures, WTOFeatures_test, WTOLabels, WTOLabels_test = train_test_split(MarkerGene_WTOFeatures_origin, WTOLabels_origin, test_size=(1-(TrainingPercentage/100)), shuffle=True, random_state=1)
IkBKOFeatures, IkBKOFeatures_test, IkBKOLabels, IkBKOLabels_test = train_test_split(MarkerGene_IkBKOFeatures_origin, IkBKOLabels_origin, test_size=(1-(TrainingPercentage/100)), shuffle=True, random_state=1)
AllGene_WTYFeatures, AllGene_WTYFeatures_test, WTYLabels, WTYLabels_test = train_test_split(AllGene_WTYFeatures_origin, WTYLabels_origin, test_size=(1-(TrainingPercentage/100)), shuffle=True, random_state=1)
AllGene_WTOFeatures, AllGene_WTOFeatures_test, WTOLabels, WTOLabels_test = train_test_split(AllGene_WTOFeatures_origin, WTOLabels_origin, test_size=(1-(TrainingPercentage/100)), shuffle=True, random_state=1)
AllGene_IkBKOFeatures, AllGene_IkBKOFeatures_test, IkBKOLabels, IkBKOLabels_test = train_test_split(AllGene_IkBKOFeatures_origin, IkBKOLabels_origin, test_size=(1-(TrainingPercentage/100)), shuffle=True, random_state=1)
## Validation Dataset splitting
WTYFeatures_train, WTYFeatures_valid, WTYLabels_train, WTYLabels_valid = train_test_split( WTYFeatures, WTYLabels, test_size=0.33,shuffle=True, random_state=1)
AllGene_WTYFeatures_train, AllGene_WTYFeatures_valid, WTYLabels_train, WTYLabels_valid = train_test_split( AllGene_WTYFeatures, WTYLabels, test_size=0.33,shuffle=True, random_state=1)
WTOFeatures_train, WTOFeatures_valid, WTOLabels_train, WTOLabels_valid = train_test_split( WTOFeatures, WTOLabels, test_size=0.33,shuffle=True, random_state=1)
AllGene_WTOFeatures_train, AllGene_WTOFeatures_valid, WTOLabels_train, WTOLabels_valid = train_test_split( AllGene_WTOFeatures, WTOLabels, test_size=0.33,shuffle=True, random_state=1)
IkBKOFeatures_train, IkBKOFeatures_valid, IkBKOLabels_train, IkBKOLabels_valid = train_test_split( IkBKOFeatures, IkBKOLabels, test_size=0.33,shuffle=True, random_state=1)
AllGene_IkBKOFeatures_train, AllGene_IkBKOFeatures_valid, IkBKOLabels_train, IkBKOLabels_valid = train_test_split( AllGene_IkBKOFeatures, IkBKOLabels, test_size=0.33,shuffle=True, random_state=1)
print('Data Splitting Completed')
# %%
# WTY MarkerGene SGDC K-fold validation and scanning
KFoldSplit = 10
setAlpha = np.array( [0.000000001,0.000000003,0.00000001,0.00000003,0.0000001,0.0000003,0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.,10.,30.,100.,300] )
#setAlpha = np.array([0.001,0.003,0.01])
kf = KFold(n_splits=KFoldSplit, shuffle=True, random_state=1)
WTY_MarkerGene_SGDC_AcuuracySummary=pd.DataFrame()
WTOPrediction_MarkerGene_SGDC_AcuuracySummary=pd.DataFrame()
IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary=pd.DataFrame()
WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary=pd.DataFrame()
WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
for Alpha in setAlpha:
    WTY_MarkerGene_SGDC_AlphaAccuracies_Kfold = []
    WTOPrediction_MarkerGene_SGDC_Kfold_AlphaAccuracies = []
    IkBKOPrediction_MarkerGene_SGDC_Kfold_AlphaAccuracies = []
    Kn = 1
    for train_index, valid_index in kf.split( WTYFeatures ):
        # Splitting Data for K-fold validation
        SectionStart = time.time()
        X_train = WTYFeatures[train_index]
        X_valid  = WTYFeatures[valid_index]
        Y_train = WTYLabels[train_index]
        Y_valid  = WTYLabels[valid_index]
        # Training for SGDC model on WTY dataset
        WTY_MarkerGene_SGDC_Kfold = SGDClassifier( n_jobs=-1, random_state=1, max_iter=1000, shuffle=True, alpha=Alpha, loss='perceptron', learning_rate='optimal')
        WTY_MarkerGene_SGDC_Kfold.fit(X_train, Y_train)
        WTY_MarkerGene_SGDC_AlphaAccuracies_Kfold.append(WTY_MarkerGene_SGDC_Kfold.score(X_valid, Y_valid))  
        dump(WTY_MarkerGene_SGDC_Kfold, 'ML_WTYMarkerGene_SGDC_KFold_A' + str(Alpha) + 'K' + str(Kn) + '_v2.joblib') 
        SectionEnd = time.time()
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        print("Fitted model MarkerGene SGDC A" + str(Alpha) + 'K' + str(Kn) +" saved")
        SectionStart = time.time()
        WTY_MarkerGene_SGDC_Kfold_PredictionTest = WTY_MarkerGene_SGDC_Kfold.predict(WTYFeatures_test)
        WTY_MarkerGene_SGDC_Kfold_PredictionTestAddition = pd.DataFrame(data=WTY_MarkerGene_SGDC_Kfold_PredictionTest)
        WTY_MarkerGene_SGDC_Kfold_PredictionTestAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary = pd.concat([WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary,WTY_MarkerGene_SGDC_Kfold_PredictionTestAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_test_MarkerGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTY_MarkerGene_SGDC_Kfold_PredictionOrigin = WTY_MarkerGene_SGDC_Kfold.predict(MarkerGene_WTYFeatures_origin)
        WTY_MarkerGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTY_MarkerGene_SGDC_Kfold_PredictionOrigin)
        WTY_MarkerGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary,WTY_MarkerGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_origin_MarkerGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTO_MarkerGene_SGDC_Kfold_PredictionOrigin = WTY_MarkerGene_SGDC_Kfold.predict(MarkerGene_WTOFeatures_origin)
        WTO_MarkerGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTO_MarkerGene_SGDC_Kfold_PredictionOrigin)
        WTO_MarkerGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary,WTO_MarkerGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        WTOPrediction_MarkerGene_SGDC_Kfold_AlphaAccuracies.append( accuracy_score(WTOLabels_origin, WTO_MarkerGene_SGDC_Kfold_PredictionOrigin) )
        SectionEnd = time.time()
        print("WTOFeature_origin_MarkerGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin = WTY_MarkerGene_SGDC_Kfold.predict(MarkerGene_IkBKOFeatures_origin)
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin)
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary,IkBKO_MarkerGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        IkBKOPrediction_MarkerGene_SGDC_Kfold_AlphaAccuracies.append( accuracy_score(IkBKOLabels_origin, IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin) )
        SectionEnd = time.time()
        print("IkBKOFeature_origin_MarkerGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        Kn = Kn + 1
    # Combination of accuracy matrix into dataframe
    WTY_MarkerGene_SGDC_AcuuracyAddition = pd.DataFrame(data=WTY_MarkerGene_SGDC_AlphaAccuracies_Kfold)
    WTY_MarkerGene_SGDC_AcuuracyAddition.columns = ['A' + str(Alpha)]
    WTY_MarkerGene_SGDC_AcuuracySummary = pd.concat([WTY_MarkerGene_SGDC_AcuuracySummary,WTY_MarkerGene_SGDC_AcuuracyAddition], axis=1, join='outer')
    WTOPrediction_MarkerGene_SGDC_AcuuracyAddition = pd.DataFrame(data=WTOPrediction_MarkerGene_SGDC_Kfold_AlphaAccuracies)
    WTOPrediction_MarkerGene_SGDC_AcuuracyAddition.columns = ['A' + str(Alpha)]
    WTOPrediction_MarkerGene_SGDC_AcuuracySummary = pd.concat([WTOPrediction_MarkerGene_SGDC_AcuuracySummary,WTOPrediction_MarkerGene_SGDC_AcuuracyAddition], axis=1, join='outer')
    IkBKOPrediction_MarkerGene_SGDC_AcuuracyAddition = pd.DataFrame(data=IkBKOPrediction_MarkerGene_SGDC_Kfold_AlphaAccuracies)
    IkBKOPrediction_MarkerGene_SGDC_AcuuracyAddition.columns = ['A' + str(Alpha)]
    IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary = pd.concat([IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary,IkBKOPrediction_MarkerGene_SGDC_AcuuracyAddition], axis=1, join='outer')
# Calculating mean, STD, and STE of accuracy values from K-fold fitted models and predictions  
WTY_MarkerGene_SGDC_AcuuracySummary.loc['Mean'] = WTY_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].mean()
WTY_MarkerGene_SGDC_AcuuracySummary.loc['STD'] = WTY_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()
WTY_MarkerGene_SGDC_AcuuracySummary.loc['STE'] = WTY_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()/math.sqrt(KFoldSplit)
WTY_MarkerGene_SGDC_AcuuracySummary.to_csv( 'WTY_MarkerGene_SGDC_Kfold_AcuuracySummary.csv', index=False)
print("WTY_MarkerGene_SGDC_Kfold_AcuuracySummary saved.")
WTOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['Mean'] = WTOPrediction_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].mean()
WTOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['STD'] = WTOPrediction_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()
WTOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['STE'] = WTOPrediction_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()/math.sqrt(KFoldSplit)
WTOPrediction_MarkerGene_SGDC_AcuuracySummary.to_csv( 'WTOPrediction_MarkerGene_SGDC_Kfold_AcuuracySummary.csv', index=False)
print("WTOPrediction_MarkerGene_SGDC_Kfold_AcuuracySummary saved.")
IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['Mean'] = IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].mean()
IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['STD'] = IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()
IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['STE'] = IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()/math.sqrt(KFoldSplit)
IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.to_csv( 'IkBKOPrediction_MarkerGene_SGDC_Kfold_AcuuracySummary.csv', index=False)
print("IkBKOPrediction_MarkerGene_SGDC_Kfold_AcuuracySummary saved.")
# %%
# Plotting the line graph of alpha vs accuracy in fitted models and predictions
x = setAlpha
WTYLine = WTY_MarkerGene_SGDC_AcuuracySummary.loc['Mean']
WTYError = WTY_MarkerGene_SGDC_AcuuracySummary.loc['STE']
WTYLineU = np.add(WTYLine,WTYError)
WTYLineL = np.subtract(WTYLine,WTYError)
WTOLine = WTOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['Mean']
WTOError = WTOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['STE']
WTOLineU = np.add(WTOLine,WTOError)
WTOLineL = np.subtract(WTOLine,WTOError)
IkBKOLine = IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['Mean']
IkBKOError = IkBKOPrediction_MarkerGene_SGDC_AcuuracySummary.loc['STE']
IkBKOLineU = np.add(IkBKOLine,IkBKOError)
IkBKOLineL = np.subtract(IkBKOLine,IkBKOError)

fig = plt.figure()
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['font.size'] = 20
plt.figure(figsize=(8, 12))
plt.xscale('log')
plt.ylim(0.7,1.0)
fig = plt.plot(x, WTYLine, 'k', color='#267F00', linewidth=2.5)
fig = plt.fill_between(x, WTYLineU, WTYLineL, alpha=0.5, edgecolor='#267F00', facecolor='#267F00', antialiased=True)
fig = plt.plot(x, WTOLine, 'k', color='#FF6A00', linewidth=2.5)
fig = plt.fill_between(x, WTOLineU, WTOLineL, alpha=0.5, edgecolor='#FF6A00', facecolor='#FF6A00', antialiased=True)
fig = plt.plot(x, IkBKOLine, 'k', color='#FF0000', linewidth=2.5)
fig = plt.fill_between(x, IkBKOLineU, IkBKOLineL, alpha=0.5, edgecolor='#FF0000', facecolor='#FF0000', antialiased=True)
plt.show()

output = fig.get_figure()
output.savefig("ML_SGDC_AlphaAccuracy_MarkerGene_Kfoldv1.png", format='png', dpi=300)
# %%
# WTY AllGene SGDC K-fold validation and scanning
KFoldSplit = 10
setAlpha = np.array( [0.000000001,0.000000003,0.00000001,0.00000003,0.0000001,0.0000003,0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.,10.,30.,100.,300] )
#setAlpha = np.array([0.001,0.003,0.01])
kf = KFold(n_splits=KFoldSplit, shuffle=True, random_state=1)
WTY_AllGene_SGDC_AcuuracySummary=pd.DataFrame()
WTOPrediction_AllGene_SGDC_AcuuracySummary=pd.DataFrame()
IkBKOPrediction_AllGene_SGDC_AcuuracySummary=pd.DataFrame()
WTY_AllGene_SGDC_Kfold_PredictionTest_Summary=pd.DataFrame()
WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
for Alpha in setAlpha:
    WTY_AllGene_SGDC_AlphaAccuracies_Kfold = []
    WTOPrediction_AllGene_SGDC_Kfold_AlphaAccuracies = []
    IkBKOPrediction_AllGene_SGDC_Kfold_AlphaAccuracies = []
    Kn = 1
    for train_index, valid_index in kf.split( AllGene_WTYFeatures ):
        # Splitting Data for K-fold validation
        SectionStart = time.time()
        X_train = AllGene_WTYFeatures[train_index]
        X_valid  = AllGene_WTYFeatures[valid_index]
        Y_train = WTYLabels[train_index]
        Y_valid  = WTYLabels[valid_index]
        # Training for SGDC model on WTY dataset
        WTY_AllGene_SGDC_Kfold = SGDClassifier( n_jobs=-1, random_state=1, max_iter=1000, shuffle=True, alpha=Alpha, loss='perceptron', learning_rate='optimal')
        WTY_AllGene_SGDC_Kfold.fit(X_train, Y_train)
        WTY_AllGene_SGDC_AlphaAccuracies_Kfold.append(WTY_AllGene_SGDC_Kfold.score(X_valid, Y_valid))  
        dump(WTY_AllGene_SGDC_Kfold, 'ML_WTYAllGene_SGDC_KFold_A' + str(Alpha) + 'K' + str(Kn) + '_v2.joblib') 
        SectionEnd = time.time()
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        print("Fitted model AllGene SGDC A" + str(Alpha) + 'K' + str(Kn) +" saved")
        SectionStart = time.time()
        WTY_AllGene_SGDC_Kfold_PredictionTest = WTY_AllGene_SGDC_Kfold.predict(AllGene_WTYFeatures_test)
        WTY_AllGene_SGDC_Kfold_PredictionTestAddition = pd.DataFrame(data=WTY_AllGene_SGDC_Kfold_PredictionTest)
        WTY_AllGene_SGDC_Kfold_PredictionTestAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        WTY_AllGene_SGDC_Kfold_PredictionTest_Summary = pd.concat([WTY_AllGene_SGDC_Kfold_PredictionTest_Summary,WTY_AllGene_SGDC_Kfold_PredictionTestAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_test_AllGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTY_AllGene_SGDC_Kfold_PredictionOrigin = WTY_AllGene_SGDC_Kfold.predict(AllGene_WTYFeatures_origin)
        WTY_AllGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTY_AllGene_SGDC_Kfold_PredictionOrigin)
        WTY_AllGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary,WTY_AllGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_origin_AllGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTO_AllGene_SGDC_Kfold_PredictionOrigin = WTY_AllGene_SGDC_Kfold.predict(AllGene_WTOFeatures_origin)
        WTO_AllGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTO_AllGene_SGDC_Kfold_PredictionOrigin)
        WTO_AllGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary,WTO_AllGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        WTOPrediction_AllGene_SGDC_Kfold_AlphaAccuracies.append( accuracy_score(WTOLabels_origin, WTO_AllGene_SGDC_Kfold_PredictionOrigin) )
        SectionEnd = time.time()
        print("WTOFeature_origin_AllGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        IkBKO_AllGene_SGDC_Kfold_PredictionOrigin = WTY_AllGene_SGDC_Kfold.predict(AllGene_IkBKOFeatures_origin)
        IkBKO_AllGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=IkBKO_AllGene_SGDC_Kfold_PredictionOrigin)
        IkBKO_AllGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(Alpha) + 'K' + str(Kn)]
        IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary,IkBKO_AllGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        IkBKOPrediction_AllGene_SGDC_Kfold_AlphaAccuracies.append( accuracy_score(IkBKOLabels_origin, IkBKO_AllGene_SGDC_Kfold_PredictionOrigin) )
        SectionEnd = time.time()
        print("IkBKOFeature_origin_AllGene SGDC A" + str(Alpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        Kn = Kn + 1
    # Combination of accuracy matrix into dataframe
    WTY_AllGene_SGDC_AcuuracyAddition = pd.DataFrame(data=WTY_AllGene_SGDC_AlphaAccuracies_Kfold)
    WTY_AllGene_SGDC_AcuuracyAddition.columns = ['A' + str(Alpha)]
    WTY_AllGene_SGDC_AcuuracySummary = pd.concat([WTY_AllGene_SGDC_AcuuracySummary,WTY_AllGene_SGDC_AcuuracyAddition], axis=1, join='outer')
    WTOPrediction_AllGene_SGDC_AcuuracyAddition = pd.DataFrame(data=WTOPrediction_AllGene_SGDC_Kfold_AlphaAccuracies)
    WTOPrediction_AllGene_SGDC_AcuuracyAddition.columns = ['A' + str(Alpha)]
    WTOPrediction_AllGene_SGDC_AcuuracySummary = pd.concat([WTOPrediction_AllGene_SGDC_AcuuracySummary,WTOPrediction_AllGene_SGDC_AcuuracyAddition], axis=1, join='outer')
    IkBKOPrediction_AllGene_SGDC_AcuuracyAddition = pd.DataFrame(data=IkBKOPrediction_AllGene_SGDC_Kfold_AlphaAccuracies)
    IkBKOPrediction_AllGene_SGDC_AcuuracyAddition.columns = ['A' + str(Alpha)]
    IkBKOPrediction_AllGene_SGDC_AcuuracySummary = pd.concat([IkBKOPrediction_AllGene_SGDC_AcuuracySummary,IkBKOPrediction_AllGene_SGDC_AcuuracyAddition], axis=1, join='outer')
# Calculating mean, STD, and STE of accuracy values from K-fold fitted models and predictions  
WTY_AllGene_SGDC_AcuuracySummary.loc['Mean'] = WTY_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].mean()
WTY_AllGene_SGDC_AcuuracySummary.loc['STD'] = WTY_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()
WTY_AllGene_SGDC_AcuuracySummary.loc['STE'] = WTY_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()/math.sqrt(KFoldSplit)
WTY_AllGene_SGDC_AcuuracySummary.to_csv( 'WTY_AllGene_SGDC_Kfold_AcuuracySummary.csv', index=False)
print("WTY_AllGene_SGDC_Kfold_AcuuracySummary saved.")
WTOPrediction_AllGene_SGDC_AcuuracySummary.loc['Mean'] = WTOPrediction_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].mean()
WTOPrediction_AllGene_SGDC_AcuuracySummary.loc['STD'] = WTOPrediction_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()
WTOPrediction_AllGene_SGDC_AcuuracySummary.loc['STE'] = WTOPrediction_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()/math.sqrt(KFoldSplit)
WTOPrediction_AllGene_SGDC_AcuuracySummary.to_csv( 'WTOPrediction_AllGene_SGDC_Kfold_AcuuracySummary.csv', index=False)
print("WTOPrediction_AllGene_SGDC_Kfold_AcuuracySummary saved.")
IkBKOPrediction_AllGene_SGDC_AcuuracySummary.loc['Mean'] = IkBKOPrediction_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].mean()
IkBKOPrediction_AllGene_SGDC_AcuuracySummary.loc['STD'] = IkBKOPrediction_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()
IkBKOPrediction_AllGene_SGDC_AcuuracySummary.loc['STE'] = IkBKOPrediction_AllGene_SGDC_AcuuracySummary.iloc[0:KFoldSplit].std()/math.sqrt(KFoldSplit)
IkBKOPrediction_AllGene_SGDC_AcuuracySummary.to_csv( 'IkBKOPrediction_AllGene_SGDC_Kfold_AcuuracySummary.csv', index=False)
print("IkBKOPrediction_AllGene_SGDC_Kfold_AcuuracySummary saved.")
# %%
# Plotting the line graph of alpha vs accuracy in fitted AllGene models and predictions
x = setAlpha
WTYLine = WTY_AllGene_SGDC_AcuuracySummary.loc['Mean']
WTYError = WTY_AllGene_SGDC_AcuuracySummary.loc['STE']
WTYLineU = np.add(WTYLine,WTYError)
WTYLineL = np.subtract(WTYLine,WTYError)
WTOLine = WTOPrediction_AllGene_SGDC_AcuuracySummary.loc['Mean']
WTOError = WTOPrediction_AllGene_SGDC_AcuuracySummary.loc['STE']
WTOLineU = np.add(WTOLine,WTOError)
WTOLineL = np.subtract(WTOLine,WTOError)
IkBKOLine = IkBKOPrediction_AllGene_SGDC_AcuuracySummary.loc['Mean']
IkBKOError = IkBKOPrediction_AllGene_SGDC_AcuuracySummary.loc['STE']
IkBKOLineU = np.add(IkBKOLine,IkBKOError)
IkBKOLineL = np.subtract(IkBKOLine,IkBKOError)

fig = plt.figure()
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['font.size'] = 20
plt.figure(figsize=(8, 12))
plt.xscale('log')
plt.ylim(0.7,1.0)
fig = plt.plot(x, WTYLine, 'k', color='#267F00', linewidth=2.5)
fig = plt.fill_between(x, WTYLineU, WTYLineL, alpha=0.5, edgecolor='#267F00', facecolor='#267F00', antialiased=True)
fig = plt.plot(x, WTOLine, 'k', color='#FF6A00', linewidth=2.5)
fig = plt.fill_between(x, WTOLineU, WTOLineL, alpha=0.5, edgecolor='#FF6A00', facecolor='#FF6A00', antialiased=True)
fig = plt.plot(x, IkBKOLine, 'k', color='#FF0000', linewidth=2.5)
fig = plt.fill_between(x, IkBKOLineU, IkBKOLineL, alpha=0.5, edgecolor='#FF0000', facecolor='#FF0000', antialiased=True)
plt.show()

output = fig.get_figure()
output.savefig("ML_SGDC_AlphaAccuracy_AllGene_Kfoldv1.png", format='png', dpi=300)
# %%
# Loading the trained MarkerGene model and performing prediction on datasets
TargetAlpha = 0.01
Kn = 1
KFoldSplit = 10
kf = KFold(n_splits=KFoldSplit, shuffle=True, random_state=1)
WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary=pd.DataFrame()
WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
for train_index, valid_index in kf.split( WTYFeatures ):
        SectionStart = time.time()
        # Loading trained SGDC model on WTY dataset with indicated Alpha
        WTY_MarkerGene_SGDC_Kfold = load('ML_WTYMarkerGene_SGDC_KFold_A' + str(TargetAlpha) + 'K' + str(Kn) + '_v2.joblib') 
        SectionEnd = time.time()
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        print("Fitted model MarkerGene SGDC A" + str(TargetAlpha) + 'K' + str(Kn) +" loaded")      
        SectionStart = time.time()
        WTY_MarkerGene_SGDC_Kfold_PredictionTest = WTY_MarkerGene_SGDC_Kfold.predict(WTYFeatures_test)
        WTY_MarkerGene_SGDC_Kfold_PredictionTestAddition = pd.DataFrame(data=WTY_MarkerGene_SGDC_Kfold_PredictionTest)
        WTY_MarkerGene_SGDC_Kfold_PredictionTestAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary = pd.concat([WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary,WTY_MarkerGene_SGDC_Kfold_PredictionTestAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_test_MarkerGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTY_MarkerGene_SGDC_Kfold_PredictionOrigin = WTY_MarkerGene_SGDC_Kfold.predict(MarkerGene_WTYFeatures_origin)
        WTY_MarkerGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTY_MarkerGene_SGDC_Kfold_PredictionOrigin)
        WTY_MarkerGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary,WTY_MarkerGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_origin_MarkerGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTO_MarkerGene_SGDC_Kfold_PredictionOrigin = WTY_MarkerGene_SGDC_Kfold.predict(MarkerGene_WTOFeatures_origin)
        WTO_MarkerGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTO_MarkerGene_SGDC_Kfold_PredictionOrigin)
        WTO_MarkerGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary,WTO_MarkerGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("WTOFeature_origin_MarkerGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin = WTY_MarkerGene_SGDC_Kfold.predict(MarkerGene_IkBKOFeatures_origin)
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin)
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary,IkBKO_MarkerGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("IkBKOFeature_origin_MarkerGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        Kn = Kn + 1
# %%
# Calculate frequency of each cluster in model predictions
TargetAlphaColumnList = [col for col in WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary if col.startswith('A'+str(TargetAlpha))]
Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary = WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary.apply(pd.value_counts)*100/len(WTYLabels_test)
Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary = Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary[TargetAlphaColumnList]
Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary.to_csv( 'Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionTest_a001_Summaryv2.csv', index=False)
print("Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary saved.")
Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary.apply(pd.value_counts)*100/len(WTYLabels_origin)
Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary.to_csv( 'Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summaryv2.csv', index=False)
print("Filtered_Frequency_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary saved.")
Frequency_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary.apply(pd.value_counts)*100/len(WTOLabels_origin)
Filtered_Frequency_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = Frequency_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_Frequency_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary.to_csv( 'Filtered_Frequency_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summaryv2.csv', index=False)
print("Filtered_Frequency_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary saved.")
Frequency_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary.apply(pd.value_counts)*100/len(IkBKOLabels_origin)
Filtered_Frequency_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = Frequency_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_Frequency_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary.to_csv( 'Filtered_Frequency_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summaryv2.csv', index=False)
print("Filtered_Frequency_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary saved.")
# %%
# Drawing comparison table for predicted class vs original label in MarkerGene model
Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary = WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary[TargetAlphaColumnList]
Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary['WTYLabels_test'] = WTYLabels_test
Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary['WTYLabels_origin'] = WTYLabels_origin
Filtered_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary['WTOLabels_origin'] = WTOLabels_origin
Filtered_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary = IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary['IkBKOLabels_origin'] = IkBKOLabels_origin
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ColorMap)

# Comparison table for WTY_test dataset prediction on trained MarkerGene model
dfPrediction = Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionTest_Summary
ReferenceLabel = 'WTYLabels_test'
ColumnNames = dfPrediction.columns.to_list()
WTY_MarkerGene_SGDC_Kfold_PredictionTest_MeanTable = pd.DataFrame()
WTY_MarkerGene_SGDC_Kfold_PredictionTest_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in range(0,5):
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class]) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class])])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class]) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class])])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class]) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class])])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class]) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class])])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class]) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == ClassList[Class])])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    WTY_MarkerGene_SGDC_Kfold_PredictionTest_MeanTable['Label'+ str(Class)] = zdata['Mean']
    WTY_MarkerGene_SGDC_Kfold_PredictionTest_STETable['Label'+ str(Class)] = zdata['STE']
WTY_MarkerGene_SGDC_Kfold_PredictionTest_MeanTable.to_csv( 'Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionTest_a001_Summary_MeanTable_v1.csv', index=False)
WTY_MarkerGene_SGDC_Kfold_PredictionTest_STETable.to_csv( 'Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionTest_a001_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionTest_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionTest_a001_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionTest_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionTest_a001_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionTest_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionTest_a001_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionTest_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionTest_a001_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)

# Comparison table for WTY_test dataset prediction on trained MarkerGene model
dfPrediction = Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary
ReferenceLabel = 'WTYLabels_origin'
ColumnNames = dfPrediction.columns.to_list()
WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable = pd.DataFrame()
WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in ClassList:
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable['Label'+ str(Class)] = zdata['Mean']
    WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable['Label'+ str(Class)] = zdata['STE']
WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable.to_csv( 'Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summary_MeanTable_v1.csv', index=False)
WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable.to_csv( 'Filtered_WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)

# Comparison table for WTO_test dataset prediction on trained MarkerGene model
dfPrediction = Filtered_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary
ReferenceLabel = 'WTOLabels_origin'
ColumnNames = dfPrediction.columns.to_list()
WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable = pd.DataFrame()
WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in ClassList:
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable['Label'+ str(Class)] = zdata['Mean']
    WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable['Label'+ str(Class)] = zdata['STE']
WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable.to_csv( 'Filtered_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summary_MeanTable_v1.csv', index=False)
WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable.to_csv( 'Filtered_WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)

# Comparison table for IkBKO_test dataset prediction on trained MarkerGene model
dfPrediction = Filtered_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_Summary
ReferenceLabel = 'IkBKOLabels_origin'
ColumnNames = dfPrediction.columns.to_list()
IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable = pd.DataFrame()
IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in ClassList:
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable['Label'+ str(Class)] = zdata['Mean']
    IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable['Label'+ str(Class)] = zdata['STE']
IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable.to_csv( 'Filtered_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summary_MeanTable_v1.csv', index=False)
IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable.to_csv( 'Filtered_IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("IkBKO_MarkerGene_SGDC_Kfold_PredictionOrigin_a001_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
# %%
# Loading the trained MarkerGene model and performing prediction on datasets
TargetAlpha = 0.03
Kn = 1
KFoldSplit = 10
kf = KFold(n_splits=KFoldSplit, shuffle=True, random_state=1)
WTY_AllGene_SGDC_Kfold_PredictionTest_Summary=pd.DataFrame()
WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary=pd.DataFrame()
for train_index, valid_index in kf.split( WTYFeatures ):
        SectionStart = time.time()
        # Loading trained SGDC model on WTY dataset with indicated Alpha
        WTY_AllGene_SGDC_Kfold = load('ML_WTYAllGene_SGDC_KFold_A' + str(TargetAlpha) + 'K' + str(Kn) + '_v2.joblib') 
        SectionEnd = time.time()
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        print("Fitted model AllGene SGDC A" + str(TargetAlpha) + 'K' + str(Kn) +" loaded")      
        SectionStart = time.time()
        WTY_AllGene_SGDC_Kfold_PredictionTest = WTY_AllGene_SGDC_Kfold.predict(AllGene_WTYFeatures_test)
        WTY_AllGene_SGDC_Kfold_PredictionTestAddition = pd.DataFrame(data=WTY_AllGene_SGDC_Kfold_PredictionTest)
        WTY_AllGene_SGDC_Kfold_PredictionTestAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        WTY_AllGene_SGDC_Kfold_PredictionTest_Summary = pd.concat([WTY_AllGene_SGDC_Kfold_PredictionTest_Summary,WTY_AllGene_SGDC_Kfold_PredictionTestAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_test_AllGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTY_AllGene_SGDC_Kfold_PredictionOrigin = WTY_AllGene_SGDC_Kfold.predict(AllGene_WTYFeatures_origin)
        WTY_AllGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTY_AllGene_SGDC_Kfold_PredictionOrigin)
        WTY_AllGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary,WTY_AllGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("WTYFeature_origin_AllGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        WTO_AllGene_SGDC_Kfold_PredictionOrigin = WTY_AllGene_SGDC_Kfold.predict(AllGene_WTOFeatures_origin)
        WTO_AllGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=WTO_AllGene_SGDC_Kfold_PredictionOrigin)
        WTO_AllGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary,WTO_AllGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("WTOFeature_origin_AllGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        SectionStart = time.time()
        IkBKO_AllGene_SGDC_Kfold_PredictionOrigin = WTY_AllGene_SGDC_Kfold.predict(AllGene_IkBKOFeatures_origin)
        IkBKO_AllGene_SGDC_Kfold_PredictionOriginAddition = pd.DataFrame(data=IkBKO_AllGene_SGDC_Kfold_PredictionOrigin)
        IkBKO_AllGene_SGDC_Kfold_PredictionOriginAddition.columns = ['A' + str(TargetAlpha) + 'K' + str(Kn)]
        IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = pd.concat([IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary,IkBKO_AllGene_SGDC_Kfold_PredictionOriginAddition], axis=1)
        SectionEnd = time.time()
        print("IkBKOFeature_origin_AllGene SGDC A" + str(TargetAlpha) + "K" + str(Kn) + " predicted.")
        print('Section Running Time:',SectionEnd - SectionStart, 'seconds.')
        Kn = Kn + 1
# %%
# Calculate frequency of each cluster in AllGene model predictions
TargetAlphaColumnList = [col for col in WTY_AllGene_SGDC_Kfold_PredictionTest_Summary if col.startswith('A'+str(TargetAlpha))]

Frequency_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary = WTY_AllGene_SGDC_Kfold_PredictionTest_Summary.apply(pd.value_counts)*100/len(WTYLabels_test)
Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary = Frequency_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary[TargetAlphaColumnList]
Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary.to_csv( 'Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionTest_Summaryv2.csv', index=False)
print("Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary saved.")
Frequency_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary = WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary.apply(pd.value_counts)*100/len(WTYLabels_origin)
Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary = Frequency_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary.to_csv( 'Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summaryv2.csv', index=False)
print("Filtered_Frequency_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary saved.")
Frequency_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary.apply(pd.value_counts)*100/len(WTOLabels_origin)
Filtered_Frequency_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = Frequency_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_Frequency_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary.to_csv( 'Filtered_Frequency_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summaryv2.csv', index=False)
print("Filtered_Frequency_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary saved.")
Frequency_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary.apply(pd.value_counts)*100/len(IkBKOLabels_origin)
Filtered_Frequency_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = Frequency_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_Frequency_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary.to_csv( 'Filtered_Frequency_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summaryv2.csv', index=False)
print("Filtered_Frequency_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary saved.")
# %%
# Drawing comparison table for predicted class vs original label in AllGene model
Filtered_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary = WTY_AllGene_SGDC_Kfold_PredictionTest_Summary[TargetAlphaColumnList]
Filtered_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary['WTYLabels_test'] = WTYLabels_test
Filtered_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary = WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary['WTYLabels_origin'] = WTYLabels_origin
Filtered_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary['WTOLabels_origin'] = WTOLabels_origin
Filtered_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary = IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary[TargetAlphaColumnList]
Filtered_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary['IkBKOLabels_origin'] = IkBKOLabels_origin
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ColorMap)

# Comparison table for WTY_test dataset prediction on trained AllGene model
dfPrediction = Filtered_WTY_AllGene_SGDC_Kfold_PredictionTest_Summary
ReferenceLabel = 'WTYLabels_test'
ColumnNames = dfPrediction.columns.to_list()
WTY_AllGene_SGDC_Kfold_PredictionTest_MeanTable = pd.DataFrame()
WTY_AllGene_SGDC_Kfold_PredictionTest_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in ClassList:
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    WTY_AllGene_SGDC_Kfold_PredictionTest_MeanTable['Label'+ str(Class)] = zdata['Mean']
    WTY_AllGene_SGDC_Kfold_PredictionTest_STETable['Label'+ str(Class)] = zdata['STE']
WTY_AllGene_SGDC_Kfold_PredictionTest_MeanTable.to_csv( 'Filtered_WTY_AllGene_SGDC_Kfold_PredictionTest_a003_Summary_MeanTable_v1.csv', index=False)
WTY_AllGene_SGDC_Kfold_PredictionTest_STETable.to_csv( 'Filtered_WTY_AllGene_SGDC_Kfold_PredictionTest_a003_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionTest_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionTest_a003_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionTest_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionTest_a003_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionTest_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionTest_a003_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionTest_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionTest_a003_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)

# Comparison table for WTY_test dataset prediction on trained AllGene model
dfPrediction = Filtered_WTY_AllGene_SGDC_Kfold_PredictionOrigin_Summary
ReferenceLabel = 'WTYLabels_origin'
ColumnNames = dfPrediction.columns.to_list()
WTY_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable = pd.DataFrame()
WTY_AllGene_SGDC_Kfold_PredictionOrigin_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in ClassList:
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    WTY_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable['Label'+ str(Class)] = zdata['Mean']
    WTY_AllGene_SGDC_Kfold_PredictionOrigin_STETable['Label'+ str(Class)] = zdata['STE']
WTY_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable.to_csv( 'Filtered_WTY_AllGene_SGDC_Kfold_PredictionOrigin_a003_Summary_MeanTable_v1.csv', index=False)
WTY_AllGene_SGDC_Kfold_PredictionOrigin_STETable.to_csv( 'Filtered_WTY_AllGene_SGDC_Kfold_PredictionOrigin_a003_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionOrigin_a003_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionOrigin_a003_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionOrigin_a003_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTY_AllGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTY_AllGene_SGDC_Kfold_PredictionOrigin_a003_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)

# Comparison table for WTO_test dataset prediction on trained AllGene model
dfPrediction = Filtered_WTO_AllGene_SGDC_Kfold_PredictionOrigin_Summary
ReferenceLabel = 'WTOLabels_origin'
ColumnNames = dfPrediction.columns.to_list()
WTO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable = pd.DataFrame()
WTO_AllGene_SGDC_Kfold_PredictionOrigin_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in ClassList:
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    WTO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable['Label'+ str(Class)] = zdata['Mean']
    WTO_AllGene_SGDC_Kfold_PredictionOrigin_STETable['Label'+ str(Class)] = zdata['STE']
WTO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable.to_csv( 'Filtered_WTO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Summary_MeanTable_v1.csv', index=False)
WTO_AllGene_SGDC_Kfold_PredictionOrigin_STETable.to_csv( 'Filtered_WTO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_AllGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("WTO_AllGene_SGDC_Kfold_PredictionOrigin_a003_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(WTO_AllGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("WTO_AllGene_SGDC_Kfold_PredictionOrigin_a003_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)

# Comparison table for IkBKO_test dataset prediction on trained AllGene model
dfPrediction = Filtered_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_Summary
ReferenceLabel = 'IkBKOLabels_origin'
ColumnNames = dfPrediction.columns.to_list()
IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable = pd.DataFrame()
IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_STETable = pd.DataFrame()
ClassList = [1,2,3,4,5]
for Class in ClassList:
    zdata = pd.DataFrame()
    for Col in range(0,10):
        a = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 1)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        b = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 2)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        c = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 3)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        d = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 4)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        e = len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class) & (dfPrediction[ColumnNames[Col]] == 5)])*100/len(dfPrediction[ColumnNames[Col]].loc[(dfPrediction[ReferenceLabel] == Class)])
        data1 = [a,b,c,d,e]
        zdata = pd.concat([zdata,(pd.DataFrame(data1))], axis=1)
        zdata['Mean'] = zdata.iloc[:,0:KFoldSplit].mean(axis=1)
        zdata['STE'] = zdata.iloc[:,0:KFoldSplit].std(axis=1)/math.sqrt(KFoldSplit)
    IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable['Label'+ str(Class)] = zdata['Mean']
    IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_STETable['Label'+ str(Class)] = zdata['STE']
IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable.to_csv( 'Filtered_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Summary_MeanTable_v1.csv', index=False)
IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_STETable.to_csv( 'Filtered_IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Summary_STETable_v1.csv', index=False)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Meantable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_MeanTable, norm=divnorm, vmax=90, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_a003_Meantable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, annot=True)
output = fig.get_figure()
output.savefig("IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_a003_STEtable.png", format='png', dpi=300)
plt.close(output)
plt.figure(figsize=(10,8))
fig = sns.heatmap(IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_STETable, vmax=10, vmin=0, cmap=cmap, cbar=False, linewidths=2, linecolor = 'white', xticklabels=False, yticklabels=False, annot=False)
output = fig.get_figure()
output.savefig("IkBKO_AllGene_SGDC_Kfold_PredictionOrigin_a003_STEtable_NoAnnot.png", format='png', dpi=300)
plt.close(output)
# %%
end = time.time()
print('Total Running Time:',end - start, 'seconds.')
