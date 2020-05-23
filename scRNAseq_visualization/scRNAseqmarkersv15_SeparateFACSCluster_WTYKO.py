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
import seaborn as sns
import time
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
CellClusterToShow = ['WTY_FrA_ClusterA', 'WTY_FrB_ClusterB', 'WTY_FrC_ClusterC', 'WTY_FrCp_ClusterCp', 'WTY_FrD_ClusterD', 'IkBKO_FrA_ClusterA', 'IkBKO_FrB_ClusterB', 'IkBKO_FrB_ClusterC', 'IkBKO_FrB_ClusterCp', 'IkBKO_FrCCp_ClusterC', 'IkBKO_FrCCp_ClusterCp', 'IkBKO_FrD_ClusterD']
# Enter the pseudocount value for the calculation of Z-score
PseudoCount = 0.0001
# Enter the length and width of the heatmap
HM_Length = 120
HM_Width = 60
# Enter the color for the minimum, middle, and maximum based on the https://matplotlib.org/examples/color/named_colors.html
ColorMap = ["dodgerblue","black","yellow"]
# Enter top and bottom boundary of the colormap scales for the heatmap
Hm_Max = '25%'
Hm_Min = '75%'

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
#df3=pd.read_csv('DBT22PCA10_Barcodes_mapped_withdoublet.csv', sep=',',header=0)
#df101=pd.read_csv( Samplename + '_MarkerGeneQuery_output.csv', sep=',',header=0, usecols = ['GeneID'])
#df100=pd.read_csv( Samplename + '_MarkerGeneQuery_output.csv', sep=',',header=0, usecols = ['Markers'])

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
    
    
FullCellList = []
for counter in range(0,len(CellClusterList)):
    List1 = globals()[CellClusterList[counter]]
    FullCellList = FullCellList + List1

CellListShown = []
for counter in range(0,len(CellClusterToShow)):
    List2 = globals()[CellClusterToShow[counter]]
    CellListShown = CellListShown + List2
ExclusionList = [i for i in FullCellList if i not in CellListShown] 

print('Barcode-Cluster information Loading Completed')

# %%
print('Exclusion of Unwanted Clusters Started')
# Include all the unwanted clusters that need to be removed
df6 = df5.copy()
df6 = df6.drop(columns = ExclusionList)

print('Exclusion of Unwanted Clusters Completed')
# %%
print('Marker Gene Extraction Started')
GeneList = df101.loc[:,'GeneID'].tolist()
df7=df6.loc[GeneList]
df8 = pd.merge(df101,df7, how='left', on='GeneID')
print('Marker Gene Extraction Completed')
df9 = df8.drop('GeneID', axis = 1)
print('Alignment of Cell-ID to Clusters Started')
df10 = df9[CellListShown]
print('Alignment of Cell-ID to Clusters Completed')

# %%
print('Overall Z-score calculation Started')
df10 = df10.fillna(0)
df10 +=PseudoCount
df10=df10.astype(np.float64)
df11 = np.log2(df10)
df12 = df11.copy()
df13=df12.transpose()
df14 = (df13 - df13.mean())/df13.std(ddof=0)
print('Z-score Caculation Completed.')
df16 = df14.transpose()
# Enable the next two lines if export of the Z-score matrix for all clusters is desired
#df16.to_csv( Samplename + 'Zscore_AllClusterMerged.csv', index=False)
#print('Overall Z-score matrix Saved')
# %%
print('Overall Heatmap Plotting Started')
hmmax = (df16.max(axis=1)).describe()
hmmaxvalue = (df16.max(axis=1)).describe()[[Hm_Max]]
hmmin = (df16.min(axis=1)).describe()
hmminvalue = (df16.min(axis=1)).describe()[[Hm_Min]]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ColorMap)
plt.figure(figsize=(HM_Length*(len(CellListShown)/len(FullCellList)), HM_Width*(len(Marker)/len(GeneList))))
ax = sns.heatmap(df16, vmax=hmmaxvalue[0] , vmin=hmminvalue[0], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
output = ax.get_figure()
output.savefig("scRNAseq1HM_" + Samplename + "_All.png", format='png', dpi=300)
plt.close(output)
print('Overall Heatmap Plotting Completed and Saved')
# %%
print('Z-score matrix Sorting and Plotting Started')
counter = 0
for counter in range(0,len(CellClusterToShow)):
    List3 = globals()[CellClusterToShow[counter]]
    HM1 = df16.loc[:, List3]
# Enable the next line if exporting separate Z-score files is desired
#    HM1.to_csv( Samplename +'Zscore_' + CellClusterToShow[counter] + '.csv', index=False)
    plt.figure(figsize=(HM_Length*(len(List3)/len(CellListShown)), HM_Width*(len(Marker)/len(GeneList))))
    ax = sns.heatmap(HM1, vmax=hmmaxvalue[0] , vmin=hmminvalue[0], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    output = ax.get_figure()
    output.savefig( Samplename + "_" + CellClusterToShow[counter] + ".png", format='png', dpi=300)
    plt.close(output)
    print(CellClusterToShow[counter] + ': Heatmap Plotted and Saved')

else:
    print('Z-score matrix Sorted.')
    print('Separate Heatmap Plotting Completed and Saved')
# %%
print('Colormap Maximum:  ' + str(hmmaxvalue[0]))
print('Colormap Minimum:  ' + str(hmminvalue[0]))
end = time.time()
print('Total Running Time:',end - start, 'seconds.')
