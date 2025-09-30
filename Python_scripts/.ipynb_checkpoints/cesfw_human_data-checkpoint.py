###Libraries
import numpy as np
import pandas as pd
import cESFW
import os
import anndata

###Base path to folder
## Set path for retreiving and depositing data
base_path = "/camp/home/depottj/working/depottj/projects/briscoej/giulia.boezio/LARRY_spinal_cord/data"

#Get file name
file_name = "preprocessing_larry_human.h5ad"

###Get folder name
folder_name = file_name[:-5]

#Adata file path
file_path = os.path.join(base_path , "anndata", file_name)

#Read in adata
adata = anndata.read_h5ad(file_path)

#Meta data
metadata = adata.obs

#Get the counts data
adata.X = adata.layers["counts"].copy()
adata_counts = adata.to_df()

#Genes need to be expressed in at least 20 cells
keep_genes = adata_counts.columns[np.where(np.sum(adata_counts,axis=0) > 20)[0]]

#New adata_counts based on the selected genes
adata_counts = adata_counts[keep_genes]

#Only cells are kept that express between a 3000 and 6000 genes
cell_genes = np.sum(adata_counts > 0,axis=1)
keep_cells = np.where((cell_genes > 3000) & (cell_genes < 6000))[0]
adata_counts = adata_counts.iloc[keep_cells]
metadata = metadata.iloc[keep_cells]

###Feature normalisation

## Prior to using cESFW, data must be scaled/normalised such that every feature only has values between 0 and 1.
# How this is done is ultimitely up to the user. However, for scRNA-seq data, we tend to find that the following relitively simple
# normalisation approach yeilds good results.

## Note, cESFW takes each row to be a sample and each column to be a feature. Hence, in this example, each row of Human_Embryo_Counts
# is a cell and each colum is gene.

## Optional: Log transform the data. Emperically we find that in most cases, log transformation of the data
# appears to lead to poorer results further downstream in most cases. However, in some datasets
# we have worked with, it has lead to improved results. This is obviously dependent on what downstream analysis
# the user chooses to do and how they do it, but we recommend starting without any log transformation (hence the
# next line of code being commented out).
#I did this because clones are combinations of single cells. This can lead to distributions with long tail

scaled_matrix = np.log2(adata_counts.copy()+1)

###Clipping and scaling
## Clip the top 2.5 percent of observed values for each gene to mitigate the effect of unusually high
# counts observations.

#This has been an updated clipping version. When upper equals 0, then the maximum is takedn anyway instead of 97.5 percentage
Upper = np.percentile(scaled_matrix,97.5,axis=0)
Upper[np.where(Upper == 0)[0]] = np.max(scaled_matrix,axis=0)[np.where(Upper == 0)[0]]
scaled_matrix = scaled_matrix.clip(upper=Upper,axis=1)

## Normalise each feature/gene of the clipped matrix.
normalisation_values = np.max(scaled_matrix,axis=0)
scaled_matrix = scaled_matrix / normalisation_values

###Run cESFW
## Given the scaled matrix, cESFW will use the following function to extract all the non-zero values into a single vector. We do this
# because ES calculations can completely ignore 0 values in the data. For sparse data like scRNA-seq data, this dramatically reduces the memory
# required, and the number of calculations that need to be carried out. For relitively dense data, this step will still need to be carried
# out to use cESFW, but will provide little benifit computationally.

## path: A string path pre-designated folder to deposit the computationally efficient objects. E.g. "/mnt/c/Users/arthu/Test_Folder/"
## scaled_matrix: The high dimensional DataFrame whose features have been scaled to values between 0 and 1. Format must be a Pandas DataFrame.
## Min_Minority_State_Cardinality: The minimum value of the total minority state mass that a feature contains before it will be automatically
# removed from the data, and hence analysis.
folder_path = os.path.join(base_path , "cesfw" , folder_name)
if not os.path.exists(folder_path):
    os.mkdir(folder_path) 
cESFW.Create_ESFW_Objects((os.path.join(base_path, "cesfw" , folder_name) + "/"), scaled_matrix, Min_Minority_State_Cardinality = 20)

## Now that we have the compute efficient object, we can calculate the ESSs and EPs matricies. The ESSs matrix provides the pairwise 
# Entropy Sort Scores for each gene in the data. THe EPs matrix provides the EPs pairwise for each gene.
ESSs, EPs = cESFW.Parallel_Calculate_ESS_EPs(os.path.join(base_path , "cesfw" ,folder_name) + "/" , Use_Cores=64)

## Un-comment below lines to save results for future useage.
np.save(os.path.join(base_path , "cesfw" ,folder_name ,"ESSs.npy"),ESSs)
np.save(os.path.join(base_path, "cesfw", folder_name,"EPs.npy"),EPs)