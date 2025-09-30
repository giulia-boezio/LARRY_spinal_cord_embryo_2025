import numpy as np
import itertools
import math
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import random
import copy

#There are two interection types: between and between_within
def freq_to_link_table(freq_table , interaction_type = "between" , cell_combo_method = "sum"):
    '''
    Converts cell type frequency table into a pairwise connection frequency table
    
        Parameters:
            freq_table (pandas.core.frame.DataFrame): DataFrame with cell types being the columns and clones the rows. The values indicate the amount of cells.
            interaction_type (str): two options (1) between: excludes interactions between cells of the same type
                                                (2) between_within: includes interactions between cells of the same type
            cell_combo_method (str): four options   (1) combination: The weight of the connection between two different cell types within a clones is the product of their amount.
                                                    (2) min_link: The weight of the connection between two different cell types within a clones is the minimum of their amount.
                                                    (3) sum: The weight of the connection between two different cell types within a clones is the sum of their amount, both amounts should be bigger than 0.
                                                    (4) connection: The weight of the connection between two different cell types within a clones is 1/0, based on the presence/absence of their co-occurrence. 
            
        Returns:
            cell_combo_table (pandas.core.frame.DataFrame): DataFrame with cell type pairs being the columns and clones the rows. The values are calculated with the cell_combo_method. We call this interaction table.
    
    '''
    
    
    #Only clones with more than 1 cell
    freq_table_filter = freq_table[np.sum(freq_table, axis = 1) > 1]
    
    #Only select clones that connect two different cell types if interaction_type == "between"
    if interaction_type == "between":
        freq_table_filter = freq_table_filter[np.sum(freq_table_filter > 0 , axis = 1) > 1]        
        
    #Get all the cell type combinations
    combination_function = itertools.combinations if interaction_type == "between" else itertools.combinations_with_replacement
    
    #Join the cell type combinations with "|"
    cell_type_combos = ["|".join(el) for el in combination_function(freq_table_filter.columns, 2)]
    
    #Initialise empty numpy array.
    cell_combo_table = np.empty((0, len(cell_type_combos)))
    
    triu_k = {"between" : 1, "between_within" : 0}

    for clone in freq_table_filter.index:
        #Cell type frequency of a specific clone
        clone_freq = np.array(freq_table_filter.loc[clone]).reshape(len(freq_table_filter.columns),1)

        #Code for combination method
        if cell_combo_method == "combination":
            #Take product of frequency tables to get the combination table.
            clone_product = clone_freq * clone_freq.T

            #Correct the diagonal, because combinations with self are not product but combinations of 2
            comb_values = [math.comb(int(value), 2) for value in clone_freq]
            np.fill_diagonal(clone_product , comb_values)

        #Code for min_link method
        elif cell_combo_method == "min_link":
            
            # Generate all combinations using meshgrid
            X, Y = np.meshgrid(freq_table_filter.loc[clone], freq_table_filter.loc[clone])

            # Apply np.min() to find the minimum between the combinations
            clone_product = np.minimum(X, Y)
        
        #Code for the sum method
        elif cell_combo_method == "sum":
            
            #Sum over the different cell combinations
            clone_product = clone_freq + clone_freq.T
            
            #Interactions with self we do sum - 1
            comb_values = [(2*clone_freq) - 1 for value in clone_freq]
            np.fill_diagonal(clone_product , comb_values)
            
            #Construct the binary combination matrix. When there is a 1: both cell types are present, 0 one of the two cell types are not present.
            clone_freq_bin = (clone_freq > 0).astype(int)
            clone_product_bin = clone_freq_bin * clone_freq_bin.T
            
            #Take product. So only sum values remain for cell type combinations with a 1 (both cell types are present).
            clone_product = clone_product * clone_product_bin

        #Code for connection method
        elif cell_combo_method == "connection":
            #Convert the frequency table into a binary table and take product the get combination table.
            #Here you have a binary question: is there a cell connection or not.
            clone_freq_bin = (clone_freq > 0).astype(int)
            clone_product = clone_freq_bin * clone_freq_bin.T

            #Correct the diagonal, because cell type with only one cell in the clone has no connection with self.
            connection_self_bin = (clone_freq > 1).astype(int)
            np.fill_diagonal(clone_product , connection_self_bin)
        
        
        #Select the upper triangle
        tri_upper_indices = np.triu_indices(len(clone_product), k=triu_k[interaction_type])

        # Extract the elements using these indices
        flattened_upper_with_diag = clone_product[tri_upper_indices]

        cell_combo_table = np.concatenate((cell_combo_table, flattened_upper_with_diag.reshape(1, -1)), axis=0)
        
    #Convert interaction table to pandas
    cell_combo_table = pd.DataFrame(cell_combo_table)
    cell_combo_table.columns = cell_type_combos
    cell_combo_table.index = freq_table_filter.index
    
    return cell_combo_table


def calculate_statistic(cell_combo_table , weighting_factor = 1 , scale_factor = 1e4):
    '''
    Calculates the statistic for cell type pairs using the pairwise connection table (output of freq_to_link_table)
    
        Parameters:
            cell_combo_table (pandas.core.frame.DataFrame): DataFrame with cell type pairs being the columns and clones the rows.
            weighting_factor (int or list int): (1) int = all clones get same weight
                                                (2) list int = list size equals number of clones enabling differentiation in weight attributed to every clone
            scale_factor (int): sets the scaling factor
            
        Returns:
            statistic (dict): keys are cell type pairs and values are the calculated statistic
    '''
    
    if isinstance(weighting_factor, int):
        weighting_factor = [weighting_factor] * cell_combo_table.shape[0]

    #Normalise data
    cell_combo_table_norm = cell_combo_table.div(cell_combo_table.sum(axis=1), axis=0)
    
    #Weight the clones
    cell_combo_table_weight = np.multiply(cell_combo_table_norm, np.array(weighting_factor)[:, np.newaxis])
    
    #Take to sum and normalise for the number of clones to have the test statistic
    statistic_dataframe = (cell_combo_table_weight.sum(axis = 0)/cell_combo_table_weight.shape[0]) * scale_factor
    
    #Convert dataframe to dictionary
    statistic = {statistic_dataframe.index[connection] : statistic for connection , statistic in enumerate(statistic_dataframe)}
    
    return statistic


def perform_permutation(permutation_number , cell_type_list_test, freq_table , clone_sizes , interaction_type, cell_combo_method):
    '''
    Runs one permutation 
    
        Parameters:
            permutation_number (int): number of permutation
            cell_type_list_test (list str): List of all the cells named as their cell type.
            freq_table (pandas.core.frame.DataFrame): DataFrame with cell types being the columns and clones the rows. The values indicate the amount of cells.
            clone_sizes (pandas.core.series.Series): Series of clone sizes
            interaction_type (str): two options (1) between: excludes interactions between cells of the same type
                                                (2) between_within: includes interactions between cells of the same type
            cell_combo_method (str): four options   (1) combination: The weight of the connection between two different cell types within a clones is the product of their amount.
                                                    (2) min_link: The weight of the connection between two different cell types within a clones is the minimum of their amount.
                                                    (3) sum: The weight of the connection between two different cell types within a clones is the sum of their amount, both amounts should be bigger than 0.
                                                    (4) connection: The weight of the connection between two different cell types within a clones is 1/0, based on the presence/absence of their co-occurrence. 
            
            
        Returns:
            permutation_statistic (dict): keys are cell type pairs and values are the calculated statistic
    '''

    cell_type_list = copy.deepcopy(cell_type_list_test)
    
    #Randomly shuffle cell_type_list
    random.shuffle(cell_type_list)
    
    #Initialise empty frequency table
    permutation_freq_table = pd.DataFrame(columns=freq_table.columns)
    
    #Populate the clones with the shuffled cells
    for clone_size in clone_sizes:
        
        clone_list = [cell_type_list.pop() for _ in range(clone_size)]
        
        clone_dict = {el : clone_list.count(el)  for el in freq_table.columns}
        
        permutation_freq_table = pd.concat([permutation_freq_table, pd.DataFrame([clone_dict])], ignore_index=True)

    #Obtain interaction table
    cell_combo_table = freq_to_link_table(permutation_freq_table , interaction_type = interaction_type, cell_combo_method = cell_combo_method)
    
    #Calculate statistic
    permutation_statistic = calculate_statistic(cell_combo_table)
    
    return permutation_statistic

def run_permutation_parallel(freq_table , interaction_type = "between" , n_jobs = 32 , number_permutations = 1000 , cell_combo_method = "sum"):
    '''
    Runs permutation to calculate the statistic of the random co-occurence of cells in clones. Clone sizes, amount of clones and number cells per cell type are kept constant during the permutations and are the same as in the observed dataset (freq_table). 
    
        Parameters:
            freq_table (pandas.core.frame.DataFrame): DataFrame with cell types being the columns and clones the rows. The values indicate the amount of cells.
            interaction_type (str): two options (1) between: excludes interactions between cells of the same type
                                                (2) between_within: includes interactions between cells of the same type
            interaction_type (int): numbers of jobs that are done in parallel
            number_permutations (int): number of permutations that are performed
            
        Returns:
            permutation_results (list dict): list of dictionaries with keys being cell type pairs and values being the calculated statistics
    '''
    
    #Frequency table of all the cell types
    cell_frequency_table = pd.melt(freq_table.reset_index() , id_vars=["clone"], var_name='Cell_type', value_name='amount').groupby("Cell_type").sum("amount").reset_index()
    
    #List of all the cells named as their cell type
    cell_type_list = [cell_type for cell_type, amount in zip(cell_frequency_table["Cell_type"] , cell_frequency_table["amount"]) for _ in range(amount)]
    
    #Get the sizes of the different clones
    clone_sizes = freq_table.sum(axis = 1)
    
    #Run the permutation tests in parallel.
    permutation_results = Parallel(n_jobs=n_jobs)(
    delayed(perform_permutation)(permutation, cell_type_list, freq_table, clone_sizes , interaction_type = interaction_type, cell_combo_method = cell_combo_method)
    for permutation in tqdm(range(number_permutations), desc="Processing elements")
    )
    
    return permutation_results 
