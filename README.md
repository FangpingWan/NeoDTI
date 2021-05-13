# NeoDTI
NeoDTI: Neural integration of neighbor information from a heterogeneous network for discovering new drug-target interactions [(Bioinformatics)](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty543/5047760).

# Recent Update 09/06/2018
L2 regularization is added.

# Requirements
* Tensorflow (tested on version 1.0.1 and version 1.2.0)
* tflearn
* numpy (tested on version 1.13.3 and version 1.14.0)
* sklearn (tested on version 0.18.1 and version 0.19.0)

# Quick start
To reproduce our results:
1. Unzip data.zip in ./data.
2. Run <code>NeoDTI_cv.py</code> to reproduce the cross validation results of NeoDTI. Options are:  
`-d: The embedding dimension d, default: 1024.`  
`-n: Global norm to be clipped, default: 1.`  
`-k: The dimension of project matrices, default: 512.`  
`-r: Positive and negative. Two choices: ten and all, the former one sets the positive:negative = 1:10, the latter one considers all unknown DTIs as negative examples. Default: ten.`  
`-t: Test scenario. The DTI matrix to be tested. Choices are: o, mat_drug_protein.txt will be tested; homo, mat_drug_protein_homo_protein_drug.txt will be tested; drug, mat_drug_protein_drug.txt will be tested; disease, mat_drug_protein_disease.txt will be tested; sideeffect, mat_drug_protein_sideeffect.txt will be tested; unique, mat_drug_protein_drug_unique.txt will be tested. Default: o.`
3. Run <code>NeoDTI_cv_with_aff.py</code> to reproduce the cross validation results of NeoDTI with additional compound-protein binding affinity data. Options are:  
`-d: The embedding dimension d, default: 1024.`  
`-n: Global norm to be clipped, default: 1.`  
`-k: The dimension of project matrices, default: 512.`  

# Data description
* drug.txt: list of drug names.
* protein.txt: list of protein names.
* disease.txt: list of disease names.
* se.txt: list of side effect names.
* drug_dict_map: a complete ID mapping between drug names and DrugBank ID.
* protein_dict_map: a complete ID mapping between protein names and UniProt ID.
* mat_drug_se.txt : Drug-SideEffect association matrix.
* mat_protein_protein.txt : Protein-Protein interaction matrix.
* mat_drug_drug.txt : Drug-Drug interaction matrix.
* mat_protein_disease.txt : Protein-Disease association matrix.
* mat_drug_disease.txt : Drug-Disease association matrix.
* mat_protein_drug.txt : Protein-Drug interaction matrix.
* mat_drug_protein.txt : Drug-Protein interaction matrix.
* Similarity_Matrix_Drugs.txt : Drug & compound similarity scores based on chemical structures of drugs (\[0,708) are drugs, the rest are compounds).
* Similarity_Matrix_Proteins.txt : Protein similarity scores based on primary sequences of proteins.
* mat_drug_protein_homo_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper).
* mat_drug_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_sideeffect.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar side effects (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_disease.txt: Drug-Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_unique: Drug-Protein interaction matrix, in which known unique and non-unique DTIs were labelled as 3 and 1, respectively, the corresponding unknown ones were labelled as 2 and 0 (see the paper for the definition of unique). 
* mat_compound_protein_bindingaffinity.txt: Compound-Protein binding affinity matrix (measured by negative logarithm of _Ki_).

All entities (i.e., drugs, compounds, proteins, diseases and side-effects) are organized in the same order across all files. These files: drug.txt, protein.txt, disease.txt, se.txt, drug_dict_map, protein_dict_map, mat_drug_se.txt, mat_protein_protein.txt, mat_drug_drug.txt, mat_protein_disease.txt, mat_drug_disease.txt, mat_protein_drug.txt, mat_drug_protein.txt, Similarity_Matrix_Proteins.txt, are extracted from https://github.com/luoyunan/DTINet.



# Contacts
If you have any questions or comments, please feel free to email Fangping Wan (wfp15[at]tsinghua[dot]org[dot]cn) and/or Jianyang Zeng (zengjy321[at]tsinghua[dot]edu[dot]cn).

