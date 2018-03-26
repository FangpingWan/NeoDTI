# NeoDTI
NeoDTI: Neural integration of neighbor information from a heterogeneous network for discovering new drug-target interactions [(preprint)](https://www.biorxiv.org/content/early/2018/02/07/261396).

# Quick start
To reproduce our results:
1. Unzip data.zip in ./data.
2. Run <code>NeoDTI_cv.py</code> to reproduce the cross validation results of NeoDTI. Options are:  
`-d:The embedding dimension d, default: 1024"` 
`-d:The embedding dimension d, default: 1024"` 

3. Run <code>NeoDTI_cv_with_aff.py</code> to reproduce the cross validation results of NeoDTI with additional compound-protein binding affinity data.

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
* mat_drug_protein.txt : Drug_Protein interaction matrix.
* Similarity_Matrix_Drugs.txt : Drug & compound similarity scores based on chemical structures of drugs (\[0,708) are drugs, the rest are compounds).
* Similarity_Matrix_Proteins.txt : Protein similarity scores based on primary sequences of proteins.
* mat_drug_protein_homo_protein_drug.txt: Drug_Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper).
* mat_drug_protein_drug.txt: Drug_Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_sideeffect.txt: Drug_Protein interaction matrix, in which DTIs with drugs sharing similar side effects (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_disease.txt: Drug_Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_compound_protein_bindingaffinity.txt: Compound-Protein binding affinity matrix (measured by negative logarithm of _Ki_).

All entities (i.e., drugs, compounds, proteins, diseases and side-effects) are organized in the same order across all files.


# Contacts
If you have any questions or comments, please feel free to email Fangping Wan (wfp15[at]mails[dot]tsinghua[dot]edu[dot]cn) and/or Jianyang Zeng (zengjy321[at]tsinghua[dot]edu[dot]cn).

