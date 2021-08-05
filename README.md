## VS-SVM

**VS-SVM** is a MATLAB script for ligand-based **V**irtual **S**creening using support vector machine (**SVM**) learning models. We designed this model for the purpose of predicting functional (chemical-genetic) similarities of compounds from their chemical structures. Although, we used this model to predict the pairwise functional similarities of the compounds in our RIKEN and NCI/NIH/GSK high-confidence sets [(Safizadeh et al. 2021)](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00993), the script is scalable to any compound collection for which chemical-genetic interaction profiles exist.

> Features

- The model uses pairwise structural similarities of compounds for prediction of their pairwise functional similarities.
- The model extracts supervised principal components (S-PCs) from chemical-genetic interaction profiles (for the training data only) to be used as learning labels.
-	The model uses a number of bootstraps (200 bootstraps for our RIKEN high-confidence set) to predict pairwise functional similarities. Each bootstrap assigns ~0.632N unique compounds to the training set and the rest to the test set using uniformly random drawing with replacement, where N represents the total number of compounds in the collection.
-	The model predicts the functional similarity of a pair of compounds by averaging the predictions over all bootstraps where the two compounds are both in the test set.
-	The SVM model parameters in the script are optimized for our RIKEN high-confidence set. The model parameters need to be tuned/optimized for other compound collections.
-	The script is developed in the MATLAB environment with no other dependencies.

> Execution details

*/VS-SVM/CG_Gold/ASP-8*<br/>
This directory includes five matfiles as follows:

-	`Artemisinins.mat:` contains a group of 20 compounds from the RIKEN NPDepo collection with very similar functional profiles. We removed these compounds from our data to prevent any bias in our analysis.
-	`Profs_New.mat:` contains the chemical-genetic interaction data for our compounds in the RIKEN high-confidence set and Artemisinins (846 compounds). 
-	`Frags_Ord.mat:` contains the ASP fingerprints (depth 8) for our compounds in the RIKEN high-confidence set and Artemisinins ordered as they are listed in the Profs_New matfile. Frags_Ord_Spr matfile is the sparse form of the Frags_Ord matfile, where the many zeros of the ASP fingerprints are removed for the purpose of storage efficiency.
-	`Cpds_Prf_New.mat:` contains the chemical names of our compounds in the same order that their functional and structural data are stored in the Prof_New and Frags_Ord matfiles respectively.
-	`Cpds_Frg_Ord.mat:` contains the same data as the Cpds_Prf_New matfile.

*/VS-SVM/Learning_Model*<br/>
This directory includes four matfiles and a log.txt file to store the experimental results of 200 bootstraps for our RIKEN high-confidence set. Please note that our machine learning model creates a single subdirectory for every bootstrap; however, we have not included those 200 bootstrap subdirectories for the sake of storage efficiency.

-	`Cpds_Ref_Bts.mat:` indicates whether a compound is in the training set or the test set in a bootstrap.
-	`Cpds_Trn_Tst.mat:` indicates whether a compound is in the training set or the test set in a bootstrap. This matfile also includes the number of S-PCs that are required to explain 95% of the variance in chemical-genetic interaction data (for the training set only).
-	`Prd_Bts.mat:` contains the predictions for the pairwise functional similarities of our compounds in the test set for each bootstrap. Both compounds have to belong to the test set in order to have a valid prediction; otherwise, the invalid value of -10 is considered for all the compound pairs where at least one compound belongs to the training set. The compound pair indices are generated by the nchoosek MATLAB command.
-	`Prd_Avg.mat:` contains the final prediction of functional similarity for every compound pair by averaging all the bootstrap prediction values where the test prediction values are valid.
-	`log.txt:` contains the step-by-step execution details of every bootstrap for our RIKEN high-confidence set.  

> Prediction for a new/query compound

The new compound should be added to the test set in all existing bootstraps. Our machine learning model will compute the functional similarity of the new compound with all other compounds that are assigned to the test set in a bootstrap. The functional similarities of the new compound with other compounds in the collection will be predicted by averaging the functional similarities of the new compound over all bootstraps where the test prediction values will be valid. Please note that the machine learning model for each bootstrap will not need to be recomputed in the presence of a new compound because the training set for each bootstrap will not change.

> License

VS-SVM is free for academic use. Please contact [Hamid Safizadeh](mailto:hamid@umn.edu) or [Chad L. Myers](mailto:chadm@umn.edu) for any commercial use.

> Citation

Please cite the following article:

Hamid Safizadeh, Scott W. Simpkins, Justin Nelson, Sheena C. Li, Jeff S. Piotrowski, Mami Yoshimura, Yoko Yashiroda, Hiroyuki Hirano, Hiroyuki Osada, Minoru Yoshida, Charles Boone, and Chad L. Myers. [Improving Measures of Chemical Structural Similarity Using Machine Learning on Chemical-Genetic Interactions](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00993). Journal of Chemical Information and Modeling, 2021.
