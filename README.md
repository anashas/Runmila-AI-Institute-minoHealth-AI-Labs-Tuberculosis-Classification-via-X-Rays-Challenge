# Runmila-AI-Institute-minoHealth-AI-Labs-Tuberculosis-Classification-via-X-Rays-Challenge

# competition [website](https://zindi.africa/competitions/runmila-ai-institute-minohealth-ai-labs-tuberculosis-classification-via-x-rays-challenge)

Tools: Google Colab, Pytorch, Albumentation, Pytorch pretrained models

See TB.ipynb for instalation details, stored in notebook folder

The data used is the small version: train_small.zip/ test_small.zip stored in inputs folder

Training: 5 fold cross validation , however fold 2 is the one that achieved the highest submitted accuracy 

TB_stratified_kfold.csv is the 5 folds cross validation file stored in inputs folder, you can generate it using train_fold.py file stored in the folder
Submission results are stored in nsub.TB.csv in inputs folder

Training engine is in utils.py stored in source folder alongside with train_folds.py

The training models are saved in models folder

To run the code please store your data according to the folders as disccused above and don't forget to change paths in the main notebook.

You can add inputs folder to store your data and another folder called models to save your best trained models

My Zindi [Profile](https://zindi.africa/users/Anas_Hasni).
 
