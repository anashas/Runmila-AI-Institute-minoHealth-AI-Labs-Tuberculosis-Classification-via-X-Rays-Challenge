# Runmila-AI-Institute-minoHealth-AI-Labs-Tuberculosis-Classification-via-X-Rays-Challenge

I used google colab to train the model

Libraries:

pytorch

Albumentation for data augmentation

pytorch pretrained models

see TB.ipynb for instalation details, stored in notebook folder


the data used is the small version: train_small.zip/ test_small.zip stored in inputs folder

training: 5 fold cross validation , however fold 2 is the one that achieved the highest submitted accuracy 

TB_stratified_kfold.csv is the 5 folds cross validation file stored in inputs folder, you can generate it using train_fold.py file stored in the folder
submission results are stored in nsub.TB.csv in inputs folder

training engine is in utils.py stored in source folder alongside with train_folds.py

the training models are saved in models folder

to run the code please store your data according to the folders as disccused above and don't forget to change paths in the main notebook.
 
