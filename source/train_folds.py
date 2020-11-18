import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    df = pd.read_csv('./input/Train.csv')
    df['ID'] = df['ID'] + '.png'
    df = df.drop(columns=['filename'],axis=0)
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.LABEL.values

    kf = StratifiedKFold(n_splits=5)

    for fold, (train_idx,val_idx) in enumerate(kf.split(X=df,y=y)):
        df.loc[val_idx,'kfold'] = fold

    df.to_csv('./input/TB_stratified_kfold.csv',index=False)     



