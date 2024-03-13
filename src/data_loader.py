import pandas as pd
import numpy as np
import os

def import_data(folder_path):
    data = pd.read_csv(os.path.join(folder_path, 'X_train.csv'), index_col=0)
    target = pd.read_csv(os.path.join(folder_path, 'Y_train.csv'), index_col=0)
    data = target.join(data)

    return data