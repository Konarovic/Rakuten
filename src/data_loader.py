import pandas as pd
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin


def import_data(folder_path):
    data = pd.read_csv(os.path.join(folder_path, 'X_train.csv'), index_col=0)
    target = pd.read_csv(os.path.join(folder_path, 'Y_train.csv'), index_col=0)
    data = target.join(data)

    return data


class CatNameExtractor(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, data_dir) -> None:
        self.mapper = pd.read_csv(os.path.join(data_dir, "prdtype.csv")).set_index(
            keys="prdtypecode").prdtypedesignation.to_dict()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(self.mapper)
        X["prdtypename"] = X["prdtypecode"].map(self.mapper)
        return X


class ClassMerger(BaseEstimator, TransformerMixin):
    """
    WIP : not working yet used for testing a system based on simple rules
    """

    def __init__(self) -> None:
        self.mapper = {
            10: 2403,
            2705: 2403,
            2280: 2403,
            2905: 40,
            2462: 40
        }
