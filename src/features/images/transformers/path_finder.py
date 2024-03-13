import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin


class PathFinder(BaseEstimator, TransformerMixin):
    def __init__(self, img_folder):
        self.img_folder = img_folder
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_path = X.apply(lambda row: self.generate_image_path(row["imageid"], row["productid"]), axis=1)
        return X_path
    
    def generate_image_path(self, image_id, product_id):
        return os.path.join(self.img_folder, f'image_{image_id}_product_{product_id}.jpg')
