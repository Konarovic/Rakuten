import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class YearExtractor(BaseEstimator, TransformerMixin):
    """Extracts years from a text column and returns a dataframe with the years and a boolean column indicating if the text contains a year."""
    def __init__(self, text_column):
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        year_pattern = r"(1[6-9][0-9]{2}|20[0-1][0-9]|202[0-4])"
        years = X[self.text_column].str.extract(year_pattern, expand=False).fillna(0).astype("uint16").to_frame("year")
        years["has_year"] = years["year"] != 0
        return years