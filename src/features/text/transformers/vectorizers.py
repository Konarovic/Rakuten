import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF(TfidfVectorizer):
    """Transforms a text column into a TF-IDF matrix."""
    def __init__(self, text_column, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return super().fit(X[self.text_column], y)
    
    def transform(self, X):
        return pd.DataFrame(data=self.transform(X[self.text_column]).toarray(), columns=self.get_feature_names_out())