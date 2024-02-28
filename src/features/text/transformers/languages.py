import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import py3langid as langid
from spellchecker import SpellChecker

def Rakuten_txt_language(data, method='langid'):
    """
    Detect the most likely language of text data in each row of a DataFrame or
    Series.

    Parameters:
    data (DataFrame or Series): Data containing text to analyze.

    Returns:
    Series: A Series indicating the detected language for each row.

    Usage:
    languages = Rakuten_txt_language(dataframe_with_text_columns)
    """

    # concatenating text from multiple columns if necessary to get a series
    if data.ndim == 2:
        data = data.apply(
            lambda row: ' '.join([s for s in row.loc[:]
                                  if isinstance(s, str)]), axis=1)
    # Replacing NaNs with empty string
    data = data.fillna(' ')

    # getting the most likely language for each row of the data series
    # langid.classify(row) returns ('language', score). We only keep the
    # language here
    if method == 'langid':
        # Subsetting possible languages
        langid.set_languages(['fr', 'en', 'de'])
        lang = data.apply(lambda row: langid.classify(row)[0])

    elif method == 'pyspell':
        spell_fr = SpellChecker(language='fr', distance=1)
        spell_en = SpellChecker(language='en', distance=1)
        spell_de = SpellChecker(language='de', distance=1)

        err_fr = data.apply(lambda row: len(spell_fr.known(row.split())))
        err_en = data.apply(lambda row: len(spell_en.known(row.split())))
        err_de = data.apply(lambda row: len(spell_de.known(row.split())))
        lang = pd.concat([err_fr.rename('fr'), err_en.rename(
            'en'), err_de.rename('de')], axis=1)
        lang = lang.idxmax(axis=1)

    elif method == 'bert':
        tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer_fr = BertTokenizer.from_pretrained(
            'dbmdz/bert-base-french-europeana-cased')
        tokenizer_de = BertTokenizer.from_pretrained('bert-base-german-cased')

        err_fr = data.apply(lambda row: ' '.join(
            tokenizer_fr.convert_ids_to_tokens(tokenizer_fr(row)['input_ids'])))
        err_en = data.apply(lambda row: ' '.join(
            tokenizer_en.convert_ids_to_tokens(tokenizer_en(row)['input_ids'])))
        err_de = data.apply(lambda row: ' '.join(
            tokenizer_de.convert_ids_to_tokens(tokenizer_de(row)['input_ids'])))
        lang = pd.concat([err_fr.rename('fr'), err_en.rename(
            'en'), err_de.rename('de')], axis=1)
        lang = lang.idxmin(axis=1)

    return lang

class LangDetector(BaseEstimator, TransformerMixin):
    pass

class LangIdDetector(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        langid.set_languages(['fr', 'en', 'de'])
        lang = X.apply(lambda row: langid.classify(row)[0])
        return lang

class PyspellDetector(BaseEstimator, TransformerMixin):
    pass