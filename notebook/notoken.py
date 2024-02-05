# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:29:23 2024

@author: Julien Fournier
"""
import py3langid as langid
import spacy
import nltk
import os
import pandas as pd
#import swifter
import numpy as np
import re


def Rakuten_txt_nottokenized(data, lang):
    """

    """

    # concatenating text from multiple columns if necessary
    if data.ndim > 1:
        data = data.apply(lambda row: ' '.join(
            [s for s in row if isinstance(s, str)]), axis=1)

    # joining text and language data
    data = pd.concat([data, lang], axis=1)

    # Checking if spacy language data have been  downloaded
    if not spacy.util.is_package('en_core_web_sm'):
        spacy.cli.download('en_core_web_sm')

    if not spacy.util.is_package('fr_core_news_sm'):
        spacy.cli.download('fr_core_news_sm')

    if not spacy.util.is_package('de_core_news_sm'):
        spacy.cli.download('de_core_news_sm')

    # Loading spacy language models
    nlpdict = {'en': spacy.load('en_core_web_sm'),
               'fr': spacy.load('fr_core_news_sm'),
               'de': spacy.load('de_core_news_sm')}

    # Applying tokenisation using spacy
    data = data.apply(lambda row: nottokens_from_spacy(
        row.iloc[0], row.iloc[1], nlpdict), axis=1)

    return data


def nottokens_from_spacy(txt, lang, nlpdict):
    """
    Generate a list of unique word tokens from a text string using spaCy.

    Parameters:
    txt (str): Text string to tokenize.
    lang (str): Language of the text.
    nlpdict (dict): Dictionary mapping language codes to spaCy models.

    Returns:
    str: String of unique, lemmatized tokens.

    Usage:
    tokens = tokens_from_spacy(text_to_tokenize, 'en', nlp_dictionary)
    """
    if isinstance(txt, str):
        # using the appropriate language
        txt = txt.lower()
        tokens = nlpdict[lang](txt)

        # Remove punctuation, and perform lemmatization
        filtered_tokens = [token.text
                           for token in tokens
                           if token.is_alpha
                           and len(token) > 2
                           and any(vowel in token.text.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]

        # Keeping a unique list of words not tokenized,
        # in the same order they appeared in
        # the text
        seen = set()
        allwords = re.findall(r'\b\w+\b', txt)
        not_a_tokens = [seen.add(word) or word
                        for word in allwords if word not in filtered_tokens and not (word in seen or seen.add(word))]

        # returning result as a single string
        return ' '.join(not_a_tokens)
    else:
        # Return empty lists if the input is not a string
        return np.nan
