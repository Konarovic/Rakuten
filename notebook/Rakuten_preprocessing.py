# -*- coding: utf-8 -*-
"""
Rakuten Preprocessing Module

This module provides functions for importing, cleaning, tokenizing, and
analyzing text data, specifically tailored for the Rakuten dataset. It includes
functionalities to import data, clean up text, detect language, tokenize and
lemmatize text, and perform word count analysis.

Usage:
    import Rakuten_preprocessing as rkt
    
    - data = rkt.Rakuten_txt_import('../Data/')

    - data['designation'] = rkt.Rakuten_txt_cleanup(data['designation'])

    - data['description'] = rkt.Rakuten_txt_cleanup(data['description'])

    - data['language'] = rkt.Rakuten_txt_language(
        data.loc[:, ['designation', 'description']])

    - data['description_tokens'] = rkt.Rakuten_txt_tokenize(
        data['description'], lang=data['language'])

    - data['designation_tokens'] = rkt.Rakuten_txt_tokenize(
        data['designation'], lang=data['language'])

    - data['all_tokens'] = data.apply(
        lambda row: rkt.merge_tokens(row['designation_tokens'],
                                 row['designation_tokens']), axis=1)

    - wordcount, wordlabels = rkt.Rakuten_txt_wordcount(data['designation_tokens'])

    - data['image_path'] = rkt.Rakuten_img_path('../Data/images/image_train/',
                                            data['imageid'], data['productid'])
    
    - data = data.join(rkt.Rakuten_img_size(data['image_path']), axis=1)

    - df_words = pd.DataFrame()
      for c in target['prdtypecode'].unique():
        cnt, wrd = rkt.Rakuten_txt_wordcount(
            data.[data['prdtypecode'] == c, 'designation_tokens'])
        df_words = df_words.join(pd.DataFrame(
            cnt, index=wrd, columns=['code_' + str(c)]))


Dependencies:
    - pandas: Used for data manipulation and analysis.
    - numpy: For numerical computations.
    - seaborn, matplotlib, plotly: For data visualization.
    - spacy: For natural language processing tasks.
    - langid: For language detection.
    - BeautifulSoup: For HTML text cleanup.

@author: Julien Fournier
"""

import cv2
from wordcloud import WordCloud
import re
from collections import Counter
from bs4 import BeautifulSoup
import langid
import spacy
import nltk
import os
import pandas as pd
import swifter
import numpy as np

from plotly.subplots import make_subplots
from plotly import graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'


def Rakuten_txt_import(folder_path):
    """
    Import text data from specified folder path.

    Parameters:
    folder_path (str): The path to the folder containing the X_train.csv and
    Y_train.csv files.

    Returns:
    tuple: A tuple containing two pandas DataFrames, the first for the text
    data and the second for the target data.
    """

    data = pd.read_csv(os.path.join(folder_path, 'X_train.csv'), index_col=0)
    target = pd.read_csv(os.path.join(folder_path, 'Y_train.csv'), index_col=0)

    data = target.join(data)

    return data


def Rakuten_target_factorize(code):
    """
    Factorize a given input (typically 'prdtypecode'),
    assigning each unique value in the column a unique integer.
    This function can be useful for plotting.

    Parameters:
    code (Series or array-like): The column from the DataFrame (e.g.,
                                'prdtypecode') that you wish to factorize.

    Returns:
    ndarray: A numpy array of the same length as the input, containing the
             numerical codes corresponding to the factorized values. Each
             unique value in the input is mapped to a unique integer.
    """

    code = pd.factorize(code)[0]

    return code


def Rakuten_txt_cleanup(data):
    """
    Clean up text data by removing HTML tags, URLs, and filenames.

    This function iterates through each column (if a DataFrame) or each
    entry (if a Series) in the provided data, applying a cleanup process
    to remove unwanted HTML elements, URLs, and filenames from the text.

    Parameters:
    data (DataFrame or Series): Text data to be cleaned. This can be a
                                pandas DataFrame or Series containing
                                text entries.

    Returns:
    DataFrame or Series: The cleaned text data, with HTML tags, URLs, and
                         filenames removed. The structure (DataFrame or
                         Series) matches the input.

    Usage:
    # For a DataFrame
    cleaned_df = Rakuten_txt_cleanup(dataframe_with_text)

    # For a Series
    cleaned_series = Rakuten_txt_cleanup(series_with_text)
    """

    url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|\
                           [!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    filename_regex = re.compile(r'\b(?<!\d\.)\w+\.(txt|jpg|png|docx|pdf)\b')

    if data.ndim == 2:
        # if data if a dataframe
        for col in data.columns:
            data.loc[:, col] = data[col].apply(
                lambda row: txt_cleanup(row, url_regex, filename_regex))

            data.loc[data[col].astype(str).str.len() == 0, col] = np.nan

    else:
        # if data is a series
        data = data.apply(
            lambda row: txt_cleanup(row, url_regex, filename_regex))

        data.loc[data.astype(str).str.len() == 0] = np.nan

    return data


def txt_cleanup(txt, url_regex, filename_regex):
    """
    Remove HTML tags, URLs, and filenames from a given text string.

    Parameters:
    txt (str): Text to be cleaned.
    url_regex (compiled regex): Regex pattern to identify URLs.
    filename_regex (compiled regex): Regex pattern to identify filenames.

    Returns:
    str: Cleaned text with HTML tags, URLs, and filenames removed.

    Usage:
    cleaned_text = txt_cleanup(some_text, url_regex, filename_regex)
    """
    if isinstance(txt, str):
        # Remove URLs
        txt = url_regex.sub(' ', txt)

        # # remove filenames
        txt = filename_regex.sub(' ', txt)

        # Remove HTML tags
        soup = BeautifulSoup(txt, 'html.parser')
        txt = soup.get_text(separator=' ')

        # Remove lxml markers
        soup = BeautifulSoup(txt, 'lxml')
        txt = soup.get_text(separator=' ')

        # removing all text shorter than 4 characters (eg ..., 1), -, etc)
        if len(txt.strip()) < 4:
            txt = ''

    return txt


def Rakuten_txt_language(data):  # ajouter une condition sur le score de fin
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
    # Subsetting possible languages
    langid.set_languages(['fr', 'en', 'de'])

    # concatenating text from multiple columns if necessary to get a series
    if data.ndim == 2:
        data = data.apply(
            lambda row: ' '.join([s for s in row.loc[:]
                                  if isinstance(s, str)]), axis=1)

    # getting the most likely language for each row of the data series
    # langid.classify(row) returns ('language', score). We only keep the
    # language here
    lang = data.apply(lambda row: langid.classify(row)[0])

    return lang


def Rakuten_txt_tokenize(data, lang=None, method='spacy'):
    """
    Tokenize and lemmatize text data, returning a list of unique tokens, in the
    same order they appeared.

    Parameters:
    data (DataFrame or Series): Text data to tokenize and lemmatize.
    lang (Series, optional): Series containing language information for each
    entry.
    method: 'spacy', 'spacy_accuracy' or 'nltk'. Method to use for tokenization and lemmatization.

    Returns:
    DataFrame or Series: Data with tokenized and lemmatized text.

    Usage:
    tokenized_data = Rakuten_txt_tokenize(data, language_series)
    """

    if lang is None:
        lang = Rakuten_txt_language(data)

    # concatenating text from multiple columns if necessary
    if data.ndim > 1:
        data = data.apply(lambda row: ' '.join(
            [s for s in row if isinstance(s, str)]), axis=1)

    # joining text and language data
    data = pd.concat([data, lang], axis=1)

    if (method == 'spacy') or (method == 'spacy_accuracy'):
        # Checking if spacy language data have been  downloaded
        suffix = 'trf' if method == 'spacy_accuracy' else 'sm'
        core = 'dep' if method == 'spacy_accuracy' else 'core'

        if not spacy.util.is_package('en_core_web_'+suffix):
            spacy.cli.download('en_core_web_'+suffix)

        if not spacy.util.is_package('fr_'+core+'_news_'+suffix):
            spacy.cli.download('fr_'+core+'_news_'+suffix)

        if not spacy.util.is_package('de_'+core+'_news_'+suffix):
            spacy.cli.download('de_'+core+'_news_'+suffix)

        # Loading spacy language models
        nlpdict = {'en': spacy.load('en_core_web_'+suffix),
                'fr': spacy.load('fr_'+core+'_news_'+suffix),
                'de': spacy.load('de_'+core+'_news_'+suffix)}

        # Applying tokenisation using spacy
        data = data.apply(lambda row: tokens_from_spacy(
            row.iloc[0], row.iloc[1], nlpdict), axis=1)

    elif method == 'nltk':
        # downloading nltk ressources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        if lang.str.len().max() == 2:
            language_mapping = {'fr': 'french',
                                'en': 'english',
                                'de': 'german'}
            data.iloc[:, 1] = data.iloc[:, 1].replace(language_mapping)

            # Applying tokenisation using spacy
            data = data.apply(lambda row: tokens_from_nltk(
                row.iloc[0], row.iloc[1]), axis=1)

    return data


def tokens_from_spacy(txt, lang, nlpdict):
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
        tokens = nlpdict[lang](txt)

        # Remove stopwords, punctuation, and perform lemmatization
        filtered_tokens = [token.lemma_.lower()
                           for token in tokens
                           if token.is_alpha and not token.is_stop
                           and len(token) > 3]

        # Keeping a unique list of tokens, in the same order they appeared in
        # the text
        seen = set()
        filtered_tokens = [seen.add(token) or token
                           for token in filtered_tokens if token not in seen]

        # returning result as a single string
        return ' '.join(filtered_tokens)
    else:
        # Return empty lists if the input is not a string
        return np.nan


def tokens_from_nltk(txt, lang):
    """
    Generate a list of unique word tokens from a text string using NLTK.

    Parameters:
    txt (str): Text string to tokenize.
    lang (str): Language of the text.

    Returns:
    str: String of unique, lemmatized tokens.

    Usage:
    tokens = tokens_from_nltk(text_to_tokenize, 'en')
    """
    if isinstance(txt, str):
        # tokenization with the appropriate language
        tokens = nltk.tokenize.word_tokenize(txt, lang)

        # Remove stopwords, punctuation, and perform lemmatization
        stop_words = set(nltk.corpus.stopwords.words(lang))
        stemmer = nltk.stem.SnowballStemmer(lang)
        filtered_tokens = [stemmer.stem(token.lower())
                           for token in tokens
                           if token.isalpha()
                           and token.lower() not in stop_words
                           and len(token) > 3]

        # Keeping a unique list of tokens, in the same order they appeared in
        # the text
        seen = set()
        filtered_tokens = [seen.add(token) or token
                           for token in filtered_tokens if token not in seen]

        # returning result as a single string
        return ' '.join(filtered_tokens)
    else:
        # Return empty lists if the input is not a string
        return ''


def merge_tokens(*args):
    """ merge two strings, keeping a unique list of the words, in the order
    they appeared"""
    # Replacing non-string inputs with empty string
    args = [arg if isinstance(arg, str) else '' for arg in args]

    # joining input strings
    alltokens = (' '.join(args)).split()

    # Keeping a unique list of tokens, in the same order they appeared in
    # the text
    seen = set()
    alltokens = [seen.add(token) or token
                 for token in alltokens if token not in seen]

    return ' '.join(alltokens)


# add correlation heat map across categories
def Rakuten_txt_wordcount(data, nmax_words=None, Normalize=False):
    """
    Count the frequency of each word in the text data.

    Parameters:
    data (DataFrame or Series): Text data to analyze.
    nmax_words (int, optional): Maximum number of words to return.

    Returns:
    tuple: Two lists, one of word counts and the other of corresponding words,
    ordered from most to least frequent.

    Usage:
    counts, words = Rakuten_wordcount(data_with_text, 100)
    """

    # concatenating text from multiple columns if necessary
    if data.ndim > 1:
        data = data.apply(lambda row: ' '.join(
            [s for s in row if isinstance(s, str)]), axis=1)

    # Replacing NaNs by spaces
    data = data.fillna(' ')

    # Merging all strings into a single one
    data = ' '.join(data)

    # Removing non-alpha characters, converting to lower case, returning a
    # string of words
    data = re.findall(r'\b\w+\b', data.lower())

    # Counting the number of occurences of each word
    counts = dict(Counter(data).most_common(nmax_words))

    # Converting dictionary values and keys into lists
    countlist = np.array(list(counts.values()))
    wordlist = np.array(list(counts.keys()))

    # Normalizing vounts if necessary
    if Normalize:
        countlist = countlist / sum(countlist)

    # Sorting indices according to frequency and reversing to get it from most
    # to  least frequent
    idx = np.argsort(countlist)[::-1]

    # Reordering values and labels according to sorted idx
    countlist = countlist[idx].tolist()
    wordlist = wordlist[idx].tolist()

    return countlist, wordlist


def Rakuten_img_path(img_folder, imageid, productid):
    """ retrurns the path to the image of a given productid and imageid"""

    df = pd.DataFrame(pd.concat([imageid, productid], axis=1))

    img_path = df.apply(lambda row:
                        os.path.join(img_folder, 'image_'
                                     + str(row['imageid'])
                                     + '_product_'
                                     + str(row['productid'])
                                     + '.jpg'),
                        axis=1)

    return img_path


def Rakuten_img_size(img_path):
    """return the size of the full image and the the ratio of the non-padded
    image to the full size"""

    # Applying get_img_size to all images
    df = pd.DataFrame()
    df['size'] = img_path.apply(lambda row: get_img_size(row))

    # Calculating the ratio of the non-padded image size to the full size
    df['size_actual'] = df['size'].apply(
        lambda row: max(row[2]/row[0], row[3]/row[1]))

    # Calculating the actual aspect ratio of the non-padded image
    df['ratio_actual'] = df['size'].apply(
        lambda row: row[2]/row[3] if row[3] > 0 else 0)

    # Keeping in size only the size of the full image
    df['size'] = df['size'].apply(lambda row: row[0:2])

    return df


def get_img_size(img_path):
    """ return the actual width and height of images without padding"""
    # Reading the image
    img = cv2.imread(img_path)

    # full size of the image
    width, height = img.shape[:2]

    # converting to gray scale to threshold white padding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Padding the gray image with a white rectangle around the full image to
    # make sure there is at least this contour to find
    border_size = 1
    gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size,
                              border_size, cv2.BORDER_CONSTANT,
                              value=[255, 255, 255])

    # Threshold the image to get binary image (white pixels will be black)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Finding the contours of the non-white area
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Getting the bounding rectangle for the largest contour, if contours
        # is not empty
        _, _, width_actual, height_actual = cv2.boundingRect(
            max(contours, key=cv2.contourArea))
    else:
        width_actual, height_actual = 0, 0

    return [width, height, width_actual, height_actual]
