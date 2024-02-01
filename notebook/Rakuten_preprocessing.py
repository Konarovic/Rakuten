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
        data[['designation', 'description']])

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
    
    - data = data.join(rkt.Rakuten_img_size(data['image_path']))

    - #Proportion of each tokens across all product of each category
    df_words = pd.DataFrame()
    for code in data['prdtypecode'].unique():
        cnt, wrd = rkt.Rakuten_txt_wordcount(data.loc[data['prdtypecode'] == code, 'designation_tokens'])
        cnt = cnt / (data['prdtypecode']==code).sum()
        df_words = df_words.join(pd.DataFrame(cnt, index=wrd, columns=['code_' + str(code)]), how='outer')
        df_words = df_words.fillna(0)
        
    from scipy.cluster.hierarchy import linkage, leaves_list
    Z = linkage(df_words.corr(), 'ward')
    order = leaves_list(Z)
    px.imshow(df_words.corr().iloc[order, order])
    
    #multiplying the word frequency by the relative proportion of each token
    across categories
    df_words_rel = df_words * df_words.apply(lambda row: row / row.sum(), axis=1)
    

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
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import re
from collections import Counter
from bs4 import BeautifulSoup
#import langid
import py3langid as langid
from spellchecker import SpellChecker
import spacy
import nltk
import os
import pandas as pd
import numpy as np

import unidecode
from transformers import BertTokenizer
import torch

from plotly.subplots import make_subplots
from plotly import graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from PIL import Image
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
    # All regex to remove
    subregex = []
    # url patterns
    subregex.append(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|\
                   [!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # filename patterns
    subregex.append(r'\b(?<!\d\.)\w+\.(txt|jpg|png|docx|pdf)\b')
    # badly formatted html markups
    subregex.append(r'nbsp|&amp|& nbsp|')
    # Converting subregex to regex pattern object
    subregex = re.compile('|'.join(subregex), re.IGNORECASE)

    # All regex to add space around
    spacearound = []
    # Add spaces around numbers and punctuations except -, ', / and ¿
    spacearound.append(r'(\d+|[.,!;:(){}\[\]"@#$%^&*+=|<>~`¬])')
    # Converting spacearound to regex pattern object
    spacearound = re.compile('|'.join(spacearound))

    # All regex to add space before
    spacebefore = []
    # Add spaces before uppercase letters if they're both preceded and followed
    # by lowercase letters
    spacebefore.append(r'(?<=[a-z])([A-Z])(?=[a-z])')
    # Converting spacebefore to regex pattern object
    spacebefore = re.compile('|'.join(spacebefore))

    if data.ndim == 2:
        # if data if a dataframe
        for col in data.columns:
            data.loc[:, col] = data[col].apply(
                lambda row: txt_cleanup(row, subregex, spacearound, spacebefore))
            # Replacing empty strings with NaNs
            data.loc[data[col].astype(str).str.len() == 0, col] = np.nan

    else:
        # if data is a series
        data = data.apply(
            lambda row: txt_cleanup(row, subregex, spacearound, spacebefore))
        # Replacing empty strings with NaNs
        data.loc[data.astype(str).str.len() == 0] = np.nan

    return data


def txt_cleanup(txt, subregex, spacearound, spacebefore):
    """
    Remove HTML tags, URLs, and filenames from a given text string.

    Parameters:
    txt (str): Text to be cleaned.
    subregex (compiled regex): Regex patterns to remove.
    spacearound (compiled regex): Regex patterns to split.
    spacebefore (compiled regex): Regex patterns where space should be added before.

    Returns:
    str: Cleaned text with HTML tags, URLs, filenames, etc removed.

    Usage:
    cleaned_text = txt_cleanup(some_text, subregex, splitregex)
    """
    if isinstance(txt, str):
        
        # Remove HTML tags
        soup = BeautifulSoup(txt, 'html.parser')
        txt = soup.get_text(separator=' ')

        # Remove lxml markers
        soup = BeautifulSoup(txt, 'lxml')
        txt = soup.get_text(separator=' ')

        # Remove according to subregex
        txt = subregex.sub(' ', txt)

        # Split according to spacearound
        txt = spacearound.sub(r' \1 ', txt)

        # Add space before according to spacebefore
        txt = spacebefore.sub(r' \1', txt)

        # cleaning up extra spaces
        txt = re.sub(r'\s+', ' ', txt).strip()

        # cleanning specific . . .
        txt = re.sub(r'\.\s\.\s\.', '...', txt).strip()

        # removing all text shorter than 4 characters (eg ..., 1), -, etc)
        if len(txt.strip()) < 4:
            txt = ''

    return txt


# ajouter une condition sur le score de fin
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
    # Subsetting possible languages
    langid.set_languages(['fr', 'en', 'de'])

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


def Rakuten_txt_tokenize(data, lang=None, method='spacy'):
    """
    Tokenize and lemmatize text data, returning a list of unique tokens, in the
    same order they appeared.

    Parameters:
    data (DataFrame or Series): Text data to tokenize and lemmatize.
    lang (Series, optional): Series containing language information for each
    entry.

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

    if method == 'spacy':
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
                           if token.is_alpha
                           and not token.is_stop
                           and len(token) > 2
                           and any(vowel in token.text.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]

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
                           and token.lower()
                           not in stop_words
                           and len(token) > 2
                           and any(vowel in token.text.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]

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


def Rakuten_txt_wordcloud(data, token_col_name, categories):
    plt_rows = len(categories)
    plt_idx = 0
    fig, axs = plt.subplots(plt_rows, 1, figsize=(12, 6*plt_rows))
    stopwords = set(STOPWORDS)
    for code in categories.index:
        img = Image.open('./wordcloud-masks/console.jpg')
        mask_coloring = np.array(img)
        wordcloud = WordCloud(
                        background_color='white',
                        mask=mask_coloring,
                        min_font_size=5,
                        max_font_size=30,
                        contour_width=1,
                        random_state=42,
                        max_words=4000,
                        stopwords=stopwords,
                    ).generate(' '.join(data[data.prdtypecode == code][token_col_name]))
        # img_colors=ImageColorGenerator(mask_coloring)
        # axs[plt_idx].imshow(wordcloud.recolor(color_func=img_colors), interpolation="bilinear")
        wc_img = Image.fromarray(wordcloud.to_array())
        # back_img= img.resize(wc_img.size)
        # img_new = Image.alpha_composite(back_img, wc_img)
        axs[plt_idx].imshow(wc_img)
        axs[plt_idx].set_title(str(code) + ' ' + categories.loc[code][0])
        axs[plt_idx].axis("off")
        plt_idx += 1

    plt.tight_layout()
    plt.show()


def Rakuten_txt_translate(data, lang=None, target_lang='fr'):
    """
    Translate text data from German or English to French using Google Translate.

    Parameters:
    data (DataFrame or Series): Text data to translate.

    Returns:
    DataFrame or Series: Translated text data.

    Usage:
    translated_data = Rakuten_txt_translate(data_with_text)
    """
    # Checking if googletrans has been installed
    try:
        from googletrans import Translator
    except ImportError:
        print('googletrans not installed. Please install it using pip.')

    if lang is None:
        lang = Rakuten_txt_language(data)

    # concatenating text from multiple columns if necessary
    if data.ndim > 1:
        data = data.apply(lambda row: ' '.join(
            [s for s in row if isinstance(s, str)]), axis=1)

    # joining text and language data
    data = pd.concat([data, lang], axis=1)
    # Instantiating translator
    translator = Translator()

    # Translating text
    return data.apply(lambda row: txt_translate(translator, row.iloc[0], target_lang) if row.iloc[1] != target_lang else row.text_clean, axis=1)


def txt_translate(translator, text, target_lang):
    """
    Translate a text string from German or English to French using Google Translate.

    Parameters:
    translator (googletrans.Translator): Translator object.
    text (str): Text string to translate.
    target_lang (str): Target language code.

    Returns:
    str: Translated text string.

    Usage:
    translated_text = txt_translate(translator, text_to_translate, 'fr')
    """
    # Checking if googletrans has been installed
    try:
        translated = translator.translate(text, dest=target_lang).text
        return translated
    except ImportError:
        print('translation error for : ' + text)

    return ''


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


def img_resize(folder_path, save_path='./resized/'):
    """
    Resize image files in a specified folder to remove extra padding areas and resize the image

    Parameters:
        folder_path (str): The path to the folder containing the image files to be resized.
        save_path (str, optional): The path to the folder where the resized images will be saved.
                                   Default is './resized/'.

    Returns:
        None

    This function iterates over all image files in the specified folder, resizes each image
    while removing extra padding areas, and saves the resized images in the 'save_path' folder.
    The resized images are named with '_resized' suffix and have the same file format as the
    original images.

    Example:
        img_resize('/path/to/images_folder', '/path/to/save_resized_images/')

    Note:
        - The function assumes that the input images are in JPEG format (.jpg).
        - The 'save_path' folder will be created if it does not exist.
    """

    # list of all files
    all_files = os.listdir(folder_path)

    # Image files
    image_files = [f for f in all_files if f.lower().endswith(('.jpg'))]

    # Making sure save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Iterating over images
    for img_name in image_files:
        # Reading the image
        img = cv2.imread(os.path.join(folder_path, img_name))

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
            x, y, width_actual, height_actual = cv2.boundingRect(
                max(contours, key=cv2.contourArea))

            # Compute scaling factors along x and y depending on the largest dim
            if width_actual >= height_actual:
                scale_x = width / width_actual
                scale_y = scale_x
            else:
                scale_y = height / height_actual
                scale_x = scale_y

            # Cropping and resizing the image
            img = img[y:y+height_actual, x:x+width_actual]
            img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)

            # Padding the image with white to reach original dimension
            # (usually 500 x 500)
            pad_top = (height - img.shape[0]) // 2
            pad_bottom = height - img.shape[0] - pad_top
            pad_left = (width - img.shape[1]) // 2
            pad_right = width - img.shape[1] - pad_left
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left,
                                     pad_right, cv2.BORDER_CONSTANT,
                                     value=[255, 255, 255])
        # Saving the resized image to the same folder with the suffix "_resized"
        output_path = os.path.join(
            save_path, os.path.splitext(img_name)[0] + '_resized.jpg')
        cv2.imwrite(output_path, img)
