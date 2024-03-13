import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin

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
    # Add spaces around numbers and punctuations except ', / and ¿
    spacearound.append(
        r'(\d+|[-.,!¡;；:¯…„“\§«»—°•£❤☆(){}\[\]"@#$%^&*+=|<>~`‘’¬])')
    # Converting spacearound to regex pattern object
    spacearound = re.compile('|'.join(spacearound))

    # All regex to add space before
    spacebefore = []
    # Add spaces before uppercase letters if they're both preceded and followed
    # by lowercase letters or preceded by a punctuation and followed by a lower
    # case letter
    spacebefore.append(
        r'(?<=[a-zÀ-ÿ]|[.,!;:\§«»°])([A-Z])(?=[a-zÀ-ÿ])')
    # Converting spacebefore to regex pattern object
    spacebefore = re.compile('|'.join(spacebefore))

    if data.ndim == 2:
        # if data is a dataframe
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
    !!!!!!!!!!!!!!!!!!!!!also check/remove  Â· ¢
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
        # Convert HTML markups
        soup = BeautifulSoup(txt, 'html.parser')
        txt = soup.get_text(separator=' ')

        # Convert lxml markers
        soup = BeautifulSoup(txt, 'lxml')
        txt = soup.get_text(separator=' ')

        # Remove according to subregex
        txt = subregex.sub(' ', txt)

        # Split according to spacearound
        #txt = spacearound.sub(r' \1 ', txt)

        # Add space before according to spacebefore
        txt = spacebefore.sub(r' \1', txt)

        # cleaning up extra spaces
        # txt = re.sub(r'\s+', ' ', txt).strip()

        # removing all text shorter than 4 characters (eg ..., 1), -, etc)
        if len(txt.strip()) < 4:
            txt = ''

    return txt


def Rakuten_txt_fixencoding(data, lang):
    """
    Cleans up text data by correcting badly encoded words.

    This function joins text data with language information, applies encoding corrections,
    and utilizes spell checking for various languages to correct misspelled words.
    It specifically targets issues with special characters like '?', '¿', 'º', '¢', '©', and '́'.

    Parameters:
    data (DataFrame): A pandas DataFrame containing text data to be cleaned.
    lang (Series): A pandas Series containing language information for each row in the data.

    Returns:
    DataFrame: The cleaned DataFrame with bad encodings corrected and language column dropped.

    Usage Example:
    df = Rakuten_txt_cleanup(data[['designation', 'description']],
                                       data['language'])
    """
    # joining text and language data
    data = pd.concat([data, lang.rename('lang')], axis=1)

    # Correction of bad encoding relies in part on spell checker, to correct
    # misspelled words.
    spellers = {'fr': SpellChecker(language='fr'),
                'en': SpellChecker(language='en'),
                'de': SpellChecker(language='de')}

    for col in data.columns[:-1]:
        data.loc[:, col] = data.apply(
            lambda row: txt_fixencoding(row[col], row['lang'], spellers), axis=1)

    return data.drop(columns='lang')


def txt_fixencoding(txt, lang, spellers):
    """
    Corrects badly encoded words within a given text string.

    This function applies multiple regex substitutions to fix common encoding issues 
    in text data. It handles duplicates of badly encoded characters, replaces specific 
    incorrectly encoded words with their correct forms, and utilizes a spell checker for further corrections. 
    The function also handles special cases of encoding errors after certain character sequences.

    Parameters:
    txt (str): The text string to be cleaned.
    lang (str): The language of the text, used for spell checking.
    spellers (dict): A dictionary of SpellChecker instances for different languages.

    Returns:
    str: The cleaned text string with encoding issues corrected.

    Usage Example:
    ----------------
    # Spell checkers for different languages
    spellers = {'fr': SpellChecker(language='fr'),
                'en': SpellChecker(language='en'),
                'de': SpellChecker(language='de')}

    # Correct the encoding in the text
    corrected_text = txt_fixencoding(example_text, language, spellers)
    """

    # returning the original value if not a str or no special characters
    if not isinstance(txt, str) or len(re.findall(r'[\?¿º¢©́]', txt)) == 0:
        return txt

    # replace duplicates of badly encoded markers and some weird combinations
    # with a single one
    pattern = r'([?¿º¢©́])\1'
    txt = re.sub(pattern, r'\1', txt)
    txt = re.sub(r'\[å¿]', '¿', txt)

    # Replacing words that won't be easily corrected by the spell checker
    replace_dict = {'c¿ur': 'coeur', '¿uvre': 'oeuvre', '¿uf': 'oeuf',
                    'n¿ud': 'noeud', '¿illets': 'oeillets',
                    'v¿ux': 'voeux', 's¿ur': 'soeur', '¿il': 'oeil',
                    'man¿uvre': 'manoeuvre',
                    '¿ºtre': 'être', 'à¢me': 'âme',
                    'm¿ºme': 'même', 'grà¢ce': 'grâçe',
                    'con¿u': 'conçu', 'don¿t': "don't",
                    'lorsqu¿': "lorsqu'", 'jusqu¿': "jusqu'",
                    'durabilit¿avec': 'durabilité avec',
                    'dâ¿hygiène': "d'hygiène", 'à¿me': 'âme',
                    'durabilit¿': 'durabilité', 'm¿urs': 'moeurs',
                    'd¿coration': 'décoration', 'tiss¿e': 'tissée',
                    '¿cran': 'écran', '¿Lastique': 'élastique',
                    '¿Lectronique': 'électronique', 'Capacit¿': 'capacité',
                    'li¿ge': 'liège', 'Kã?Â¿Rcher': 'karcher',
                    'Ber¿Ante': 'berçante',

                    'durabilitéavec': 'durabilité avec',
                    'cahiercaract¿re': 'cahier caractère',
                    'Cahierembl¿Me': 'Cahier emblème',

                    'c?ur': 'coeur', '?uvre': 'oeuvre', '?uf': 'oeuf',
                    'n?ud': 'noeud', '?illets': 'oeillets',
                    'v?ux': 'voeux', 's?ur': 'soeur', '?il': 'oeil',
                    'man?uvre': 'manoeuvre',
                    '?ºtre': 'être',
                    'm?ºme': 'même',
                    'con?u': 'conçu', 'don?t': "don't",
                    'lorsqu¿': "lorsqu'", 'jusqu¿': "jusqu'",
                    'durabilit¿avec': 'durabilité avec',
                    'dâ?hygiène': "d'hygiène", 'à?me': 'âme',
                    'durabilit?': 'durabilité', 'm?urs': 'moeurs',
                    'd?coration': 'décoration', 'tiss?e': 'tissée',
                    '?cran': 'écran', '?Lastique': 'élastique',
                    '?Lectronique': 'électronique', 'Capacit?': 'capacité',
                    'li?ge': 'liège', 'Ber?Ante': 'berçante',
                    "Lâ¿¿Incroyable": "l'incroyable", 'Creì¿Ateur': 'créateur',

                    'cahiercaract?re': 'cahier caractère',
                    'Cahierembl?Me': 'Cahier emblème'}

    for badword, correction in replace_dict.items():
        txt = re.sub(re.escape(badword), correction, txt, flags=re.IGNORECASE)

    # Not sure why but the following doesn't work at once so we do it again
    # (It is quite common in the data set...)
    pattern = re.escape('durabilitéavec')
    txt = re.sub(pattern, 'durabilité avec', txt)
    pattern = re.escape('cahiercaractère')
    txt = re.sub(pattern, 'cahier caractère', txt)
    pattern = re.escape('Cahieremblème')
    txt = re.sub(pattern, 'cahier emblème', txt)

    # Replacing badly encoded character by apostrophe when following in second
    # position a d, l, c or n.
    pattern = r'\b([dlcn])[¿?](?=[aeiouyh])'
    txt = re.sub(pattern, r"\1'", txt, flags=re.IGNORECASE)

    # Replacing badly encoded character by apostrophe when following in third
    # position after qu.
    pattern = r'\bqu[¿?]'
    txt = re.sub(pattern, "qu'", txt, flags=re.IGNORECASE)

    # Finding all remaining words with special characters at the start, end or
    # within
    pattern = r'\b\w*[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*|\b\w*[\?¿º¢©́]|[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*'
    badword_list = re.findall(pattern, txt)
    # Since this ends up with some special characters alone (for instance à
    # would become single ?), we make sure they are the last to be corrected.
    # Otherwise, the other motifs wouldn't be detectable anymore in the next
    # loop
    badword_list.sort(key=len, reverse=True)

    # correction function with lru_cache to enable caching
    @lru_cache(maxsize=1000)
    def cached_spell_correction(word, language):
        return spellers[lang].correction(word)

    # Replacing each of these word by the correction from the spell checker (if
    # available), if it is different from the original word without special
    # character (in case this character is an actual punctuation)
    for badword in badword_list:
        badword = badword.lower()
        badword_corrected = cached_spell_correction(badword, lang)
        badword_cleaned = re.sub(r'[^a-zA-Z0-9]', '', badword)
        if badword_corrected and badword_corrected != badword_cleaned:
            pattern = re.escape(badword)
            #txt = txt.replace(badword, badword_corrected)
            txt = re.sub(pattern, badword_corrected, txt, flags=re.IGNORECASE)

    # for debugging purpose
    pattern = r'\b\w*[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*|\b\w*[\?¿º¢©́]|[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*'
    badword_list = re.findall(pattern, txt)

    # at the end we remove all remaining special characters except ? (as those)
    # at the end of words may correspond to actual punctuations
    pattern = r'[\¿º¢©́]'
    txt = re.sub(pattern, ' ', txt)

    # Adding a space after ? if necessary
    pattern = r'\?(?!\s)'
    txt = re.sub(pattern, '? ', txt)

    return txt

class HtmlCleaner(BaseEstimator, TransformerMixin):
    """A transformer to clean HTML markups"""

    def __init__(self) -> None:
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.apply(lambda row: self.clean_markups(row))
        return X_cleaned
    
    @staticmethod
    def clean_markups(text):
        if isinstance(text, str):
            soup = BeautifulSoup(text, 'html.parser')
            cleaned_text = soup.get_text(separator=' ')
            return cleaned_text
        else:
            return text
        

class LxmlCleaner(BaseEstimator, TransformerMixin):
    """A transformer to clean LXML markups"""

    def __init__(self) -> None:
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.apply(lambda row: self.clean_markups(row))
        return X_cleaned
    
    @staticmethod
    def clean_markups(text):
        if isinstance(text, str):
            soup = BeautifulSoup(text, 'lxml')
            cleaned_text = soup.get_text(separator=' ')
            return cleaned_text
        else:
            return text
        
class TextCleaner(BaseEstimator, TransformerMixin):
    """A Transformer class to clean text"""
    def __init__(self, pattern) -> None:
        self.pattern = re.compile(pattern)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace(pat=self.pattern, repl=" ", regex=True)
    

class UrlCleaner(TextCleaner):
    """A transformer to clean URLs from text"""
    def __init__(self):
        super().__init__(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")


class FileNameCleaner(TextCleaner):
    """A transformer to clean Filenames from text"""
    def __init__(self) -> None:
        super().__init__(r'\b(?<!\d\.)\w+\.(txt|jpg|png|docx|pdf)\b')

class BadHTMLCleaner(TextCleaner):
    """A transformer to clean Bad HTML from text"""
    def __init__(self) -> None:
        super().__init__(r'nbsp|&amp|& nbsp|')
