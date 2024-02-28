

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