import re
from typing import Any, List, Union

import nltk
import numpy.typing as npt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from text_analytics.config import ACRONYMS


def convert_lowercase(series: Union[pd.Series, str]) -> Union[pd.Series, str]:
    if isinstance(series, str):
        return series.lower()
    return series.str.lower()


def remove_html_tags(series: Union[pd.Series, str]) -> Union[pd.Series, str]:
    if isinstance(series, str):
        return re.sub(pattern=r"<.*?>", repl="", string=series)

    return series.str.replace(pat=r"<.*?>", repl="", regex=True)


def remove_punctuation(word_arr: Union[pd.Series, str]) -> Union[pd.Series, str]:
    import string

    if isinstance(word_arr, str):
        return " ".join(word for word in word_arr if word not in string.punctuation)

    return word_arr.apply(
        lambda arr: [word for word in arr if word not in string.punctuation]
    )


def convert_abbreviations(series: Union[pd.Series, str]) -> Union[pd.Series, str]:

    if isinstance(series, str):
        return " ".join(
            ACRONYMS.get(word) if word in ACRONYMS.keys() else word
            for word in series.split()
        )

    return series.apply(
        lambda sentence: " ".join(
            ACRONYMS.get(word) if word in ACRONYMS.keys() else word
            for word in sentence.split()
        )
    )


def remove_stopwords(
    series: Union[pd.Series, str],
    stop_words: Union[nltk.corpus.reader.wordlist.WordListCorpusReader, List] = None,
) -> Union[pd.Series, str]:
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    if isinstance(series, str):
        return " ".join(word for word in series.split() if word not in stop_words)

    return series.apply(
        lambda sentence: " ".join(
            word for word in sentence.split() if word not in stop_words
        )
    )


def tokenize_words(text: str, tokenizer: str = "word") -> npt.ArrayLike:

    if tokenizer not in ("word", "sentence"):
        raise ValueError(f"{tokenizer} must be one of (word, sentence)")

    tokens = {"word": word_tokenize(text), "sentence": sent_tokenize(text)}
    try:
        return tokens.get(tokenizer)
    except BaseException as err:
        print(f"Unexpected err: {err}, Type: {type(err)}")
        raise


def remove_non_alnum(word_arr: Union[pd.Series, str]) -> Union[pd.Series, str]:
    if isinstance(word_arr, str):
        return " ".join(word for word in word_arr.split() if word.isalnum())
    return word_arr.apply(lambda arr: [word for word in arr if word.isalnum()])


def stemming(word_arr: npt.ArrayLike, stemmer: Any = None) -> npt.ArrayLike:
    if stemmer is None:
        stemmer = PorterStemmer()
    try:
        return [stemmer.stem(word) for word in word_arr]
    except BaseException as err:
        print(f"Unexpected err: {err}, Type: {type(err)}")
        raise


def lemmatizer(word_arr: npt.ArrayLike, lemmatizer: Any = None) -> npt.ArrayLike:
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    try:
        return [lemmatizer.lemmatize(word) for word in word_arr]
    except BaseException as err:
        print(f"Unexpected err: {err}, Type: {type(err)}")
        raise


if __name__ == "__main__":
    pass
