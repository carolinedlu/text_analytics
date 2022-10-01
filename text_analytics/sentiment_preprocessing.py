import logging
import os
import re
import string
from typing import Any, List, Union

import nltk
import numpy as np
import numpy.typing as npt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from text_analytics.config import ACRONYMS, RAW_DATA_PATH, SENTIMENT_CLEANED_DATA_PATH


def convert_lowercase(series: pd.Series) -> pd.Series:
    return series.str.lower()


def remove_html_tags(series: pd.Series) -> pd.Series:
    return series.str.replace(pat=r"<.*?>", repl="", regex=True)


def remove_punctuation(series: pd.Series) -> pd.Series:
    import string

    punctuations = str.maketrans(dict.fromkeys(string.punctuation))
    return series.str.translate(punctuations)


def convert_abbreviations(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda sentence: " ".join(
            ACRONYMS.get(word) if word in ACRONYMS.keys() else word
            for word in sentence.split()
        )
    )


def remove_stopwords(
    series: pd.Series,
    stop_words: Union[nltk.corpus.reader.wordlist.WordListCorpusReader, List] = None,
) -> pd.Series:
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    return series.apply(
        lambda sentence: " ".join(
            word for word in sentence.split() if word not in stop_words
        )
    )


def tokenize_words(text: str, tokenizer: Any = "word") -> npt.ArrayLike:

    if tokenizer not in ("word", "sentence"):
        raise ValueError(f"{tokenizer} must be one of (word, sentence)")

    tokens = {"word": word_tokenize(text), "sentence": sent_tokenize(text)}
    try:
        return tokens.get(tokenizer)
    except BaseException as err:
        print(f"Unexpected err: {err}, Type: {type(err)}")
        raise


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


def sentiment_text_processing(series: pd.Series) -> pd.Series:
    """
    Takes in a string of text, then performs the following:
    - remove line break
    - Lowercase
    - remove punctuation
    - Remove stopwords
    - Stemming / Lemmatizing (seems stemming works better)
    Returns a list of the cleaned text
    """

    logger.info("Removing html tags")
    series = remove_html_tags(series)

    logger.info("Converting to lowercase")
    series = convert_lowercase(series)

    logger.info("Converting abbreviations")
    series = convert_abbreviations(series)

    logger.info("Remove stopwords")
    series = remove_stopwords(series)

    logger.info("Replacing film / movie")
    series = series.str.replace(pat=r"film|movie", repl="", regex=True)

    logger.info("Tokenizing words")
    series = series.apply(lambda sentence: tokenize_words(sentence, tokenizer="word"))

    logger.info("Stemming / Lemmatizing")
    series = series.apply(lambda arr: lemmatizer(arr))

    return series


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Preprocessing reviews")
    df.drop_duplicates(inplace=True)
    df["preprocessed_review"] = sentiment_text_processing(series=df["review"])
    df["length"] = df["preprocessed_review"].apply(len)
    df["class"] = np.where(df["sentiment"] == "positive", 1, 0)
    df.drop(columns=["review", "sentiment"], inplace=True)

    print(df.head())
    logger.info("Writing to project bucket")
    df.to_csv(SENTIMENT_CLEANED_DATA_PATH)
