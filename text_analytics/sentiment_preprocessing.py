import logging

import numpy as np
import pandas as pd

from text_analytics.config import RAW_DATA_PATH, SENTIMENT_CLEANED_DATA_PATH
from text_analytics.preprocessing import (
    convert_abbreviations,
    convert_lowercase,
    lemmatizer,
    remove_html_tags,
    remove_non_alnum,
    remove_punctuation,
    remove_stopwords,
    tokenize_words,
)


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

    logger.info("Remove punctuation")
    series = remove_punctuation(series)

    logger.info("Stemming / Lemmatizing")
    series = series.apply(lambda arr: lemmatizer(arr))

    logger.info("Remove non-alphabetic/numeric")
    series = remove_non_alnum(series)

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
