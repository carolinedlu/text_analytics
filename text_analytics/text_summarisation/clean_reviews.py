import logging

import pandas as pd
from text_analytics.config import RAW_DATA_PATH, SUMMARISER_CLEANED_DATA_PATH
from text_analytics.preprocessing import convert_abbreviations, remove_html_tags


def summariser_text_processing(series: pd.Series) -> pd.Series:

    logger.info("Removing html tags")
    series = remove_html_tags(series)

    logger.info("Converting abbreviations")
    series = convert_abbreviations(series)

    return series


if __name__ == "__main__":
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()

    df = pd.read_parquet(RAW_DATA_PATH)

    logger.info("Preprocessing reviews")
    df.drop_duplicates(inplace=True)
    df["cleaned_reviews"] = summariser_text_processing(series=df["review"])
    df["cleaned_reviews"] = df["cleaned_reviews"].astype(str)
    df.drop(columns=["review", "sentiment"], inplace=True)

    print(df.head())
    logger.info("Writing to project bucket")
    df.to_parquet(SUMMARISER_CLEANED_DATA_PATH)
