import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from text_analytics.config import DATA_PATH, RANDOM_STATE, SENTIMENT_CLEANED_DATA_PATH

if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()

    logger.info("Reading in file")

    sentiment_bow = pd.read_csv(SENTIMENT_CLEANED_DATA_PATH)

    logger.info("Train test splitting")
    train, test = train_test_split(
        sentiment_bow,
        test_size=0.2,
        stratify=sentiment_bow["class"],
        random_state=RANDOM_STATE,
    )

    logger.info("Writing to project bucket")

    train.to_parquet(DATA_PATH / "sentiment_train.parquet")
    test.to_parquet(DATA_PATH / "sentiment_test.parquet")
