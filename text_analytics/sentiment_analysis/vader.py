import logging
from typing import Union

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from text_analytics.config import RAW_DATA_PATH
from text_analytics.preprocessing import (
    convert_abbreviations,
    convert_lowercase,
    remove_html_tags,
    remove_stopwords,
)


class VaderReviews:
    def __init__(self, data: Union[str, pd.DataFrame]) -> None:

        try:
            self.model = SentimentIntensityAnalyzer()
        except BaseException:
            nltk.download("vader_lexicon")
            self.model = SentimentIntensityAnalyzer()

        self.data = data
        self.polarity_scores = self.compound_score = self.prediction = None

    def calculate_polarity_score(self) -> None:

        if isinstance(self.data, str):
            self.polarity_scores = self.model.polarity_scores(self.data)
        else:
            self.polarity_scores = self.data["review"].apply(
                lambda sentence: self.model.polarity_scores(sentence)
            )

    def extract_compound_score(self) -> None:
        if isinstance(self.data, str):
            self.compound_score = self.polarity_scores.get("compound")
        else:
            self.compound_score = self.polarity_scores.apply(
                lambda score: score.get("compound")
            )

    def extract_prediction(self) -> None:
        if isinstance(self.data, str):
            self.prediction = "positive" if self.compound_score > 0 else "negative"
        else:
            self.prediction = self.compound_score.apply(
                lambda c_score: "positive" if c_score > 0 else "negative"
            )

    def return_vader_scores(self) -> None:

        if self.polarity_scores is None:
            self.calculate_polarity_score()
        elif self.compound_score is None:
            self.extract_compound_score()
        elif self.prediction is None:
            self.extract_prediction()
        if isinstance(self.data, str):
            return (self.compound_score, self.prediction)

        result = pd.concat(
            [self.data, self.compound_score, self.prediction], axis="columns"
        )
        result.columns = ["review", "sentiment", "compound_score", "prediction"]
        print(result.head())


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()

    logger.info(f"Reading in file: {RAW_DATA_PATH}")
    reviews = pd.read_parquet(RAW_DATA_PATH)

    logger.info("Preprocessing file")
    reviews["review"] = remove_html_tags(reviews["review"])
    reviews["review"] = convert_lowercase(reviews["review"])
    reviews["review"] = convert_abbreviations(reviews["review"])
    reviews["review"] = remove_stopwords(reviews["review"])

    logger.info("Calculating sentiment score")
    vr = VaderReviews(data=reviews)

    vr.calculate_polarity_score()
    vr.extract_compound_score()
    vr.extract_prediction()
    vr.return_vader_scores()
    print("done")
