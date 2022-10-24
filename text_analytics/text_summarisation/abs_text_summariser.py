# importing libraries
from collections import defaultdict
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge import Rouge
from text_analytics.config import SUMMARISER_CLEANED_DATA_PATH

# pip install transformers
from transformers import pieline
import warnings
warnings.filterwarnings('ignore')


class AbstractiveTextSummarizer:
    def __init__(self, summarizer1, summarizer2) -> None:
        self.article = article
        self.frequency_table = defaultdict(int)
        self.summarizer1 = summarizer1
        self.summarizer2 = summarizer2

    def run_article_summary(self, article):
        intermediate_output = self.summarizer1(article)[0]['summary_text']
        output = self.summarizer2(intermediate_output, min_length=25, max_length=60)[0]['summary_text']
        return output

        return article_summary

    def get_rouge_score(
        self, hypothesis_text: str, reference_text: str
    ) -> npt.ArrayLike:
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis_text, reference_text)
        return scores


if __name__ == "__main__":
    summarizer1 = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    summarizer2 = pipeline("summarization", model="facebook/bart-large-cnn")
    abstracive_summarizer = AbstractiveTextSummarizer(summarizer1, summarizer2)

    df = pd.read_parquet(SUMMARISER_CLEANED_DATA_PATH)
    result = []
    articles = df.loc[:, "cleaned_reviews"].sample(3).values

    for review in articles:
        print(f"Original Review: \n{review}")
        print("-" * 200)
        review_summary = abstractive_summarizer.run_article_summary(review)
        result.append(review_summary)

        print(f"Summarised Review: \n{review_summary}")
        print("-" * 200)

        # this line is POC for now since we don't have the reference text
        print(
            extractive_summarizer.get_rouge_score(
                hypothesis_text=review_summary, reference_text=review_summary
            )
        )