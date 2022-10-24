# importing libraries
import warnings

import numpy.typing as npt
import pandas as pd
from rouge import Rouge
from text_analytics.config import DATA_PATH
from transformers import pipeline

warnings.filterwarnings("ignore")


class AbstractiveTextSummarizer:
    def __init__(
        self, rough_summariser: pipeline = None, refine_summariser: pipeline = None
    ) -> None:
        if rough_summariser is None:
            rough_summariser = pipeline(
                "summarization", model="t5-base", tokenizer="t5-base", framework="tf"
            )
        if refine_summariser is None:
            refine_summariser = pipeline(
                "summarization", model="facebook/bart-large-cnn"
            )

        self.rough_summariser = rough_summariser
        self.refine_summariser = refine_summariser
        self.rouge = Rouge()

    def run_article_summary(self, article: str) -> str:
        intermediate_output = self.rough_summariser(
            article, min_length=60, max_length=200
        )[0]["summary_text"]
        output = self.refine_summariser(
            intermediate_output, min_length=25, max_length=60
        )[0]["summary_text"]
        return output

    def get_rouge_score(
        self, hypothesis_text: str, reference_text: str
    ) -> npt.ArrayLike:
        scores = self.rouge.get_scores(hypothesis_text, reference_text)
        return scores


if __name__ == "__main__":

    rough_summariser = pipeline(
        "summarization", model="t5-base", tokenizer="t5-base", framework="tf"
    )

    refine_summariser = pipeline("summarization", model="facebook/bart-large-cnn")
    abstractive_summarizer = AbstractiveTextSummarizer(
        rough_summariser=rough_summariser, refine_summariser=refine_summariser
    )

    df = pd.read_csv(DATA_PATH / "review_evaluation.csv")
    result = []
    articles = df.loc[:, "review"]
    reference_summary = df.loc[:, "Summary"]

    for review, reference_text in zip(articles, reference_summary):
        print(f"Original Review: \n{review}")
        print("-" * 200)
        review_summary = abstractive_summarizer.run_article_summary(review)
        rouge_score = abstractive_summarizer.get_rouge_score(
            hypothesis_text=review_summary, reference_text=reference_text
        )
        result.append(review_summary)

        print(f"Summarised Review: \n{review_summary}")
        print("-" * 200)
        print(rouge_score)
