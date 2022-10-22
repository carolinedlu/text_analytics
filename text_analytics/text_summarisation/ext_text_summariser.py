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


class ExtractiveTextSummarizer:
    def __init__(self, article: Union[str, pd.DataFrame]) -> None:
        self.article = article
        self.frequency_table = defaultdict(int)

    def _create_dictionary_table(self, stemmer: Any = None) -> dict:

        # removing stop words
        stop_words = set(stopwords.words("english"))
        word_vector = word_tokenize(self.article)

        # instantiate the stemmer
        if stemmer is None:
            stemmer = PorterStemmer()

        stemmed_word_vector = [stemmer.stem(word) for word in word_vector]
        for word in stemmed_word_vector:
            if word not in stop_words:
                self.frequency_table[word] += 1

        return self.frequency_table

    def _calculate_sentence_scores(self, sentences: npt.ArrayLike) -> dict:

        # algorithm for scoring a sentence by its words
        sentence_weights = defaultdict(int)

        for sentence in sentences:
            sentence_wordcount_without_stop_words = 0

            for word_weight in self.frequency_table:
                sentence_weights[sentence[:7]] += self.frequency_table[word_weight]

                if word_weight in sentence.lower():
                    sentence_wordcount_without_stop_words += 1

            sentence_weights[sentence[:7]] /= sentence_wordcount_without_stop_words

        return sentence_weights

    def _calculate_threshold_score(self, sentence_weight: dict) -> float:
        return np.mean(list(sentence_weight.values()))

    def _get_article_summary(
        self, sentences: npt.ArrayLike, sentence_weights: dict, threshold: float
    ) -> str:
        article_summary = [
            sentence
            for sentence in sentences
            if sentence[:7] in sentence_weights
            and sentence_weights.get(sentence[:7]) >= threshold
        ]

        return " ".join(article_summary)

    def run_article_summary(self):

        # creating a dictionary for the word frequency table
        _ = self._create_dictionary_table()

        # tokenizing the sentences
        sentences = sent_tokenize(self.article)

        # algorithm for scoring a sentence by its words
        sentence_scores = self._calculate_sentence_scores(sentences)

        # getting the threshold
        threshold = self._calculate_threshold_score(sentence_scores)

        # producing the summary
        article_summary = self._get_article_summary(
            sentences, sentence_scores, 0.95 * threshold
        )

        return article_summary

    def get_rouge_score(
        self, hypothesis_text: str, reference_text: str
    ) -> npt.ArrayLike:
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis_text, reference_text)
        return scores


if __name__ == "__main__":
    df = pd.read_parquet(SUMMARISER_CLEANED_DATA_PATH)

    result = []
    articles = df.loc[:, "cleaned_reviews"].sample(3).values

    for review in articles:
        print(f"Original Review: \n{review}")
        print("-" * 200)
        extractive_summarizer = ExtractiveTextSummarizer(article=review)
        review_summary = extractive_summarizer.run_article_summary()
        result.append(review_summary)

        print(f"Summarised Review: \n{review_summary}")
        print("-" * 200)

        # this line is POC for now since we don't have the reference text
        print(
            extractive_summarizer.get_rouge_score(
                hypothesis_text=review_summary, reference_text=review_summary
            )
        )
