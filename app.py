# --------
# PACKAGES
# --------

import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import DistilBertTokenizerFast, pipeline

from text_analytics.config import (
    RAW_DATA_PATH,
    SENTIMENT_CLEANED_DATA_PATH,
    SUMMARISER_CLEANED_DATA_PATH,
)
from text_analytics.preprocessing import (
    convert_abbreviations,
    convert_lowercase,
    remove_html_tags,
    remove_non_alnum,
    remove_punctuation,
    remove_stopwords,
    stemming,
    tokenize_words,
)
from text_analytics.sentiment_analysis.bert_finetuned import (
    fast_encode,
    get_sentiment_bert_finetuned,
)
from text_analytics.sentiment_analysis.logistic_regression import (
    LogisticRegressionReviews,
)
from text_analytics.sentiment_analysis.naive_bayes import NaiveBayesReviews
from text_analytics.text_summarisation.abs_text_summariser import (
    AbstractiveTextSummarizer,
)
from text_analytics.text_summarisation.ext_text_summariser import (
    ExtractiveTextSummarizer,
)

st.set_page_config(layout="wide")


# -------
# BACKEND
# -------


@st.cache
def read_in_data():

    raw_data = pd.read_parquet(RAW_DATA_PATH)
    sentiment_cleaned_data = pd.read_parquet(SENTIMENT_CLEANED_DATA_PATH)
    summariser_cleaned_data = pd.read_parquet(SUMMARISER_CLEANED_DATA_PATH)

    return raw_data, sentiment_cleaned_data, summariser_cleaned_data


@st.cache
def tokeniser(input_text: str, tokeniser: str) -> str:

    input_text = convert_lowercase(input_text)
    input_text = remove_html_tags(input_text)
    input_text = remove_stopwords(input_text)
    print(tokeniser)

    if tokeniser == "Word":
        return word_tokenize(input_text)

    return sent_tokenize(input_text)


@st.cache
def named_entity_recogniser(input_text: str) -> str:
    pass


@st.cache(suppress_st_warning=True)
def load_in_bert():
    sentiment_bert_model = get_sentiment_bert_finetuned()
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    return sentiment_bert_model, tokenizer


def sentiment_preprocessing(input_text: str) -> str:

    article_transformed = remove_html_tags(pd.Series([input_text]))
    article_transformed = convert_lowercase(article_transformed)
    article_transformed = convert_abbreviations(article_transformed)
    article_transformed = remove_stopwords(article_transformed)
    article_transformed = article_transformed.apply(
        lambda review: tokenize_words(review, tokenizer="word")
    )
    article_transformed = remove_punctuation(article_transformed)
    article_transformed = article_transformed.apply(lambda review: stemming(review))
    article_transformed = remove_non_alnum(article_transformed)
    article_transformed = " ".join(article_transformed[0])
    return article_transformed


def sentiment_analysis(input_text: str, model_to_use: str) -> Tuple[str]:

    if model_to_use == "Pre-trained BERT":
        sentiment_bert_model, tokenizer = load_in_bert()
        ids, attention = fast_encode(tokenizer=tokenizer, texts=[input_text])
        sentiment_bert_model.ids = ids
        sentiment_bert_model.attention = attention
        predictions = sentiment_bert_model.predict_value()
        output = ["Positive" if score[0] > 0.5 else "Negative" for score in predictions]
        return output

    article_transformed = sentiment_preprocessing(input_text)
    if model_to_use == "Logistic Regression":
        model = LogisticRegressionReviews()
        model.load_model(file_name="log_reg_best")
    if model_to_use == "Random Forest":
        model = RandomForestReviews()
        model.load_model(file_name="random_forest_best")
    if model_to_use == "Naive Bayes":
        model = NaiveBayesReviews()
        model.load_model(file_name="naive_bayes_best")

    prediction, token_scores = model.predict_single_review(
        article=[article_transformed]
    )
    return prediction, token_scores, f"{model.best_model[1]}"


def text_summarisation(input_text: str, model_to_use: str) -> str:
    if model_to_use == "Extractive":
        extractive_summarizer = ExtractiveTextSummarizer(article=input_text)
        result = extractive_summarizer.run_article_summary()

        return result

    if model_to_use == "Abstractive":
        abstractive_summarizer = AbstractiveTextSummarizer(
            rough_summariser=rough_summariser, refine_summariser=refine_summariser
        )
        input_text_summary = abstractive_summarizer.run_article_summary(input_text)
        return input_text_summary
    return


# ---------
# FRONT END
# ---------


def main():

    load_in_bert()
    rough_summariser = pipeline(
        "summarization", model="t5-base", tokenizer="t5-base", framework="tf"
    )
    refine_summariser = pipeline("summarization", model="facebook/bart-large-cnn")

    with st.container():
        st.markdown("## Group 4 - Text Analytics with Movie Reviews")

        with st.expander("ℹ️ - About this app", expanded=False):

            st.write(
                """
                Sub-caption
                """
            )
        st.subheader("NLP Processing Tasks: ")

    tokenise, sentiment, summarise, eda = st.tabs(
        ["Tokenise Text", "Sentiment Analysis", "Text Summarisation", "Source Data"]
    )

    with tokenise:

        st.markdown("#### Text Tokeniser")
        test_button = st.radio("Tokeniser to use", ["Word", "Sentence"])
        message = st.text_area("Enter review to tokenize", "Text Here", height=200)

        if st.button("Tokenise"):
            with st.spinner("Tokenising..."):
                token_result = tokeniser(message, test_button)
                time.sleep(2)
            st.success(f"Tokens: {token_result}")

    with sentiment:

        st.markdown("#### Perform Sentiment Analysis")
        sentiment_analysis_choice = st.radio(
            "Sentiment Analyser to use",
            ["Logistic Regression", "Random Forest", "Naive Bayes", "Pre-trained BERT"],
        )
        message = st.text_area(
            "Enter review to perform Sentiment Analysis on", "Text Here", height=200
        )

        if st.button("Analyse"):
            with st.spinner("Calculating sentiment..."):
                sentiment_result = sentiment_analysis(
                    input_text=message, model_to_use=sentiment_analysis_choice
                )
                time.sleep(2)
            st.success(f"Review sentiment: \n{sentiment_result}")

    with summarise:

        st.markdown("#### Perform Text Summarisation")
        summariser_choice = st.radio(
            "Text summariser to use", ["Extractive", "Abstractive"]
        )

        message = st.text_area(
            "Enter review to perform Text Summarisation on", "Text Here", height=200
        )

        if st.button("Summarise"):
            with st.spinner("Summarising..."):
                summarise_result = text_summarisation(
                    input_text=message, model_to_use=summariser_choice
                )
                time.sleep(2)
            st.success(f"Review summary: \n{summarise_result}")

    with eda:

        st.markdown("#### Our data is from: ")
        st.markdown("Raw data here")

        raw_data, sentiment_cleaned_data, _ = read_in_data()

        with st.expander("More info about data here", expanded=False):
            st.dataframe(data=raw_data, height=300)

        st.markdown("Post preprocessing")

        with st.container():
            st.dataframe(data=sentiment_cleaned_data, width=3000, height=300)

        positive_sentiments = sentiment_cleaned_data.loc[
            sentiment_cleaned_data["class"] == 1, "length"
        ]
        negative_sentiments = sentiment_cleaned_data.loc[
            sentiment_cleaned_data["class"] == 0, "length"
        ]

        # Plot histogram
        hist_data = [positive_sentiments, negative_sentiments]

        # create distplot
        fig = ff.create_distplot(
            hist_data, group_labels=["Positive", "Negative"], bin_size=[0.1, 0.25, 0.5]
        )

        st.plotly_chart(fig, use_container_width=True)

    hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
