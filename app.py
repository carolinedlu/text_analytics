# --------
# PACKAGES
# --------

import time
from random import Random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
import transformers
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import DistilBertTokenizerFast

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
from text_analytics.sentiment_analysis.random_forest import RandomForestReviews
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


@st.cache(ttl=120)
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


@st.cache
def convert_df(df: pd.DataFrame):
    return df.to_csv().encode("utf-8")


@st.cache(ttl=120)
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


@st.cache(ttl=120)
def sentiment_column_preprocessing(input_series: pd.Series) -> pd.Series:

    article_transformed = remove_html_tags(input_series)
    article_transformed = convert_lowercase(article_transformed)
    article_transformed = convert_abbreviations(article_transformed)
    article_transformed = remove_stopwords(article_transformed)
    article_transformed = article_transformed.apply(
        lambda review: tokenize_words(review, tokenizer="word")
    )
    article_transformed = remove_punctuation(article_transformed)
    article_transformed = article_transformed.apply(lambda review: stemming(review))
    article_transformed = remove_non_alnum(article_transformed)
    article_transformed = article_transformed.apply(lambda review: " ".join(review))
    return article_transformed


@st.cache(allow_output_mutation=True)
def load_bert_model():
    return get_sentiment_bert_finetuned()


@st.cache(allow_output_mutation=True)
def load_bert_tokeniser():
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


@st.cache(allow_output_mutation=True)
def load_lr_model():
    lr = LogisticRegressionReviews()
    lr.load_model(file_name="log_reg_best")
    return lr


@st.cache(allow_output_mutation=True)
def load_nb_model():
    nb = NaiveBayesReviews()
    nb.load_model(file_name="naive_bayes_best")
    return nb


@st.cache(allow_output_mutation=True)
def load_rf_model():
    rf = RandomForestReviews()
    rf.load_model(file_name="random_forest_best")
    return rf


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_ext_text_summariser():
    extractive_summarizer = ExtractiveTextSummarizer(article="")
    return extractive_summarizer


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_abs_text_summariser():
    rough_summariser = transformers.pipeline(
        "summarization", model="t5-base", tokenizer="t5-base", framework="tf"
    )
    refine_summariser = transformers.pipeline(
        "summarization", model="facebook/bart-large-cnn"
    )

    abstractive_summariser = AbstractiveTextSummarizer(
        rough_summariser=rough_summariser, refine_summariser=refine_summariser
    )
    return abstractive_summariser


# ---------
# FRONT END
# ---------


def main():

    with st.container():
        st.markdown("## Group 4 - Text Analytics with Movie Reviews")

        with st.expander("ℹ️ - About this app", expanded=False):

            st.write(
                """
                Sub-caption
                """
            )
        st.subheader("NLP Processing Tasks: ")

    tokenise, sentiment, summarise, upload_csv_file, eda = st.tabs(
        [
            "Tokenise Text",
            "Sentiment Analysis",
            "Text Summarisation",
            "Upload File",
            "Source Data",
        ]
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

                if sentiment_analysis_choice == "Random Forest":

                    rf = load_rf_model()
                    article_transformed = sentiment_preprocessing(input_text=message)

                    prediction, tokens = rf.predict_single_review(
                        article=[article_transformed]
                    )

                elif sentiment_analysis_choice == "Logistic Regression":
                    lr = load_lr_model()
                    article_transformed = sentiment_preprocessing(input_text=message)
                    prediction, tokens = lr.predict_single_review(
                        article=[article_transformed]
                    )

                elif sentiment_analysis_choice == "Naive Bayes":
                    nb = load_nb_model()
                    article_transformed = sentiment_preprocessing(input_text=message)
                    prediction, tokens = nb.predict_single_review(
                        article=[article_transformed]
                    )

                else:
                    sentiment_bert_model = load_bert_model()
                    tokenizer = load_bert_tokeniser()

                    ids, attention = fast_encode(tokenizer=tokenizer, texts=[message])
                    sentiment_bert_model.ids = ids
                    sentiment_bert_model.attention = attention
                    predictions = sentiment_bert_model.predict_value()
                    prediction = [
                        "Positive" if score[0] > 0.5 else "Negative"
                        for score in predictions
                    ][0]
                    tokens = ["-"]

                token_disp = " / ".join(tokens)
                time.sleep(1)
            st.success(f"Review sentiment: \n{prediction}")
            st.success(f"Tokens: {token_disp}")

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

                if summariser_choice == "Extractive":
                    ext_summariser = load_ext_text_summariser()
                    ext_summariser.article = message
                    summarise_result = ext_summariser.run_article_summary()
                else:
                    abstractive_summariser = load_abs_text_summariser()
                    summarise_result = abstractive_summariser.run_article_summary(
                        message
                    )
                time.sleep(2)
            st.success(f"Review summary: \n{summarise_result}")

    with upload_csv_file:
        st.markdown("#### Upload a CSV and perform a text analytics task")
        task_choice = st.radio(
            "Task to perform", ["Sentiment Analysis", "Text Summarisation"]
        )

        file_uploaded = st.file_uploader(
            label="File to upload", accept_multiple_files=False
        )

        if st.button("Analyse reviews") and file_uploaded:
            with st.spinner("Analysing..."):
                if task_choice == "Sentiment Analysis":
                    df_reviews = pd.read_csv(file_uploaded, header=0, index_col=None)

                    st.write(df_reviews)
                    processed_series = sentiment_column_preprocessing(
                        df_reviews.loc[:, "review"]
                    )

                    lr = load_lr_model()
                    sentiment_result = lr.best_model.predict(processed_series)
                    df_reviews["sentiment_result"] = sentiment_result

                    csv = convert_df(df_reviews)
                    if st.download_button(
                        label="Download file",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    ):
                        st.write("Download successful")

                else:
                    df_reviews = pd.read_csv(file_uploaded, header=0, index_col=None)
                    st.write(df_reviews)

                    summarised_results = []
                    for review in df_reviews["review"].values:
                        ext_summariser = load_ext_text_summariser()
                        ext_summariser.article = review

                        _result = ext_summariser.run_article_summary()
                        summarised_results.append(_result)

                    st.write(summarised_results)
                    df_reviews["summarised_result"] = summarised_results
                    csv = convert_df(df_reviews)
                    if st.download_button(
                        label="Download file",
                        data=csv,
                        file_name="summarised_results.csv",
                        mime="text/csv",
                    ):
                        st.write("Download successful")

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
