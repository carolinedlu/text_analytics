# --------
# PACKAGES
# --------

import time

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from text_analytics.config import RAW_DATA_PATH, SENTIMENT_CLEANED_DATA_PATH
from text_analytics.preprocessing import (
    convert_lowercase,
    remove_html_tags,
    remove_stopwords,
)
from text_analytics.text_summariser import ExtractiveTextSummarizer

st.set_page_config(layout="wide")


# -------
# BACKEND
# -------


@st.cache
def read_in_data():

    raw_data = pd.read_csv(RAW_DATA_PATH)
    cleaned_data = pd.read_csv(SENTIMENT_CLEANED_DATA_PATH)

    return raw_data, cleaned_data


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


def sentiment_analysis(input_text: str):
    pass


def text_summarisation(input_text: str) -> str:
    extractive_summarizer = ExtractiveTextSummarizer(article=input_text)
    result = extractive_summarizer.run_article_summary()

    return result


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
        sentiment_analysis_button = st.radio(
            "Sentiment Analyser to use",
            ["Logistic Regression", "Random Forest", "LSTM", "Pre-trained BERT"],
        )
        message = st.text_area(
            "Enter review to perform Sentiment Analysis on", "Text Here", height=200
        )

        if st.button("Analyse"):
            with st.spinner("Calculating sentiment..."):
                sentiment_result = "placeholder"
                time.sleep(2)
            st.success(f"Review sentiment: {sentiment_result}")

    with summarise:

        st.markdown("#### Perform Text Summarisation")
        summariser = st.radio("Text summariser to use", ["Extractive", "Abstractive"])

        message = st.text_area(
            "Enter review to perform Text Summarisation on", "Text Here", height=200
        )

        if st.button("Summarise"):
            with st.spinner("Summarising..."):
                summarise_result = text_summarisation(message)
                time.sleep(2)
            st.success(f"Review summary: {summarise_result}")

    with eda:

        st.markdown("#### Our data is from: ")
        st.markdown("Raw data here")

        raw_data, cleaned_data = read_in_data()

        with st.expander("More info about data here", expanded=False):
            st.dataframe(data=raw_data, height=300)

        st.markdown("Post preprocessing")

        with st.container():
            st.dataframe(data=cleaned_data, width=3000, height=300)

        positive_sentiments = cleaned_data.loc[cleaned_data["class"] == 1, "length"]
        negative_sentiments = cleaned_data.loc[cleaned_data["class"] == 0, "length"]

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
