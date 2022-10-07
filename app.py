# --------
# PACKAGES
# --------

import numpy as np
import pandas as pd
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
            st.write("""Sub-caption""")

        st.subheader("NLP Processing Tasks: ")

    tokenise, sentiment, summarise, eda = st.tabs(
        ["Tokenise Text", "Sentiment Analysis", "Text Summarisation", "Source Data"]
    )

    with tokenise:
        pass

    with sentiment:
        pass

    with summarise:
        pass

    with eda:
        pass

    hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
