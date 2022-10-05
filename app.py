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


def read_in_data():
    pass


def tokeniser(input_text: str, tokeniser: str) -> str:
    pass


def named_entity_recogniser(input_text: str) -> str:
    pass


def sentiment_analysis(input_text: str):
    pass


def text_summarisation(input_text: str) -> str:
    pass


# ---------
# FRONT END
# ---------


def main():

    hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
