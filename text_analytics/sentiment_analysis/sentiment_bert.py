########################
# VERSION NOT FUNCTIONAL
########################

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model, load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from text_analytics.config import DATA_PATH, SENTIMENT_CLEANED_DATA_PATH
from tokenizers import BertWordPieceTokenizer
from transformers import DistilBertTokenizer, TFDistilBertModel

warnings.filterwarnings("ignore")


class SentimentBert:
    def __init__(self):
        self.model = None
        # First load the real tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", lower=True
        )

        # bert-base

        # Save the loaded tokenizer locally
        self.tokenizer.save_pretrained(".")
        # Reload it with the huggingface tokenizers library
        self.fast_tokenizer = BertWordPieceTokenizer("vocab.txt", lowercase=True)
        self.bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

        # get_data
        train = pd.read_parquet(DATA_PATH / "sentiment_train.parquet")
        test = pd.read_parquet(DATA_PATH / "sentiment_test.parquet")

        X_train, y_train = train["preprocessed_review"], train["class"]
        X_test, y_test = test["preprocessed_review"], test["class"]

        self.y_train = y_train
        self.y_test = y_test

        self.x_train = self.fast_encode(X_train.values, self.fast_tokenizer, maxlen=400)

        self.x_test = self.fast_encode(X_test.values, self.fast_tokenizer, maxlen=400)

    def fast_encode(self, texts, tokenizer, chunk_size=256, maxlen=400):

        tokenizer.enable_truncation(max_length=maxlen)
        tokenizer.enable_padding(length=maxlen)
        all_ids = []

        for i in range(0, len(texts), chunk_size):
            text_chunk = texts[i : i + chunk_size].tolist()
            encs = tokenizer.encode_batch(text_chunk)
            all_ids.extend([enc.ids for enc in encs])

        return np.array(all_ids)

    def build_model(self, transformer, max_len=400):

        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        sequence_output = transformer(input_word_ids)[0]
        cls_token = sequence_output[:, 0, :]
        out = Dense(1, activation="sigmoid")(cls_token)

        model = Model(inputs=input_word_ids, outputs=out)
        model.compile(Adam(lr=2e-5), loss="binary_crossentropy", metrics=["accuracy"])

        return model

    def load_model(self):
        self.model = load_model(
            "/content/drive/MyDrive/ISSS609/bert_base_01.h5",
            custom_objects={"TFDistilBertModel": self.bert_model},
        )

    def evaluate(self):
        print(
            "Accuracy of the model on Testing Data is - ",
            self.model.evaluate(self.x_test, self.y_test)[1] * 100,
            "%",
        )
        y_pred = self.model.predict(self.x_test)
        y_pred = np.round(pred).astype(int)
        print(
            classification_report(
                self.y_test, y_pred, target_names=["Negative", "Positive"]
            )
        )

    def test_single(self, review):
        review = self.fast_encode(np.array([review]), self.fast_tokenizer, maxlen=400)
        prediction = np.round(self.model.predict(review)).astype(int)

        return "Negative" if prediction == 0 else "Positive"


if __name__ == "__main__":

    movie_reviews = pd.read_csv(SENTIMENT_CLEANED_DATA_PATH)
    movie_reviews.head()

    sentiment_bert = SentimentBert()
    sentiment_bert.load_model()
    # sentiment_bert.evaluate() need GPU, CPU take too long time

    record = movie_reviews.sample(1)
    article = record["REVIEW"].values[0]
    rating = record["RATING"].values[0]

    print("Original Review: ")
    print(article)

    print("====================================================")
    print("RATING:", rating)

    print("====================================================")
    print(sentiment_bert.test_single(article))

    """
    summaraizer = Ext_text_summarizer(article)
    result = summaraizer._run_article_summary(summaraizer.para)
    print(result)
    print('====================================================')
    print(summaraizer._get_rouge_score(article, result))
    """
