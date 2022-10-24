# tensorflow related libraries
from typing import Optional

import tensorflow as tf
from text_analytics.config import MODEL_PATH
from transformers import DistilBertConfig, DistilBertTokenizerFast, TFDistilBertModel

MAX_LENGTH = 512
MAX_LENGTH_FULL = 128
CHUNK_SIZE = 256
LAYER_DROPOUT = 0.1
RANDOM_STATE = 2022
LEARNING_RATE = 1e-4
DISTILBERT_DROPOUT = 0.1
DISTILBERT_ATT_DROPOUT = 0.1
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE_FULL = 2e-5
EPOCHS_FULL = 1
BATCH_SIZE_FULL = 32
NUM_STEPS = 39665 // BATCH_SIZE  # length of X_train
NUM_STEPS_FULL = 39665 // BATCH_SIZE


class BertReviews:
    def __init__(self, bert_model, ids: Optional = None, attention: Optional = None):

        self.model = build_model(bert_model)
        self.model.load_weights(f"{MODEL_PATH}/model_weights/model_weights_final")
        self.ids = ids
        self.attention = attention

    def predict_value(self):

        return self.model.predict([self.ids, self.attention])


def build_model(transformer, max_length=MAX_LENGTH):
    """ "
    Build a model off of the DistilBERT architecture

    Input:
      - transformer:  a base Hugging Face transformer model object (DistilBERT or BERT-like)
      - max_length:   integer controlling the maximum number of encoded tokens
                      in a given sequence.

    Output:
      - model:        a compiled tf.keras.Model with added classification layers
                      on top of the base pre-trained model architecture.
    """

    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=RANDOM_STATE)

    # Define input layers
    input_ids_layer = tf.keras.layers.Input(
        shape=(max_length,), name="input_ids", dtype="int32"
    )
    input_attention_layer = tf.keras.layers.Input(
        shape=(max_length,), name="input_attention", dtype="int32"
    )

    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]

    # We only care about DistilBERT's output for the [CLS] token, which is located
    # at index 0.  Splicing out the [CLS] tokens gives us 2D data.
    cls_token = last_hidden_state[:, 0, :]

    D1 = tf.keras.layers.Dropout(LAYER_DROPOUT, seed=RANDOM_STATE)(cls_token)

    X = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_initializer=weight_initializer,
        bias_initializer="zeros",
    )(D1)

    D2 = tf.keras.layers.Dropout(LAYER_DROPOUT, seed=RANDOM_STATE)(X)

    X = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer=weight_initializer,
        bias_initializer="zeros",
    )(D2)

    D3 = tf.keras.layers.Dropout(LAYER_DROPOUT, seed=RANDOM_STATE)(X)

    # Define a single node that makes up the output layer (for binary classification)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=weight_initializer,
        bias_initializer="zeros",
    )(D3)

    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

    # Compile the model
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def fast_encode(tokenizer, texts, chunk_size=CHUNK_SIZE, maxlen=MAX_LENGTH):
    """
    Encodes a batch of texts and returns the corresponding encodings and attention masks that can be
    fed into a pre-trained transformer model
    """

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), chunk_size):
        batch = texts[i : i + chunk_size]
        inputs = tokenizer.batch_encode_plus(
            batch,
            max_length=maxlen,
            padding="max_length",  # implements dynamic padding
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        input_ids.extend(inputs["input_ids"])
        attention_mask.extend(inputs["attention_mask"])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def get_sentiment_bert_finetuned():
    # Configure DistilBERT's initialization
    config = DistilBertConfig(
        dropout=DISTILBERT_DROPOUT,
        attention_dropout=DISTILBERT_ATT_DROPOUT,
        output_hidden_states=True,
    )

    distilBERT = TFDistilBertModel.from_pretrained(
        "distilbert-base-uncased", config=config
    )

    model = BertReviews(bert_model=distilBERT)

    return model


if __name__ == "__main__":

    texts = ["Really love this movie!"]

    sentiment_bert_model = get_sentiment_bert_finetuned()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    ids, attention = fast_encode(tokenizer=tokenizer, texts=texts)
    sentiment_bert_model.ids = ids
    sentiment_bert_model.attention = attention
    predictions = sentiment_bert_model.predict_value()

    output = ["Positive" if score[0] > 0.5 else "Negative" for score in predictions]
    print(f"Prediction: {output}")
