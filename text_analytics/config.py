import os
from pathlib import Path

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit

# ------
# PATHS
# ------

ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
DATA_PATH = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_PATH / "imdb_data.parquet"
SENTIMENT_CLEANED_DATA_PATH = DATA_PATH / "sentiment_cleaned_data.parquet"
SUMMARISER_CLEANED_DATA_PATH = DATA_PATH / "summariser_cleaned_data.parquet"
MODEL_PATH = ROOT_DIR / "models"
ARTIFACTS_PATH = ROOT_DIR / "artifacts"
RANDOM_STATE = 123
N_JOBS = -1
BASE_SCORER = {"AUC": "roc_auc", "F_score": make_scorer(fbeta_score, beta=1)}
CV_SPLIT = StratifiedShuffleSplit(
    n_splits=5, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE
)


ACRONYMS = {
    "asap": "as soon as possible",
    "btw": "by the way",
    "diy": "do it yourself",
    "fb": "facebook",
    "fomo": "fear of missing out",
    "fyi": "for your information",
    "g2g": "got to go",
    "idk": "i don't know",
    "imo": "in my opinion",
    "irl": "in real life",
    "lmao": "laughing my ass off",
    "lmk": "let me know",
    "lol": "laugh out loud",
    "msg": "message",
    "noyb": "none of your business",
    "omg": "oh my god",
    "rofl": "rolling on the floor laughing",
    "smh": "shaking my head",
    "tmi": "too much information",
    "ttyl": "talk to you later",
    "wth": "what the hell",
    "yolo": "you only live once",
}


def print_config() -> None:
    print(
        f"""
    Model parameters
    ------------------
    ROOT_DIR: {ROOT_DIR}
    DATA_PATH: {DATA_PATH}
    RAW_DATA_PATH: {RAW_DATA_PATH}
    SENTIMENT_CLEANED_DATA_PATH: {SENTIMENT_CLEANED_DATA_PATH}
    MODEL_PATH: {MODEL_PATH}
    RANDOM_STATE: {RANDOM_STATE}
    N_JOBS: {N_JOBS}
    BASE_SCORER: {BASE_SCORER}
    """
    )


if __name__ == "__main__":
    print_config()
