import os
from pathlib import Path

ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
DATA_PATH = ROOT_DIR / "data"
RAW_DATA_PATH = ROOT_DIR / "data" / "imdb_data.csv"
SENTIMENT_CLEANED_DATA_PATH = ROOT_DIR / "data" / "sentiment_cleaned_data.csv"

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
    """
    )


if __name__ == "__main__":
    print_config()
