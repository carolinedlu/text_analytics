import argparse
import logging
import pickle
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from text_analytics.config import (
    BASE_SCORER,
    CV_SPLIT,
    DATA_PATH,
    MODEL_PATH,
    N_JOBS,
    RANDOM_STATE,
)
from text_analytics.helpers import (
    calculate_report_metrics,
    evaluate_tuning,
    save_confusion_matrix,
    save_roc_curve,
)

warnings.filterwarnings("ignore")


class LogisticRegressionReviews:
    def __init__(
        self,
        X_train: pd.DataFrame = None,
        y_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
        model: LogisticRegression = None,
        scorer: dict = BASE_SCORER,
        cv_split: StratifiedShuffleSplit = CV_SPLIT,
        vectoriser: Optional = None,
        hpt: Optional = None,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scorer = scorer
        self.cv_split = cv_split

        if model is None:
            model = LogisticRegression(
                solver="liblinear",
                penalty="none",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                max_iter=2000,
                warm_start=True,
            )

        self.model = model
        self.trainer = None
        self.best_model = None

        if vectoriser is None:
            vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=0.005)

        if hpt is None:
            hpt = {
                # regularization param: higher C = less regularization
                "log_reg__C": [0.01, 0.1, 1, 10],
                # specifies kernel type to be used
                "log_reg__penalty": ["l1", "l2"],
            }

        self.hpt = hpt
        self.vec = vectoriser

        self.timestamp = datetime.now().strftime("%d_%H_%M")
        self.file_name = f"sent_logreg_{self.timestamp}"

    def save_model(self) -> None:
        save_path = MODEL_PATH / f"{self.file_name}.pkl"

        with open(save_path, "wb") as file:
            pickle.dump(self.best_model, file)

    def load_model(self, file_name: str) -> None:

        model_path = MODEL_PATH / f"{file_name}.pkl"
        with open(model_path, "rb") as file:
            self.best_model = pickle.load(file)

    def train(self):

        # build the pipeline
        log_reg_pipe = Pipeline([("vec", self.vec), ("log_reg", self.model)])

        # cross validate model with GridSearch
        self.trainer = GridSearchCV(
            estimator=log_reg_pipe,
            param_grid=self.hpt,
            scoring=self.scorer,
            refit="F_score",
            cv=self.cv_split,
            return_train_score=True,
            verbose=10,
        )

        self.trainer.fit(self.X_train, self.y_train)

        self.best_model = self.trainer.best_estimator_

        return self.trainer, self.best_model

    def evaluate(self) -> None:
        try:
            evaluate_tuning(tuner=self.trainer)
        except AttributeError as err:
            print(
                f"{type(err)}: Fitting must be done before evaluating of hyperparameter tuning process"
            )
            raise

        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        report = calculate_report_metrics(
            y_test=self.y_test, y_pred=y_pred, y_pred_prob=y_pred_proba
        )

        save_confusion_matrix(cf_matrix=report["cf_matrix"], model_name=self.file_name)

        save_roc_curve(
            fpr=report.get("roc")[0],
            tpr=report.get("roc")[1],
            model_name=self.file_name,
            auc=report.get("auroc"),
        )

    def predict_csv(self, csv_to_predict: pd.DataFrame) -> pd.DataFrame:
        X = csv_to_predict["review"]
        y_pred = self.best_model.predict(X)
        return pd.concat([csv_to_predict, y_pred], axis=1)

    def predict_single_review(self, article: List[str]) -> Tuple[str, List]:

        vectorised_features = self.best_model[0].transform(article)
        explainable_tokens = self.best_model[0].inverse_transform(vectorised_features)[
            0
        ]

        logits = dict(
            zip(
                self.best_model[0].get_feature_names_out(),
                self.best_model[1].coef_[0],
            )
        )

        token_scores = sorted(
            [f"{token}, {logits.get(token)}" for token in explainable_tokens],
            key=lambda score: abs(float(score.split(", ")[1])),
            reverse=True,
        )

        prediction = self.best_model.predict(article)

        return "Positive" if prediction > 0.5 else "Negative", token_scores


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vec", choices=["count", "tfidf"], default="tfidf", required=False
    )
    args = parser.parse_args()

    logger.info("Reading in files")
    train = pd.read_parquet(DATA_PATH / "sentiment_train.parquet")
    test = pd.read_parquet(DATA_PATH / "sentiment_test.parquet")

    X_train, y_train = train["preprocessed_review"], train["class"]
    X_test, y_test = test["preprocessed_review"], test["class"]

    logger.info(
        f"Instantiating vectoriser: {'TfidfVectorizer' if str(args.vec) == 'tfidf' else 'CountVectoriser'}"
    )
    vectoriser = (
        TfidfVectorizer(ngram_range=(1, 2), min_df=0.005)
        if str(args.vec) == "tfidf"
        else CountVectorizer(ngram_range=(1, 2), min_df=0.005)
    )

    lr = LogisticRegressionReviews(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        vectoriser=vectoriser,
    )

    logger.info("Training model")
    lr.train()

    logger.info("Evaluating model")
    lr.evaluate()

    logger.info("Saving model")
    lr.save_model()
