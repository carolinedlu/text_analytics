import argparse
import logging
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from text_analytics.config import (
    BASE_SCORER,
    CV_SPLIT,
    DATA_PATH,
    MODEL_PATH,
    N_JOBS,
    RANDOM_STATE,
    SENTIMENT_CLEANED_DATA_PATH,
)
from text_analytics.helpers import (
    calculate_report_metrics,
    evaluate_tuning,
    save_confusion_matrix,
    save_roc_curve,
)

warnings.filterwarnings("ignore")


class RandomForestReviews:
    def __init__(
        self,
        X_train: pd.DataFrame = None,
        y_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
        scorer: dict = BASE_SCORER,
        cv_split: StratifiedShuffleSplit = CV_SPLIT,
        model: xgb.XGBRFClassifier = None,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scorer = scorer
        self.cv_split = cv_split

        if model is None:
            model = xgb.XGBRFClassifier(
                objective="binary:logistic",
                booster="gbtree",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                verbosity=0,
            )

        self.model = model
        self.trainer = None
        self.best_model = None

        self.timestamp = datetime.now().strftime("%d_%H_%M")
        self.file_name = f"sentiment_xgb_rf_{self.timestamp}"

    def save_model(self) -> None:
        save_path = MODEL_PATH / f"{self.file_name}.pkl"

        with open(save_path, "wb") as file:
            pickle.dump(self.best_model, file)

        print(f"Model saved to {save_path}")

    def load_model(self, file_name: str) -> None:

        model_path = MODEL_PATH / f"{file_name}.pkl"
        with open(model_path, "rb") as file:
            self.best_model = pickle.load(file)

        print(f"Model loaded from {model_path}")

    def train(self, iters: int = 20):

        # setup the hyperparameter grid
        rf_param_grid = {
            "rf__n_estimators": np.arange(100, 500, 100),
            "rf__learning_rate": np.linspace(0.3, 1, 8),
            "rf__subsample": np.arange(0.6, 0.9, 0.1),
            "rf__colsample_bynode": np.linspace(0.6, 0.9, 5),
            "rf__max_depth": np.arange(3, 10, 1),
            "rf__reg_lambda": np.linspace(0, 1, 9),
        }

        # build the pipeline
        rf_pipe = Pipeline([("rf", self.model)])

        # cross validate model with RandomizedSearch
        self.trainer = RandomizedSearchCV(
            estimator=rf_pipe,
            param_distributions=rf_param_grid,
            n_iter=iters,
            scoring=self.scorer,
            refit="F_score",
            cv=self.cv_split,
            return_train_score=True,
            n_jobs=N_JOBS,
            verbose=10,
            random_state=RANDOM_STATE,
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


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", default=15, required=False)
    args = parser.parse_args()

    logger.info("Reading in files")
    train = pd.read_parquet(DATA_PATH / "sentiment_train.parquet")
    test = pd.read_parquet(DATA_PATH / "sentiment_test.parquet")

    X_train, y_train = train["preprocessed_review"], train["class"]
    X_test, y_test = test["preprocessed_review"], test["class"]

    logger.info("Vectorising files")
    count_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=10)

    X_train = count_vectorizer.fit_transform(X_train)
    X_test = count_vectorizer.transform(X_test)

    rf = RandomForestReviews(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    logger.info("Training model")
    rf.train(int(args.iters))

    logger.info("Evaluating model")
    rf.evaluate()

    logger.info("Saving model")
    rf.save_model()
