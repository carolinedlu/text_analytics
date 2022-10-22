from typing import Type, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from text_analytics.config import ARTIFACTS_PATH, ROOT_DIR

plt.style.use(ROOT_DIR / "styles" / "base.mplstyle")


def evaluate_tuning(tuner: Union[Type[RandomizedSearchCV], Type[GridSearchCV]]) -> None:
    print(
        f"""
    --------------
    TUNING RESULTS
    --------------
    ESTIMATOR: {tuner.estimator}
    BEST SCORE: {tuner.best_score_:.2%}
    BEST PARAMS: {tuner.best_params_}
    TRAIN AUC: {tuner.cv_results_["mean_train_AUC"][tuner.best_index_]:.2%}
    TRAIN AUC SD: {tuner.cv_results_["std_train_AUC"][tuner.best_index_]:.2%}
    TEST AUC: {tuner.cv_results_["mean_test_AUC"][tuner.best_index_]:.2%}
    TEST AUC SD: {tuner.cv_results_["std_test_AUC"][tuner.best_index_]:.2%}
    TRAIN F_score: {tuner.cv_results_['mean_train_F_score'][tuner.best_index_]:.2%}
    TEST F_score: {tuner.cv_results_['mean_test_F_score'][tuner.best_index_]:.2%}
    """
    )


def calculate_report_metrics(
    y_test: pd.Series, y_pred: npt.ArrayLike, y_pred_prob: npt.ArrayLike
) -> dict:
    report = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "auroc": roc_auc_score(y_test, y_pred_prob),
        "cf_matrix": confusion_matrix(y_test, y_pred, normalize="all"),
        "roc": roc_curve(y_test, y_pred_prob),
    }

    print(
        f"""
    -----------
    PERFORMANCE
    -----------
    ACCURACY: {report.get("accuracy"):.2%}
    PRECISION: {report.get("precision"):.2%}
    RECALL: {report.get("recall"):.2%}
    F1: {report.get("F1"):.2%}
    ROC AUC: {report.get("auroc"):.2%}
    """
    )
    return report


def save_confusion_matrix(cf_matrix: npt.ArrayLike, model_name: str) -> None:
    _, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        np.eye(2),
        annot=cf_matrix,
        fmt=".2%",
        annot_kws={"size": 50},
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )

    ax.set_xlabel("Predicted Sentiment", size=20)
    ax.set_ylabel("Actual Sentiment", size=20)
    plt.savefig(f"{ARTIFACTS_PATH}/confusion_matrix/{model_name}.jpeg")
    plt.clf()


def save_roc_curve(
    fpr: npt.ArrayLike, tpr: npt.ArrayLike, model_name: str, auc: float
) -> None:

    plt.plot([0, 1], [0, 1], ls="--", color="black")
    plt.plot(fpr, tpr, linestyle="solid", color="blue")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{model_name} ROC curve")
    plt.text(
        0.95,
        0.01,
        f"AUC: {auc:.2%}",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    plt.savefig(f"{ARTIFACTS_PATH}/roc/{model_name}.jpg")
    plt.clf()
