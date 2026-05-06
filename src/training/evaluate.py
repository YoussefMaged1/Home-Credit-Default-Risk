import json
import logging
import os

import hydra
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_models(cfg: DictConfig, logger):
    SAVE_PATH = cfg.artifacts.model_path
    REPORT_PATH = cfg.artifacts.report_path
    SOURCE = cfg.data.processed_dir

    logger.info("Starting Ensemble Evaluation and Visualizations")

    try:
        y_train = np.load(os.path.join(SAVE_PATH, "y_train.npy"))
        xgb_oof = np.load(os.path.join(SAVE_PATH, "xgb_oof.npy"))
        cbm_oof = np.load(os.path.join(SAVE_PATH, "cbm_oof.npy"))
    except FileNotFoundError:
        logger.error(
            "OOF files not found! Make sure to save y_train, xgb_oof, and cbm_oof in train.py"
        )
        return

    ensemble_oof_preds = (cfg.model_weights.xgboost * xgb_oof) + (
        cfg.model_weights.catboost * cbm_oof
    )
    ensemble_auc = roc_auc_score(y_train, ensemble_oof_preds)

    logger.info(f"XGBoost OOF AUC: {roc_auc_score(y_train, xgb_oof):.5f}")
    logger.info(f"CatBoost OOF AUC: {roc_auc_score(y_train, cbm_oof):.5f}")
    logger.info(
        f"Final Ensemble AUC ({cfg.model_weights.xgboost * 100:.0f}/{cfg.model_weights.catboost * 100:.0f}): {ensemble_auc:.5f}"
    )

    threshold = cfg.threshold
    y_pred_binary = (ensemble_oof_preds > threshold).astype(int)

    os.makedirs(REPORT_PATH, exist_ok=True)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_train, y_pred_binary)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Ensemble Confusion Matrix at Threshold {threshold}")

    cm_path = os.path.join(REPORT_PATH, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    try:
        logger.info("Loading models for Permutation Importance...")
        xgb_model = joblib.load(os.path.join(SAVE_PATH, "xgb_model_pipeline.pkl"))
        cbm_model = joblib.load(os.path.join(SAVE_PATH, "cbm_model_pipeline.pkl"))

        df_sample = pd.read_csv(os.path.join(SOURCE, "train.csv")).sample(
            2000, random_state=cfg.training.random_seed
        )
        X_sample = df_sample.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
        y_sample = df_sample["TARGET"]

        def ensemble_predict_fixed(X):
            x_probs = xgb_model.predict_proba(X)[:, 1]
            c_probs = cbm_model.predict_proba(X)[:, 1]
            return (cfg.model_weights.xgboost * x_probs) + (
                cfg.model_weights.catboost * c_probs
            )

        logger.info("Calculating Permutation Importance (Sampling 2000 rows)...")
        results = permutation_importance(
            estimator=cbm_model,
            X=X_sample,
            y=y_sample,
            scoring=lambda est, X, y: roc_auc_score(y, ensemble_predict_fixed(X)),
            n_repeats=2,
            n_jobs=-1,
        )

        perm_importance_df = pd.DataFrame(
            {"Feature": X_sample.columns, "Importance": results.importances_mean}
        ).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=perm_importance_df.head(15),
            x="Importance",
            y="Feature",
            hue="Feature",
            legend=False,
        )
        plt.title("Top 15 Features (Ensemble Permutation Importance)")

        feat_path = os.path.join(REPORT_PATH, "feature_importance.png")
        plt.savefig(feat_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {feat_path}")

    except Exception as e:
        logger.error(f"Could not complete Permutation Importance: {e}")

    final_summary = {
        "ensemble_auc": float(ensemble_auc),
        "threshold": threshold,
        "classification_report": classification_report(
            y_train, y_pred_binary, output_dict=True
        ),
    }
    with open(os.path.join(REPORT_PATH, "final_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=4)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    evaluate_models(cfg, logger)


if __name__ == "__main__":
    main()
