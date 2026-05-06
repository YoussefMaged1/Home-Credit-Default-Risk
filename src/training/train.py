import json
import logging
import os

import dagshub
import joblib
import mlflow
import mlflow.catboost
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
SAVE_PATH = os.path.join(MODEL_PATH, "final_models_pipelines")


def train_model(cfg, logger) -> None:
    load_dotenv()

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=cfg.model.repo_name,
        mlflow=cfg.model.use_mlflow,
    )

    # dagshub.init(repo_owner='youssef.maged237', repo_name='Home-Credit-Default-Risk', mlflow=True)
    mlflow.set_experiment("Ensemble_Training_HomeCredit")
    with mlflow.start_run(run_name="Full_Ensemble_Training"):
        logger.info("Model Training started")

        df_sample = pd.read_csv(os.path.join(SOURCE, "train.csv"))
        test_full = pd.read_csv(os.path.join(SOURCE, "test.csv"))
        X_train = df_sample.drop(columns=["TARGET"])
        y_train = df_sample["TARGET"]

        X_test = test_full.drop(columns=["SK_ID_CURR"])

        numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )

        def XGB_pipeline():
            logger.info("XGBoost Pipeline started")
            with mlflow.start_run(run_name="XGBoost_Model", nested=True):
                XGB_cat_pipeline = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Unknown",
                                add_indicator=True,
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                min_frequency=0.01,
                            ),
                        ),
                    ]
                )

                XGB_preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", num_pipeline, numeric_cols),
                        ("cat", XGB_cat_pipeline, categorical_cols),
                    ]
                )

                def xgb_objective(trial):
                    params = {
                        **cfg.xgboost.static_params,
                        "learning_rate": trial.suggest_float(
                            "learning_rate",
                            cfg.xgboost.search_space.learning_rate_min,
                            cfg.xgboost.search_space.learning_rate_max,
                            log=True,
                        ),
                        "max_depth": trial.suggest_int(
                            "max_depth",
                            cfg.xgboost.search_space.max_depth_min,
                            cfg.xgboost.search_space.max_depth_max,
                        ),
                        "min_child_weight": trial.suggest_int(
                            "min_child_weight",
                            cfg.xgboost.search_space.min_child_weight_min,
                            cfg.xgboost.search_space.min_child_weight_max,
                        ),
                        "subsample": trial.suggest_float(
                            "subsample",
                            cfg.xgboost.search_space.subsample_min,
                            cfg.xgboost.search_space.subsample_max,
                        ),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree",
                            cfg.xgboost.search_space.colsample_bytree_min,
                            cfg.xgboost.search_space.colsample_bytree_max,
                        ),
                        "scale_pos_weight": trial.suggest_float(
                            "scale_pos_weight",
                            cfg.xgboost.search_space.scale_pos_weight_min,
                            cfg.xgboost.search_space.scale_pos_weight_max,
                        ),
                    }

                    skf_opt = StratifiedKFold(
                        n_splits=3, shuffle=True, random_state=cfg.training.random_seed
                    )
                    scores = []
                    for t_idx, v_idx in skf_opt.split(X_train, y_train):
                        xt, xv = X_train.iloc[t_idx], X_train.iloc[v_idx]
                        yt, yv = y_train.iloc[t_idx], y_train.iloc[v_idx]

                        xt_p = XGB_preprocessor.fit_transform(xt)
                        xv_p = XGB_preprocessor.transform(xv)

                        model = xgb.XGBClassifier(**params)
                        model.fit(xt_p, yt, eval_set=[(xv_p, yv)], verbose=False)
                        preds = model.predict_proba(xv_p)[:, 1]
                        scores.append(roc_auc_score(yv, preds))
                    return np.mean(scores)

                XGB_study = optuna.create_study(direction=cfg.optuna.direction)
                XGB_study.optimize(xgb_objective, n_trials=cfg.optuna.n_trials)

                logger.info(f"XGB best params : {XGB_study.best_params}")

                XGB_final_params = {
                    **cfg.xgboost.static_params,
                    **XGB_study.best_params,
                    "n_estimators": 2000,
                }

                XGB_full_pipeline = Pipeline(
                    steps=[
                        ("preprocessor", XGB_preprocessor),
                        ("classifier", xgb.XGBClassifier(**XGB_final_params)),
                    ]
                )

                skf = StratifiedKFold(
                    n_splits=cfg.training.n_folds,
                    shuffle=True,
                    random_state=cfg.training.random_seed,
                )
                XGB_oof_preds = np.zeros(len(X_train))
                XGB_test_preds_total = np.zeros(len(X_test))

                for fold, (train_idx, val_idx) in enumerate(
                    skf.split(X_train, y_train)
                ):
                    logger.info(f"Started Fold {fold + 1}")
                    X_fold_train, X_fold_val = (
                        X_train.iloc[train_idx],
                        X_train.iloc[val_idx],
                    )
                    y_fold_train, y_fold_val = (
                        y_train.iloc[train_idx],
                        y_train.iloc[val_idx],
                    )

                    XGB_preprocessor.fit(X_fold_train, y_fold_train)
                    X_val_transformed = XGB_preprocessor.transform(X_fold_val).astype(
                        "float32"
                    )

                    XGB_full_pipeline.fit(
                        X_fold_train,
                        y_fold_train,
                        classifier__eval_set=[(X_val_transformed, y_fold_val)],
                        classifier__verbose=False,
                    )

                    XGB_oof_preds[val_idx] = XGB_full_pipeline.predict_proba(
                        X_fold_val
                    )[:, 1]
                    XGB_test_preds_total += (
                        XGB_full_pipeline.predict_proba(X_test)[:, 1]
                        / cfg.training.n_folds
                    )
                    logger.info(f"✅ Fold {fold + 1} Done!")

                logger.info(
                    f"XGB Overall AUC: {roc_auc_score(y_train, XGB_oof_preds):.5f}"
                )

                os.makedirs(SAVE_PATH, exist_ok=True)
                joblib.dump(
                    XGB_full_pipeline, os.path.join(SAVE_PATH, "xgb_model_pipeline.pkl")
                )
                logger.info(
                    f"XGB model pipeline saved to {os.path.join(SAVE_PATH, 'xgb_model_pipeline.pkl')}"
                )

                final_auc = float(roc_auc_score(y_train, XGB_oof_preds))

                mlflow.log_params(XGB_study.best_params)
                mlflow.log_params(cfg.xgboost.static_params)
                mlflow.log_params({"n_folds": cfg.training.n_folds})
                mlflow.log_params({"blending_weight": cfg.model_weights.xgboost})
                mlflow.log_metric("overall_auc", final_auc)
                mlflow.sklearn.log_model(XGB_full_pipeline, "xgb_pipeline")

                xgb_meta = {
                    "model_type": "XGBoost",
                    "final_oof_auc": final_auc,
                    "best_params": XGB_study.best_params,
                    "n_estimators": XGB_final_params["n_estimators"],
                    "n_folds": cfg.training.n_folds,
                }

                with open(os.path.join(SAVE_PATH, "xgb_metrics.json"), "w") as f:
                    json.dump(xgb_meta, f, indent=4)

                logger.info(
                    f"XGB Metrics saved to {os.path.join(SAVE_PATH, 'xgb_metrics.json')}"
                )
                return XGB_oof_preds, XGB_test_preds_total

        def CBM_pipeline():
            logger.info("CatBoost Pipeline started")
            with mlflow.start_run(run_name="CatBoost_Model", nested=True):
                CBM_cat_pipeline = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Unknown",
                                add_indicator=True,
                            ),
                        )
                    ]
                )

                CBM_preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", num_pipeline, numeric_cols),
                        ("cat", CBM_cat_pipeline, categorical_cols),
                    ]
                )
                CBM_preprocessor.set_output(transform="pandas")

                def cat_objective(trial):
                    params = {
                        **cfg.catboost.static_params,
                        "learning_rate": trial.suggest_float(
                            "learning_rate",
                            cfg.catboost.search_space.learning_rate_min,
                            cfg.catboost.search_space.learning_rate_max,
                            log=True,
                        ),
                        "depth": trial.suggest_int(
                            "depth",
                            cfg.catboost.search_space.depth_min,
                            cfg.catboost.search_space.depth_max,
                        ),
                        "l2_leaf_reg": trial.suggest_float(
                            "l2_leaf_reg",
                            cfg.catboost.search_space.l2_leaf_reg_min,
                            cfg.catboost.search_space.l2_leaf_reg_max,
                        ),
                        "random_strength": trial.suggest_float(
                            "random_strength",
                            cfg.catboost.search_space.random_strength_min,
                            cfg.catboost.search_space.random_strength_max,
                        ),
                        "bagging_temperature": trial.suggest_float(
                            "bagging_temperature",
                            cfg.catboost.search_space.bagging_temperature_min,
                            cfg.catboost.search_space.bagging_temperature_max,
                        ),
                        "border_count": trial.suggest_int(
                            "border_count",
                            cfg.catboost.search_space.border_count_min,
                            cfg.catboost.search_space.border_count_max,
                        ),
                        "scale_pos_weight": trial.suggest_float(
                            "scale_pos_weight",
                            cfg.catboost.search_space.scale_pos_weight_min,
                            cfg.catboost.search_space.scale_pos_weight_max,
                        ),
                    }

                    skf_opt = StratifiedKFold(
                        n_splits=3, shuffle=True, random_state=cfg.training.random_seed
                    )
                    scores = []
                    for t_idx, v_idx in skf_opt.split(X_train, y_train):
                        xt, xv = X_train.iloc[t_idx], X_train.iloc[v_idx]
                        yt, yv = y_train.iloc[t_idx], y_train.iloc[v_idx]

                        xt_p = CBM_preprocessor.fit_transform(xt)
                        xv_p = CBM_preprocessor.transform(xv)

                        current_cats = [
                            col for col in xt_p.columns if col.startswith("cat__")
                        ]
                        model = CatBoostClassifier(**params, cat_features=current_cats)
                        model.fit(xt_p, yt, eval_set=[(xv_p, yv)], verbose=False)

                        preds = model.predict_proba(xv_p)[:, 1]
                        scores.append(roc_auc_score(yv, preds))
                    return np.mean(scores)

                CBM_study = optuna.create_study(direction=cfg.optuna.direction)
                CBM_study.optimize(cat_objective, n_trials=cfg.optuna.n_trials)
                logger.info(f"CBM Best params: {CBM_study.best_params}")

                temp_df = CBM_preprocessor.fit_transform(X_train.head())
                final_cat_cols = [
                    col for col in temp_df.columns if col.startswith("cat__")
                ]

                CBM_final_params = {
                    **cfg.catboost.static_params,
                    **CBM_study.best_params,
                    "iterations": 5000,
                    "early_stopping_rounds": 200,
                    "cat_features": final_cat_cols,
                }

                CBM_full_pipeline = Pipeline(
                    steps=[
                        ("preprocessor", CBM_preprocessor),
                        ("classifier", CatBoostClassifier(**CBM_final_params)),
                    ]
                )

                skf = StratifiedKFold(
                    n_splits=cfg.training.n_folds,
                    shuffle=True,
                    random_state=cfg.training.random_seed,
                )
                CBM_oof_preds = np.zeros(len(X_train))
                CBM_test_preds_total = np.zeros(len(X_test))

                for fold, (train_idx, val_idx) in enumerate(
                    skf.split(X_train, y_train)
                ):
                    logger.info(f"Started CBM Fold {fold + 1}")
                    X_fold_train, X_fold_val = (
                        X_train.iloc[train_idx],
                        X_train.iloc[val_idx],
                    )
                    y_fold_train, y_fold_val = (
                        y_train.iloc[train_idx],
                        y_train.iloc[val_idx],
                    )

                    CBM_preprocessor.fit(X_fold_train, y_fold_train)
                    X_val_transformed = CBM_preprocessor.transform(X_fold_val)

                    CBM_full_pipeline.fit(
                        X_fold_train,
                        y_fold_train,
                        classifier__eval_set=[(X_val_transformed, y_fold_val)],
                        classifier__verbose=False,
                    )

                    CBM_oof_preds[val_idx] = CBM_full_pipeline.predict_proba(
                        X_fold_val
                    )[:, 1]
                    CBM_test_preds_total += (
                        CBM_full_pipeline.predict_proba(X_test)[:, 1]
                        / cfg.training.n_folds
                    )
                    logger.info(f"CBM Fold {fold + 1} Done!")

                logger.info(
                    f"CBM Overall AUC: {roc_auc_score(y_train, CBM_oof_preds):.5f}"
                )

                os.makedirs(SAVE_PATH, exist_ok=True)
                joblib.dump(
                    CBM_full_pipeline, os.path.join(SAVE_PATH, "cbm_model_pipeline.pkl")
                )
                logger.info(
                    f"CBM model pipeline saved to {os.path.join(SAVE_PATH, 'cbm_model_pipeline.pkl')}"
                )

                final_auc = float(roc_auc_score(y_train, CBM_oof_preds))

                mlflow.log_params(CBM_study.best_params)
                mlflow.log_params(cfg.catboost.static_params)
                mlflow.log_params({"n_folds": cfg.training.n_folds})
                mlflow.log_params({"blending_weight": cfg.model_weights.catboost})
                mlflow.log_metric("overall_auc", final_auc)
                mlflow.sklearn.log_model(CBM_full_pipeline, "cbm_pipeline")

                cbm_meta = {
                    "model_type": "CatBoost",
                    "final_oof_auc": final_auc,
                    "best_params": CBM_study.best_params,
                    "iterations": CBM_final_params["iterations"],
                    "n_folds": cfg.training.n_folds,
                }

                with open(os.path.join(SAVE_PATH, "cbm_metrics.json"), "w") as f:
                    json.dump(cbm_meta, f, indent=4)

                logger.info(
                    f"CBM Metrics saved to {os.path.join(SAVE_PATH, 'cbm_metrics.json')}"
                )
                return CBM_oof_preds, CBM_test_preds_total

        xgb_oof, xgb_test = XGB_pipeline()
        cbm_oof, cbm_test = CBM_pipeline()
        ensemble_oof_preds = (cfg.model_weights.xgboost * xgb_oof) + (
            cfg.model_weights.catboost * cbm_oof
        )
        ensemble_auc = roc_auc_score(y_train, ensemble_oof_preds)
        mlflow.log_metric("overall_auc", ensemble_auc)
        mlflow.log_metric("threshold", cfg.threshold)
        
        logger.info(
            f"Final Ensemble AUC ({cfg.model_weights.xgboost * 100:.0f}/{cfg.model_weights.catboost * 100:.0f}): {ensemble_auc:.5f}"
        )

        np.save(os.path.join(SAVE_PATH, "xgb_oof.npy"), xgb_oof)
        np.save(os.path.join(SAVE_PATH, "cbm_oof.npy"), cbm_oof)
        np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train.values)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main(cfg: DictConfig):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        train_model(cfg, logger)

    main()
