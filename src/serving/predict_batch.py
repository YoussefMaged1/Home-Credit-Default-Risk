import os
from datetime import timedelta

import dagshub
import duckdb
import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task
from prefect.tasks import task_input_hash

from src.training.process_data import (
    apply_feature_engineering,
    get_bureau_features,
    get_installments_features,
    get_prev_apps_features,
)

load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=2),
)
def extract_data():
    con = duckdb.connect(f"md:home_credit_db?motherduck_token={MOTHERDUCK_TOKEN}")

    test_df = con.execute("SELECT * FROM test_data").df()
    bureau = con.execute("SELECT * FROM bureau").df()
    prev = con.execute("SELECT * FROM previous_application").df()
    ins = con.execute("SELECT * FROM installments_payments").df()

    con.close()
    return test_df, bureau, prev, ins


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=2))
def transform_data(test_df, bureau, prev, ins):
    bureau_df = get_bureau_features(bureau)
    prev_df = get_prev_apps_features(prev)
    ins_df = get_installments_features(ins)

    sk_ids = test_df["SK_ID_CURR"].tolist() if "SK_ID_CURR" in test_df.columns else None

    df = test_df.merge(bureau_df, on="SK_ID_CURR", how="left")
    df = df.merge(prev_df, on="SK_ID_CURR", how="left")
    df = df.merge(ins_df, on="SK_ID_CURR", how="left")
    df = df.drop(columns=["SK_ID_CURR"], errors="ignore")
    df = apply_feature_engineering(df)    

    return df, sk_ids


@task
def predict(cfg, df: pd.DataFrame):

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name="Home-Credit-Default-Risk",
        mlflow=True,
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace({pd.NA: np.nan})
    df = df.astype(np.float64)

    for col in df.columns:
        if np.array_equal(df[col].dropna(), df[col].dropna().astype(int)):
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = df[col].astype(np.float64)

    xgb_model = mlflow.sklearn.load_model("models:/home-credit-ensemble_xgb@production")
    cbm_model = mlflow.sklearn.load_model("models:/home-credit-ensemble_cbm@production")

    xgb_preds = xgb_model.predict_proba(df)[:, 1]
    cbm_preds = cbm_model.predict_proba(df)[:, 1]
    ensemble_preds = (cfg.model_weights.xgboost * xgb_preds) + (
        cfg.model_weights.catboost * cbm_preds
    )

    return ensemble_preds


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=2),
)
def save_predictions(cfg, sk_ids, preds):
    results_df = pd.DataFrame(
        {
            "SK_ID_CURR": sk_ids,
            "TARGET": preds,
            "PREDICTION": [
                "NO DEFAULT" if p > cfg.threshold else "DEFAULT" for p in preds
            ],
        }
    )

    con = duckdb.connect(f"md:home_credit_db?motherduck_token={MOTHERDUCK_TOKEN}")
    con.execute("""
        CREATE OR REPLACE TABLE predictions AS
        SELECT * FROM results_df
    """)
    count = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    con.close()

    print(f"✅ Saved {count} predictions to MotherDuck!")
    return results_df


@flow(name="home-credit-batch-prediction")
def home_credit_batch_flow():
    hydra.initialize(version_base=None, config_path="../../conf")
    cfg = hydra.compose(config_name="config")

    test_df, bureau, prev, ins = extract_data()
    df_processed, sk_ids = transform_data(test_df, bureau, prev, ins)
    preds = predict(cfg, df_processed)
    save_predictions(cfg, sk_ids, preds)


if __name__ == "__main__":
    home_credit_batch_flow()
