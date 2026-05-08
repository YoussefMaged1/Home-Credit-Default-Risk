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
    RAW = "data/raw"

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        {
            "DAYS_CREDIT": ["mean", "max", "min"],
            "DAYS_CREDIT_ENDDATE": ["mean"],
            "AMT_CREDIT_SUM": ["mean", "sum"],
            "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
            "AMT_CREDIT_MAX_OVERDUE": ["mean"],
            "CNT_CREDIT_PROLONG": ["sum"],
        }
    )
    bureau_agg.columns = pd.Index(
        ["BUREAU_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()]
    )
    bureau_agg["BUREAU_DEBT_CREDIT_RATIO"] = bureau_agg[
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM"
    ] / (bureau_agg["BUREAU_AMT_CREDIT_SUM_SUM"] + 1e-6)
    bureau_agg["BUREAU_LOAN_COUNT"] = bureau.groupby("SK_ID_CURR").size()
    bureau_df = bureau_agg.reset_index()

    prev_agg = prev.groupby("SK_ID_CURR").agg(
        {
            "AMT_ANNUITY": ["mean", "max"],
            "AMT_APPLICATION": ["mean", "max"],
            "AMT_CREDIT": ["mean", "max"],
            "AMT_DOWN_PAYMENT": ["mean", "max"],
            "DAYS_DECISION": ["mean", "max"],
            "CNT_PAYMENT": ["mean", "sum"],
        }
    )
    prev_agg.columns = pd.Index(
        ["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()]
    )
    prev_agg["PREV_APPROVED_COUNT"] = (
        prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR").size()
    )
    prev_agg["PREV_REFUSED_COUNT"] = (
        prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size()
    )
    prev_agg = prev_agg.fillna(0)
    prev_df = prev_agg.reset_index()

    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
    ins_agg = ins.groupby("SK_ID_CURR").agg(
        {
            "PAYMENT_DIFF": ["mean", "max", "sum"],
            "DPD": ["mean", "max", "sum"],
            "AMT_PAYMENT": ["mean", "sum"],
            "DAYS_ENTRY_PAYMENT": ["max", "mean"],
        }
    )
    ins_agg.columns = pd.Index(
        ["INS_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()]
    )
    ins_df = ins_agg.reset_index()

    df = test_df.merge(bureau_df, on="SK_ID_CURR", how="left")
    df = df.merge(prev_df, on="SK_ID_CURR", how="left")
    df = df.merge(ins_df, on="SK_ID_CURR", how="left")

    sk_ids = df["SK_ID_CURR"]
    df = df.drop(columns=["SK_ID_CURR"], errors="ignore")

    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["EXT_SOURCES_PROD"] = (
        df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    )
    df["EXT_SOURCES_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(
        axis=1
    )
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["INCOME_ANNUITY_CHUNKS"] = df["AMT_INCOME_TOTAL"] / df["AMT_ANNUITY"]
    df["CREDIT_DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["AGE_INT"] = (df["DAYS_BIRTH"] / -365).astype(int)
    df["EXT_SOURCE_1_OVER_3"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_3"] + 1e-6)
    df["EXT_SOURCE_2_OVER_3"] = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-6)
    df["EXT_SOURCES_SUM"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].sum(
        axis=1
    )
    df["EXT_SOURCES_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(
        axis=1
    )

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
            "SK_ID_CURR": sk_ids.values,
            "TARGET": preds,
            "PREDICTION": [
                "DEFAULT" if p > cfg.threshold else "NO DEFAULT" for p in preds
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
