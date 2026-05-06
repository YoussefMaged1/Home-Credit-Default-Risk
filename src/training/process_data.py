import os

import pandas as pd


def read_process_data(config, logger) -> None:
    logger.info("Data Processing started")
    SOURCE = config.data.raw_dir
    DESTINATION = config.data.processed_dir

    def get_bureau_features():
        path = os.path.join(SOURCE, "bureau.csv")
        bureau = pd.read_csv(path)

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

        return bureau_agg.reset_index()

    def get_prev_apps_features():
        path = os.path.join(SOURCE, "previous_application.csv")
        prev = pd.read_csv(path)

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
            prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]
            .groupby("SK_ID_CURR")
            .size()
        )
        prev_agg["PREV_REFUSED_COUNT"] = (
            prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size()
        )
        prev_agg = prev_agg.fillna(0)

        return prev_agg.reset_index()

    def get_installments_features():
        path = os.path.join(SOURCE, "installments_payments.csv")
        ins = pd.read_csv(path)

        ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

        ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
        ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)  # Days Past Due

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

        return ins_agg.reset_index()

    logger.info("Extracting features from other tables...")
    bureau_df = get_bureau_features()
    prev_df = get_prev_apps_features()
    ins_df = get_installments_features()

    def merge_all(main_df, bureau_df, prev_df, ins_df):
        main_df = main_df.merge(bureau_df, on="SK_ID_CURR", how="left")
        main_df = main_df.merge(prev_df, on="SK_ID_CURR", how="left")
        main_df = main_df.merge(ins_df, on="SK_ID_CURR", how="left")
        return main_df

    logger.info("Merging all tables...")
    df_sample = pd.read_csv(os.path.join(SOURCE, "application_train.csv"))
    test_full = pd.read_csv(os.path.join(SOURCE, "application_test.csv"))

    df_sample = merge_all(df_sample, bureau_df, prev_df, ins_df)
    test_full = merge_all(test_full, bureau_df, prev_df, ins_df)

    X_train = df_sample.drop(columns=["TARGET", "SK_ID_CURR"])
    y_train = df_sample["TARGET"]

    X_test = pd.concat([test_full[X_train.columns], test_full["SK_ID_CURR"]], axis=1)
    logger.info(f"Done! New feature count: {X_train.shape[1]}")

    def apply_feature_engineering(df):
        df = df.copy()
        df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
        df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
        df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        df["EXT_SOURCES_PROD"] = (
            df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        )
        df["EXT_SOURCES_STD"] = df[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].std(axis=1)
        df["EXT_SOURCES_STD"] = df["EXT_SOURCES_STD"]
        df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
        df["INCOME_ANNUITY_CHUNKS"] = df["AMT_INCOME_TOTAL"] / df["AMT_ANNUITY"]
        df["CREDIT_DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
        df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        df["AGE_INT"] = (df["DAYS_BIRTH"] / -365).astype(int)
        df["EXT_SOURCE_1_OVER_3"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_3"] + 1e-6)
        df["EXT_SOURCE_2_OVER_3"] = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-6)
        df["EXT_SOURCES_SUM"] = df[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].sum(axis=1)
        df["EXT_SOURCES_MEAN"] = df[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].mean(axis=1)
        return df

    X_train = apply_feature_engineering(X_train)
    test = apply_feature_engineering(X_test)

    train = pd.concat([X_train, y_train], axis=1)

    logger.info("Saving processed data...")
    os.makedirs(DESTINATION, exist_ok=True)
    train.to_csv(os.path.join(DESTINATION, "train.csv"), index=False)
    test.to_csv(os.path.join(DESTINATION, "test.csv"), index=False)


if __name__ == "__main__":
    import logging

    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main(cfg: DictConfig):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        read_process_data(cfg, logger)

    main()
