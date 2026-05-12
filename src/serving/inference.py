import os

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name="Home-Credit-Default-Risk",
    mlflow=True,
)

xgb_model = mlflow.sklearn.load_model("models:/home-credit-ensemble_xgb@production")
cbm_model = mlflow.sklearn.load_model("models:/home-credit-ensemble_cbm@production")


def predict(cfg, df: pd.DataFrame) -> np.ndarray:
    xgb_preds = xgb_model.predict_proba(df)[:, 1]
    cbm_preds = cbm_model.predict_proba(df)[:, 1]

    ensemble_preds = (cfg.model_weights.xgboost * xgb_preds) + (cfg.model_weights.catboost * cbm_preds)
    return ensemble_preds


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main(cfg: DictConfig):

        test_df = pd.read_csv("data/processed/test.csv")

        sk_ids = test_df["SK_ID_CURR"]
        test_features = test_df.drop(columns=["SK_ID_CURR"])
        predictions = predict(cfg, test_features)

        output = pd.DataFrame({"SK_ID_CURR": sk_ids, "TARGET": predictions})

        output.to_csv("submissions/submission.csv", index=False)
        print("✅ Done! submission.csv saved.")
        print(output.head())

    main()
