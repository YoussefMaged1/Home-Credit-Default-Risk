import os
from typing import List, Optional

import dagshub
import litserve as ls
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.training.process_data import (
    apply_feature_engineering,
)


class ApplicantFeatures(BaseModel):
    SK_ID_CURR: Optional[int] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    CNT_CHILDREN: Optional[int] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    NAME_TYPE_SUITE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    DAYS_BIRTH: Optional[int] = None
    DAYS_EMPLOYED: Optional[int] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[int] = None
    OWN_CAR_AGE: Optional[float] = None
    FLAG_MOBIL: Optional[int] = None
    FLAG_EMP_PHONE: Optional[int] = None
    FLAG_WORK_PHONE: Optional[int] = None
    FLAG_CONT_MOBILE: Optional[int] = None
    FLAG_PHONE: Optional[int] = None
    FLAG_EMAIL: Optional[int] = None
    OCCUPATION_TYPE: Optional[str] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    REGION_RATING_CLIENT: Optional[int] = None
    REGION_RATING_CLIENT_W_CITY: Optional[int] = None
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    HOUR_APPR_PROCESS_START: Optional[int] = None
    REG_REGION_NOT_LIVE_REGION: Optional[int] = None
    REG_REGION_NOT_WORK_REGION: Optional[int] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[int] = None
    REG_CITY_NOT_LIVE_CITY: Optional[int] = None
    REG_CITY_NOT_WORK_CITY: Optional[int] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[int] = None
    ORGANIZATION_TYPE: Optional[str] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    APARTMENTS_AVG: Optional[float] = None
    BASEMENTAREA_AVG: Optional[float] = None
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = None
    YEARS_BUILD_AVG: Optional[float] = None
    COMMONAREA_AVG: Optional[float] = None
    ELEVATORS_AVG: Optional[float] = None
    ENTRANCES_AVG: Optional[float] = None
    FLOORSMAX_AVG: Optional[float] = None
    FLOORSMIN_AVG: Optional[float] = None
    LANDAREA_AVG: Optional[float] = None
    LIVINGAPARTMENTS_AVG: Optional[float] = None
    LIVINGAREA_AVG: Optional[float] = None
    NONLIVINGAPARTMENTS_AVG: Optional[float] = None
    NONLIVINGAREA_AVG: Optional[float] = None
    APARTMENTS_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = None
    YEARS_BUILD_MODE: Optional[float] = None
    COMMONAREA_MODE: Optional[float] = None
    ELEVATORS_MODE: Optional[float] = None
    ENTRANCES_MODE: Optional[float] = None
    FLOORSMAX_MODE: Optional[float] = None
    FLOORSMIN_MODE: Optional[float] = None
    LANDAREA_MODE: Optional[float] = None
    LIVINGAPARTMENTS_MODE: Optional[float] = None
    LIVINGAREA_MODE: Optional[float] = None
    NONLIVINGAPARTMENTS_MODE: Optional[float] = None
    NONLIVINGAREA_MODE: Optional[float] = None
    APARTMENTS_MEDI: Optional[float] = None
    BASEMENTAREA_MEDI: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = None
    YEARS_BUILD_MEDI: Optional[float] = None
    COMMONAREA_MEDI: Optional[float] = None
    ELEVATORS_MEDI: Optional[float] = None
    ENTRANCES_MEDI: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    FLOORSMIN_MEDI: Optional[float] = None
    LANDAREA_MEDI: Optional[float] = None
    LIVINGAPARTMENTS_MEDI: Optional[float] = None
    LIVINGAREA_MEDI: Optional[float] = None
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = None
    NONLIVINGAREA_MEDI: Optional[float] = None
    FONDKAPREMONT_MODE: Optional[str] = None
    HOUSETYPE_MODE: Optional[str] = None
    TOTALAREA_MODE: Optional[float] = None
    WALLSMATERIAL_MODE: Optional[str] = None
    EMERGENCYSTATE_MODE: Optional[str] = None
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    FLAG_DOCUMENT_2: Optional[int] = None
    FLAG_DOCUMENT_3: Optional[int] = None
    FLAG_DOCUMENT_4: Optional[int] = None
    FLAG_DOCUMENT_5: Optional[int] = None
    FLAG_DOCUMENT_6: Optional[int] = None
    FLAG_DOCUMENT_7: Optional[int] = None
    FLAG_DOCUMENT_8: Optional[int] = None
    FLAG_DOCUMENT_9: Optional[int] = None
    FLAG_DOCUMENT_10: Optional[int] = None
    FLAG_DOCUMENT_11: Optional[int] = None
    FLAG_DOCUMENT_12: Optional[int] = None
    FLAG_DOCUMENT_13: Optional[int] = None
    FLAG_DOCUMENT_14: Optional[int] = None
    FLAG_DOCUMENT_15: Optional[int] = None
    FLAG_DOCUMENT_16: Optional[int] = None
    FLAG_DOCUMENT_17: Optional[int] = None
    FLAG_DOCUMENT_18: Optional[int] = None
    FLAG_DOCUMENT_19: Optional[int] = None
    FLAG_DOCUMENT_20: Optional[int] = None
    FLAG_DOCUMENT_21: Optional[int] = None
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None


class PredictRequest(BaseModel):
    applicants: List[ApplicantFeatures]


class HomeCreditAPI(ls.LitAPI):

    def setup(self, device):
        cfg = OmegaConf.load("conf/config.yaml")

        self.xgb_weight = cfg.model_weights.xgboost
        self.cbm_weight = cfg.model_weights.catboost
        self.threshold = cfg.threshold

        load_dotenv()

        dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_USERNAME"),
            repo_name="Home-Credit-Default-Risk",
            mlflow=True,
        )

        self.xgb_model = mlflow.sklearn.load_model(
            "models:/home-credit-ensemble_xgb@production"
        )
        self.cbm_model = mlflow.sklearn.load_model(
            "models:/home-credit-ensemble_cbm@production"
        )

        self.bureau_df = pd.read_csv("data/processed/bureau_features.csv")
        self.prev_df = pd.read_csv("data/processed/prev_features.csv")
        self.ins_df = pd.read_csv("data/processed/installments_features.csv")

    def decode_request(self, request: PredictRequest):
        df = pd.DataFrame([a.model_dump() for a in request.applicants])

        self.sk_ids = df["SK_ID_CURR"].tolist() if "SK_ID_CURR" in df.columns else None

        df = df.merge(self.bureau_df, on="SK_ID_CURR", how="left")
        df = df.merge(self.prev_df, on="SK_ID_CURR", how="left")
        df = df.merge(self.ins_df, on="SK_ID_CURR", how="left")
        df = df.drop(columns=["SK_ID_CURR"], errors="ignore")
        df = apply_feature_engineering(df)

        return df

    def predict(self, df: pd.DataFrame):
        xgb_preds = self.xgb_model.predict_proba(df)[:, 1]
        cbm_preds = self.cbm_model.predict_proba(df)[:, 1]
        ensemble_preds = (self.xgb_weight * xgb_preds) + (self.cbm_weight * cbm_preds)
        return ensemble_preds

    def encode_response(self, preds):
        results = []
        for i, score in enumerate(preds):
            record = {
                "default_probability": round(float(score), 4),
                "prediction": "DEFAULT" if score > self.threshold else "NO DEFAULT",
            }
            if self.sk_ids:
                record["SK_ID_CURR"] = int(self.sk_ids[i])
            results.append(record)

        return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    api = HomeCreditAPI(api_path="/predict")
    server = ls.LitServer(api)
    server.run(port=8000)
