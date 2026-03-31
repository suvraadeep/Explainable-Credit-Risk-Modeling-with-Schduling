import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Optional, Tuple
import joblib
import os
 
warnings.filterwarnings("ignore")
 
 
# ─── Bureau ───────────────────────────────────────────────────────────────────
 
def _engineer_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau balance status into per-bureau features."""
    STATUS_MAP = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
 
    bb = bureau_balance.copy()
    bb["STATUS_NUM"] = bb["STATUS"].map(STATUS_MAP).fillna(0)
 
    agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_STATUS_WORST   = ("STATUS_NUM", "max"),
        BB_STATUS_MEAN    = ("STATUS_NUM", "mean"),
        BB_STATUS_STD     = ("STATUS_NUM", "std"),
        BB_MONTHS_COUNT   = ("MONTHS_BALANCE", "count"),
        BB_DPD_MONTHS     = ("STATUS_NUM", lambda x: (x > 0).sum()),
    ).reset_index()
 
    bb["DPD_FLAG"]   = (bb["STATUS_NUM"] > 0).astype(int)
    bb["SEVERE_DPD"] = (bb["STATUS_NUM"] >= 3).astype(int)
 
    agg2 = bb.groupby("SK_ID_BUREAU").agg(
        BB_DPD_RATE       = ("DPD_FLAG",   "mean"),
        BB_SEVERE_DPD_RATE= ("SEVERE_DPD", "mean"),
    ).reset_index()
 
    return agg.merge(agg2, on="SK_ID_BUREAU", how="left")
 
 
def engineer_bureau_features(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame
) -> pd.DataFrame:
    """
    Full bureau feature engineering.
    Returns a DataFrame indexed by SK_ID_CURR.
    """
    bb_agg = _engineer_bureau_balance(bureau_balance)
    bur    = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
 
    # Derived ratios
    bur["CREDIT_ACTIVE_BINARY"]  = (bur["CREDIT_ACTIVE"] == "Active").astype(int)
    bur["CREDIT_CLOSED_BINARY"]  = (bur["CREDIT_ACTIVE"] == "Closed").astype(int)
    bur["DEBT_CREDIT_RATIO"]     = bur["AMT_CREDIT_SUM_DEBT"] / (bur["AMT_CREDIT_SUM"] + 1)
    bur["CREDIT_UTIL_RATE"]      = bur["AMT_CREDIT_SUM_OVERDUE"] / (bur["AMT_CREDIT_SUM"] + 1)
    bur["DAYS_CREDIT_ENDDATE"]   = bur["DAYS_CREDIT_ENDDATE"].clip(-3000, 3000)
    bur["CREDIT_LENGTH"]         = bur["DAYS_CREDIT_ENDDATE"] - bur["DAYS_CREDIT"]
    bur["OVERDUE_CREDIT_RATIO"]  = bur["CREDIT_DAY_OVERDUE"] / (bur["AMT_CREDIT_SUM"] + 1)
 
    aggregations: Dict = {
        "DAYS_CREDIT":              ["mean", "min", "max", "std"],
        "CREDIT_DAY_OVERDUE":       ["mean", "max", "sum"],
        "DAYS_CREDIT_ENDDATE":      ["mean", "min", "max"],
        "DAYS_CREDIT_UPDATE":       ["mean"],
        "AMT_CREDIT_SUM":           ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_DEBT":      ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_OVERDUE":   ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_LIMIT":     ["mean", "max"],
        "DEBT_CREDIT_RATIO":        ["mean", "max"],
        "CREDIT_UTIL_RATE":         ["mean", "max"],
        "CREDIT_ACTIVE_BINARY":     ["mean", "sum"],
        "CREDIT_CLOSED_BINARY":     ["sum"],
        "BB_STATUS_WORST":          ["mean", "max"],
        "BB_STATUS_MEAN":           ["mean"],
        "BB_DPD_RATE":              ["mean", "max"],
        "BB_SEVERE_DPD_RATE":       ["mean", "max"],
        "BB_MONTHS_COUNT":          ["mean", "sum"],
        "CNT_CREDIT_PROLONG":       ["sum", "mean"],
        "CREDIT_LENGTH":            ["mean", "max"],
    }
 
    agg_df = bur.groupby("SK_ID_CURR").agg(aggregations)
    agg_df.columns = ["BUREAU_" + "_".join(c).upper() for c in agg_df.columns]
 
    # Counts
    agg_df["BUREAU_COUNT"]        = bur.groupby("SK_ID_CURR").size()
    agg_df["BUREAU_ACTIVE_COUNT"] = bur.groupby("SK_ID_CURR")["CREDIT_ACTIVE_BINARY"].sum()
    agg_df["BUREAU_CLOSED_COUNT"] = bur.groupby("SK_ID_CURR")["CREDIT_CLOSED_BINARY"].sum()
 
    # Credit type diversity
    credit_type_counts = bur.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique()
    agg_df["BUREAU_CREDIT_TYPE_DIVERSITY"] = credit_type_counts
 
    return agg_df.reset_index()
 
 
# ─── Previous Applications ────────────────────────────────────────────────────
 
def engineer_prev_app_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous application history per applicant."""
    p = prev.copy()
 
    p["APP_CREDIT_RATIO"]      = p["AMT_APPLICATION"] / (p["AMT_CREDIT"] + 1)
    p["DOWN_PAYMENT_RATIO"]    = p["AMT_DOWN_PAYMENT"]  / (p["AMT_CREDIT"] + 1)
    p["ANNUITY_CREDIT_RATIO"]  = p["AMT_ANNUITY"]       / (p["AMT_CREDIT"] + 1)
    p["GOODS_CREDIT_RATIO"]    = p["AMT_GOODS_PRICE"]   / (p["AMT_CREDIT"] + 1)
    p["APPROVED"]              = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["REFUSED"]               = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["CANCELED"]              = (p["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)
    p["HOUR_APPR_PROCESS_START_LATE"] = (p["HOUR_APPR_PROCESS_START"] >= 18).astype(int)
 
    agg = p.groupby("SK_ID_CURR").agg(
        PREV_COUNT                    = ("SK_ID_PREV",              "count"),
        PREV_APPROVED_COUNT           = ("APPROVED",                "sum"),
        PREV_REFUSED_COUNT            = ("REFUSED",                 "sum"),
        PREV_CANCELED_COUNT           = ("CANCELED",                "sum"),
        PREV_APPROVED_RATE            = ("APPROVED",                "mean"),
        PREV_REFUSED_RATE             = ("REFUSED",                 "mean"),
        PREV_APP_CREDIT_RATIO_MEAN    = ("APP_CREDIT_RATIO",        "mean"),
        PREV_APP_CREDIT_RATIO_MAX     = ("APP_CREDIT_RATIO",        "max"),
        PREV_DOWN_PAYMENT_MEAN        = ("DOWN_PAYMENT_RATIO",      "mean"),
        PREV_ANNUITY_MEAN             = ("AMT_ANNUITY",             "mean"),
        PREV_ANNUITY_MAX              = ("AMT_ANNUITY",             "max"),
        PREV_CREDIT_MEAN              = ("AMT_CREDIT",              "mean"),
        PREV_CREDIT_MAX               = ("AMT_CREDIT",              "max"),
        PREV_CREDIT_SUM               = ("AMT_CREDIT",              "sum"),
        PREV_DAYS_DECISION_MEAN       = ("DAYS_DECISION",           "mean"),
        PREV_DAYS_DECISION_MIN        = ("DAYS_DECISION",           "min"),
        PREV_DAYS_LAST_DUE_MEAN       = ("DAYS_LAST_DUE",          "mean"),
        PREV_GOODS_PRICE_MEAN         = ("AMT_GOODS_PRICE",         "mean"),
        PREV_HOUR_LATE_RATE           = ("HOUR_APPR_PROCESS_START_LATE", "mean"),
        PREV_TERM_MEAN                = ("CNT_PAYMENT",             "mean"),
    ).reset_index()
 
    # Most recent prev application features
    last_prev = p.sort_values("DAYS_DECISION").groupby("SK_ID_CURR").last().reset_index()
    last_prev = last_prev[["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO"]].rename(
        columns={
            "AMT_CREDIT":        "PREV_LAST_CREDIT",
            "AMT_ANNUITY":       "PREV_LAST_ANNUITY",
            "APP_CREDIT_RATIO":  "PREV_LAST_APP_CREDIT_RATIO",
        }
    )
    agg = agg.merge(last_prev, on="SK_ID_CURR", how="left")
    return agg
 
 
# ─── Installments ─────────────────────────────────────────────────────────────
 
def engineer_installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Derive payment behaviour from installments history."""
    i = inst.copy()
 
    i["PAYMENT_DIFF"]      = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]
    i["PAYMENT_RATIO"]     = i["AMT_PAYMENT"]    / (i["AMT_INSTALMENT"] + 1)
    i["DAYS_ENTRY_DIFF"]   = i["DAYS_INSTALMENT"] - i["DAYS_ENTRY_PAYMENT"]
    i["LATE_PAYMENT"]      = (i["DAYS_ENTRY_DIFF"] > 0).astype(int)
    i["EARLY_PAYMENT"]     = (i["DAYS_ENTRY_DIFF"] < 0).astype(int)
    i["SHORT_PAYMENT"]     = (i["PAYMENT_DIFF"] > 0).astype(int)
    i["OVER_PAYMENT"]      = (i["PAYMENT_DIFF"] < 0).astype(int)
 
    agg = i.groupby("SK_ID_CURR").agg(
        INST_PAYMENT_DIFF_MEAN    = ("PAYMENT_DIFF",    "mean"),
        INST_PAYMENT_DIFF_MAX     = ("PAYMENT_DIFF",    "max"),
        INST_PAYMENT_DIFF_SUM     = ("PAYMENT_DIFF",    "sum"),
        INST_PAYMENT_RATIO_MEAN   = ("PAYMENT_RATIO",   "mean"),
        INST_PAYMENT_RATIO_MIN    = ("PAYMENT_RATIO",   "min"),
        INST_DAYS_ENTRY_DIFF_MEAN = ("DAYS_ENTRY_DIFF", "mean"),
        INST_DAYS_ENTRY_DIFF_MAX  = ("DAYS_ENTRY_DIFF", "max"),
        INST_LATE_PAYMENT_RATE    = ("LATE_PAYMENT",    "mean"),
        INST_LATE_PAYMENT_COUNT   = ("LATE_PAYMENT",    "sum"),
        INST_EARLY_PAYMENT_RATE   = ("EARLY_PAYMENT",   "mean"),
        INST_SHORT_PAYMENT_RATE   = ("SHORT_PAYMENT",   "mean"),
        INST_OVER_PAYMENT_RATE    = ("OVER_PAYMENT",    "mean"),
        INST_COUNT                = ("SK_ID_PREV",      "count"),
        INST_NUM_DISTINCT_LOANS   = ("SK_ID_PREV",      "nunique"),
        INST_AMT_PAYMENT_MEAN     = ("AMT_PAYMENT",     "mean"),
        INST_AMT_PAYMENT_STD      = ("AMT_PAYMENT",     "std"),
    ).reset_index()
 
    return agg
 
 
# ─── POS Cash ─────────────────────────────────────────────────────────────────
 
def engineer_pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS Cash balance signals."""
    p = pos.copy()
    p["DPD_BINARY"]  = (p["SK_DPD"] > 0).astype(int)
    p["DPD_SEVERE"]  = (p["SK_DPD"] > 30).astype(int)
    p["DPD_RATIO"]   = p["SK_DPD"] / (p["CNT_INSTALMENT"] + 1)
 
    agg = p.groupby("SK_ID_CURR").agg(
        POS_MONTHS_COUNT          = ("MONTHS_BALANCE",   "count"),
        POS_SK_DPD_MEAN           = ("SK_DPD",           "mean"),
        POS_SK_DPD_MAX            = ("SK_DPD",           "max"),
        POS_SK_DPD_SUM            = ("SK_DPD",           "sum"),
        POS_DPD_RATE              = ("DPD_BINARY",        "mean"),
        POS_SEVERE_DPD_RATE       = ("DPD_SEVERE",        "mean"),
        POS_CNT_INSTALMENT_MEAN   = ("CNT_INSTALMENT",   "mean"),
        POS_CNT_INSTALMENT_FUTURE_MEAN = ("CNT_INSTALMENT_FUTURE", "mean"),
        POS_NAME_CONTRACT_STATUS  = ("NAME_CONTRACT_STATUS", lambda x: (x == "Active").mean()),
        POS_NUM_DISTINCT_LOANS    = ("SK_ID_PREV",        "nunique"),
    ).reset_index()
 
    return agg
 
 
# ─── Credit Card ──────────────────────────────────────────────────────────────
 
def engineer_credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit card usage signals."""
    c = cc.copy()
 
    c["UTIL_RATE"]      = c["AMT_BALANCE"]        / (c["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    c["DRAWING_RATE"]   = c["AMT_DRAWINGS_CURRENT"]/ (c["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    c["PAYMENT_RATE"]   = c["AMT_PAYMENT_CURRENT"] / (c["AMT_BALANCE"] + 1)
    c["RECEIVABLE_RATE"]= c["AMT_RECEIVABLE_PRINCIPAL"] / (c["AMT_BALANCE"] + 1)
 
    agg = c.groupby("SK_ID_CURR").agg(
        CC_UTIL_RATE_MEAN         = ("UTIL_RATE",          "mean"),
        CC_UTIL_RATE_MAX          = ("UTIL_RATE",          "max"),
        CC_UTIL_RATE_STD          = ("UTIL_RATE",          "std"),
        CC_DRAWING_RATE_MEAN      = ("DRAWING_RATE",       "mean"),
        CC_PAYMENT_RATE_MEAN      = ("PAYMENT_RATE",       "mean"),
        CC_PAYMENT_RATE_MIN       = ("PAYMENT_RATE",       "min"),
        CC_AMT_BALANCE_MEAN       = ("AMT_BALANCE",        "mean"),
        CC_AMT_BALANCE_MAX        = ("AMT_BALANCE",        "max"),
        CC_AMT_DRAWINGS_MEAN      = ("AMT_DRAWINGS_CURRENT","mean"),
        CC_AMT_DRAWINGS_ATM_MEAN  = ("AMT_DRAWINGS_ATM_CURRENT","mean"),
        CC_COUNT                  = ("SK_ID_PREV",         "count"),
        CC_DPD_MEAN               = ("SK_DPD",             "mean"),
        CC_DPD_MAX                = ("SK_DPD",             "max"),
        CC_DPD_DEF_MEAN           = ("SK_DPD_DEF",         "mean"),
        CC_DISTINCT_MONTHS        = ("MONTHS_BALANCE",     "nunique"),
    ).reset_index()
 
    return agg
 
 
# ─── Main Application Table ───────────────────────────────────────────────────
 
def engineer_app_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core feature engineering on application_train / application_test.
    Returns a new DataFrame — does not modify in place.
    """
    d = df.copy()
 
    # ── Financial ratios ───────────────────────────────────────────────────
    d["CREDIT_INCOME_RATIO"]        = d["AMT_CREDIT"]  / (d["AMT_INCOME_TOTAL"] + 1)
    d["ANNUITY_INCOME_RATIO"]       = d["AMT_ANNUITY"] / (d["AMT_INCOME_TOTAL"] + 1)
    d["CREDIT_TERM"]                = d["AMT_ANNUITY"] / (d["AMT_CREDIT"] + 1)
    d["GOODS_CREDIT_RATIO"]         = d["AMT_GOODS_PRICE"] / (d["AMT_CREDIT"] + 1)
    d["GOODS_INCOME_RATIO"]         = d["AMT_GOODS_PRICE"] / (d["AMT_INCOME_TOTAL"] + 1)
    d["INCOME_CREDIT_PCT"]          = d["AMT_INCOME_TOTAL"] / (d["AMT_CREDIT"] + 1)
 
    # ── Age & employment ──────────────────────────────────────────────────
    d["AGE_YEARS"]                  = d["DAYS_BIRTH"].abs() / 365.25
    d["EMPLOYMENT_YEARS"]           = d["DAYS_EMPLOYED"].apply(lambda x: abs(x) / 365.25 if x < 0 else 0)
    d["EMPLOYED_RATIO"]             = d["EMPLOYMENT_YEARS"] / (d["AGE_YEARS"] + 1)
    d["CREDIT_TO_AGE"]              = d["AMT_CREDIT"] / (d["AGE_YEARS"] + 1)
    d["REGISTRATION_YEARS"]         = d["DAYS_REGISTRATION"].abs() / 365.25
    d["ID_PUBLISH_YEARS"]           = d["DAYS_ID_PUBLISH"].abs() / 365.25
    d["DAYS_LAST_PHONE_CHANGE_YEARS"] = d["DAYS_LAST_PHONE_CHANGE"].abs() / 365.25
 
    # ── Family ────────────────────────────────────────────────────────────
    d["INCOME_PER_PERSON"]          = d["AMT_INCOME_TOTAL"] / (d["CNT_FAM_MEMBERS"] + 1)
    d["CREDIT_PER_PERSON"]          = d["AMT_CREDIT"]       / (d["CNT_FAM_MEMBERS"] + 1)
    d["CHILDREN_RATIO"]             = d["CNT_CHILDREN"]     / (d["CNT_FAM_MEMBERS"] + 1)
    d["HAS_CHILDREN"]               = (d["CNT_CHILDREN"] > 0).astype(int)
 
    # ── External scores (most predictive features in Home Credit) ─────────
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    d["EXT_SOURCE_MEAN"]            = d[ext_cols].mean(axis=1)
    d["EXT_SOURCE_MIN"]             = d[ext_cols].min(axis=1)
    d["EXT_SOURCE_MAX"]             = d[ext_cols].max(axis=1)
    d["EXT_SOURCE_PROD"]            = d["EXT_SOURCE_1"] * d["EXT_SOURCE_2"] * d["EXT_SOURCE_3"]
    d["EXT_SOURCE_STD"]             = d[ext_cols].std(axis=1)
    d["EXT_SOURCE_RANGE"]           = d["EXT_SOURCE_MAX"] - d["EXT_SOURCE_MIN"]
    d["EXT1_EXT2_INTERACTION"]      = d["EXT_SOURCE_1"] * d["EXT_SOURCE_2"]
    d["EXT2_EXT3_INTERACTION"]      = d["EXT_SOURCE_2"] * d["EXT_SOURCE_3"]
    d["EXT1_EXT3_INTERACTION"]      = d["EXT_SOURCE_1"] * d["EXT_SOURCE_3"]
    d["EXT_CREDIT_RATIO"]           = d["EXT_SOURCE_MEAN"] * d["CREDIT_INCOME_RATIO"]
    d["EXT_AGE_INTERACTION"]        = d["EXT_SOURCE_MEAN"] * d["AGE_YEARS"]
 
    # ── Document flags ────────────────────────────────────────────────────
    doc_cols = [c for c in d.columns if "FLAG_DOCUMENT" in c]
    d["DOCUMENT_COUNT"]             = d[doc_cols].sum(axis=1)
    d["DOCUMENT_RATE"]              = d["DOCUMENT_COUNT"] / len(doc_cols)
 
    # ── Enquiry signals ───────────────────────────────────────────────────
    enq_cols = [c for c in d.columns if "AMT_REQ_CREDIT_BUREAU" in c]
    d["TOTAL_ENQUIRIES"]            = d[enq_cols].sum(axis=1)
    if "AMT_REQ_CREDIT_BUREAU_WEEK" in d.columns and "TOTAL_ENQUIRIES" in d.columns:
        d["RECENT_ENQUIRY_RATIO"]   = d["AMT_REQ_CREDIT_BUREAU_WEEK"] / (d["TOTAL_ENQUIRIES"] + 1)
    if "AMT_REQ_CREDIT_BUREAU_YEAR" in d.columns:
        d["YEAR_ENQUIRY_RATE"]      = d["AMT_REQ_CREDIT_BUREAU_YEAR"] / (d["AGE_YEARS"] + 1)
 
    # ── Asset flags ───────────────────────────────────────────────────────
    d["HAS_CAR"]                    = (d["FLAG_OWN_CAR"] == "Y").astype(int)
    d["HAS_REALTY"]                 = (d["FLAG_OWN_REALTY"] == "Y").astype(int)
    d["HAS_CAR_REALTY"]             = (d["HAS_CAR"] & d["HAS_REALTY"]).astype(int)
 
    # ── Contact flags ─────────────────────────────────────────────────────
    contact_cols = [c for c in d.columns if "FLAG_CONT_MOBILE" in c or "FLAG_PHONE" in c or "FLAG_EMAIL" in c]
    d["CONTACT_COUNT"]              = d[contact_cols].sum(axis=1)
 
    # ── Social circle ─────────────────────────────────────────────────────
    if "OBS_30_CNT_SOCIAL_CIRCLE" in d.columns:
        d["SOCIAL_CIRCLE_DEF_RATE"] = d["DEF_30_CNT_SOCIAL_CIRCLE"] / (d["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    if "OBS_60_CNT_SOCIAL_CIRCLE" in d.columns:
        d["SOCIAL_CIRCLE_DEF_RATE_60"] = d["DEF_60_CNT_SOCIAL_CIRCLE"] / (d["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
 
    # ── Label encode categoricals ─────────────────────────────────────────
    cat_cols = d.select_dtypes("object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        d[col] = d[col].fillna("Unknown")
        d[col] = le.fit_transform(d[col].astype(str))
 
    return d
 
 
# ─── Full pipeline class ──────────────────────────────────────────────────────
 
class FeatureEngineer:
    """
    End-to-end feature engineering orchestrator.
 
    Parameters
    ----------
    cfg : dataclass  — project config with OUTPUT_DIR, MODEL_DIR, SEED fields.
    """
 
    def __init__(self, cfg):
        self.cfg = cfg
 
    def fit_transform(
        self,
        tables: Dict[str, pd.DataFrame],
        mode: str = "train",
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Build the full feature matrix.
 
        Returns
        -------
        (train_df, test_df) — both indexed by SK_ID_CURR.
        test_df is None when mode == "score".
        """
        print("⚙️  Engineering application features...")
        train_eng = engineer_app_features(tables["app_train"])
        test_eng  = engineer_app_features(tables["app_test"]) if "app_test" in tables else None
 
        print("⚙️  Engineering bureau features...")
        bureau_feat = engineer_bureau_features(tables["bureau"], tables["bureau_balance"])
 
        print("⚙️  Engineering previous application features...")
        prev_feat = engineer_prev_app_features(tables["prev_app"])
 
        print("⚙️  Engineering installments features...")
        inst_feat = engineer_installments_features(tables["installments"])
 
        print("⚙️  Engineering POS Cash features...")
        pos_feat  = engineer_pos_cash_features(tables["pos_cash"])
 
        print("⚙️  Engineering credit card features...")
        cc_feat   = engineer_credit_card_features(tables["credit_card"])
 
        def _merge(app_df):
            df = app_df.copy()
            for feat, name in [
                (bureau_feat, "bureau"),
                (prev_feat,   "prev_app"),
                (inst_feat,   "installments"),
                (pos_feat,    "pos_cash"),
                (cc_feat,     "credit_card"),
            ]:
                df = df.merge(feat, on="SK_ID_CURR", how="left")
                print(f"   Merged {name}: {df.shape}")
            return df
 
        train_full = _merge(train_eng)
        test_full  = _merge(test_eng) if test_eng is not None else None
 
        return train_full, test_full