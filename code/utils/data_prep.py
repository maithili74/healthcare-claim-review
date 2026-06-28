import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
OUT_PATH = f"{OUT_DIR}/claim_review_dataset.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# loading data
def load_data():
    patients = pd.read_csv(f"{RAW_DIR}/patients.csv")
    claims = pd.read_csv(f"{RAW_DIR}/claims.csv")
    conditions = pd.read_csv(f"{RAW_DIR}/conditions.csv")
    procedures = pd.read_csv(f"{RAW_DIR}/procedures.csv")
    providers = pd.read_csv(f"{RAW_DIR}/providers.csv")
    return patients, claims, conditions, procedures, providers


def prepare_dataset():
    patients, claims, conditions, procedures, providers = load_data()

    patients = patients.rename(columns={
        "Id": "patient_id",
        "BIRTHDATE": "birthdate",
        "GENDER": "gender",
        "HEALTHCARE_EXPENSES": "healthcare_expenses",
        "HEALTHCARE_COVERAGE": "healthcare_coverage"
    })

    claims = claims.rename(columns={
        "Id": "claim_id",
        "PATIENTID": "patient_id",
        "PROVIDERID": "provider_id",
        "SERVICEDATE": "service_date",
        "DIAGNOSIS1": "diagnosis_code",
        "STATUS1": "claim_status",
        "OUTSTANDING1": "outstanding_amount"
    })

    providers = providers.rename(columns={
        "Id": "provider_id",
        "NAME": "provider_name",
        "SPECIALITY": "provider_speciality",
        "UTILIZATION": "provider_utilization"
    })

    conditions = conditions.rename(columns={
        "PATIENT": "patient_id",
        "CODE": "condition_code",
        "DESCRIPTION": "condition_description"
    })

    procedures = procedures.rename(columns={
        "PATIENT": "patient_id",
        "CODE": "procedure_code",
        "DESCRIPTION": "procedure_description",
        "BASE_COST": "base_cost",
        "REASONDESCRIPTION": "reason_description"
    })

    patients = patients[[
        "patient_id", "birthdate", "gender",
        "healthcare_expenses", "healthcare_coverage"
    ]]

    claims = claims[[
        "claim_id", "patient_id", "provider_id",
        "service_date", "diagnosis_code",
        "claim_status", "outstanding_amount"
    ]]

    providers = providers[[
        "provider_id", "provider_name",
        "provider_speciality", "provider_utilization"
    ]]

    conditions = conditions[[
        "patient_id", "condition_code", "condition_description"
    ]]

    procedures = procedures[[
        "patient_id", "procedure_code", "procedure_description",
        "base_cost", "reason_description"
    ]]

    # Use one condition and one procedure per patient for simple POC
    conditions_simple = conditions.drop_duplicates(subset=["patient_id"])
    procedures_simple = procedures.drop_duplicates(subset=["patient_id"])

    df = claims.merge(patients, on="patient_id", how="left")
    df = df.merge(providers, on="provider_id", how="left")
    df = df.merge(conditions_simple, on="patient_id", how="left")
    df = df.merge(procedures_simple, on="patient_id", how="left")

    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce", utc=True).dt.tz_localize(None)

    df["age"] = ((df["service_date"] - df["birthdate"]).dt.days / 365.25).round()

    numeric_cols = [
        "base_cost", "outstanding_amount", "provider_utilization",
        "healthcare_expenses", "healthcare_coverage"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


    ## feature engineering
    # synthea costs repeat a lot, so added realistic payer variation for POC
    np.random.seed(42)
    df["claim_amount"] = (df["base_cost"] * np.random.uniform(0.8, 1.3, len(df))).round(2)

    if len(df) > 20:
        anomaly_idx = df.sample(frac=0.03, random_state=42).index
        df.loc[anomaly_idx, "claim_amount"] = (
            df.loc[anomaly_idx, "claim_amount"] *
            np.random.uniform(2.0, 4.0, len(anomaly_idx))
        ).round(2)

    df["month"] = df["service_date"].dt.to_period("M").astype(str)

    # coverage ratio
    df["coverage_ratio"] = np.where(
        df["healthcare_expenses"] > 0,
        df["healthcare_coverage"] / df["healthcare_expenses"],
        0
    ).round(3)

    # provider-level features
    provider_stats = df.groupby("provider_id").agg(
        provider_avg_claim_amount=("claim_amount", "mean"),
        provider_claim_count=("claim_id", "count")
    ).reset_index()

    df = df.merge(provider_stats, on="provider_id", how="left")

    # specialty-level features
    specialty_stats = df.groupby("provider_speciality").agg(
        specialty_avg_claim_amount=("claim_amount", "mean")
    ).reset_index()

    df = df.merge(specialty_stats, on="provider_speciality", how="left")

    df["amount_vs_provider_avg"] = (
        df["claim_amount"] / df["provider_avg_claim_amount"].replace(0, np.nan)
    ).fillna(0).round(2)

    df["amount_vs_specialty_avg"] = (
        df["claim_amount"] / df["specialty_avg_claim_amount"].replace(0, np.nan)
    ).fillna(0).round(2)

    # z-score
    proc_mean = df.groupby("procedure_description")["claim_amount"].transform("mean")
    proc_std = df.groupby("procedure_description")["claim_amount"].transform("std").replace(0, np.nan)

    df["procedure_amount_zscore"] = (
        (df["claim_amount"] - proc_mean) / proc_std
    ).fillna(0).round(2)

    # duplicate claim flag
    duplicate_cols = [
        "patient_id", "provider_id", "procedure_description",
        "service_date", "claim_amount"
    ]

    df["duplicate_flag"] = df.duplicated(subset=duplicate_cols, keep=False).astype(int)

    # repeating procedure within 30 days for same patient
    df = df.sort_values(["patient_id", "procedure_description", "service_date"])

    df["previous_same_procedure_date"] = (
        df.groupby(["patient_id", "procedure_description"])["service_date"].shift(1)
    )

    df["days_since_same_procedure"] = (
        df["service_date"] - df["previous_same_procedure_date"]
    ).dt.days

    df["repeat_procedure_30d"] = (
        (df["days_since_same_procedure"].notna()) &
        (df["days_since_same_procedure"] <= 30)
    ).astype(int)

    # monthly provider claim volume
    monthly_counts = df.groupby(["provider_id", "month"]).agg(
        provider_monthly_claim_count=("claim_id", "count")
    ).reset_index()

    df = df.merge(monthly_counts, on=["provider_id", "month"], how="left")

    # cleaning text fields
    text_cols = [
        "provider_name", "provider_speciality",
        "condition_description", "procedure_description", "reason_description",
        "claim_status"
    ]

    for col in text_cols:
        df[col] = df[col].fillna("Unknown")

    final_cols = [
        "claim_id", "patient_id", "provider_id", "provider_name", "provider_speciality",
        "service_date", "month", "age", "gender",
        "condition_code", "condition_description",
        "procedure_code", "procedure_description", "reason_description",
        "base_cost", "claim_amount", "outstanding_amount", "claim_status",
        "healthcare_expenses", "healthcare_coverage", "coverage_ratio",
        "provider_utilization", "provider_avg_claim_amount", "provider_claim_count",
        "specialty_avg_claim_amount", "amount_vs_provider_avg", "amount_vs_specialty_avg",
        "procedure_amount_zscore", "duplicate_flag", "repeat_procedure_30d",
        "days_since_same_procedure", "provider_monthly_claim_count"
    ]

    df = df[final_cols]

    df = df.dropna(subset=["claim_id", "patient_id", "provider_id"])
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved processed dataset to: {OUT_PATH}")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    prepare_dataset()