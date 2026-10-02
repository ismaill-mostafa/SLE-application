

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent
CLINICAL_CSV = DATA_DIR / "cleaned_clinical_dataset.csv"
RAW_EXCEL = DATA_DIR / "Coded full patients.xlsx"


def normalize_value(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "na", "n/a", "none", "null"}:
            return 0
        if text in {"positive", "yes", "y", "true", "1", "1.0"}:
            return 1
        if text in {"negative", "no", "n", "false", "0", "0.0"}:
            return 0
        try:
            return float(text)
        except ValueError:
            return 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


def clean_nephritis(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.map(lambda x: normalize_value(x))
    return cleaned.astype(float)


def get_reference_data():
    if not CLINICAL_CSV.exists():
        raise FileNotFoundError("cleaned_clinical_dataset.csv was not found in the project folder.")

    df_clean = pd.read_csv(CLINICAL_CSV)
    df_clean = df_clean.copy()

    drop_cols = [
        "Age",
        "Age of diagnosis",
        "Age of onset of the disease",
        "Disease duration",
        "Target_SLE",
        "Diagnosis_Original",
    ]

    feature_cols = [col for col in df_clean.columns if col not in drop_cols]
    X = df_clean[feature_cols].copy()
    y = df_clean["Target_SLE"].astype(int)

    if RAW_EXCEL.exists():
        try:
            df_raw = pd.read_excel(RAW_EXCEL)
            if "Nephritis" in df_raw.columns:
                neph = clean_nephritis(df_raw["Nephritis"].iloc[: len(df_clean)].copy())
                X["Nephritis"] = neph.to_numpy()
            else:
                X["Nephritis"] = 0.0
        except Exception:
            X["Nephritis"] = 0.0
    else:
        X["Nephritis"] = 0.0

    feature_cols = list(X.columns)
    return X, y, feature_cols


def train_sle_model():
    X, y, feature_cols = get_reference_data()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    C=1.0,
                    solver="saga",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)

    coefs = model.named_steps["clf"].coef_[0]
    active_mask = np.abs(coefs) >= 1e-4
    active_features = [feature for feature, keep in zip(feature_cols, active_mask) if keep]
    active_coefs = coefs[active_mask]

    if len(active_coefs) == 0:
        min_abs_coef = 1.0
    else:
        min_abs_coef = np.min(np.abs(active_coefs))

    raw_points = active_coefs / min_abs_coef
    int_points = np.rint(raw_points).astype(int)

    nomogram_df = pd.DataFrame(
        {
            "Clinical Feature": active_features,
            "Lasso Coefficient": active_coefs,
            "Assigned Points": int_points,
        }
    ).sort_values(by="Assigned Points", ascending=False)

    return model, feature_cols, nomogram_df


def make_patient_record_from_dict(raw_patient: Dict[str, object], feature_cols: Iterable[str]) -> pd.DataFrame:
    feature_map = {str(k).strip(): v for k, v in raw_patient.items()}
    row = {}
    for col in feature_cols:
        if col in feature_map:
            row[col] = normalize_value(feature_map[col])
        else:
            row[col] = 0.0
    return pd.DataFrame([row], columns=feature_cols)


def make_patient_record_from_text(text_value: str, feature_cols: List[str]) -> pd.DataFrame:
    raw_parts = [part.strip() for part in text_value.split(",")]
    if len(raw_parts) != len(feature_cols):
        raise ValueError(
            f"Expected {len(feature_cols)} values, but received {len(raw_parts)}. "
            "Paste the values in the exact feature order shown below."
        )

    row = {feature: normalize_value(value) for feature, value in zip(feature_cols, raw_parts)}
    return pd.DataFrame([row], columns=feature_cols)


def score_patient(patient_df: pd.DataFrame, model, nomogram_df) -> Dict[str, object]:
    patient_df = patient_df.copy()
    patient_df = patient_df.reindex(columns=list(model.named_steps["scaler"].scale_.shape[0] if False else patient_df.columns), fill_value=0)

    prob = float(model.predict_proba(patient_df)[0, 1])
    pred = int(model.predict(patient_df)[0])

    if prob >= 0.75:
        category = "High likelihood of SLE"
    elif prob >= 0.5:
        category = "Intermediate likelihood of SLE"
    else:
        category = "Low likelihood of SLE"

    present_signs = [
        feature for feature in patient_df.columns if feature in nomogram_df["Clinical Feature"].values and patient_df.iloc[0][feature] == 1.0
    ]

    points = 0
    for _, row in nomogram_df.iterrows():
        feat = row["Clinical Feature"]
        pts = int(row["Assigned Points"])
        if feat in patient_df.columns and patient_df.iloc[0][feat] == 1.0:
            points += pts

    return {
        "Probability": prob,
        "Prediction": pred,
        "Category": category,
        "Nomogram Score": points,
        "Present Signs": present_signs,
        "Nephritis": 1 if "Nephritis" in patient_df.columns and patient_df.iloc[0]["Nephritis"] == 1.0 else 0,
    }


def format_patient_row_for_display(patient_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    display_df = patient_df.copy()
    display_df = display_df.reindex(columns=feature_cols, fill_value=0)
    return display_df


def get_manual_choices(feature: str):
    if feature == "Sex":
        return {"Male": 0, "Female": 1}
    if feature == "Marital status":
        return {"Single": 0, "Married": 1, "Other": 2}
    if feature == "Nephritis":
        return {"Negative": 0, "Positive": 1}
    return {"Not Exist": 0, "Exist": 1}


def run_single_patient_mode(model, feature_cols, nomogram_df):
    st.subheader("Single Patient Assessment")

    option = st.radio("Choose input method", ["Manual form", "Paste values as text"], horizontal=True)

    if option == "Manual form":
        with st.form("manual_patient_form"):
            inputs = {}
            for col in feature_cols:
                choices = get_manual_choices(col)
                selected_label = st.selectbox(
                    col,
                    options=list(choices),
                    index=0,
                    key=f"manual_{col}",
                )
                inputs[col] = choices[selected_label]
            submitted = st.form_submit_button("Score patient")

        if submitted:
            patient_df = make_patient_record_from_dict(inputs, feature_cols)
            result = score_patient(patient_df, model, nomogram_df)
            st.dataframe(format_patient_row_for_display(patient_df, feature_cols), use_container_width=True)
            display_result(result)

    else:
        st.caption("Paste a single patient row as comma-separated values in this exact order:")
        st.write(feature_cols)
        text_value = st.text_area("Patient features", value=",".join(["0"] * len(feature_cols)))

        if st.button("Score this patient"):
            try:
                patient_df = make_patient_record_from_text(text_value, feature_cols)
                result = score_patient(patient_df, model, nomogram_df)
                st.dataframe(format_patient_row_for_display(patient_df, feature_cols), use_container_width=True)
                display_result(result)
            except ValueError as exc:
                st.error(str(exc))


def display_result(result: Dict[str, object]):
    prob_pct = round(result["Probability"] * 100, 2)
    st.markdown(f"### Probability of SLE: {prob_pct}%")
    st.markdown(f"### Risk category: {result['Category']}")
    st.markdown(f"### Nomogram score: {result['Nomogram Score']} points")

    if result["Nephritis"] == 1:
        st.warning("Lupus nephritis is positive.")

    signs = result["Present Signs"]
    if signs:
        st.write("Present clinical features:")
        st.write(", ".join(signs[:10]))
    else:
        st.write("No active clinical features from the model were selected.")


def run_batch_mode(model, feature_cols, nomogram_df):
    st.subheader("Upload CSV / Excel File")
    uploaded_file = st.file_uploader("Upload patient dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return

        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        if "Target_SLE" in df.columns:
            df = df.drop(columns=["Target_SLE"])
        if "Diagnosis_Original" in df.columns:
            df = df.drop(columns=["Diagnosis_Original"])

        if "Nephritis" not in df.columns:
            df["Nephritis"] = 0

        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            for c in missing_cols:
                df[c] = 0

        detailed_rows = []
        for idx, row in df.iterrows():
            row_map = row.to_dict()
            patient_df = make_patient_record_from_dict(row_map, feature_cols)
            result = score_patient(patient_df, model, nomogram_df)
            detailed_rows.append(
                {
                    "Patient Index": idx,
                    "Probability (%)": round(result["Probability"] * 100, 2),
                    "Risk Category": result["Category"],
                    "Nomogram Score": result["Nomogram Score"],
                    "Nephritis": result["Nephritis"],
                    "Predicted SLE": result["Prediction"],
                    "Present Signs": ", ".join(result["Present Signs"][:10]),
                    "_detail_result": result,
                }
            )

        results_df = pd.DataFrame([
            {k: v for k, v in row.items() if k != "_detail_result"}
            for row in detailed_rows
        ])

        st.dataframe(results_df, use_container_width=True)

        if not results_df.empty:
            selected_patient = st.selectbox(
                "Choose a patient to inspect in detail",
                options=results_df["Patient Index"].tolist(),
                format_func=lambda value: f"Patient {value}",
            )

            selected_result = next(
                item["_detail_result"]
                for item in detailed_rows
                if item["Patient Index"] == selected_patient
            )

            st.subheader(f"Patient {selected_patient} detail")
            display_result(selected_result)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download prediction table",
            data=csv_bytes,
            file_name="sle_predictions.csv",
            mime="text/csv",
        )



def main():
    st.set_page_config(page_title="SLE Clinical Risk Scorer", layout="wide")
    st.title("SLE Clinical Risk Scoring App")
    st.caption("Clinical-only model using the same logic as the notebook workflow")

    try:
        model, feature_cols, nomogram_df = train_sle_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2 = st.tabs(["Single patient", "Upload batch file"])

    with tab1:
        run_single_patient_mode(model, feature_cols, nomogram_df)

    with tab2:
        run_batch_mode(model, feature_cols, nomogram_df)


if __name__ == "__main__":
    main()
