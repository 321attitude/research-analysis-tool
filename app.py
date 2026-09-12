import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Research Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Research Analysis Tool")
st.write("SPSS + Stata style research data analysis")

uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file:

    try:
        # =========================
        # READ EXCEL WORKBOOK
        # =========================
        excel = pd.ExcelFile(uploaded_file)
        sheets = excel.sheet_names

        st.success(f"{len(sheets)} sheet(s) detected.")

        overview = []

        for sheet in sheets:
            data = pd.read_excel(
                uploaded_file,
                sheet_name=sheet
            )

            overview.append({
                "Sheet": sheet,
                "Rows": len(data),
                "Columns": len(data.columns),
                "Missing cells": int(data.isna().sum().sum()),
                "Duplicate rows": int(data.duplicated().sum())
            })

        # =========================
        # WORKBOOK OVERVIEW
        # =========================
        st.subheader("📁 Workbook Overview")

        st.dataframe(
            pd.DataFrame(overview),
            use_container_width=True
        )

        # =========================
        # SELECT SHEET
        # =========================
        selected_sheet = st.selectbox(
            "Select sheet",
            sheets
        )

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )

        # =========================
        # DATA PREVIEW
        # =========================
        st.subheader("📋 Data Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        # =========================
        # VARIABLE INFORMATION
        # =========================
        st.subheader("🔎 Variable Information")

        info = pd.DataFrame({
            "Variable": df.columns,
            "Type": df.dtypes.astype(str),
            "Missing": df.isna().sum().values,
            "Missing %": (
                df.isna().mean().values * 100
            ).round(2),
            "Unique": [
                df[col].nunique(dropna=True)
                for col in df.columns
            ]
        })

        st.dataframe(
            info,
            use_container_width=True
        )

        # =========================
        # ANALYSIS
        # =========================
        st.header("📊 Statistical Analysis")

        analysis_option = st.selectbox(
            "Choose Analysis",
            [
                "Descriptive Statistics",
                "Frequencies",
                "Group-wise Descriptive Statistics"
            ]
        )

        # =========================
        # DESCRIPTIVE STATISTICS
        # =========================
        if analysis_option == "Descriptive Statistics":

            st.subheader("📈 Descriptive Statistics")

            numeric = df.select_dtypes(
                include=np.number
            )

            if not numeric.empty:

                descriptive = pd.DataFrame({
                    "N": numeric.count(),
                    "Mean": numeric.mean(),
                    "Median": numeric.median(),
                    "SD": numeric.std(),
                    "Minimum": numeric.min(),
                    "Maximum": numeric.max(),
                    "Range": numeric.max() - numeric.min(),
                    "IQR": (
                        numeric.quantile(0.75)
                        - numeric.quantile(0.25)
                    )
                })

                st.dataframe(
                    descriptive.round(3),
                    use_container_width=True
                )

            else:
                st.info(
                    "No numeric variables detected."
                )

        # =========================
        # FREQUENCIES
        # =========================
        elif analysis_option == "Frequencies":

            st.subheader("🔢 Frequencies & Percentages")

            selected_variable = st.selectbox(
                "Select variable",
                df.columns
            )

            frequency = (
                df[selected_variable]
                .value_counts(dropna=False)
                .reset_index()
            )

            frequency.columns = [
                "Value",
                "Frequency"
            ]

            frequency["Percentage"] = (
                frequency["Frequency"]
                / len(df)
                * 100
            ).round(2)

            st.dataframe(
                frequency,
                use_container_width=True
            )

        # =========================
        # GROUP-WISE DESCRIPTIVES
        # =========================
        elif analysis_option == "Group-wise Descriptive Statistics":

            st.subheader(
                "📊 Group-wise Descriptive Statistics"
            )

            numeric_columns = list(
                df.select_dtypes(
                    include=np.number
                ).columns
            )

            if not numeric_columns:

                st.info(
                    "No numeric variables available."
                )

            else:

                group_variable = st.selectbox(
                    "Select grouping variable",
                    df.columns
                )

                analysis_variable = st.selectbox(
                    "Select numeric variable",
                    numeric_columns
                )

                grouped = (
                    df.groupby(group_variable)[
                        analysis_variable
                    ]
                    .agg(
                        N="count",
                        Mean="mean",
                        Median="median",
                        SD="std",
                        Minimum="min",
                        Maximum="max"
                    )
                    .reset_index()
                )

                grouped["Range"] = (
                    grouped["Maximum"]
                    - grouped["Minimum"]
                )

                st.dataframe(
                    grouped.round(3),
                    use_container_width=True
                )

        # =========================
        # DATA QUALITY
        # =========================
        st.subheader("⚠️ Data Quality")

        missing = int(
            df.isna().sum().sum()
        )

        duplicates = int(
            df.duplicated().sum()
        )

        if missing:
            st.warning(
                f"{missing} missing cell(s) detected."
            )
        else:
            st.success(
                "No missing cells."
            )

        if duplicates:
            st.warning(
                f"{duplicates} duplicate row(s) detected."
            )
        else:
            st.success(
                "No duplicate rows."
            )

    except Exception as error:

        st.error(
            f"Excel file could not be read: {error}"
        )

else:

    st.info(
        "Upload an Excel .xlsx file to start."
        )
