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

        st.subheader("📁 Workbook Overview")
        st.dataframe(
            pd.DataFrame(overview),
            use_container_width=True
        )

        selected_sheet = st.selectbox(
            "Select sheet",
            sheets
        )

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )

        st.subheader("📋 Data Preview")
        st.dataframe(
            df.head(100),
            use_container_width=True
        )

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

        st.subheader("📈 Descriptive Statistics")

        numeric = df.select_dtypes(
            include=np.number
        )

        if not numeric.empty:
            st.dataframe(
                numeric.describe().T,
                use_container_width=True
            )
        else:
            st.info("No numeric variables detected.")

        st.subheader("⚠️ Data Quality")

        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())

        if missing:
            st.warning(
                f"{missing} missing cell(s) detected."
            )
        else:
            st.success("No missing cells.")

        if duplicates:
            st.warning(
                f"{duplicates} duplicate row(s) detected."
            )
        else:
            st.success("No duplicate rows.")

    except Exception as error:
        st.error(
            f"Excel file could not be read: {error}"
        )

else:
    st.info(
        "Upload an Excel .xlsx file to start."
    )
