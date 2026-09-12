"""
Research Analysis Tool
-----------------------
An SPSS-style Streamlit app for exploring and analysing research data
stored in Excel workbooks: descriptive statistics, frequencies,
group comparisons, t-tests, chi-square, and correlation.
"""

import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Research Analysis Tool", page_icon="📊", layout="wide")

ID_PATTERN = re.compile(r"(?:^|_)id(?:_|$)")
MAX_CATEGORICAL_LEVELS = 20
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_id_column(name: str) -> bool:
    """Flag columns that look like row identifiers.

    Matches 'id', 'participant_id', 'record_id' etc. as whole tokens,
    so it will NOT wrongly flag things like 'Grid', 'Paid', 'Avoid'.
    """
    return bool(ID_PATTERN.search(name.lower()))


@st.cache_data(show_spinner=False)
def list_sheets(file_bytes: bytes) -> list:
    """Cached: sheet names for an uploaded workbook."""
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    """Cached: read one sheet. Re-reads only when file or sheet changes."""
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


def numeric_candidates(df: pd.DataFrame) -> list:
    return list(df.select_dtypes(include=np.number).columns)


def categorical_candidates(df: pd.DataFrame, exclude=()) -> list:
    """Columns usable as a grouping/categorical variable.

    Works for BOTH numeric-coded groups (e.g. sex as 1/2) and text
    categories, as long as they aren't ID-like and have a manageable
    number of levels. This is the fix for the old bug where a numeric
    grouping column (like a 0/1 treatment flag) was silently excluded.
    """
    out = []
    for col in df.columns:
        if col in exclude or is_id_column(col):
            continue
        if df[col].nunique(dropna=True) <= MAX_CATEGORICAL_LEVELS:
            out.append(col)
    return out


def download_button_for(df: pd.DataFrame, label: str, filename: str, key: str):
    st.download_button(
        label=f"⬇️ Download {label} (CSV)",
        data=df.to_csv(index=True).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def cohens_d_independent(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    n1, n2 = len(a), len(b)
    pooled_var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_sd


def cohens_d_paired(diff: pd.Series) -> float:
    """Cohen's d for paired samples: mean difference / SD of differences."""
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return diff.mean() / sd


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    """Cramer's V effect size for a contingency table."""
    n = table.to_numpy().sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    if denom == 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))


def has_constant_column(clean: pd.DataFrame, cols) -> bool:
    return any(clean[c].nunique(dropna=True) <= 1 for c in cols)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("file_bytes", None)
st.session_state.setdefault("variable_meta", {})  # {column_name: custom_label}

st.title("📊 Research Analysis Tool")
st.caption("SPSS-style research data analysis")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 File")
    uploaded_file = st.file_uploader("Open Excel file", type=["xlsx"])

    if uploaded_file is not None:
        st.session_state.file_bytes = uploaded_file.getvalue()

    st.divider()
    st.header("🧭 Navigation")
    page = st.radio("Go to", ["📋 Data View", "⚙️ Variable View", "📊 Analyze"])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if st.session_state.file_bytes is None:
    st.info("📁 Please upload an Excel (.xlsx) file from the sidebar.")
    st.markdown(
        """
        ### How to use
        1. Upload your `.xlsx` file.
        2. Select a worksheet.
        3. Use **Data View** to inspect your data and data quality.
        4. Use **Variable View** to review/label variables.
        5. Use **Analyze** to run statistical tests.
        """
    )
    st.stop()

try:
    sheets = list_sheets(st.session_state.file_bytes)
    selected_sheet = st.sidebar.selectbox("Select Sheet", sheets)
    df = load_excel(st.session_state.file_bytes, selected_sheet)
except Exception as error:
    st.error(f"Excel file could not be read: {error}")
    st.stop()

if df.empty:
    st.warning("This sheet has no rows.")
    st.stop()

numeric_columns = numeric_candidates(df)


# ---------------------------------------------------------------------------
# DATA VIEW
# ---------------------------------------------------------------------------

if page == "📋 Data View":
    st.header("📋 Data View")
    st.caption("SPSS-style data table")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Variables", len(df.columns))
    col3.metric("Missing Cells", int(df.isna().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.divider()
    st.dataframe(df, use_container_width=True, height=550)

    st.divider()
    st.subheader("🔎 Data Quality")

    quality = pd.DataFrame({
        "Variable": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True)
    download_button_for(quality, "data quality report", "data_quality.csv", "dl_quality")


# ---------------------------------------------------------------------------
# VARIABLE VIEW
# ---------------------------------------------------------------------------

elif page == "⚙️ Variable View":
    st.header("⚙️ Variable View")
    st.caption("SPSS-style variable definition — edit the Label column to rename "
               "variables for display in charts and result tables.")

    rows = []
    for i, column in enumerate(df.columns):
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        is_id = is_id_column(column)
        rows.append({
            "No.": i + 1,
            "Name": column,
            "Type": "Numeric" if is_numeric else "String",
            "Label": st.session_state.variable_meta.get(column, column),
            "Missing": int(df[column].isna().sum()),
            "Unique": int(df[column].nunique(dropna=True)),
            "Measure": "ID" if is_id else ("Scale" if is_numeric else "Nominal"),
            "Role": "Identifier" if is_id else "Analysis",
        })
    variable_view = pd.DataFrame(rows)

    edited = st.data_editor(
        variable_view,
        use_container_width=True,
        height=550,
        disabled=["No.", "Name", "Type", "Missing", "Unique", "Measure", "Role"],
        key="variable_editor",
    )

    # Persist any label edits back into session state
    for _, row in edited.iterrows():
        if row["Label"] != row["Name"]:
            st.session_state.variable_meta[row["Name"]] = row["Label"]
        elif row["Name"] in st.session_state.variable_meta:
            del st.session_state.variable_meta[row["Name"]]

    st.info(
        "💡 Variables detected as IDs are treated as identifiers and should not "
        "be used as statistical outcomes, predictors, or grouping variables."
    )


# ---------------------------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------------------------

elif page == "📊 Analyze":
    st.header("📊 Analyze")
    st.caption("Statistical analysis")

    analysis = st.selectbox(
        "Select analysis",
        [
            "Descriptive Statistics",
            "Frequencies",
            "Group-wise Descriptive Statistics",
            "Independent t-test",
            "Paired t-test",
            "Chi-square test",
            "Correlation",
        ],
    )

    # --- Descriptive Statistics ------------------------------------------
    if analysis == "Descriptive Statistics":
        st.subheader("📈 Descriptive Statistics")
        if not numeric_columns:
            st.warning("No numeric variables detected.")
        else:
            selected = st.multiselect("Select numeric variables", numeric_columns,
                                       default=numeric_columns)
            if selected:
                data = df[selected]
                result = pd.DataFrame({
                    "N": data.count(),
                    "Mean": data.mean(),
                    "Median": data.median(),
                    "SD": data.std(),
                    "Minimum": data.min(),
                    "Maximum": data.max(),
                    "Range": data.max() - data.min(),
                    "IQR": data.quantile(0.75) - data.quantile(0.25),
                    "Skewness": data.skew(),
                    "Kurtosis": data.kurtosis(),
                }).round(3)
                st.dataframe(result, use_container_width=True)
                download_button_for(result, "descriptives", "descriptive_statistics.csv", "dl_desc")

    # --- Frequencies --------------------------------------------------
    elif analysis == "Frequencies":
        st.subheader("🔢 Frequencies")
        variable = st.selectbox("Select variable", df.columns)

        frequency = df[variable].value_counts(dropna=False).reset_index()
        frequency.columns = ["Value", "Frequency"]
        frequency["Percent"] = (frequency["Frequency"] / len(df) * 100).round(2)

        st.dataframe(frequency, use_container_width=True)
        download_button_for(frequency, "frequencies", "frequencies.csv", "dl_freq")

        if frequency["Value"].nunique() <= MAX_CATEGORICAL_LEVELS:
            st.bar_chart(frequency.set_index("Value")["Frequency"])

    # --- Group-wise Descriptive -----------------------------------------
    elif analysis == "Group-wise Descriptive Statistics":
        st.subheader("📊 Group-wise Descriptive Statistics")

        if not numeric_columns:
            st.warning("No numeric outcome variables detected.")
        else:
            outcome = st.selectbox("Select numeric variable", numeric_columns)
            group_candidates = categorical_candidates(df, exclude=[outcome])

            if not group_candidates:
                st.warning("No suitable grouping variable detected.")
            else:
                group_variable = st.selectbox("Select grouping variable", group_candidates)

                result = (
                    df.groupby(group_variable)[outcome]
                    .agg(N="count", Mean="mean", Median="median", SD="std",
                         Minimum="min", Maximum="max")
                    .reset_index()
                )
                result["Range"] = result["Maximum"] - result["Minimum"]
                result = result.round(3)

                st.dataframe(result, use_container_width=True)
                download_button_for(result, "group-wise descriptives", "groupwise_descriptives.csv", "dl_group")

    # --- Independent t-test ------------------------------------------
    elif analysis == "Independent t-test":
        st.subheader("🧪 Independent t-test")

        if not numeric_columns:
            st.warning("No numeric outcome variable available.")
        else:
            outcome = st.selectbox("Select numeric outcome", numeric_columns)
            candidates = [
                c for c in categorical_candidates(df, exclude=[outcome])
                if df[c].dropna().nunique() == 2
            ]

            if not candidates:
                st.warning("No suitable two-group variable detected.")
            else:
                group_variable = st.selectbox("Select grouping variable", candidates)
                groups = df[group_variable].dropna().unique()
                group1, group2 = groups[0], groups[1]

                data1 = pd.to_numeric(df.loc[df[group_variable] == group1, outcome],
                                       errors="coerce").dropna()
                data2 = pd.to_numeric(df.loc[df[group_variable] == group2, outcome],
                                       errors="coerce").dropna()

                if len(data1) < 2 or len(data2) < 2:
                    st.warning("Each group needs at least 2 valid observations.")
                else:
                    levene_p = stats.levene(data1, data2).pvalue
                    result = stats.ttest_ind(data1, data2, equal_var=False)  # Welch's, robust default
                    d = cohens_d_independent(data1, data2)

                    summary = pd.DataFrame({
                        "Group": [str(group1), str(group2)],
                        "N": [len(data1), len(data2)],
                        "Mean": [data1.mean(), data2.mean()],
                        "SD": [data1.std(), data2.std()],
                    }).round(3)
                    st.subheader("Group Summary")
                    st.dataframe(summary, use_container_width=True)

                    test_result = pd.DataFrame({
                        "Mean Difference": [data1.mean() - data2.mean()],
                        "t-value": [result.statistic],
                        "p-value": [result.pvalue],
                        "Cohen's d": [d],
                    }).round(4)
                    st.subheader("Test Result (Welch's t-test)")
                    st.dataframe(test_result, use_container_width=True)
                    download_button_for(test_result, "t-test result", "independent_ttest.csv", "dl_ttest")

                    st.caption(
                        f"Levene's test for equal variances: p = {levene_p:.4f} "
                        + ("(variances differ — Welch's correction is appropriate here)."
                           if levene_p < ALPHA else
                           "(variances are similar; Welch's t-test is still valid and safe as a default).")
                    )

                    if result.pvalue < ALPHA:
                        st.success(f"Statistically significant difference (p < {ALPHA}).")
                    else:
                        st.info(f"No statistically significant difference (p ≥ {ALPHA}).")

    # --- Paired t-test --------------------------------------------------
    elif analysis == "Paired t-test":
        st.subheader("🧪 Paired t-test")

        if len(numeric_columns) < 2:
            st.warning("At least two numeric variables are required.")
        else:
            before = st.selectbox("Select first measurement", numeric_columns)
            after = st.selectbox("Select second measurement",
                                  [c for c in numeric_columns if c != before])

            paired = pd.DataFrame({
                "Before": pd.to_numeric(df[before], errors="coerce"),
                "After": pd.to_numeric(df[after], errors="coerce"),
            }).dropna()

            if len(paired) < 2:
                st.warning("At least two complete paired observations are required.")
            else:
                result = stats.ttest_rel(paired["Before"], paired["After"])
                difference = paired["After"] - paired["Before"]
                d = cohens_d_paired(difference)

                summary = pd.DataFrame({
                    "Measurement": [before, after],
                    "N": [len(paired)] * 2,
                    "Mean": [paired["Before"].mean(), paired["After"].mean()],
                    "SD": [paired["Before"].std(), paired["After"].std()],
                }).round(3)
                st.subheader("Paired Summary")
                st.dataframe(summary, use_container_width=True)

                test_result = pd.DataFrame({
                    "Mean Difference (After-Before)": [difference.mean()],
                    "t-value": [result.statistic],
                    "df": [len(paired) - 1],
                    "p-value": [result.pvalue],
                    "Cohen's d": [d],
                }).round(4)
                st.subheader("Test Result")
                st.dataframe(test_result, use_container_width=True)
                download_button_for(test_result, "paired t-test result", "paired_ttest.csv", "dl_paired")

                if result.pvalue < ALPHA:
                    st.success(f"Statistically significant difference (p < {ALPHA}).")
                else:
                    st.info(f"No statistically significant difference (p ≥ {ALPHA}).")

    # --- Chi-square -------------------------------------------------
    elif analysis == "Chi-square test":
        st.subheader("🧮 Chi-square Test")

        categorical = categorical_candidates(df)
        if len(categorical) < 2:
            st.warning("At least two categorical variables are required.")
        else:
            row_variable = st.selectbox("Select row variable", categorical)
            column_variable = st.selectbox(
                "Select column variable", [c for c in categorical if c != row_variable]
            )

            table = pd.crosstab(df[row_variable], df[column_variable])
            st.subheader("Contingency Table")
            st.dataframe(table, use_container_width=True)

            chi2, p, dof, expected = stats.chi2_contingency(table)
            v = cramers_v(table, chi2)
            low_expected_pct = (expected < 5).mean() * 100

            result = pd.DataFrame({
                "Chi-square": [chi2], "df": [dof], "p-value": [p], "Cramer's V": [v],
            }).round(4)
            st.subheader("Test Result")
            st.dataframe(result, use_container_width=True)
            download_button_for(result, "chi-square result", "chi_square.csv", "dl_chi")

            if low_expected_pct > 20:
                st.warning(
                    f"⚠️ {low_expected_pct:.0f}% of expected cell counts are below 5 — "
                    "the chi-square approximation may not be reliable here."
                )
                if table.shape == (2, 2):
                    odds_ratio, fisher_p = stats.fisher_exact(table)
                    st.caption(f"Fisher's exact test (recommended for this 2×2 table): "
                               f"p = {fisher_p:.4f}")

            if p < ALPHA:
                st.success(f"Statistically significant association (p < {ALPHA}).")
            else:
                st.info(f"No statistically significant association (p ≥ {ALPHA}).")

    # --- Correlation --------------------------------------------------
    elif analysis == "Correlation":
        st.subheader("🔗 Correlation")

        if len(numeric_columns) < 2:
            st.warning("At least two numeric variables are required.")
        else:
            variable1 = st.selectbox("Select first variable", numeric_columns)
            variable2 = st.selectbox("Select second variable",
                                      [c for c in numeric_columns if c != variable1])
            method = st.selectbox("Correlation method", ["Pearson", "Spearman", "Kendall"])

            clean = df[[variable1, variable2]].apply(pd.to_numeric, errors="coerce").dropna()

            if len(clean) < 3:
                st.warning("At least three complete observations are required.")
            elif has_constant_column(clean, [variable1, variable2]):
                st.warning("One of the selected variables has no variation (constant "
                           "values) — correlation is undefined.")
            else:
                if method == "Pearson":
                    coefficient, p = stats.pearsonr(clean[variable1], clean[variable2])
                elif method == "Spearman":
                    coefficient, p = stats.spearmanr(clean[variable1], clean[variable2])
                else:
                    coefficient, p = stats.kendalltau(clean[variable1], clean[variable2])

                result = pd.DataFrame({
                    "Method": [method], "Correlation": [coefficient],
                    "p-value": [p], "N": [len(clean)],
                }).round(4)
                st.dataframe(result, use_container_width=True)
                download_button_for(result, "correlation result", "correlation.csv", "dl_corr")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("Research Analysis Tool | SPSS-style workflow")
