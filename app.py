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

try:
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Research Analysis Tool", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# "Classic desktop stats software" chrome — title bar / menu bar / toolbar.
# Purely decorative (the menu/toolbar items aren't wired to actions); the
# real navigation is the tab strip rendered further down.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .spss-shell {
        font-family: "Segoe UI", Arial, sans-serif;
        border: 1px solid #94a3b8;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 0.75rem;
    }
    .spss-titlebar {
        background: linear-gradient(180deg, #3b6ea5, #2c5686);
        color: #ffffff;
        padding: 7px 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
    }
    .spss-titlebar-text { flex: 1; font-weight: 500; }
    .spss-titlebar-dots span {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background: rgba(255,255,255,0.55); margin-left: 5px;
    }
    .spss-menubar {
        background: #eef2f7; border-bottom: 1px solid #cbd5e1;
        padding: 5px 12px; display: flex; gap: 20px; font-size: 13px; color: #1e293b;
    }
    .spss-menubar span { cursor: default; padding: 2px 5px; border-radius: 3px; }
    .spss-menubar span:hover { background: #dbe6f3; }
    .spss-menubar span.active { background: #2c5686; color: #ffffff; }
    .spss-toolbar {
        background: #f8fafc; border-bottom: 1px solid #cbd5e1;
        padding: 7px 12px; display: flex; gap: 12px; align-items: center; font-size: 16px;
    }
    .spss-toolbar-sep { width: 1px; height: 18px; background: #cbd5e1; }

    /* Restyle the page-navigation radio (in the main area) as a classic tab strip */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 0; border-bottom: 2px solid #94a3b8; background: #eef2f7;
    }
    div[data-testid="stRadio"] label {
        border: 1px solid #cbd5e1; border-bottom: none; border-radius: 6px 6px 0 0;
        padding: 6px 16px; margin-right: 2px; background: #e2e8f0;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #ffffff; border-bottom: 2px solid #ffffff; margin-bottom: -2px;
        font-weight: 600; color: #2c5686;
    }
    div[data-testid="stRadio"] input[type="radio"] { display: none; } /* hide the radio circle, tab-strip look */
    </style>

    <div class="spss-shell">
        <div class="spss-titlebar">
            <span>📊</span>
            <span class="spss-titlebar-text">Research Analysis Tool — Data Editor</span>
            <span class="spss-titlebar-dots"><span></span><span></span><span></span></span>
        </div>
        <div class="spss-menubar">
            <span>File</span><span>Edit</span><span>View</span><span>Data</span>
            <span>Transform</span><span class="active">Analyze</span><span>Graphs</span>
            <span>Utilities</span><span>Extensions</span><span>Window</span><span>Help</span>
        </div>
        <div class="spss-toolbar">
            <span title="Open">📂</span><span title="Save">💾</span><span title="Print">🖨️</span>
            <span class="spss-toolbar-sep"></span>
            <span title="Undo">↩️</span><span title="Redo">↪️</span>
            <span class="spss-toolbar-sep"></span>
            <span title="Find">🔍</span><span title="Insert Variable">➕</span><span title="Split File">🔀</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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


def variable_type_icon(df: pd.DataFrame, column: str) -> str:
    """Small icon reflecting a variable's role, echoing SPSS's Measure-type icons
    (a ruler for Scale, a tag for Nominal, an ID badge for identifiers)."""
    if is_id_column(column):
        return "🆔"
    if pd.api.types.is_numeric_dtype(df[column]):
        return "📏"
    return "🏷️"


def has_constant_column(clean: pd.DataFrame, cols) -> bool:
    return any(clean[c].nunique(dropna=True) <= 1 for c in cols)


def eta_squared_oneway(groups: list) -> float:
    """Eta-squared effect size for a one-way ANOVA (SS_between / SS_total)."""
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum((all_data - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def cronbachs_alpha(item_df: pd.DataFrame) -> float:
    """Cronbach's alpha for a set of scale items (rows = respondents, columns = items)."""
    item_df = item_df.dropna()
    k = item_df.shape[1]
    if k < 2 or len(item_df) < 2:
        return np.nan
    item_variances = item_df.var(ddof=1, axis=0)
    total_variance = item_df.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_variances.sum() / total_variance)


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal-rule integration, written by hand so it doesn't depend on
    np.trapz (removed) vs np.trapezoid (its replacement) across numpy versions."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def roc_curve_manual(y_true: pd.Series, y_score: pd.Series):
    """ROC curve (FPR, TPR) and AUC computed from scratch (no scikit-learn
    dependency needed). Sweeps every distinct predicted-probability threshold,
    matching the standard step-function definition of the ROC curve."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    positives = y_true_sorted.sum()
    negatives = len(y_true_sorted) - positives
    if positives == 0 or negatives == 0:
        return None, None, np.nan

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    threshold_idx = np.r_[distinct_idx, len(y_true_sorted) - 1]

    tpr = tps[threshold_idx] / positives
    fpr = fps[threshold_idx] / negatives

    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]
    auc = _trapezoid_area(tpr, fpr)
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("file_bytes", None)
st.session_state.setdefault("variable_meta", {})  # {column_name: custom_label}

page = st.radio(
    "Go to",
    ["📋 Data View", "⚙️ Variable View", "📊 Analyze"],
    horizontal=True,
    label_visibility="collapsed",
    key="page_nav",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 File")
    uploaded_file = st.file_uploader("Open Excel file", type=["xlsx"])

    if uploaded_file is not None:
        st.session_state.file_bytes = uploaded_file.getvalue()


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
    column_config = {
        col: st.column_config.Column(label=f"{variable_type_icon(df, col)} {col}")
        for col in df.columns
    }
    st.dataframe(df, use_container_width=True, height=550, column_config=column_config)

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
            "One-Sample t-test",
            "Independent t-test",
            "Paired t-test",
            "One-Way ANOVA",
            "Chi-square test",
            "Correlation",
            "Linear Regression",
            "Logistic Regression",
            "Nonparametric Tests",
            "Reliability Analysis (Cronbach's Alpha)",
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

    # --- One-Sample t-test ------------------------------------------
    elif analysis == "One-Sample t-test":
        st.subheader("🧪 One-Sample t-test")

        if not numeric_columns:
            st.warning("No numeric variable available.")
        else:
            variable = st.selectbox("Select numeric variable", numeric_columns, key="one_sample_var")
            test_value = st.number_input(
                "Test value (the population mean to compare against)", value=0.0, key="one_sample_val"
            )

            data = pd.to_numeric(df[variable], errors="coerce").dropna()

            if len(data) < 2:
                st.warning("At least two valid observations are required.")
            else:
                result = stats.ttest_1samp(data, test_value)
                mean_diff = data.mean() - test_value
                se = data.std(ddof=1) / np.sqrt(len(data))
                ci_low, ci_high = stats.t.interval(1 - ALPHA, len(data) - 1, loc=mean_diff, scale=se)

                summary = pd.DataFrame({
                    "N": [len(data)], "Mean": [data.mean()], "SD": [data.std()],
                    "Test Value": [test_value],
                }).round(3)
                st.subheader("Summary")
                st.dataframe(summary, use_container_width=True)

                test_result = pd.DataFrame({
                    "Mean Difference": [mean_diff],
                    "t-value": [result.statistic],
                    "df": [len(data) - 1],
                    "p-value": [result.pvalue],
                    "95% CI Lower": [ci_low],
                    "95% CI Upper": [ci_high],
                }).round(4)
                st.subheader("Test Result")
                st.dataframe(test_result, use_container_width=True)
                download_button_for(test_result, "one-sample t-test result", "one_sample_ttest.csv", "dl_one_sample")

                if result.pvalue < ALPHA:
                    st.success(f"Mean is statistically significantly different from {test_value} (p < {ALPHA}).")
                else:
                    st.info(f"No statistically significant difference from {test_value} (p ≥ {ALPHA}).")

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

    # --- One-Way ANOVA --------------------------------------------------
    elif analysis == "One-Way ANOVA":
        st.subheader("📐 One-Way ANOVA")

        if not numeric_columns:
            st.warning("No numeric outcome variable available.")
        else:
            outcome = st.selectbox("Select numeric outcome", numeric_columns, key="anova_outcome")
            candidates = [
                c for c in categorical_candidates(df, exclude=[outcome])
                if df[c].dropna().nunique() >= 3
            ]

            if not candidates:
                st.warning(
                    "No suitable grouping variable with 3 or more levels detected. "
                    "(Use Independent t-test for a 2-group comparison.)"
                )
            else:
                group_variable = st.selectbox("Select grouping variable", candidates, key="anova_group")

                clean = df[[outcome, group_variable]].copy()
                clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce")
                clean = clean.dropna()

                groups_dict = {
                    level: sub[outcome].to_numpy()
                    for level, sub in clean.groupby(group_variable)
                }
                groups_dict = {k: v for k, v in groups_dict.items() if len(v) >= 2}

                if len(groups_dict) < 3:
                    st.warning("At least three groups with 2+ valid observations each are required.")
                else:
                    group_arrays = list(groups_dict.values())
                    f_stat, p_value = stats.f_oneway(*group_arrays)
                    eta_sq = eta_squared_oneway(group_arrays)

                    summary = pd.DataFrame({
                        "Group": list(groups_dict.keys()),
                        "N": [len(v) for v in group_arrays],
                        "Mean": [v.mean() for v in group_arrays],
                        "SD": [v.std(ddof=1) for v in group_arrays],
                    }).round(3)
                    st.subheader("Group Summary")
                    st.dataframe(summary, use_container_width=True)

                    test_result = pd.DataFrame({
                        "F-value": [f_stat], "p-value": [p_value], "Eta-squared": [eta_sq],
                    }).round(4)
                    st.subheader("ANOVA Result")
                    st.dataframe(test_result, use_container_width=True)
                    download_button_for(test_result, "ANOVA result", "anova.csv", "dl_anova")

                    if p_value < ALPHA:
                        st.success(f"Statistically significant difference between groups (p < {ALPHA}).")
                        valid_mask = clean[group_variable].isin(groups_dict.keys())
                        if STATSMODELS_AVAILABLE:
                            tukey = pairwise_tukeyhsd(
                                endog=clean.loc[valid_mask, outcome],
                                groups=clean.loc[valid_mask, group_variable].astype(str),
                                alpha=ALPHA,
                            )
                            tukey_table = tukey.summary()
                            posthoc = pd.DataFrame(
                                data=tukey_table.data[1:],
                                columns=tukey_table.data[0],
                            )
                            st.subheader("Post-hoc: Tukey HSD")
                            st.dataframe(posthoc, use_container_width=True)
                            download_button_for(posthoc, "Tukey HSD post-hoc", "tukey_posthoc.csv", "dl_tukey")
                        else:
                            st.info(
                                "Install the `statsmodels` package to also see pairwise "
                                "Tukey HSD post-hoc comparisons here."
                            )
                    else:
                        st.info(f"No statistically significant difference between groups (p ≥ {ALPHA}).")

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

    # --- Linear Regression ----------------------------------------------
    elif analysis == "Linear Regression":
        st.subheader("📈 Linear Regression")

        if not STATSMODELS_AVAILABLE:
            st.error(
                "This feature requires the `statsmodels` package. Install it "
                "(it's already listed in requirements.txt) and restart the app."
            )
        elif len(numeric_columns) < 2:
            st.warning("At least two numeric variables are required.")
        else:
            dependent = st.selectbox("Select dependent variable (outcome)", numeric_columns, key="lr_dep")
            predictors = st.multiselect(
                "Select independent variable(s) (predictors)",
                [c for c in numeric_columns if c != dependent],
                key="lr_pred",
            )

            if not predictors:
                st.info("Select at least one predictor variable.")
            else:
                clean = df[[dependent] + predictors].apply(pd.to_numeric, errors="coerce").dropna()

                if len(clean) < len(predictors) + 2:
                    st.warning("Not enough complete observations for this many predictors.")
                elif has_constant_column(clean, predictors):
                    st.warning("One of the selected predictors has no variation (constant values).")
                else:
                    X = sm.add_constant(clean[predictors])
                    y = clean[dependent]
                    try:
                        model = sm.OLS(y, X).fit()
                    except Exception as error:
                        st.error(f"Model could not be fit: {error}")
                    else:
                        model_summary = pd.DataFrame({
                            "R-squared": [model.rsquared],
                            "Adj. R-squared": [model.rsquared_adj],
                            "F-statistic": [model.fvalue],
                            "F p-value": [model.f_pvalue],
                            "N": [int(model.nobs)],
                        }).round(4)
                        st.subheader("Model Summary")
                        st.dataframe(model_summary, use_container_width=True)

                        coef_table = pd.DataFrame({
                            "Coefficient": model.params,
                            "Std. Error": model.bse,
                            "t-value": model.tvalues,
                            "p-value": model.pvalues,
                            "95% CI Lower": model.conf_int()[0],
                            "95% CI Upper": model.conf_int()[1],
                        }).round(4)
                        coef_table.index.name = "Variable"
                        coef_table = coef_table.reset_index()
                        st.subheader("Coefficients")
                        st.dataframe(coef_table, use_container_width=True)
                        download_button_for(coef_table, "regression coefficients", "linear_regression.csv", "dl_linreg")

                        equation_terms = " + ".join(
                            f"{model.params[p]:.3f}×{p}" for p in predictors
                        )
                        st.caption(f"{dependent} = {equation_terms} + {model.params['const']:.3f}")

    # --- Logistic Regression ---------------------------------------------
    elif analysis == "Logistic Regression":
        st.subheader("🎯 Logistic Regression")

        if not STATSMODELS_AVAILABLE:
            st.error(
                "This feature requires the `statsmodels` package. Install it "
                "(it's already listed in requirements.txt) and restart the app."
            )
        else:
            binary_candidates = [c for c in categorical_candidates(df) if df[c].dropna().nunique() == 2]

            if not binary_candidates or not numeric_columns:
                st.warning("A binary outcome variable and at least one numeric predictor are required.")
            else:
                dependent = st.selectbox("Select binary outcome variable", binary_candidates, key="logit_dep")
                predictors = st.multiselect(
                    "Select independent variable(s) (predictors)",
                    [c for c in numeric_columns if c != dependent],
                    key="logit_pred",
                )

                if not predictors:
                    st.info("Select at least one predictor variable.")
                else:
                    clean = df[[dependent] + predictors].dropna()
                    for p in predictors:
                        clean[p] = pd.to_numeric(clean[p], errors="coerce")
                    clean = clean.dropna()

                    levels = sorted(clean[dependent].unique(), key=str)
                    positive_class = st.selectbox(
                        "Outcome level to model as the 'positive' event (1)",
                        levels, index=len(levels) - 1, key="logit_pos",
                    )
                    y = (clean[dependent] == positive_class).astype(int)

                    if y.nunique() < 2:
                        st.warning("Both outcome levels must be present after removing missing values.")
                    elif len(clean) < len(predictors) + 10:
                        st.warning("Not enough complete observations for a reliable model.")
                    elif has_constant_column(clean, predictors):
                        st.warning("One of the selected predictors has no variation (constant values).")
                    else:
                        X = sm.add_constant(clean[predictors])
                        try:
                            model = sm.Logit(y, X).fit(disp=0)
                        except Exception as error:
                            st.error(f"Model could not be fit (it may not have converged): {error}")
                        else:
                            coef_table = pd.DataFrame({
                                "Coefficient (log-odds)": model.params,
                                "Std. Error": model.bse,
                                "z-value": model.tvalues,
                                "p-value": model.pvalues,
                                "Odds Ratio": np.exp(model.params),
                            }).round(4)
                            coef_table.index.name = "Variable"
                            coef_table = coef_table.reset_index()
                            st.subheader("Coefficients")
                            st.dataframe(coef_table, use_container_width=True)
                            download_button_for(
                                coef_table, "logistic regression coefficients", "logistic_regression.csv", "dl_logreg"
                            )
                            st.caption(f"Pseudo R² (McFadden): {model.prsquared:.4f} | N = {int(model.nobs)}")

                            predicted_prob = model.predict(X)
                            predicted_class = (predicted_prob >= 0.5).astype(int)
                            accuracy = (predicted_class == y).mean()
                            st.caption(f"Classification accuracy at 0.5 threshold: {accuracy:.1%}")

                            fpr, tpr, auc = roc_curve_manual(y, predicted_prob)
                            if fpr is None:
                                st.info("ROC curve is undefined here (only one outcome class present).")
                            else:
                                st.subheader("ROC Curve")
                                roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr}).set_index("FPR")
                                st.line_chart(roc_df)
                                st.caption(f"AUC = {auc:.4f}")

    # --- Nonparametric Tests ----------------------------------------------
    elif analysis == "Nonparametric Tests":
        st.subheader("📊 Nonparametric Tests")

        test_type = st.selectbox(
            "Select test",
            ["Mann-Whitney U (2 independent groups)",
             "Kruskal-Wallis (3+ independent groups)",
             "Wilcoxon Signed-Rank (2 paired measurements)"],
            key="nonparam_test",
        )

        if test_type == "Mann-Whitney U (2 independent groups)":
            if not numeric_columns:
                st.warning("No numeric outcome variable available.")
            else:
                outcome = st.selectbox("Select numeric outcome", numeric_columns, key="mw_outcome")
                candidates = [
                    c for c in categorical_candidates(df, exclude=[outcome])
                    if df[c].dropna().nunique() == 2
                ]
                if not candidates:
                    st.warning("No suitable two-group variable detected.")
                else:
                    group_variable = st.selectbox("Select grouping variable", candidates, key="mw_group")
                    groups = df[group_variable].dropna().unique()
                    group1, group2 = groups[0], groups[1]
                    data1 = pd.to_numeric(df.loc[df[group_variable] == group1, outcome], errors="coerce").dropna()
                    data2 = pd.to_numeric(df.loc[df[group_variable] == group2, outcome], errors="coerce").dropna()

                    if len(data1) < 1 or len(data2) < 1:
                        st.warning("Each group needs at least 1 valid observation.")
                    else:
                        result = stats.mannwhitneyu(data1, data2, alternative="two-sided")
                        summary = pd.DataFrame({
                            "Group": [str(group1), str(group2)],
                            "N": [len(data1), len(data2)],
                            "Median": [data1.median(), data2.median()],
                        }).round(3)
                        st.subheader("Group Summary")
                        st.dataframe(summary, use_container_width=True)

                        test_result = pd.DataFrame({
                            "U-statistic": [result.statistic], "p-value": [result.pvalue],
                        }).round(4)
                        st.subheader("Test Result")
                        st.dataframe(test_result, use_container_width=True)
                        download_button_for(test_result, "Mann-Whitney U result", "mann_whitney.csv", "dl_mw")

                        if result.pvalue < ALPHA:
                            st.success(f"Statistically significant difference (p < {ALPHA}).")
                        else:
                            st.info(f"No statistically significant difference (p ≥ {ALPHA}).")

        elif test_type == "Kruskal-Wallis (3+ independent groups)":
            if not numeric_columns:
                st.warning("No numeric outcome variable available.")
            else:
                outcome = st.selectbox("Select numeric outcome", numeric_columns, key="kw_outcome")
                candidates = [
                    c for c in categorical_candidates(df, exclude=[outcome])
                    if df[c].dropna().nunique() >= 3
                ]
                if not candidates:
                    st.warning("No suitable grouping variable with 3+ levels detected.")
                else:
                    group_variable = st.selectbox("Select grouping variable", candidates, key="kw_group")
                    clean = df[[outcome, group_variable]].copy()
                    clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce")
                    clean = clean.dropna()

                    groups_dict = {
                        level: sub[outcome].to_numpy()
                        for level, sub in clean.groupby(group_variable)
                    }
                    groups_dict = {k: v for k, v in groups_dict.items() if len(v) >= 1}

                    if len(groups_dict) < 3:
                        st.warning("At least three groups with 1+ valid observation each are required.")
                    else:
                        group_arrays = list(groups_dict.values())
                        result = stats.kruskal(*group_arrays)

                        summary = pd.DataFrame({
                            "Group": list(groups_dict.keys()),
                            "N": [len(v) for v in group_arrays],
                            "Median": [np.median(v) for v in group_arrays],
                        }).round(3)
                        st.subheader("Group Summary")
                        st.dataframe(summary, use_container_width=True)

                        test_result = pd.DataFrame({
                            "H-statistic": [result.statistic], "p-value": [result.pvalue],
                        }).round(4)
                        st.subheader("Test Result")
                        st.dataframe(test_result, use_container_width=True)
                        download_button_for(test_result, "Kruskal-Wallis result", "kruskal_wallis.csv", "dl_kw")

                        if result.pvalue < ALPHA:
                            st.success(f"Statistically significant difference between groups (p < {ALPHA}).")
                        else:
                            st.info(f"No statistically significant difference between groups (p ≥ {ALPHA}).")

        else:  # Wilcoxon Signed-Rank
            if len(numeric_columns) < 2:
                st.warning("At least two numeric variables are required.")
            else:
                before = st.selectbox("Select first measurement", numeric_columns, key="wsr_before")
                after = st.selectbox(
                    "Select second measurement", [c for c in numeric_columns if c != before], key="wsr_after"
                )
                paired = pd.DataFrame({
                    "Before": pd.to_numeric(df[before], errors="coerce"),
                    "After": pd.to_numeric(df[after], errors="coerce"),
                }).dropna()
                paired = paired[paired["Before"] != paired["After"]]  # ties are dropped by the test

                if len(paired) < 1:
                    st.warning("At least one non-tied paired observation is required.")
                else:
                    result = stats.wilcoxon(paired["Before"], paired["After"])
                    test_result = pd.DataFrame({
                        "N (non-tied pairs)": [len(paired)],
                        "W-statistic": [result.statistic],
                        "p-value": [result.pvalue],
                    }).round(4)
                    st.subheader("Test Result")
                    st.dataframe(test_result, use_container_width=True)
                    download_button_for(test_result, "Wilcoxon result", "wilcoxon.csv", "dl_wilcoxon")

                    if result.pvalue < ALPHA:
                        st.success(f"Statistically significant difference (p < {ALPHA}).")
                    else:
                        st.info(f"No statistically significant difference (p ≥ {ALPHA}).")

    # --- Reliability Analysis ---------------------------------------------
    elif analysis == "Reliability Analysis (Cronbach's Alpha)":
        st.subheader("📏 Reliability Analysis")
        st.caption("Measures internal consistency across a set of scale items (e.g. survey questions).")

        if len(numeric_columns) < 2:
            st.warning("At least two numeric items are required.")
        else:
            items = st.multiselect(
                "Select scale items", numeric_columns, default=numeric_columns[: min(5, len(numeric_columns))],
                key="rel_items",
            )

            if len(items) < 2:
                st.info("Select at least two items.")
            else:
                item_df = df[items].apply(pd.to_numeric, errors="coerce")
                complete_rows = item_df.dropna()

                if len(complete_rows) < 2:
                    st.warning("At least two complete respondent rows are required.")
                else:
                    alpha = cronbachs_alpha(complete_rows)

                    st.metric("Cronbach's Alpha", f"{alpha:.3f}" if pd.notna(alpha) else "N/A")
                    st.caption(f"Based on N = {len(complete_rows)} complete cases, {len(items)} items.")

                    if pd.notna(alpha):
                        if alpha >= 0.9:
                            interpretation = "Excellent internal consistency."
                        elif alpha >= 0.8:
                            interpretation = "Good internal consistency."
                        elif alpha >= 0.7:
                            interpretation = "Acceptable internal consistency."
                        elif alpha >= 0.6:
                            interpretation = "Questionable internal consistency."
                        else:
                            interpretation = "Poor internal consistency — consider reviewing these items."
                        st.info(interpretation)

                    # item-total correlations, useful for spotting a weak/reversed item
                    total_score = complete_rows.sum(axis=1)
                    item_total_corr = {
                        item: complete_rows[item].corr(total_score - complete_rows[item])
                        for item in items
                    }
                    item_stats = pd.DataFrame({
                        "Item": items,
                        "Item-Total Correlation": [item_total_corr[i] for i in items],
                    }).round(3)
                    st.subheader("Item-Total Correlations")
                    st.dataframe(item_stats, use_container_width=True)
                    download_button_for(item_stats, "item-total correlations", "reliability_items.csv", "dl_reliability")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("Research Analysis Tool | SPSS-style workflow")
