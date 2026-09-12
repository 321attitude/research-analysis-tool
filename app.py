import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Research Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Research Analysis Tool")
st.caption("SPSS-style research data analysis")


# =========================================================
# SESSION STATE
# =========================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "variable_meta" not in st.session_state:
    st.session_state.variable_meta = {}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📁 File")

    uploaded_file = st.file_uploader(
        "Open Excel file",
        type=["xlsx"]
    )

    st.divider()

    st.header("🧭 Navigation")

    page = st.radio(
        "Go to",
        [
            "📋 Data View",
            "⚙️ Variable View",
            "📊 Analyze"
        ]
    )


# =========================================================
# LOAD EXCEL
# =========================================================

if uploaded_file is not None:

    try:

        excel = pd.ExcelFile(uploaded_file)

        sheets = excel.sheet_names

        selected_sheet = st.sidebar.selectbox(
            "Select Sheet",
            sheets
        )

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )

        st.session_state.data = df

    except Exception as error:

        st.error(
            f"Excel file could not be read: {error}"
        )

        st.stop()


# =========================================================
# NO DATA
# =========================================================

if st.session_state.data is None:

    st.info(
        "📁 Please upload an Excel (.xlsx) file from the sidebar."
    )

    st.markdown(
        """
        ### How to use

        1. Upload your `.xlsx` file.
        2. Select a worksheet.
        3. Use **Data View** to see your data.
        4. Use **Variable View** to define variables.
        5. Use **Analyze** to perform statistical analysis.
        """
    )

    st.stop()


df = st.session_state.data


# =========================================================
# COMMON VARIABLE LISTS
# =========================================================

numeric_columns = list(
    df.select_dtypes(
        include=np.number
    ).columns
)


# =========================================================
# DATA VIEW
# =========================================================

if page == "📋 Data View":

    st.header("📋 Data View")

    st.caption(
        "SPSS-style data table"
    )

    # Workbook information

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        len(df)
    )

    col2.metric(
        "Variables",
        len(df.columns)
    )

    col3.metric(
        "Missing Cells",
        int(df.isna().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    st.divider()

    # Data table

    st.dataframe(
        df,
        use_container_width=True,
        height=550
    )

    st.divider()

    st.subheader("🔎 Data Quality")

    quality = pd.DataFrame({

        "Variable": df.columns,

        "Type": df.dtypes.astype(str),

        "Missing": df.isna().sum().values,

        "Missing %": (
            df.isna().mean() * 100
        ).round(2).values,

        "Unique Values": [
            df[col].nunique(
                dropna=True
            )
            for col in df.columns
        ]

    })

    st.dataframe(
        quality,
        use_container_width=True
    )


# =========================================================
# VARIABLE VIEW
# =========================================================

elif page == "⚙️ Variable View":

    st.header("⚙️ Variable View")

    st.caption(
        "SPSS-style variable definition"
    )

    variable_rows = []

    for index, column in enumerate(df.columns):

        dtype = str(df[column].dtype)

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            variable_type = "Numeric"
        else:
            variable_type = "String"

        # Detect likely ID variables

        lower_name = column.lower()

        id_keywords = [
            "id",
            "participant_id",
            "patient_id",
            "subject_id",
            "record_id"
        ]

        is_id = any(
            keyword == lower_name
            or lower_name.endswith(
                "_" + keyword
            )
            for keyword in id_keywords
        )

        if is_id:
            measure = "ID"
        else:
            measure = (
                "Scale"
                if variable_type == "Numeric"
                else "Nominal"
            )

        variable_rows.append({

            "No.": index + 1,

            "Name": column,

            "Type": variable_type,

            "Label": column,

            "Missing": int(
                df[column].isna().sum()
            ),

            "Unique": int(
                df[column].nunique(
                    dropna=True
                )
            ),

            "Measure": measure,

            "Role": (
                "Identifier"
                if is_id
                else "Analysis"
            )

        })

    variable_view = pd.DataFrame(
        variable_rows
    )

    st.dataframe(
        variable_view,
        use_container_width=True,
        height=550
    )

    st.info(
        "💡 Variables detected as IDs are treated as "
        "identifiers and should not be used as statistical "
        "outcomes, predictors, or grouping variables."
    )


# =========================================================
# ANALYZE
# =========================================================

elif page == "📊 Analyze":

    st.header("📊 Analyze")

    st.caption(
        "Statistical analysis"
    )

    analysis = st.selectbox(

        "Select analysis",

        [

            "Descriptive Statistics",

            "Frequencies",

            "Group-wise Descriptive Statistics",

            "Independent t-test",

            "Paired t-test",

            "Chi-square test",

            "Correlation"

        ]

    )


    # =====================================================
    # DESCRIPTIVE STATISTICS
    # =====================================================

    if analysis == "Descriptive Statistics":

        st.subheader(
            "📈 Descriptive Statistics"
        )

        if not numeric_columns:

            st.warning(
                "No numeric variables detected."
            )

        else:

            selected = st.multiselect(

                "Select numeric variables",

                numeric_columns,

                default=numeric_columns

            )

            if selected:

                data = df[selected]

                result = pd.DataFrame({

                    "N": data.count(),

                    "Mean": data.mean(),

                    "Median": data.median(),

                    "SD": data.std(),

                    "Minimum": data.min(),

                    "Maximum": data.max(),

                    "Range": (
                        data.max()
                        - data.min()
                    ),

                    "IQR": (
                        data.quantile(0.75)
                        -
                        data.quantile(0.25)
                    ),

                    "Skewness": data.skew(),

                    "Kurtosis": data.kurtosis()

                })

                st.dataframe(
                    result.round(3),
                    use_container_width=True
                )


    # =====================================================
    # FREQUENCIES
    # =====================================================

    elif analysis == "Frequencies":

        st.subheader(
            "🔢 Frequencies"
        )

        variable = st.selectbox(
            "Select variable",
            df.columns
        )

        frequency = (

            df[variable]

            .value_counts(
                dropna=False
            )

            .reset_index()

        )

        frequency.columns = [
            "Value",
            "Frequency"
        ]

        frequency["Percent"] = (

            frequency["Frequency"]
            / len(df)
            * 100

        ).round(2)

        st.dataframe(
            frequency,
            use_container_width=True
        )


    # =====================================================
    # GROUP-WISE DESCRIPTIVE
    # =====================================================

    elif analysis == (
        "Group-wise Descriptive Statistics"
    ):

        st.subheader(
            "📊 Group-wise Descriptive Statistics"
        )

        group_candidates = []

        for col in df.columns:

            if col in numeric_columns:
                continue

            if df[col].nunique(
                dropna=True
            ) <= 20:

                group_candidates.append(
                    col
                )

        if not group_candidates:

            st.warning(
                "No suitable grouping variable detected."
            )

        elif not numeric_columns:

            st.warning(
                "No numeric outcome variables detected."
            )

        else:

            group_variable = st.selectbox(
                "Select grouping variable",
                group_candidates
            )

            outcome = st.selectbox(
                "Select numeric variable",
                numeric_columns
            )

            result = (

                df.groupby(
                    group_variable
                )[outcome]

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

            result["Range"] = (
                result["Maximum"]
                -
                result["Minimum"]
            )

            st.dataframe(
                result.round(3),
                use_container_width=True
            )


    # =====================================================
    # INDEPENDENT T-TEST
    # =====================================================

    elif analysis == "Independent t-test":

        st.subheader(
            "🧪 Independent t-test"
        )

        if not numeric_columns:

            st.warning(
                "No numeric outcome variable available."
            )

        else:

            outcome = st.selectbox(
                "Select numeric outcome",
                numeric_columns
            )

            grouping_candidates = []

            for col in df.columns:

                if col == outcome:
                    continue

                # Do not use likely ID variables

                name = col.lower()

                if (
                    name == "id"
                    or name.endswith("_id")
                    or name.endswith("id")
                ):
                    continue

                if (
                    df[col]
                    .dropna()
                    .nunique()
                    == 2
                ):

                    grouping_candidates.append(
                        col
                    )

            if not grouping_candidates:

                st.warning(
                    "No suitable two-group variable detected."
                )

            else:

                group_variable = st.selectbox(
                    "Select grouping variable",
                    grouping_candidates
                )

                groups = (
                    df[group_variable]
                    .dropna()
                    .unique()
                )

                group1 = groups[0]
                group2 = groups[1]

                data1 = pd.to_numeric(
                    df.loc[
                        df[group_variable] == group1,
                        outcome
                    ],
                    errors="coerce"
                ).dropna()

                data2 = pd.to_numeric(
                    df.loc[
                        df[group_variable] == group2,
                        outcome
                    ],
                    errors="coerce"
                ).dropna()

                if (
                    len(data1) >= 2
                    and len(data2) >= 2
                ):

                    result = stats.ttest_ind(
                        data1,
                        data2,
                        equal_var=False
                    )

                    mean_difference = (
                        data1.mean()
                        -
                        data2.mean()
                    )

                    summary = pd.DataFrame({

                        "Group": [
                            str(group1),
                            str(group2)
                        ],

                        "N": [
                            len(data1),
                            len(data2)
                        ],

                        "Mean": [
                            data1.mean(),
                            data2.mean()
                        ],

                        "SD": [
                            data1.std(),
                            data2.std()
                        ]

                    })

                    st.subheader(
                        "Group Summary"
                    )

                    st.dataframe(
                        summary.round(3),
                        use_container_width=True
                    )

                    test_result = pd.DataFrame({

                        "Mean Difference": [
                            mean_difference
                        ],

                        "t-value": [
                            result.statistic
                        ],

                        "p-value": [
                            result.pvalue
                        ]

                    })

                    st.subheader(
                        "Test Result"
                    )

                    st.dataframe(
                        test_result.round(4),
                        use_container_width=True
                    )

                    if result.pvalue < 0.05:

                        st.success(
                            "Statistically significant "
                            "difference (p < 0.05)."
                        )

                    else:

                        st.info(
                            "No statistically significant "
                            "difference (p ≥ 0.05)."
                        )

                else:

                    st.warning(
                        "Each group needs at least "
                        "2 valid observations."
                    )


    # =====================================================
    # PAIRED T-TEST
    # =====================================================

    elif analysis == "Paired t-test":

        st.subheader(
            "🧪 Paired t-test"
        )

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numeric variables are required."
            )

        else:

            before = st.selectbox(
                "Select first measurement",
                numeric_columns
            )

            after_candidates = [
                col
                for col in numeric_columns
                if col != before
            ]

            after = st.selectbox(
                "Select second measurement",
                after_candidates
            )

            paired = pd.DataFrame({

                "Before": pd.to_numeric(
                    df[before],
                    errors="coerce"
                ),

                "After": pd.to_numeric(
                    df[after],
                    errors="coerce"
                )

            }).dropna()

            if len(paired) >= 2:

                result = stats.ttest_rel(
                    paired["Before"],
                    paired["After"]
                )

                difference = (
                    paired["After"]
                    -
                    paired["Before"]
                )

                summary = pd.DataFrame({

                    "Measurement": [
                        before,
                        after
                    ],

                    "N": [
                        len(paired),
                        len(paired)
                    ],

                    "Mean": [
                        paired["Before"].mean(),
                        paired["After"].mean()
                    ],

                    "SD": [
                        paired["Before"].std(),
                        paired["After"].std()
                    ]

                })

                st.subheader(
                    "Paired Summary"
                )

                st.dataframe(
                    summary.round(3),
                    use_container_width=True
                )

                test_result = pd.DataFrame({

                    "Mean Difference (After-Before)": [
                        difference.mean()
                    ],

                    "t-value": [
                        result.statistic
                    ],

                    "df": [
                        len(paired) - 1
                    ],

                    "p-value": [
                        result.pvalue
                    ]

                })

                st.subheader(
                    "Test Result"
                )

                st.dataframe(
                    test_result.round(4),
                    use_container_width=True
                )

                if result.pvalue < 0.05:

                    st.success(
                        "Statistically significant "
                        "difference (p < 0.05)."
                    )

                else:

                    st.info(
                        "No statistically significant "
                        "difference (p ≥ 0.05)."
                    )

            else:

                st.warning(
                    "At least two complete paired "
                    "observations are required."
                )


    # =====================================================
    # CHI-SQUARE
    # =====================================================

    elif analysis == "Chi-square test":

        st.subheader(
            "🧮 Chi-square Test"
        )

        categorical = [
            col
            for col in df.columns
            if df[col].nunique(
                dropna=True
            ) <= 20
        ]

        if len(categorical) < 2:

            st.warning(
                "At least two categorical variables "
                "are required."
            )

        else:

            row_variable = st.selectbox(
                "Select row variable",
                categorical
            )

            column_candidates = [
                col
                for col in categorical
                if col != row_variable
            ]

            column_variable = st.selectbox(
                "Select column variable",
                column_candidates
            )

            table = pd.crosstab(
                df[row_variable],
                df[column_variable]
            )

            st.subheader(
                "Contingency Table"
            )

            st.dataframe(
                table,
                use_container_width=True
            )

            chi2, p, dof, expected = (
                stats.chi2_contingency(table)
            )

            result = pd.DataFrame({

                "Chi-square": [chi2],

                "df": [dof],

                "p-value": [p]

            })

            st.subheader(
                "Test Result"
            )

            st.dataframe(
                result.round(4),
                use_container_width=True
            )

            if p < 0.05:

                st.success(
                    "Statistically significant association "
                    "(p < 0.05)."
                )

            else:

                st.info(
                    "No statistically significant association "
                    "(p ≥ 0.05)."
                )


    # =====================================================
    # CORRELATION
    # =====================================================

    elif analysis == "Correlation":

        st.subheader(
            "🔗 Correlation"
        )

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numeric variables "
                "are required."
            )

        else:

            variable1 = st.selectbox(
                "Select first variable",
                numeric_columns
            )

            variable2_candidates = [
                col
                for col in numeric_columns
                if col != variable1
            ]

            variable2 = st.selectbox(
                "Select second variable",
                variable2_candidates
            )

            method = st.selectbox(
                "Correlation method",
                [
                    "Pearson",
                    "Spearman",
                    "Kendall"
                ]
            )

            clean = df[
                [variable1, variable2]
            ].apply(
                pd.to_numeric,
                errors="coerce"
            ).dropna()

            if len(clean) >= 3:

                if method == "Pearson":

                    coefficient, p = (
                        stats.pearsonr(
                            clean[variable1],
                            clean[variable2]
                        )
                    )

                elif method == "Spearman":

                    coefficient, p = (
                        stats.spearmanr(
                            clean[variable1],
                            clean[variable2]
                        )
                    )

                else:

                    coefficient, p = (
                        stats.kendalltau(
                            clean[variable1],
                            clean[variable2]
                        )
                    )

                result = pd.DataFrame({

                    "Method": [method],

                    "Correlation": [
                        coefficient
                    ],

                    "p-value": [p],

                    "N": [len(clean)]

                })

                st.dataframe(
                    result.round(4),
                    use_container_width=True
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Research Analysis Tool | SPSS-style workflow"
)
