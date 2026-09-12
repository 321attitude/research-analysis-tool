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
st.write("SPSS + Stata style research data analysis")


# =========================================================
# EXCEL UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"]
)


if uploaded_file:

    try:

        # =================================================
        # READ EXCEL WORKBOOK
        # =================================================

        excel = pd.ExcelFile(uploaded_file)

        sheets = excel.sheet_names

        st.success(
            f"{len(sheets)} sheet(s) detected."
        )


        # =================================================
        # WORKBOOK OVERVIEW
        # =================================================

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
                "Missing cells": int(
                    data.isna().sum().sum()
                ),
                "Duplicate rows": int(
                    data.duplicated().sum()
                )
            })


        st.subheader("📁 Workbook Overview")

        st.dataframe(
            pd.DataFrame(overview),
            use_container_width=True
        )


        # =================================================
        # SELECT SHEET
        # =================================================

        selected_sheet = st.selectbox(
            "Select sheet",
            sheets
        )


        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )


        # =================================================
        # DATA PREVIEW
        # =================================================

        st.subheader("📋 Data Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )


        # =================================================
        # VARIABLE INFORMATION
        # =================================================

        st.subheader(
            "🔎 Variable Information"
        )

        info = pd.DataFrame({

            "Variable": df.columns,

            "Type": df.dtypes.astype(str),

            "Missing": df.isna().sum().values,

            "Missing %": (
                df.isna().mean().values * 100
            ).round(2),

            "Unique": [
                df[col].nunique(
                    dropna=True
                )
                for col in df.columns
            ]

        })


        st.dataframe(
            info,
            use_container_width=True
        )


        # =================================================
        # NUMERIC VARIABLES
        # =================================================

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )


        # =================================================
        # STATISTICAL ANALYSIS MENU
        # =================================================

        st.header(
            "📊 Statistical Analysis"
        )

        analysis_option = st.selectbox(

            "Choose Analysis",

            [

                "Descriptive Statistics",

                "Frequencies",

                "Group-wise Descriptive Statistics",

                "Independent t-test",

                "Paired t-test"

            ]

        )


        # =================================================
        # 1. DESCRIPTIVE STATISTICS
        # =================================================

        if analysis_option == "Descriptive Statistics":

            st.subheader(
                "📈 Descriptive Statistics"
            )


            if not numeric_columns:

                st.info(
                    "No numeric variables detected."
                )

            else:

                numeric = df[
                    numeric_columns
                ]


                descriptive = pd.DataFrame({

                    "N": numeric.count(),

                    "Mean": numeric.mean(),

                    "Median": numeric.median(),

                    "SD": numeric.std(),

                    "Minimum": numeric.min(),

                    "Maximum": numeric.max(),

                    "Range": (
                        numeric.max()
                        - numeric.min()
                    ),

                    "IQR": (
                        numeric.quantile(0.75)
                        -
                        numeric.quantile(0.25)
                    )

                })


                st.dataframe(

                    descriptive.round(3),

                    use_container_width=True

                )


        # =================================================
        # 2. FREQUENCIES
        # =================================================

        elif analysis_option == "Frequencies":

            st.subheader(
                "🔢 Frequencies & Percentages"
            )


            selected_variable = st.selectbox(

                "Select variable",

                df.columns

            )


            frequency = (

                df[selected_variable]

                .value_counts(
                    dropna=False
                )

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


        # =================================================
        # 3. GROUP-WISE DESCRIPTIVE STATISTICS
        # =================================================

        elif analysis_option == (
            "Group-wise Descriptive Statistics"
        ):

            st.subheader(
                "📊 Group-wise Descriptive Statistics"
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

                    df.groupby(
                        group_variable
                    )[analysis_variable]

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

                    -
                    grouped["Minimum"]

                )


                st.dataframe(

                    grouped.round(3),

                    use_container_width=True

                )


        # =================================================
        # 4. INDEPENDENT T-TEST
        # =================================================

        elif analysis_option == "Independent t-test":

            st.subheader(
                "🧪 Independent t-test"
            )


            if not numeric_columns:

                st.info(
                    "At least one numeric variable "
                    "is required."
                )

            else:

                outcome_variable = st.selectbox(

                    "Select numeric outcome variable",

                    numeric_columns,

                    key="independent_outcome"

                )


                # Find variables having exactly
                # two non-missing categories

                grouping_candidates = []


                for col in df.columns:

                    if col == outcome_variable:
                        continue

                    unique_count = (
                        df[col]
                        .dropna()
                        .nunique()
                    )

                    if unique_count == 2:

                        grouping_candidates.append(
                            col
                        )


                if not grouping_candidates:

                    st.warning(

                        "No suitable two-group "
                        "variable was detected."

                    )

                else:

                    group_variable = st.selectbox(

                        "Select 2-group variable",

                        grouping_candidates,

                        key="independent_group"

                    )


                    groups = (

                        df[group_variable]

                        .dropna()

                        .unique()

                    )


                    if len(groups) == 2:

                        group1 = groups[0]

                        group2 = groups[1]


                        st.write(
                            f"Group 1: **{group1}**"
                        )

                        st.write(
                            f"Group 2: **{group2}**"
                        )


                        data1 = pd.to_numeric(

                            df.loc[
                                df[group_variable]
                                == group1,

                                outcome_variable
                            ],

                            errors="coerce"

                        ).dropna()


                        data2 = pd.to_numeric(

                            df.loc[
                                df[group_variable]
                                == group2,

                                outcome_variable
                            ],

                            errors="coerce"

                        ).dropna()


                        if (
                            len(data1) >= 2
                            and len(data2) >= 2
                        ):


                            # Welch independent t-test

                            result = stats.ttest_ind(

                                data1,

                                data2,

                                equal_var=False

                            )


                            mean1 = data1.mean()

                            mean2 = data2.mean()


                            mean_difference = (

                                mean1 - mean2

                            )


                            variance1 = data1.var(
                                ddof=1
                            )

                            variance2 = data2.var(
                                ddof=1
                            )


                            standard_error = np.sqrt(

                                (
                                    variance1
                                    / len(data1)
                                )
                                +
                                (
                                    variance2
                                    / len(data2)
                                )

                            )


                            # Welch-Satterthwaite df

                            numerator = (

                                (
                                    variance1
                                    / len(data1)
                                )
                                +
                                (
                                    variance2
                                    / len(data2)
                                )
                            ) ** 2


                            denominator = (

                                (
                                    (
                                        variance1
                                        / len(data1)
                                    ) ** 2
                                    /
                                    (len(data1) - 1)
                                )
                                +
                                (
                                    (
                                        variance2
                                        / len(data2)
                                    ) ** 2
                                    /
                                    (len(data2) - 1)
                                )

                            )


                            degrees_freedom = (
                                numerator
                                / denominator
                            )


                            critical_value = stats.t.ppf(

                                0.975,

                                degrees_freedom

                            )


                            ci_low = (

                                mean_difference
                                -
                                critical_value
                                * standard_error

                            )


                            ci_high = (

                                mean_difference
                                +
                                critical_value
                                * standard_error

                            )


                            # Group summary

                            result_table = pd.DataFrame({

                                "Group": [

                                    str(group1),

                                    str(group2)

                                ],

                                "N": [

                                    len(data1),

                                    len(data2)

                                ],

                                "Mean": [

                                    mean1,

                                    mean2

                                ],

                                "SD": [

                                    data1.std(),

                                    data2.std()

                                ]

                            })


                            st.subheader(
                                "📊 Group Summary"
                            )


                            st.dataframe(

                                result_table.round(3),

                                use_container_width=True

                            )


                            # Test result

                            test_result = pd.DataFrame({

                                "Mean Difference": [

                                    mean_difference

                                ],

                                "t-value": [

                                    result.statistic

                                ],

                                "df": [

                                    degrees_freedom

                                ],

                                "p-value": [

                                    result.pvalue

                                ],

                                "95% CI Lower": [

                                    ci_low

                                ],

                                "95% CI Upper": [

                                    ci_high

                                ]

                            })


                            st.subheader(
                                "📈 Test Result"
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


        # =================================================
        # 5. PAIRED T-TEST
        # =================================================

        elif analysis_option == "Paired t-test":

            st.subheader(
                "🧪 Paired t-test"
            )


            if len(numeric_columns) < 2:

                st.info(

                    "At least two numeric variables "
                    "are required."

                )

            else:

                before_variable = st.selectbox(

                    "Select first measurement",

                    numeric_columns,

                    key="paired_before"

                )


                after_candidates = [

                    col

                    for col in numeric_columns

                    if col != before_variable

                ]


                after_variable = st.selectbox(

                    "Select second measurement",

                    after_candidates,

                    key="paired_after"

                )


                before = pd.to_numeric(

                    df[before_variable],

                    errors="coerce"

                )


                after = pd.to_numeric(

                    df[after_variable],

                    errors="coerce"

                )


                paired_data = pd.DataFrame({

                    "Before": before,

                    "After": after

                }).dropna()


                if len(paired_data) >= 2:

                    result = stats.ttest_rel(

                        paired_data["Before"],

                        paired_data["After"]

                    )


                    mean_before = (

                        paired_data["Before"]
                        .mean()

                    )


                    mean_after = (

                        paired_data["After"]
                        .mean()

                    )


                    difference = (

                        paired_data["After"]

                        -
                        paired_data["Before"]

                    )


                    mean_difference = (
                        difference.mean()
                    )


                    sd_difference = (
                        difference.std()
                    )


                    standard_error = (

                        sd_difference
                        /
                        np.sqrt(
                            len(difference)
                        )

                    )


                    degrees_freedom = (
                        len(difference) - 1
                    )


                    critical_value = stats.t.ppf(

                        0.975,

                        degrees_freedom

                    )


                    ci_low = (

                        mean_difference

                        -
                        critical_value
                        * standard_error

                    )


                    ci_high = (

                        mean_difference

                        +
                        critical_value
                        * standard_error

                    )


                    # Paired summary

                    summary_table = pd.DataFrame({

                        "Measurement": [

                            before_variable,

                            after_variable

                        ],

                        "N": [

                            len(paired_data),

                            len(paired_data)

                        ],

                        "Mean": [

                            mean_before,

                            mean_after

                        ],

                        "SD": [

                            paired_data[
                                "Before"
                            ].std(),

                            paired_data[
                                "After"
                            ].std()

                        ]

                    })


                    st.subheader(
                        "📊 Paired Summary"
                    )


                    st.dataframe(

                        summary_table.round(3),

                        use_container_width=True

                    )


                    # Test result

                    test_result = pd.DataFrame({

                        "Mean Difference (After-Before)": [

                            mean_difference

                        ],

                        "t-value": [

                            result.statistic

                        ],

                        "df": [

                            degrees_freedom

                        ],

                        "p-value": [

                            result.pvalue

                        ],

                        "95% CI Lower": [

                            ci_low

                        ],

                        "95% CI Upper": [

                            ci_high

                        ]

                    })


                    st.subheader(
                        "📈 Test Result"
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


        # =================================================
        # DATA QUALITY
        # =================================================

        st.subheader(
            "⚠️ Data Quality"
        )


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
