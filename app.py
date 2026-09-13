"""
Data Cleaning & Analysis Web App
- Upload Excel (xlsx/xls) or CSV
- See data-quality report (missing, errors, duplicates, outliers, ...)
- Edit flagged cells with st.data_editor
- Fix / fill / drop / convert options
- Download cleaned file (Excel + CSV)
- Analysis suggestions + basic EDA (describe, missing heatmap, distributions, correlation)
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Data Cleaner & Analyzer", page_icon="🧹", layout="wide")


# ---------- Pure helper functions (no streamlit) ----------
def load_file(uploaded_file) -> pd.DataFrame:
    """Load xlsx / xls / csv into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Supported formats: .xlsx, .xls, .csv")


def detect_date_cols(df: pd.DataFrame) -> dict:
    """Return {col: parsed_series} for columns that look like dates."""
    date_cols = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols[col] = s
            continue
        if s.dtype == object:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().sum() >= max(1, int(len(parsed) * 0.6)):
                date_cols[col] = parsed
    return date_cols


def auto_convert(df: pd.DataFrame) -> tuple:
    """Best-effort type conversion. Returns (df, list_of_changes)."""
    df = df.copy()
    changes = []
    for col in df.columns:
        s = df[col]
        if s.dtype == object and s.dropna().nunique() > 0:
            sample = s.dropna().astype(str)
            cleaned = sample.str.replace(",", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() >= max(1, int(0.8 * len(cleaned))):
                df[col] = pd.to_numeric(cleaned, errors="coerce")
                changes.append(f"{col}: string -> number")
    for col, parsed in detect_date_cols(df).items():
        df[col] = parsed
        changes.append(f"{col}: -> date")
    return df, changes


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-column quality report table."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        missing_pct = round(missing / n * 100, 1) if n else 0
        dtype = str(s.dtype)
        uniques = int(s.nunique(dropna=True))
        count = int(s.count())
        dup_pct = "" if uniques == 0 else f"{round(100 * (1 - uniques / max(count, 1)), 1)}%"
        ws = 0
        case_mix = 0
        if s.dtype == object:
            strs = s.dropna().astype(str)
            if len(strs):
                ws = int((strs != strs.str.strip()).sum())
                case_mix = int(((strs.str.lower() != strs) & (strs.str.upper() != strs)).sum())
        outliers = ""
        if pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr and not np.isnan(iqr):
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "Column": col,
            "Dtype": dtype,
            "Missing": missing,
            "Missing %": missing_pct,
            "Unique": uniques,
            "Repeat %": dup_pct,
            "Whitespace": ws,
            "Case issues": case_mix,
            "Outliers": outliers,
        })
    return pd.DataFrame(rows)


def summarize_issues(report: pd.DataFrame) -> list:
    """Human readable bullets of what is wrong."""
    bullets = []
    for _, r in report.iterrows():
        if r["Missing"] > 0:
            bullets.append(f"{r['Column']} — {r['Missing']} missing values ({r['Missing %']}%)")
        if isinstance(r["Whitespace"], int) and r["Whitespace"] > 0:
            bullets.append(f"{r['Column']} — {r['Whitespace']} rows have leading/trailing spaces")
        if isinstance(r["Case issues"], int) and r["Case issues"] > 0:
            bullets.append(f"{r['Column']} — mixed capital / small letters ({r['Case issues']} rows)")
        if isinstance(r["Outliers"], int) and r["Outliers"] > 0:
            bullets.append(f"{r['Column']} — {r['Outliers']} possible outliers (1.5×IQR)")
    return bullets


def suggested_analysis(df: pd.DataFrame) -> list:
    """Recommend analyses based on column types present."""
    num_cols = list(df.select_dtypes(include=np.number).columns)
    cat_cols = list(df.select_dtypes(include="object").columns)
    date_cols = list(df.select_dtypes(include="datetime64").columns)
    recs = []
    if num_cols:
        recs.append("📊 Descriptive stats (mean / median / min / max / std) — " + ", ".join(num_cols[:5]))
    if num_cols and cat_cols:
        recs.append("📈 Group-wise comparison — e.g. mean of `" + num_cols[0] + "` grouped by `" + cat_cols[0] + "`")
        recs.append("🎯 Category distribution (top values + counts) — `" + cat_cols[0] + "`")
    if len(num_cols) >= 2:
        recs.append("🔗 Correlation matrix — `" + num_cols[0] + "` vs `" + num_cols[1] + "`")
    if date_cols and num_cols:
        recs.append("📅 Time-series trend — `" + num_cols[0] + "` over time (`" + date_cols[0] + "`)")
    if cat_cols and len(df) >= 5:
        recs.append("🧩 Pivot table — count / sum by `" + cat_cols[0] + "`")
    if not recs:
        recs.append("No clear numeric/category column found. Add a header row with clear names, re-upload, and reload.")
    return recs


# ---------- Session state ----------
if "df" not in st.session_state:
    st.session_state.df = None
if "original" not in st.session_state:
    st.session_state.original = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# ---------- Sidebar: upload + load ----------
st.sidebar.title("🧹 Steps")
st.sidebar.markdown("**1. Upload your file**")
up = st.sidebar.file_uploader("Excel / CSV file", type=["xlsx", "xls", "csv"], key="up")

if up is not None and getattr(up, "file_id", None) != st.session_state.last_file:
    try:
        st.session_state.df = load_file(up)
        st.session_state.original = st.session_state.df.copy()
        st.session_state.last_file = getattr(up, "file_id", None) or up.name
    except Exception as exc:
        st.sidebar.error(f"❌ Could not read file: {exc}")

# ---------- Main body ----------
if st.session_state.df is not None:
    df = st.session_state.df
    st.sidebar.markdown(f"**Loaded:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    if st.sidebar.button("↺ Reset to original"):
        st.session_state.df = st.session_state.original.copy()
        st.rerun()

    tab_data, tab_fix, tab_clean, tab_download, tab_analyze = st.tabs(
        ["📋 Data & Quality Report", "✏️ Fix Data (Edit Grid)", "🛠 Cleaning Options", "⬇️ Download", "📊 Analysis"]
    )

    # ---- TAB 1: quality report ----
    with tab_data:
        st.subheader("Data preview (first 500 rows)")
        st.dataframe(df.head(500), use_container_width=True)

        st.subheader("🚨 Data Quality Report")
        report = quality_report(df)
        st.dataframe(report, use_container_width=True)

        bullets = summarize_issues(report)
        if bullets:
            st.warning("📌 Issues found — fix these in the **Fix Data** / **Cleaning** tabs:")
            for b in bullets:
                st.markdown(f"- {b}")
        else:
            st.success("✅ Data looks clean — no missing / type / formatting issues detected.")

    # ---- TAB 2: edit grid ----
    with tab_fix:
        st.subheader("✏️ Click any cell to edit it directly")
        st.caption("Type fixes right into the table, then press **Save changes**. Blank cells = missing (NaN).")
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor")
        if st.button("💾 Save edits to data"):
            st.session_state.df = edited.reset_index(drop=True)
            st.success("✅ Edits saved! Now go to Cleaning / Download tabs.")
            st.rerun()

    # ---- TAB 3: cleaning options ----
    with tab_clean:
        st.subheader("🛠 Cleaning options")

        c1, c2, c3 = st.columns(3)
        with c1:
            drop_dups = st.checkbox("Drop duplicate rows", value=True)
        with c2:
            strip_ws = st.checkbox("Strip whitespace in text", value=True)
        with c3:
            dedupe_cat = st.checkbox("Upper-case text columns", value=False)

        st.markdown("**Fill / impute missing values:**")
        fill_col = st.selectbox("Which column?", ["-- all numeric columns --"] + list(df.columns))
        fill_method = st.selectbox("Fill method",
                                   ["Do not fill", "Fill with 0", "Fill with column mean",
                                    "Fill with column median", "Drop rows with missing"])
        if st.button("🔧 Apply cleaning"):
            work = df.copy()
            logs = []
            if strip_ws:
                for col in work.select_dtypes(include="object").columns:
                    work[col] = work[col].astype(str).str.strip().replace({"nan": "", "None": ""})
                logs.append("Whitespace stripped in text columns")
            if dedupe_cat:
                for col in work.select_dtypes(include="object").columns:
                    work[col] = work[col].astype(str).str.upper()
                logs.append("Text normalised to UPPER case")
            if drop_dups:
                before = len(work)
                work = work.drop_duplicates().reset_index(drop=True)
                logs.append(f"Dropped {before - len(work)} duplicate rows")
            if fill_method != "Do not fill":
                targets = [fill_col] if fill_col != "-- all numeric columns --" else list(
                    work.select_dtypes(include=np.number).columns)
                for col in targets:
                    s = work[col]
                    if fill_method == "Fill with 0":
                        work[col] = s.fillna(0)
                    elif fill_method == "Fill with column mean":
                        work[col] = s.fillna(s.mean())
                    elif fill_method == "Fill with column median":
                        work[col] = s.fillna(s.median())
                    else:
                        work = work.dropna(subset=[col]).reset_index(drop=True)
                logs.append(f"{fill_method} applied to {len(targets)} numeric column(s)")
            st.session_state.df = work
            for lg in logs:
                st.info(lg)
            st.success("✅ Cleaning applied. Preview below.")
            st.dataframe(st.session_state.df.head(200), use_container_width=True)

        st.markdown("---")
        st.subheader("⚡ Auto-convert types (numbers/date strings → real types)")
        if st.button("🧪 Run auto type conversion"):
            new_df, changes = auto_convert(df)
            st.session_state.df = new_df
            if changes:
                for c in changes:
                    st.info(c)
            else:
                st.info("No obvious type conversions found.")
            st.dataframe(st.session_state.df.head(200), use_container_width=True)

        st.markdown("---")
        st.subheader("🗑 Delete a column")
        drop_col = st.selectbox("Column to remove", ["-- choose --"] + list(df.columns))
        if drop_col != "-- choose --" and st.button("❌ Delete column"):
            st.session_state.df = df.drop(columns=[drop_col]).reset_index(drop=True)
            st.success(f"Deleted `{drop_col}`")
            st.rerun()

    # ---- TAB 4: download ----
    with tab_download:
        st.subheader("⬇️ Download cleaned data")
        st.markdown(f"Current data: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
        buf_x = io.BytesIO()
        with pd.ExcelWriter(buf_x, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="CleanedData")
        st.download_button(
            "📥 Download cleaned Excel (.xlsx)",
            data=buf_x.getvalue(),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        buf_c = io.StringIO()
        df.to_csv(buf_c, index=False)
        st.download_button(
            "📥 Download cleaned CSV",
            data=buf_c.getvalue().encode("utf-8-sig"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
        st.caption("CSV is UTF-8 with BOM so Excel opens Tamil/Unicode text correctly.")

    # ---- TAB 5: analysis ----
    with tab_analyze:
        st.subheader("💡 What analysis can you do with this data?")
        df_conv, _ = auto_convert(df)
        recs = suggested_analysis(df_conv)
        for r in recs:
            st.markdown(f"- {r}")

        st.markdown("---")
        st.subheader("Quick EDA (works out of the box)")
        num_cols = list(df_conv.select_dtypes(include=np.number).columns)
        if num_cols:
            st.markdown("**Descriptive statistics**")
            st.dataframe(df_conv[num_cols].describe(), use_container_width=True)

            if len(num_cols) >= 2:
                st.markdown("**Correlation heatmap (numeric columns)**")
                try:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.heatmap(df_conv[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                except Exception:
                    st.dataframe(df_conv[num_cols].corr(), use_container_width=True)

            st.markdown("**Column distribution (pick one)**")
            hist_col = st.selectbox("Numeric column", num_cols)
            import matplotlib.pyplot as plt
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            df_conv[hist_col].dropna().hist(bins=30, ax=ax2)
            ax2.set_title(f"Distribution of {hist_col}")
            st.pyplot(fig2)

        cat_cols = list(df_conv.select_dtypes(include="object").columns)
        if cat_cols:
            st.markdown("**Category counts (pick a text column)**")
            cat_pick = st.selectbox("Text column", cat_cols)
            counts = df_conv[cat_pick].value_counts().head(10)
            st.dataframe(counts, use_container_width=True)
            import matplotlib.pyplot as plt
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            counts.plot(kind="bar", ax=ax3)
            ax3.set_title(f"Top values in {cat_pick}")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

else:
    st.markdown(
        "## 👋 Welcome\n\n"
        "Upload an **Excel (.xlsx/.xls)** or **CSV** file from the sidebar to start.\n\n"
        "### What this tool does\n"
        "1. **Loads & shows** your data with a full **Quality Report**\n"
        "2. **Finds issues** — missing values, wrong types, duplicates, outliers, messy text\n"
        "3. **Lets you edit** any cell directly in the grid\n"
        "4. **One-click cleaning** — drop dupes, strip spaces, fill missing, convert types\n"
        "5. **Downloads** clean Excel / CSV\n"
        "6. **Suggests & runs** analyses (describe, correlations, charts)\n"
    )
