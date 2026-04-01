import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from scipy.stats import f_oneway

st.set_page_config(layout="wide", page_title="Oil Sands Lake Analysis Dashboard")

st.title("Oil Sands Lake Analysis Dashboard")
st.caption("Interactive dashboard for comparing lake chemistry across Near, Mid, and Far distance groups.")

# =========================
# HELPERS
# =========================
def derive_lake(site_name: str) -> str:
    if pd.isna(site_name):
        return "Unknown"
    s = str(site_name).upper()

    if "ISADORE" in s:
        return "Isadore"
    if "MILDRED" in s:
        return "Mildred"
    if "KEARL" in s:
        return "Kearl"
    if "MCCLELLAND" in s:
        return "McClelland"
    if "NAMUR" in s:
        return "Namur"
    if "GREGOIRE" in s:
        return "Gregoire"

    return "Unknown"


UNIT_MAP = {
    "Aluminum Dissolved": "µg/L",
    "Aluminum Total Recoverable": "µg/L",
    "Calcium Dissolved": "mg/L",
    "Calcium Total": "mg/L",
    "Copper Dissolved": "µg/L",
    "Copper Total Recoverable": "µg/L",
    "Lead Dissolved": "µg/L",
    "Lead Total Recoverable": "µg/L",
    "Mercury Dissolved": "µg/L",
    "Mercury Total": "µg/L",
    "Methyl Mercury": "µg/L",
    "Nickel Dissolved": "µg/L",
    "Nickel Total Recoverable": "µg/L",
    "Nitrogen Kjeldahl Dissolved": "mg/L",
    "Nitrogen Kjeldahl Total": "mg/L",
    "pH": "",
    "pH (Field)": "",
    "Phosphorus Total": "mg/L",
    "Phosphorus Total Dissolved": "mg/L",
    "Turbidity": "NTU",
    "Vanadium Total Recoverable": "µg/L",
    "Vanadium Dissolved": "µg/L",
    "OXYGEN BIOCHEMICAL DEMAND": "mg/L",
    "Oxygen Dissolved (Field Meter)": "mg/L",
    "Oxygen dissolved % saturation": "%",
}

def with_unit(var_name: str) -> str:
    unit = UNIT_MAP.get(var_name, "")
    return f"{var_name} ({unit})" if unit else var_name


# =========================
# LOAD DATA
# =========================
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("FINAL_NORMALIZED_FULL.csv")

    # Parse timestamps
    if "Sampling Timestamp" in df.columns:
        df["Sampling Timestamp"] = pd.to_datetime(df["Sampling Timestamp"], errors="coerce")

    # Derive lake
    if "Site Name" in df.columns:
        df["Lake"] = df["Site Name"].apply(derive_lake)
    else:
        df["Lake"] = "Unknown"

    return df


df = load_data()

# =========================
# COORDINATES (INCLUDING AR6)
# =========================
lake_coords = {
    "Isadore": (57.23041, -111.60697),
    "Mildred": (57.055560, -111.588890),
    "Kearl": (57.2917, -111.2333),
    "McClelland": (57.49125, -111.27844),
    "Namur": (57.4444, -112.6211),
    "Gregoire": (56.48447, -110.83511),
    "AR6": (57.02, -111.50),
}
df["Latitude"] = df["Lake"].map(lambda x: lake_coords.get(x, (np.nan, np.nan))[0])
df["Longitude"] = df["Lake"].map(lambda x: lake_coords.get(x, (np.nan, np.nan))[1])

# =========================
# DISTANCE FROM AR6 
# =========================
AR6_LAT, AR6_LON = lake_coords["AR6"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

df["Distance (km)"] = haversine(
    df["Latitude"],
    df["Longitude"],
    AR6_LAT,
    AR6_LON
)

# =========================
# CHEMISTRY VARIABLES
# =========================
chemistry_vars = [
    "Aluminum Dissolved",
    "Aluminum Total Recoverable",
    "Calcium Dissolved",
    "Calcium Total",
    "Copper Dissolved",
    "Copper Total Recoverable",
    "Lead Dissolved",
    "Lead Total Recoverable",
    "Mercury Dissolved",
    "Mercury Total",
    "Methyl Mercury",
    "Nickel Dissolved",
    "Nickel Total Recoverable",
    "Nitrogen Kjeldahl Dissolved",
    "Nitrogen Kjeldahl Total",
    "pH",
    "pH (Field)",
    "Phosphorus Total",
    "Phosphorus Total Dissolved",
    "Turbidity",
    "Vanadium Total Recoverable",
    "Vanadium Dissolved",
    "OXYGEN BIOCHEMICAL DEMAND",
    "Oxygen Dissolved (Field Meter)",
    "Oxygen dissolved % saturation",
]
chemistry_vars = [c for c in chemistry_vars if c in df.columns]

# =========================
# FILTERS
# =========================
col1, col2 = st.columns(2)

with col1:
    selected_group = st.selectbox(
        "Distance Group",
        ["All"] + sorted(df["Distance Group"].dropna().unique().tolist())
    )

with col2:
    selected_lake = st.selectbox(
        "Lake",
        ["All"] + sorted(df["Lake"].dropna().unique().tolist())
    )

filtered_df = df.copy()

if selected_group != "All":
    filtered_df = filtered_df[filtered_df["Distance Group"] == selected_group]

if selected_lake != "All":
    filtered_df = filtered_df[filtered_df["Lake"] == selected_lake]

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)


date_min = filtered_df["Sampling Timestamp"].min() if "Sampling Timestamp" in filtered_df.columns else None
date_max = filtered_df["Sampling Timestamp"].max() if "Sampling Timestamp" in filtered_df.columns else None

col1.metric("Total Samples", f"{len(filtered_df):,}")
col2.metric("Lakes in Scope", filtered_df["Lake"].nunique())
col3.metric(
    "Date Range",
    "N/A" if pd.isna(date_min) or pd.isna(date_max) else f"{date_min.date()} → {date_max.date()}"
)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Distributions", "PCA", "ANOVA", "Random Forest"]
)

# =========================
# OVERVIEW
# =========================
with tab1:
    st.subheader("Overview")

    st.markdown(
        """
        - Compare chemistry patterns across **Near**, **Mid**, and **Far** lakes.
        - Explore chemical distributions across lakes and distance groups.
        - Use PCA to assess multivariate structure.
        - Use ANOVA to test differences between groups.
        - Use Random Forest to identify the strongest chemical drivers.
        """
    )

    with st.expander("Data inconsistencies and harmonization notes", expanded=True):
        st.markdown(
            """
            **Key data issues addressed during cleaning**
            - Data were compiled from multiple publicly available monitoring sources.
            - Coverage varies across lakes, years, and parameters depending on the original sampling programs.
            - Some lakes required alternate source files, which introduced differences in naming conventions, units, and measurement methods.
            - Parameters were standardized to a common schema before analysis, and units were harmonized on a parameter-by-parameter basis.
            - Field and lab measurements were kept separate when their definitions differed.
            - In the wide-format dataset, the `Unit Symbol` field is included as a reference note only and does not represent every parameter in a row.
            - All available data were included after cleaning and standardization; however, variation in data availability and sampling coverage reflects the source monitoring programs and was not controlled within this project.
            """
        )
    
    with st.expander("Parameter Units Reference", expanded=False):
        unit_df = pd.DataFrame([
        {"Parameter": k, "Unit": v}
        for k, v in UNIT_MAP.items()
    ]).sort_values("Parameter")

    # FIX INDEX
    unit_df = unit_df.reset_index(drop=True)
    unit_df.index = unit_df.index + 1

    st.dataframe(unit_df, width="stretch")

    st.caption(
        "All variables were standardized to consistent units prior to analysis. "
        "This table provides the reference units used across all visualizations."
    )

    with st.expander("Lake groups and locations", expanded=True):
        location_summary = (
            filtered_df.groupby(["Distance Group", "Lake"])
            .size()
            .reset_index(name="Samples")
            .sort_values(["Distance Group", "Lake"])
        )
        
        location_summary.index = location_summary.index + 1

        st.dataframe(location_summary, width="stretch")

        fig_locations = px.bar(
            location_summary,
            x="Lake",
            y="Samples",
            color="Distance Group",
            barmode="group",
            title="Samples by Lake and Distance Group"
        )
        
        
        
        st.plotly_chart(fig_locations, width="stretch")

    with st.expander("Data sources", expanded=False):
        st.markdown(
            """
            - Isadore Lake, Kearl Lake, McClelland Lake, Namur Lake: [Oil Sands Monitoring / OSMP Portal](https://osmdataportal.alberta.ca/applications/public.html?publicuser=Guest#waterdata/stationoverview)
            - Mildred Lake, Gregoire Lake: [Alberta Water Quality Data Portal](https://environment.extranet.gov.ab.ca/apps/WaterQuality/dataportal/)
            - [Alberta Water Quality Data Portal](https://github.com/muntahaltaie/DATA-501-DASHBOARD/tree/main)
            """
        )
    with st.expander("Study Area & Site Classification", expanded=True):

        st.image(
        "study_area_map.png",
        caption="""
        Study lakes classified by distance from oil sands activity:
        Near-field (<20 km), Mid-field (20–50 km), Far-field (>50 km).
        Source: Kelly et al. (2010), adapted from IOP Science.
        """,
        width="stretch"
    )

    st.markdown("""
    ### Interpretation

    - **Near-field lakes** (black) are closest to oil sands mining operations and are expected to show the strongest anthropogenic influence.
    - **Mid-field lakes** (blue) represent transitional zones where both natural and industrial signals may be present.
    - **Far-field lakes** (red) act as comparison/reference systems with reduced direct exposure.

    The 50 km radius shown provides a spatial reference for defining mid vs. far-field conditions.
    """)
    
    # =========================
    # INTERACTIVE MAP
    # =========================
    st.subheader("Interactive Lake Map")

    map_df = (
        filtered_df.groupby(["Lake", "Distance Group", "Latitude", "Longitude", "Distance (km)"])
        .size()
        .reset_index(name="Samples")
    )

    fig_map = px.scatter_mapbox(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color= "Distance Group",
        size="Samples",
        hover_name="Lake",
        hover_data={
            "Distance (km)": ":.1f",  
            "Latitude": False,
            "Longitude": False,
            "Samples": True
        },
        zoom=5,
        height=500,
        title="Lake Locations Relative to Oil Sands Site (AR6)"
    )

    # =========================
    # ADD AR6 (DISTINCT)
    # =========================
    fig_map.add_scattermapbox(
        lat=[57.02],
        lon=[-111.50],
        mode="markers+text",
        marker=dict(
            size=26,
            color="#FFD700",
            opacity=1.0
        ),
        text=["AR6"],
        textposition="top center",
        name="AR6"
    )

    # Dark theme
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig_map, width="stretch", key="map_chart")

    # =========================
    # CAPTION
    # =========================
    st.caption(
        """
        AR6 represents the central oil sands operational site. 
        Distances are computed using the Haversine formula (km) from each lake to AR6.
        Distance groups are defined as: Near (<20 km), Mid (20–50 km), Far (>50 km)
        """
    )

    st.markdown("### Average Distance by Lake")

    distance_summary = (
        filtered_df.groupby("Lake")["Distance (km)"]
        .mean()
        .reset_index()
        .sort_values("Distance (km)")
    )
    
    distance_summary.index = distance_summary.index + 1

    st.dataframe(distance_summary, width="stretch")


    if "Sampling Timestamp" in filtered_df.columns:
        trend_options = [c for c in chemistry_vars if c in filtered_df.columns]

    if trend_options:
        default_trends = [c for c in ["Phosphorus Total", "Turbidity"] if c in trend_options]

        selected_trends = st.multiselect(
            "Trend Variables",
            trend_options,
            default=default_trends,
            key="trend_variables"
        )

        if selected_trends:
            trend_df = filtered_df.copy()
            trend_df["Year"] = trend_df["Sampling Timestamp"].dt.year

            for trend_var in selected_trends:
                annual = (
                    trend_df.groupby(["Year", "Distance Group"], dropna=False)[trend_var]
                    .mean()
                    .reset_index()
                )
                
                annual.index = annual.index + 1

                fig_trend = px.line(
                    annual,
                    x="Year",
                    y=trend_var,
                    color="Distance Group",
                    markers=True,
                    title=f"Annual Mean Trend: {with_unit(trend_var)}"
                )
                fig_trend.update_yaxes(title_text=with_unit(trend_var))
                st.plotly_chart(
                    fig_trend,
                    width="stretch",
                    key=f"trend_chart_{trend_var}"
                )


# =========================
# DISTRIBUTIONS
# =========================
with tab2:
    st.subheader("Parameter Distributions")

    selected_params = st.multiselect(
        "Select parameters",
        chemistry_vars,
        default=[c for c in ["Phosphorus Total", "Calcium Dissolved", "Turbidity", "pH"] if c in chemistry_vars]
    )

    if selected_params:
        cols = st.columns(2)
        for i, col in enumerate(selected_params):
            fig = px.box(
                filtered_df,
                x="Distance Group",
                y=col,
                color= "Distance Group",
                title=with_unit(col)
            )
            fig.update_yaxes(title_text=with_unit(col))
            cols[i % 2].plotly_chart(fig, width="stretch")

# =========================
# PCA
# =========================
with tab3:
    st.subheader("PCA Analysis")

    pca_vars = st.multiselect(
        "Select variables for PCA",
        chemistry_vars,
        default=[c for c in chemistry_vars if c not in ["pH (Field)"]]
    )

    st.markdown("### Variables Included")
    vars_df = pd.DataFrame({"Variable": pca_vars})
    vars_df.index = vars_df.index + 1

    st.dataframe(vars_df, width="stretch")

    if len(pca_vars) < 2:
        st.warning("Not enough chemistry variables are available to run PCA.")
    else:
        X = filtered_df[pca_vars].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all")
        X = X.loc[:, X.nunique() > 1]
        X = X.fillna(X.median())

        if X.shape[1] < 2:
            st.warning("After cleaning, not enough variables remain to compute PCA.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)

            pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
            pca_df["Distance Group"] = filtered_df["Distance Group"].values
            pca_df["Lake"] = filtered_df["Lake"].values

            pc1_var = pca.explained_variance_ratio_[0] * 100
            pc2_var = pca.explained_variance_ratio_[1] * 100

            color_col = "Distance Group" if filtered_df["Distance Group"].nunique() > 1 else "Lake"

            col1, col2 = st.columns([2, 1])

            with col1:
                fig_pca = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    color=color_col,
                    hover_data=["Lake"],
                    title=f"PCA Scatter Plot (PC1: {pc1_var:.1f}%, PC2: {pc2_var:.1f}%)"
                )
                st.plotly_chart(fig_pca, width="stretch")

            with col2:
                variance_df = pd.DataFrame({
                    "Component": ["PC1", "PC2"],
                    "Explained Variance (%)": [pc1_var, pc2_var]
                })
                variance_df = variance_df.reset_index(drop=True)
                variance_df.index = variance_df.index + 1

                st.dataframe(variance_df, width="stretch")

            loadings = pd.DataFrame(
                pca.components_.T,
                columns=["PC1", "PC2"],
                index=X.columns
            ).reset_index()
            loadings.columns = ["Variable", "PC1", "PC2"]
            loadings = loadings.reset_index(drop=True)
            loadings.index = loadings.index + 1

            st.markdown("### PCA Loadings")
            selected_pc = st.selectbox("Select component", ["PC1", "PC2"], key="pca_component")
            loadings_sorted = loadings.sort_values(selected_pc, key=lambda s: s.abs(), ascending=False)

            fig_load = px.bar(
                loadings_sorted.head(15),
                x=selected_pc,
                y="Variable",
                orientation="h",
                title=f"Top Variable Loadings for {selected_pc}"
            )
            fig_load.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_load, width="stretch")

            st.dataframe(loadings, width="stretch")

# =========================
# ANOVA
# =========================
with tab4:
    st.subheader("ANOVA Analysis")

    anova_vars = st.multiselect(
        "Select variables for ANOVA",
        chemistry_vars,
        default=[c for c in ["Phosphorus Total", "Phosphorus Total Dissolved", "Turbidity", "pH"] if c in chemistry_vars],
        key="anova_vars"
    )

    if filtered_df["Distance Group"].nunique() > 1:
        group_col = "Distance Group"
    elif filtered_df["Lake"].nunique() > 1:
        group_col = "Lake"
    else:
        group_col = None

    if group_col is None:
        st.warning("ANOVA requires at least two groups. Change the filters.")
    else:
        st.markdown("### Grouping Variable")
        st.write(group_col)

        anova_results = []

        for col in anova_vars:
            temp = filtered_df[[group_col, col]].copy()
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

            grouped = []
            for g in sorted(temp[group_col].dropna().unique()):
                vals = temp[temp[group_col] == g][col].dropna()
                if len(vals) > 1:
                    grouped.append(vals)

            if len(grouped) >= 2:
                f_stat, p_val = f_oneway(*grouped)
                anova_results.append({
                    "Variable": col,
                    "F Statistic": f_stat,
                    "p-value": p_val,
                    "Unit": UNIT_MAP.get(col, "")
                })

        if not anova_results:
            st.warning("No ANOVA results could be computed.")
        else:
            anova_df = (
                pd.DataFrame(anova_results)
                .sort_values("p-value")
                .reset_index(drop=True)
            )

            anova_df.index = anova_df.index + 1
            st.dataframe(anova_df, width="stretch")

            alpha = st.slider("Significance threshold", 0.01, 0.10, 0.05, 0.01, key="anova_alpha")
            significant = anova_df[anova_df["p-value"] < alpha]["Variable"].tolist()

            max_plots = st.slider("Max boxplots", 4, 16, 8, 2, key="anova_max_plots")
            vars_to_plot = significant[:max_plots] if significant else anova_df["Variable"].head(max_plots).tolist()

            if vars_to_plot:
                rows = math.ceil(len(vars_to_plot) / 2)
                for r in range(rows):
                    cols = st.columns(2)
                    for c in range(2):
                        idx = r * 2 + c
                        if idx < len(vars_to_plot):
                            var = vars_to_plot[idx]
                            pval = anova_df.loc[anova_df["Variable"] == var, "p-value"].values[0]

                            fig_box = px.box(
                                filtered_df,
                                x=group_col,
                                y=var,
                                color=group_col,
                                title=f"{with_unit(var)} (p = {pval:.4f})"
                            )
                            fig_box.update_yaxes(title_text=with_unit(var))

                            cols[c].plotly_chart(
                                fig_box,
                                width="stretch",
                                key=f"anova_box_{var}_{idx}"
                            )

            st.markdown("### Summary")
            st.write(
                f"ANOVA was computed across **{group_col}** for "
                f"**{len(anova_vars)}** selected variables. "
                f"**{len(significant)}** variables met alpha = {alpha:.2f}."
            )

# =========================
# RANDOM FOREST
# =========================
with tab5:
    st.subheader("Random Forest Analysis")

    rf_vars = st.multiselect(
        "Select variables for Random Forest",
        chemistry_vars,
        default=[c for c in chemistry_vars if c not in ["pH (Field)"]],
        key="rf_vars"
    )

    if filtered_df["Distance Group"].nunique() > 1:
        target_col = "Distance Group"
    elif filtered_df["Lake"].nunique() > 1:
        target_col = "Lake"
    else:
        target_col = None

    if target_col is None:
        st.warning("Random Forest requires at least two target classes. Change the filters.")
    else:
        st.markdown("### Prediction Target")
        st.write(target_col)

        X = filtered_df[rf_vars].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all")
        X = X.loc[:, X.nunique() > 1]

        y = filtered_df[target_col].copy()

        if X.shape[1] < 2 or y.nunique() < 2:
            st.warning("Not enough usable variables or target classes remain after cleaning.")
        else:
            imputer = SimpleImputer(strategy="median")
            X_imputed = imputer.fit_transform(X)

            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.slider("Number of trees", 100, 500, 300, 50)
            with col2:
                max_depth = st.selectbox("Max depth", [None, 3, 5, 10, 20], index=0)
            with col3:
                test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced"
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("Classes Predicted", y.nunique())

            st.markdown("### Classification Report")
            report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(report_df, width="stretch")

            st.markdown("### Confusion Matrix")
            labels = sorted(y.unique().tolist())
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=True
            )
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            st.plotly_chart(fig_cm, width="stretch")

            st.markdown("### Feature Importance")

            importance_df = (
                pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": rf.feature_importances_,
                    "Unit": [UNIT_MAP.get(f, "") for f in X.columns]
                })
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )

            importance_df.index = importance_df.index + 1


            top_n = st.slider("Top features to display", 5, min(20, len(importance_df)), min(10, len(importance_df)))

            importance_plot = importance_df.head(top_n).copy()
            importance_plot["Feature Label"] = importance_plot.apply(
                lambda row: with_unit(row["Feature"]),
                axis=1
            )

            fig_imp = px.bar(
                importance_plot,
                x="Importance",
                y="Feature Label",
                orientation="h",
                title=f"Top {top_n} Variable Drivers"
            )
            fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_imp, width="stretch")

            st.dataframe(importance_df, width="stretch")

# =========================
# DATA PREVIEW
# =========================
st.subheader("Data Preview")

preview_df = filtered_df.copy()

# Preview controls
with st.expander("Preview Controls", expanded=True):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        preview_group = st.selectbox(
            "Preview Distance Group",
            ["All"] + sorted(df["Distance Group"].dropna().unique().tolist()),
            key="preview_group"
        )

    with c2:
        preview_lake = st.selectbox(
            "Preview Lake",
            ["All"] + sorted(df["Lake"].dropna().unique().tolist()),
            key="preview_lake"
        )

    with c3:
        row_limit = st.selectbox(
            "Rows to display",
            options=[10, 25, 50, 100, 250, 500, 1000, "All"],
            index=3,
            key="preview_rows"
        )

    with c4:
        valid_only = st.checkbox(
            "Only show rows with valid selected data",
            value=True,
            key="preview_valid_only"
        )

    default_preview_cols = [
        c for c in [
            "Distance Group",
            "Lake",
            "Sampling Timestamp",
            "Site Name",
            "Phosphorus Total",
            "Turbidity",
            "pH",
            "Distance (km)"
        ] if c in preview_df.columns
    ]

    selected_preview_cols = st.multiselect(
        "Columns to display",
        options=preview_df.columns.tolist(),
        default=default_preview_cols,
        key="preview_columns"
    )

# Apply preview filters
if preview_group != "All":
    preview_df = preview_df[preview_df["Distance Group"] == preview_group]

if preview_lake != "All":
    preview_df = preview_df[preview_df["Lake"] == preview_lake]

# Keep only selected columns
if selected_preview_cols:
    preview_df = preview_df[selected_preview_cols].copy()

# Filter to rows with real data in selected columns
metadata_cols = {"Distance Group", "Lake", "Sampling Timestamp", "Site Name", "Distance (km)"}
data_check_cols = [c for c in selected_preview_cols if c not in metadata_cols]

if valid_only and data_check_cols:
    temp_check = preview_df[data_check_cols].copy()
    preview_df = preview_df[temp_check.notna().any(axis=1)]

# Reset index to start at 1
preview_df = preview_df.reset_index(drop=True)
preview_df.index = preview_df.index + 1

# Apply row limit
if row_limit != "All":
    preview_df = preview_df.head(row_limit)

# Create display copy without breaking numeric dtypes
preview_display = preview_df.copy()

# For object/text columns only replace missing with blank
for col in preview_display.columns:
    if preview_display[col].dtype == "object":
        preview_display[col] = preview_display[col].fillna("")

st.dataframe(preview_display, width="stretch")
