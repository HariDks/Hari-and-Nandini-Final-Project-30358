"""
dashboard.py
Chicago Streetlight Outages & Crime — Interactive Streamlit Dashboard

Pages:
  1. Crime Impact  — pre/during crime rate comparison around outage complaints and scatter plot correlations between crime rates and census tract variables
  2. Hotspot Analysis — per-tract rolling OLS: outage rate → crime-in-buffer rate
  3. Law Enforcement Dashboard- split hotspot analysis by crime type

Metrics (symmetric K-day windows, rate = crimes / (K * n_requests)):
  A) pre_rate    — crime rate in the K days BEFORE each complaint
  B) during_rate — crime rate in the first K days AFTER each complaint
  C) diff_rate   — during_rate minus pre_rate  (the key comparison)
  D) avg_outage_days — mean time-to-fix for requests in each tract
"""

import warnings
warnings.filterwarnings("ignore")

import pathlib
from scipy import stats as scipy_stats

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely.geometry
import plotly.express as px

import pydeck as pdk

_HERE    = pathlib.Path(__file__).parent
_ROOT    = _HERE.parent                        # streamlit_app/ → project root
_RAW     = _ROOT / "data" / "raw_data"
_DERIVED = _ROOT / "data" / "derived_data"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Streetlight Outages & Crime",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #F0B8A8;
    }
    .stApp {
        background-color: #FDFAF4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Label lookup ─────────────────────────────────────────────────────────────
METRIC_LABELS = {
    "diff_rate":       "Difference (During − Pre)  [crimes/day/request]",
    "during_rate":     "During-window crime rate  [crimes/day/request]",
    "pre_rate":        "Pre-window crime rate  [crimes/day/request]",
    "avg_outage_days": "Avg outage days",
}

METRIC_SHORT = {
    "diff_rate":       "During − Pre",
    "during_rate":     "During rate",
    "pre_rate":        "Pre rate",
    "avg_outage_days": "Avg outage days",
}


OVERVIEW_DESCRIPTION = """
Chicago experiences roughly **50,000 streetlight outage complaints** per year.
Each complaint marks a location where one or more lights went dark — sometimes for
days, sometimes for weeks. The prevailing theory in criminology (the "broken windows"
and "routine activity" frameworks) holds that reduced visibility creates opportunity
for crime: offenders can act with less risk of identification, and potential victims
feel less safe.

This project links **Chicago 311 streetlight outage records (2011–2018)** with the
**Chicago Police Department crime dataset** to test that theory at the census-tract
level. For every outage complaint, we measure crime rates in a spatial buffer around
the outage location both *before* and *during* the outage — holding geography constant
and letting only the lighting status vary.

Our key finding is that crime rates are **modestly higher** in the days immediately
following an outage complaint compared to the days just before it

> **Use the pages in the sidebar** to explore the pre/during crime rate comparison
> (Crime Impact), identify the tracts where crime rates are higher than predicted (Hotspot Analysis),
> or view per–crime-type breakdowns of tracts where crime rates are higher than predicted (Law Enforcement Dashboard).
"""

CRIME_TYPES = [
    "ARSON", "ASSAULT", "BATTERY", "BURGLARY",
    "CONCEALED CARRY LICENSE VIOLATION", "CRIM SEXUAL ASSAULT",
    "CRIMINAL DAMAGE", "CRIMINAL SEXUAL ASSAULT", "CRIMINAL TRESPASS",
    "DECEPTIVE PRACTICE", "GAMBLING", "HOMICIDE",
    "INTERFERENCE WITH PUBLIC OFFICER", "INTIMIDATION", "KIDNAPPING",
    "LIQUOR LAW VIOLATION", "MOTOR VEHICLE THEFT", "NARCOTICS",
    "OBSCENITY", "OFFENSE INVOLVING CHILDREN", "OTHER OFFENSE",
    "PROSTITUTION", "PUBLIC PEACE VIOLATION", "ROBBERY",
    "SEX OFFENSE", "STALKING", "THEFT", "WEAPONS VIOLATION",
]

# ─── Data loaders ─────────────────────────────────────────────────────────────
def load_data():
    gdf = gpd.read_file(_DERIVED / "streetlight_crime_events_with_tracts.geojson")
    tract_polys = (
        gpd.read_file(_DERIVED / "tract_level_crime_summary.geojson")[["tract_geoid", "geometry"]]
        .to_crs(epsg=4326)
    )

    pts_wgs = gdf.to_crs(epsg=4326)
    gdf = gdf.copy()
    gdf["lng"] = pts_wgs.geometry.x
    gdf["lat"] = pts_wgs.geometry.y

    df = pd.DataFrame(gdf.drop(columns="geometry"))

    df["creation_date"]   = pd.to_datetime(df["creation_date"])
    df["completion_date"] = pd.to_datetime(df["completion_date"])
    df["crime_date"]      = pd.to_datetime(df["crime_date"])

    crime_types = sorted(df["primary_type"].dropna().unique())

    return df, tract_polys, crime_types


@st.cache_data
def load_hotspot_panel():
    panel  = pd.read_csv(_DERIVED / "tract_week_crime_outage_panel.csv")
    tracts = pd.read_csv(_DERIVED / "chicago_streetlights_tract_data.csv")

    ref = gpd.read_file(_DERIVED / "tract_level_crime_summary.geojson")
    chicago_geoids = set(ref["tract_geoid"])
    tracts["tract_geoid"] = (tracts["GISJOIN"].str[1:3]
                             + tracts["GISJOIN"].str[4:7]
                             + tracts["GISJOIN"].str[8:])
    chi_tracts = tracts[tracts["tract_geoid"].isin(chicago_geoids)][
        ["GISJOIN", "streetlight_count_est", "tract_geoid"]
    ]
    panel = panel.merge(chi_tracts, on="GISJOIN", how="inner")
    panel["week"] = pd.to_datetime(panel["week"])
    panel = panel[(panel["streetlight_count_est"] > 0) & (panel["total_crimes"] > 0)].copy()
    panel["outage_rate"]       = panel["active_outages"] / (7 * panel["streetlight_count_est"])
    panel["crime_buffer_rate"] = panel["crimes_inside"]  / panel["total_crimes"]
    return panel[["GISJOIN", "tract_geoid", "week", "outage_rate", "crime_buffer_rate"]].dropna()


@st.cache_data
def fit_per_tract_models() -> pd.DataFrame:
    """
    Per-tract OLS trained on 2011-2017 only.
    Each tract gets its own intercept + slope.
    Requires >= 20 weeks with valid data and non-zero variance in outage_rate.
    Returns DataFrame: GISJOIN, tract_geoid, intercept, slope.
    """
    panel = load_hotspot_panel()
    train = panel[panel["week"] < pd.Timestamp("2018-01-01")].dropna(
        subset=["outage_rate", "crime_buffer_rate"]
    )
    rows = []
    for gisjoin, grp in train.groupby("GISJOIN"):
        if len(grp) < 20 or grp["outage_rate"].std() < 1e-10:
            continue
        slope, intercept, _, _, _ = scipy_stats.linregress(
            grp["outage_rate"].values,
            grp["crime_buffer_rate"].values,
        )
        rows.append({
            "GISJOIN":     gisjoin,
            "tract_geoid": grp["tract_geoid"].iloc[0],
            "intercept":   float(intercept),
            "slope":       float(slope),
        })
    return pd.DataFrame(rows)


@st.cache_data
def compute_2018_risk(cutoff_week_str: str) -> pd.DataFrame:
    """
    For each tract, apply its 2011-2017 model to 2018 weeks up to the cutoff.
    Residual = actual − predicted (tract's own model prediction).
    High positive mean residual = crime running above what THIS tract's own
    historical outage-crime relationship would predict → highest-risk tracts.
    """
    panel  = load_hotspot_panel()
    models = fit_per_tract_models().set_index("GISJOIN")

    cutoff   = pd.Timestamp(cutoff_week_str)
    data2018 = panel[
        (panel["week"] >= pd.Timestamp("2018-01-01")) &
        (panel["week"] <= cutoff) &
        (panel["GISJOIN"].isin(models.index))
    ].copy()

    if data2018.empty:
        return pd.DataFrame(columns=["GISJOIN","tract_geoid","mean_residual",
                                      "mean_actual","mean_predicted",
                                      "latest_outage_rate","n_weeks"])

    data2018["intercept"] = data2018["GISJOIN"].map(models["intercept"])
    data2018["slope"]     = data2018["GISJOIN"].map(models["slope"])
    data2018["predicted"] = data2018["intercept"] + data2018["slope"] * data2018["outage_rate"]
    data2018["residual"]  = data2018["crime_buffer_rate"] - data2018["predicted"]

    risk = (
        data2018.sort_values("week")
        .groupby(["GISJOIN", "tract_geoid"], as_index=False)
        .agg(
            mean_residual      =("residual",          "mean"),
            mean_actual        =("crime_buffer_rate",  "mean"),
            mean_predicted     =("predicted",          "mean"),
            latest_outage_rate =("outage_rate",        "last"),
            n_weeks            =("residual",           "count"),
        )
    )
    return risk


@st.cache_data
def load_acs() -> pd.DataFrame:
    df = pd.read_csv(_DERIVED / "chicago_streetlights_tract_data.csv",
                     usecols=["tract_geoid","population","population_density",
                              "pct_black","pct_white","pct_college","unemployment_rate"])
    df["tract_geoid"] = df["tract_geoid"].astype(str)
    return df.set_index("tract_geoid")


@st.cache_data
def load_full_panel() -> pd.DataFrame:
    """Weekly panel with all crime-type columns, filtered to Chicago tracts."""
    panel  = pd.read_csv(_DERIVED / "tract_week_crime_outage_panel.csv")
    tracts = pd.read_csv(_DERIVED / "chicago_streetlights_tract_data.csv")
    ref = gpd.read_file(_DERIVED / "tract_level_crime_summary.geojson")
    chicago_geoids = set(ref["tract_geoid"])
    tracts["tract_geoid"] = (tracts["GISJOIN"].str[1:3]
                             + tracts["GISJOIN"].str[4:7]
                             + tracts["GISJOIN"].str[8:])
    chi_tracts = tracts[tracts["tract_geoid"].isin(chicago_geoids)][
        ["GISJOIN", "streetlight_count_est", "tract_geoid"]
    ]
    panel = panel.merge(chi_tracts, on="GISJOIN", how="inner")
    panel["week"] = pd.to_datetime(panel["week"])
    panel = panel[panel["streetlight_count_est"] > 0].copy()
    panel["outage_rate"] = panel["active_outages"] / (7 * panel["streetlight_count_est"])
    return panel


@st.cache_data
def get_crime_type_cols():
    """Return list of (col_prefix, display_name) for crime types with ≥30 inside events."""
    panel = load_full_panel()
    result = []
    for col in sorted(panel.columns):
        if not col.endswith("_inside") or col == "crimes_inside":
            continue
        prefix = col[:-len("_inside")]
        outside_col = f"{prefix}_outside"
        if outside_col not in panel.columns:
            continue
        if panel[col].sum() < 30:
            continue
        display = prefix.replace("_", " ").title()
        result.append((prefix, display))
    return result


@st.cache_data
def load_nhgis_census() -> pd.DataFrame:

    keep = [
        "GISJOIN",
        "ADKWE001",
        "ADKXE001","ADKXE002","ADKXE003",
        "ADMZE001",
        "ADMZE002","ADMZE003","ADMZE004","ADMZE005",
        "ADMZE006","ADMZE007","ADMZE008","ADMZE009",
        "ADMZE010","ADMZE011","ADMZE012","ADMZE013",
        "ADMZE014","ADMZE015","ADMZE016",
        "ADMZE017","ADMZE018",
        "ADMZE019","ADMZE020","ADMZE021",
        "ADMZE022","ADMZE023","ADMZE024","ADMZE025",
        "ADPIE001","ADPIE002","ADPIE003",
        "ADPIE004","ADPIE005","ADPIE007",
        "STATE",
    ]

    df = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1WaiWZD1l6JUP24ZAO3nnzIpPBYKNRwTX",
        usecols=keep
    )

    df = df[df["STATE"] == "Illinois"].copy()

    df["tract_geoid"] = df["GISJOIN"].str[1:3] + df["GISJOIN"].str[4:7] + df["GISJOIN"].str[8:]

    no_hs_cols = [f"ADMZE{str(i).zfill(3)}" for i in range(2, 17)]

    df["pct_no_hs"] = df[no_hs_cols].sum(axis=1) / df["ADMZE001"].replace(0, np.nan)
    df["pct_hs_diploma"] = (df["ADMZE017"] + df["ADMZE018"]) / df["ADMZE001"].replace(0, np.nan)
    df["pct_some_college"] = (df["ADMZE019"] + df["ADMZE020"] + df["ADMZE021"]) / df["ADMZE001"].replace(0, np.nan)
    df["pct_bachelors"] = (df["ADMZE022"] + df["ADMZE023"] + df["ADMZE024"] + df["ADMZE025"]) / df["ADMZE001"].replace(0, np.nan)

    df["pct_black"] = df["ADKXE003"] / df["ADKXE001"].replace(0, np.nan)
    df["pct_white"] = df["ADKXE002"] / df["ADKXE001"].replace(0, np.nan)

    df["unemployment_rate"] = df["ADPIE005"] / df["ADPIE003"].replace(0, np.nan)
    df["pct_not_in_lf"] = df["ADPIE007"] / df["ADPIE001"].replace(0, np.nan)
    df["lf_participation"] = df["ADPIE002"] / df["ADPIE001"].replace(0, np.nan)

    df["population"] = df["ADKWE001"]

    out_cols = [
        "tract_geoid","population","pct_black","pct_white",
        "pct_no_hs","pct_hs_diploma","pct_some_college","pct_bachelors",
        "unemployment_rate","pct_not_in_lf","lf_participation"
    ]

    return df[out_cols].set_index("tract_geoid")


# Census variable display labels and axis titles
CENSUS_VAR_OPTIONS = {
    "Median income ($)":         ("median_income_estimate", "Median household income ($)"),
    "Population":                ("population",             "Census tract population"),
    "% Black":                   ("pct_black",              "Share Black population (%)"),
    "% White":                   ("pct_white",              "Share White population (%)"),
    "% Bachelor's or higher":    ("pct_bachelors",          "Share with bachelor's degree or higher (%)"),
    "% No HS diploma":           ("pct_no_hs",              "Share without HS diploma (%)"),
    "% Some college (no degree)":("pct_some_college",       "Share with some college, no degree (%)"),
    "% HS diploma / GED only":   ("pct_hs_diploma",         "Share with HS diploma / GED only (%)"),
    "Unemployment rate":         ("unemployment_rate",       "Unemployment rate (civilian LF)"),
    "% Not in labor force":      ("pct_not_in_lf",          "Share not in labor force (%)"),
    "Labor force participation": ("lf_participation",       "Labor force participation rate (%)"),
}


@st.cache_data
def load_crime_type_panel(crime_type_col: str) -> pd.DataFrame:
    """crime_buffer_rate = type_inside / (type_inside + type_outside) per tract-week."""
    panel = load_full_panel()
    inside_col  = f"{crime_type_col}_inside"
    outside_col = f"{crime_type_col}_outside"
    df = panel.copy()
    df["type_total"]        = df[inside_col] + df[outside_col]
    df = df[df["type_total"] > 0].copy()
    df["crime_buffer_rate"] = df[inside_col] / df["type_total"]
    return df[["GISJOIN","tract_geoid","week","outage_rate","crime_buffer_rate"]].dropna()


@st.cache_data
def fit_per_tract_crime_models(crime_type_col: str) -> pd.DataFrame:
    panel = load_crime_type_panel(crime_type_col)
    train = panel[panel["week"] < pd.Timestamp("2018-01-01")].dropna(
        subset=["outage_rate","crime_buffer_rate"]
    )
    rows = []
    for gisjoin, grp in train.groupby("GISJOIN"):
        if len(grp) < 20 or grp["outage_rate"].std() < 1e-10:
            continue
        slope, intercept, _, _, _ = scipy_stats.linregress(
            grp["outage_rate"].values, grp["crime_buffer_rate"].values
        )
        rows.append({"GISJOIN": gisjoin, "tract_geoid": grp["tract_geoid"].iloc[0],
                     "intercept": float(intercept), "slope": float(slope)})
    return pd.DataFrame(rows)


@st.cache_data
def compute_2018_crime_risk(crime_type_col: str, cutoff_week_str: str) -> pd.DataFrame:
    panel  = load_crime_type_panel(crime_type_col)
    models = fit_per_tract_crime_models(crime_type_col).set_index("GISJOIN")
    cutoff   = pd.Timestamp(cutoff_week_str)
    data2018 = panel[
        (panel["week"] >= pd.Timestamp("2018-01-01")) &
        (panel["week"] <= cutoff) &
        (panel["GISJOIN"].isin(models.index))
    ].copy()
    if data2018.empty:
        return pd.DataFrame(columns=["GISJOIN","tract_geoid","mean_residual",
                                      "mean_actual","mean_predicted",
                                      "latest_outage_rate","n_weeks"])
    data2018["intercept"] = data2018["GISJOIN"].map(models["intercept"])
    data2018["slope"]     = data2018["GISJOIN"].map(models["slope"])
    data2018["predicted"] = data2018["intercept"] + data2018["slope"] * data2018["outage_rate"]
    data2018["residual"]  = data2018["crime_buffer_rate"] - data2018["predicted"]
    return (
        data2018.sort_values("week")
        .groupby(["GISJOIN","tract_geoid"], as_index=False)
        .agg(mean_residual      =("residual",          "mean"),
             mean_actual        =("crime_buffer_rate",  "mean"),
             mean_predicted     =("predicted",          "mean"),
             latest_outage_rate =("outage_rate",        "last"),
             n_weeks            =("residual",           "count"))
    )


def _make_2018_weeks():
    weeks = []
    start = pd.Timestamp("2018-01-01")
    i = 1
    while start.year == 2018:
        end   = start + pd.Timedelta(days=6)
        label = (f"Week {i} of 2018  "
                 f"({start.strftime('%b %d')} – {min(end, pd.Timestamp('2018-12-31')).strftime('%b %d')})")
        weeks.append((start.date(), label))
        start += pd.Timedelta(weeks=1)
        i += 1
    return weeks

WEEKS_2018 = _make_2018_weeks()


# ─── Tract metric computation (Crime Impact page) ─────────────────────────────
def compute_tract_metrics(df: pd.DataFrame, K: int) -> pd.DataFrame:
    req_meta = (
        df.drop_duplicates(subset=["request_id", "tract_geoid"])
        .groupby("tract_geoid", as_index=False)
        .agg(n_requests=("request_id", "nunique"), avg_outage_days=("time_to_fix", "mean"))
    )
    pre_agg = (
        df[(df["days_from_outage_request"] >= -K) & (df["days_from_outage_request"] < 0)]
        .groupby("tract_geoid").size().rename("pre_crimes").reset_index()
    )
    during_agg = (
        df[(df["days_from_outage_request"] >= 0) & (df["days_from_outage_request"] < K)]
        .groupby("tract_geoid").size().rename("during_crimes").reset_index()
    )
    m = req_meta.merge(pre_agg, on="tract_geoid", how="left")
    m = m.merge(during_agg, on="tract_geoid", how="left")
    m[["pre_crimes", "during_crimes"]] = m[["pre_crimes", "during_crimes"]].fillna(0)
    m["pre_rate"]    = m["pre_crimes"]    / (K * m["n_requests"])
    m["during_rate"] = m["during_crimes"] / (K * m["n_requests"])
    m["diff_rate"]   = m["during_rate"]   - m["pre_rate"]
    return m


# ─── Load shared data ─────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    events, tract_polys, _ = load_data()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Dashboard Controls")
    page = st.selectbox("Page", ["Overview", "Crime Impact", "Hotspot Analysis", "Law Enforcement Dashboard"])
    st.markdown("---")

    if page == "Crime Impact":
        radius = st.radio("Buffer radius (m)", [15, 30, 50], index=1, horizontal=True,
                          help="Spatial buffer around each streetlight complaint used to count nearby crimes")
        K = st.radio("Symmetric window K (days)", [5, 10], index=0, horizontal=True,
                     help=("Pre-window = K days before complaint date\n"
                           "During-window = first K days after complaint date\n"
                           "(Max pre-window in data is 10 days)"))
        crime_filter  = st.selectbox("Crime type", ["All"] + CRIME_TYPES)
        st.markdown("---")
        metric_choice = st.selectbox("Metric to display on map", list(METRIC_LABELS.keys()),
                                     format_func=METRIC_LABELS.get)
        min_requests  = st.slider("Min requests per tract", 1, 20, 3,
                                  help="Hides tracts with fewer outage complaints — reduces noise from low-N averages")
        st.markdown("---")
        st.markdown("**Hex overlay (During − Pre weight)**")
        show_hex = st.toggle("Show hex overlay", value=False)
        show_3d  = st.toggle("3D extrusion", value=False, disabled=not show_hex)
        st.caption("Hex color = +1 for during-window crimes, −1 for pre-window crimes. "
                   "Red areas = more crime during outage than before.")
        st.markdown("---")
        st.caption(f"Data: Chicago 311 streetlight outages × Chicago crime reports, 2011–2018\n\n"
                   f"Buffer: {radius} m | K={K} days | Crime: {crime_filter}")

    elif page == "Hotspot Analysis":
        st.markdown("### Hotspot controls")
        n_hotspots = st.slider("Hotspots to highlight", min_value=1, max_value=20, value=5,
                               help="Number of top-coefficient tracts to color coral on the map")
        week_labels  = [label for _, label in WEEKS_2018]
        week_idx     = st.selectbox("Data cutoff (2018)", range(len(WEEKS_2018)),
                                    format_func=lambda i: week_labels[i])
        selected_date, selected_label = WEEKS_2018[week_idx]
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "For each census tract, OLS is run:\n\n"
            "`crime_in_buffer_rate ~ outage_rate`\n\n"
            "Training data: all weeks **2011–2017** plus weeks in **2018 up to the selected cutoff**.\n\n"
            "Only tracts with **p < 0.10** and a **positive coefficient** are eligible hotspots.\n\n"
            "The slider picks the **top N** by coefficient magnitude, colored **coral**."
        )
        st.caption(f"Cutoff: {selected_label}")

    elif page == "Law Enforcement Dashboard":
        st.markdown("### Law enforcement controls")
        le_n_hotspots = st.slider("Hotspots per crime type", min_value=1, max_value=20, value=5,
                                  help="Top N tracts highlighted coral on each crime-type map")
        le_week_labels = [label for _, label in WEEKS_2018]
        le_week_idx    = st.selectbox("Data cutoff (2018)", range(len(WEEKS_2018)),
                                      format_func=lambda i: le_week_labels[i],
                                      key="le_week_idx")
        le_selected_date, le_selected_label = WEEKS_2018[le_week_idx]
        st.markdown("---")
        all_crime_options = get_crime_type_cols()
        all_display_names = [d for _, d in all_crime_options]
        le_selected_display = st.multiselect(
            "Crime types to show",
            options=all_display_names,
            default=all_display_names[:6],
            help="Each selected crime type gets its own map (2 per row)",
        )
        le_selected_crimes = [(col, disp) for col, disp in all_crime_options
                              if disp in le_selected_display]
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "Per tract per crime type, OLS is fitted on **2011–2017**:  \n"
            "`(type_inside / type_total) ~ outage_rate`\n\n"
            "Coral = top N tracts where this crime type ran furthest above its own historical prediction in 2018."
        )
        st.caption(f"Cutoff: {le_selected_label}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW / LANDING
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    # ── Title + writeup ───────────────────────────────────────────────────────
    st.title("Crime and Streetlight Outage in Chicago")
    st.markdown(OVERVIEW_DESCRIPTION)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CRIME IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Crime Impact":

    df_filtered   = events[events["buffer_radius_m"] == radius].copy()
    if crime_filter != "All":
        df_filtered = df_filtered[df_filtered["primary_type"] == crime_filter]

    tract_metrics  = compute_tract_metrics(df_filtered, K)
    tract_filtered = tract_metrics[tract_metrics["n_requests"] >= min_requests].copy()

    st.title("Chicago Streetlight Outages & Crime")
    st.markdown(
        f"Comparing crime rates **{K} days before** vs **{K} days after** each streetlight "
        f"outage complaint | Buffer: **{radius} m** | Crime type: **{crime_filter}** |      \n     "
        
        f" This page shows a dynamic map of Chicago to show the variation in crime rates during an outage, before an outage and change in crime rates per tract. "
        f" We also compute correlations between demigraphic variable aggregeates for a census tract and crime rates "   
    )
    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    total_reqs = int(df_filtered.drop_duplicates("request_id")["request_id"].nunique())
    k1.metric("Total requests", f"{total_reqs:,}",
              help="Unique streetlight outage complaints after radius and crime-type filter")
    k2.metric("Tracts in view", f"{len(tract_filtered):,}",
              help=f"Census tracts with >= {min_requests} requests")
    k3.metric("Mean outage days", f"{tract_filtered['avg_outage_days'].mean():.1f}",
              help="Mean days from complaint to resolution, averaged across tracts")
    mean_diff = tract_filtered["diff_rate"].mean()
    k4.metric("Mean during − pre", f"{mean_diff:+.4f}", delta=f"{mean_diff:+.4f}",
              delta_color="inverse",
              help="Positive = more crime during the outage window than before it")
    st.markdown("---")

    st.subheader(f"Census tract map — {METRIC_SHORT[metric_choice]}")
    tract_map = tract_polys.merge(tract_filtered, on="tract_geoid", how="inner")

    if len(tract_map) == 0:
        st.warning("No tracts match the current filters. Try reducing 'Min requests per tract'.")
    else:
        col_vals = tract_map[metric_choice].values
        if metric_choice == "diff_rate":
            abs_max = max(abs(float(np.nanquantile(col_vals, 0.05))),
                          abs(float(np.nanquantile(col_vals, 0.95))), 1e-9)
            def get_fill_color(v):
                n = max(-1.0, min(1.0, float(v) / abs_max))
                if n >= 0:
                    r, g, b = 220, int(220*(1-n)), int(220*(1-n))
                else:
                    r, g, b = int(220*(1+n)), int(220*(1+n)), 220
                return [r, g, b, 175]
        else:
            vmin   = float(np.nanquantile(col_vals, 0.05))
            vmax   = float(np.nanquantile(col_vals, 0.95))
            vrange = vmax - vmin if vmax > vmin else 1e-9
            def get_fill_color(v):
                n = max(0.0, min(1.0, (float(v) - vmin) / vrange))
                return [int(240-200*n), int(240-200*n), int(100+155*n), 175]

        features = []
        for _, row in tract_map.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": shapely.geometry.mapping(row.geometry),
                "properties": {
                    "tract_geoid":     str(row["tract_geoid"]),
                    "n_requests":      int(row["n_requests"]),
                    "avg_outage_days": round(float(row["avg_outage_days"]), 1),
                    "pre_rate":        round(float(row["pre_rate"]), 5),
                    "during_rate":     round(float(row["during_rate"]), 5),
                    "diff_rate":       round(float(row["diff_rate"]), 5),
                    "fill_color":      get_fill_color(row[metric_choice]),
                },
            })

        layers = [pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": features},
                            pickable=True, stroked=True, filled=True,
                            get_fill_color="properties.fill_color",
                            get_line_color=[80, 80, 80, 120], line_width_min_pixels=0.5)]

        if show_hex:
            during_mask = ((df_filtered["days_from_outage_request"] >= 0) &
                           (df_filtered["days_from_outage_request"] < K))
            pre_mask    = ((df_filtered["days_from_outage_request"] >= -K) &
                           (df_filtered["days_from_outage_request"] < 0))
            df_hex = df_filtered[during_mask | pre_mask][["lng","lat","days_from_outage_request"]].copy()
            df_hex["weight"] = np.where(df_hex["days_from_outage_request"] >= 0, 1, -1)
            layers.append(pdk.Layer("HexagonLayer", df_hex[["lng","lat"]].to_dict("records"),
                                    get_position=["lng","lat"], radius=300, coverage=0.85,
                                    extruded=show_3d,
                                    elevation_scale=60 if show_3d else 0, elevation_range=[0,800],
                                    pickable=True,
                                    color_range=[[1,152,189],[73,227,206],[216,254,181],
                                                 [254,237,177],[254,173,84],[209,55,78]],
                                    auto_highlight=True))

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=41.83, longitude=-87.68, zoom=10,
                                             pitch=40 if (show_hex and show_3d) else 0),
            tooltip={"html": ("<b>Tract:</b> {properties.tract_geoid}<br/>"
                              "<b>Requests (n):</b> {properties.n_requests}<br/>"
                              "<b>Avg outage days:</b> {properties.avg_outage_days}<br/>"
                              "<b>Pre rate:</b> {properties.pre_rate} crimes/day/req<br/>"
                              "<b>During rate:</b> {properties.during_rate} crimes/day/req<br/>"
                              "<b>Diff (During−Pre):</b> {properties.diff_rate}"),
                     "style": {"backgroundColor":"#1a1a2e","color":"white",
                               "fontSize":"13px","padding":"8px"}},
            map_style="mapbox://styles/mapbox/light-v9",
        ), height=520)

        if metric_choice == "diff_rate":
            st.caption("Color: blue = less crime during outage | red = more crime | white = no change")
        else:
            st.caption(f"Color: light → dark blue = low → high {METRIC_SHORT[metric_choice]}. "
                       f"5th–95th percentile range.")

    # ── Census correlation scatter ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Census variable correlation with crime rate change")

    census_df = load_nhgis_census()

    # Build merged table: tract metrics + income (already in events) + nhgis vars
    income_lookup = (
        events[["tract_geoid", "median_income_estimate"]]
        .dropna(subset=["median_income_estimate"])
        .query("median_income_estimate >= 0")
        .drop_duplicates("tract_geoid")
        .set_index("tract_geoid")
    )
    census_merged = tract_filtered.set_index("tract_geoid").join(census_df, how="left")
    census_merged = census_merged.join(income_lookup, how="left")
    census_merged = census_merged.reset_index()

    # Selector row
    cv_col1, cv_col2 = st.columns([3, 1])
    with cv_col1:
        chosen_label = st.selectbox(
            "Census variable — x-axis",
            options=list(CENSUS_VAR_OPTIONS.keys()),
            index=0,
            key="census_var_select",
        )
    with cv_col2:
        pct_as_100 = st.toggle("Show % as 0–100", value=True, key="pct_toggle")

    cv_col, cv_axis_label = CENSUS_VAR_OPTIONS[chosen_label]
    plot_df = census_merged[["tract_geoid", cv_col, "diff_rate", "during_rate",
                              "pre_rate", "n_requests"]].dropna()

    # Scale percentages to 0–100 for readability
    is_pct_var = cv_col not in ("median_income_estimate", "population")
    x_vals = plot_df[cv_col].copy()
    if is_pct_var and pct_as_100:
        x_vals = x_vals * 100
        x_label = cv_axis_label.replace("(%)", "").strip() + " (% of tract)"
    else:
        x_label = cv_axis_label

    plot_df = plot_df.copy()
    plot_df["_x"] = x_vals

    # Pearson r and p-value
    valid = plot_df[["_x", "diff_rate"]].dropna()
    if len(valid) >= 5:
        r_val, p_val = scipy_stats.pearsonr(valid["_x"], valid["diff_rate"])
        p_str  = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        r_str  = f"r = {r_val:+.3f},  {p_str}  (n = {len(valid)} tracts)"
    else:
        r_str = "(insufficient data)"

    sc_left, sc_right = st.columns([2, 1])
    with sc_left:
        fig_sc = px.scatter(
            plot_df,
            x="_x", y="diff_rate",
            size="n_requests",
            color="diff_rate",
            color_continuous_scale="RdBu_r",
            hover_data={"tract_geoid": True, "n_requests": True,
                        "_x": ":.3f", "diff_rate": ":.5f"},
            labels={"_x": x_label,
                    "diff_rate": "During − Pre  [crimes/day/request]",
                    "n_requests": "# Requests"},
            trendline="ols",
            trendline_color_override="#333333",
            opacity=0.75,
            height=420,
        )
        fig_sc.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.5)
        fig_sc.update_layout(
            coloraxis_showscale=False, plot_bgcolor="white",
            xaxis_gridcolor="#eeeeee", yaxis_gridcolor="#eeeeee",
            margin=dict(l=10, r=10, t=30, b=20),
            title=dict(text=f"Pearson {r_str}", font_size=12, x=0),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Each dot = one census tract. Bubble size = # outage requests. "
            "Red = more crime during outage than before; blue = less. "
            "Black line = OLS trendline."
        )

    with sc_right:
        st.markdown("**Variable summary (tract-level)**")
        if len(valid) >= 2:
            summary = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Std dev", "Min", "Max"],
                x_label[:25]: [
                    f"{valid['_x'].mean():.3f}",
                    f"{valid['_x'].median():.3f}",
                    f"{valid['_x'].std():.3f}",
                    f"{valid['_x'].min():.3f}",
                    f"{valid['_x'].max():.3f}",
                ],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("**Correlation with During − Pre**")
        if len(valid) >= 5:
            direction = "positive" if r_val > 0 else "negative"
            sig_note  = "statistically significant" if p_val < 0.05 else "not significant at 5%"
            st.markdown(
                f"- Pearson **r = {r_val:+.3f}**\n"
                f"- {p_str}\n"
                f"- {direction.capitalize()} association\n"
                f"- {sig_note}"
            )

    st.markdown("---")
    with st.expander("View full tract data table"):
        display_cols = ["tract_geoid","n_requests","avg_outage_days",
                        "pre_crimes","during_crimes","pre_rate","during_rate","diff_rate"]
        tbl = (tract_filtered[display_cols].sort_values(metric_choice, ascending=False)
               .reset_index(drop=True))
        tbl = tbl.rename(columns={"tract_geoid":"Tract GEOID","n_requests":"Requests (n)",
                                   "avg_outage_days":"Avg outage days",
                                   "pre_crimes":f"Pre crimes (K={K})",
                                   "during_crimes":f"During crimes (K={K})",
                                   "pre_rate":"Pre rate","during_rate":"During rate",
                                   "diff_rate":"Diff (During−Pre)"})
        st.dataframe(tbl.style.format({"Avg outage days":"{:.1f}","Pre rate":"{:.5f}",
                                       "During rate":"{:.5f}","Diff (During−Pre)":"{:+.5f}"}),
                     use_container_width=True)
        st.caption(f"Rates = crimes / (K={K} days × n_requests). "
                   "Showing only tracts with >= min_requests filter.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HOTSPOT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Hotspot Analysis":

    st.title("Hotspot Analysis")
    st.markdown(
        "This policy tool helps the streetlight maintainance department track areas which are experiencing higher crime rates in streetlight outage areas than historic averages. This helps in finding hotspots which need immediate attention. Understanding that departments may have limited budgets, we allow to filter the number of hospots to be shown.     \n "
        "Each tract's OLS model is trained on its **own 2011–2017** history " 
        "(`crime_buffer_rate ~ outage_rate`).  \n"
        f"The map shows how much each tract's **2018 crime-near-outages** exceeds "
        f"its own historical prediction, through **{selected_label}**.  \n"
        f"Top **{n_hotspots}** tracts with the highest excess are highlighted **coral**."
    )
    st.markdown("---")

    with st.spinner("Fitting per-tract models and scoring 2018 data…"):
        models = fit_per_tract_models()
        risk   = compute_2018_risk(str(selected_date))

    if risk.empty:
        st.warning("No 2018 data available up to this cutoff, or no tracts had enough 2011–2017 history to fit a model.")
        st.stop()

    risk_sorted      = risk.sort_values("mean_residual", ascending=False)
    hotspot_geoids   = set(risk_sorted.head(n_hotspots)["tract_geoid"])
    n_hotspots_found = len(hotspot_geoids)
    risk_lookup      = risk.set_index("tract_geoid").to_dict("index")

    # ── Map: coral hotspots only, grey everything else ────────────────────────
    COLOR_HOTSPOT = [255, 100, 60, 240]
    COLOR_OTHER   = [210, 210, 210, 70]

    features_hs = []
    for _, row in tract_polys.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        gid  = str(row["tract_geoid"])
        info = risk_lookup.get(gid, {})
        resid  = round(float(info.get("mean_residual",      0)), 5)
        actual = round(float(info.get("mean_actual",        0)), 5)
        pred   = round(float(info.get("mean_predicted",     0)), 5)
        outage = round(float(info.get("latest_outage_rate", 0)), 5)
        n_wks  = int(info.get("n_weeks", 0))
        is_spot = gid in hotspot_geoids

        features_hs.append({
            "type": "Feature",
            "geometry": shapely.geometry.mapping(row.geometry),
            "properties": {
                "tract_geoid":    gid,
                "mean_residual":  resid,
                "mean_actual":    actual,
                "mean_predicted": pred,
                "outage_rate":    outage,
                "n_weeks":        n_wks,
                "fill_color":     COLOR_HOTSPOT if is_spot else COLOR_OTHER,
            },
        })

    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("GeoJsonLayer",
                          {"type": "FeatureCollection", "features": features_hs},
                          pickable=True, stroked=True, filled=True,
                          get_fill_color="properties.fill_color",
                          get_line_color=[120, 120, 120, 60],
                          line_width_min_pixels=0.3)],
        initial_view_state=pdk.ViewState(latitude=41.83, longitude=-87.68, zoom=10, pitch=0),
        tooltip={"html": (
                    "<b>Tract:</b> {properties.tract_geoid}<br/>"
                    "<b>Mean excess (actual−pred):</b> {properties.mean_residual}<br/>"
                    "<b>Mean actual:</b> {properties.mean_actual}<br/>"
                    "<b>Mean predicted:</b> {properties.mean_predicted}<br/>"
                    "<b>Latest outage rate:</b> {properties.outage_rate}<br/>"
                    "<b>2018 weeks:</b> {properties.n_weeks}"
                 ),
                 "style": {"backgroundColor":"#1a1a2e","color":"white",
                           "fontSize":"13px","padding":"8px"}},
        map_style="mapbox://styles/mapbox/light-v9",
    ), height=540)
    st.caption(f"Coral = top {n_hotspots} highest-risk tracts (crime running furthest above tract's own historical model). Grey = all other tracts.")

    # ── Hotspot tracts: risk stats + ACS demographics ─────────────────────────
    st.markdown("---")
    st.subheader(f"Top {n_hotspots_found} hotspot tracts — risk scores & community profile")

    acs = pd.read_csv(_DERIVED / "chicago_streetlights_tract_data.csv",
                      usecols=["tract_geoid","population","population_density",
                               "pct_black","pct_white","pct_college","unemployment_rate"])
    acs["tract_geoid"] = acs["tract_geoid"].astype(str)

    top_risk = risk_sorted.head(n_hotspots)[
        ["tract_geoid","mean_residual","mean_actual","mean_predicted","latest_outage_rate","n_weeks"]
    ].copy()
    top_risk.insert(0, "Rank", range(1, len(top_risk) + 1))
    top_risk = top_risk.merge(acs, on="tract_geoid", how="left")

    # ── Risk summary table ─────────────────────────────────────────────────────
    st.markdown("**Risk scores**")
    risk_disp = top_risk[["Rank","tract_geoid","mean_residual","mean_actual",
                           "mean_predicted","latest_outage_rate","n_weeks"]].copy()
    risk_disp = risk_disp.rename(columns={
        "tract_geoid":        "Tract GEOID",
        "mean_residual":      "Mean Excess",
        "mean_actual":        "Mean Actual",
        "mean_predicted":     "Mean Predicted",
        "latest_outage_rate": "Latest Outage Rate",
        "n_weeks":            "2018 Weeks",
    })
    st.dataframe(
        risk_disp.set_index("Rank")
        .style.format({"Mean Excess":"{:+.5f}","Mean Actual":"{:.5f}",
                       "Mean Predicted":"{:.5f}","Latest Outage Rate":"{:.5f}"}),
        use_container_width=True,
    )

    # ── ACS demographics table ─────────────────────────────────────────────────
    st.markdown("**Community demographics (ACS 2015 5-year estimates)**")
    acs_disp = top_risk[["Rank","tract_geoid","population","population_density",
                          "pct_black","pct_white","pct_college","unemployment_rate"]].copy()
    acs_disp = acs_disp.rename(columns={
        "tract_geoid":        "Tract GEOID",
        "population":         "Population",
        "population_density": "Pop. Density (per km²)",
        "pct_black":          "% Black",
        "pct_white":          "% White",
        "pct_college":        "% College+",
        "unemployment_rate":  "Unemployment Rate",
    })
    st.dataframe(
        acs_disp.set_index("Rank")
        .style.format({
            "Population":             "{:,.0f}",
            "Pop. Density (per km²)": "{:,.0f}",
            "% Black":                "{:.1%}",
            "% White":                "{:.1%}",
            "% College+":             "{:.1%}",
            "Unemployment Rate":      "{:.1%}",
        }),
        use_container_width=True,
    )

    # ── Demographics bar charts ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Demographic profile of hotspot tracts")
    demo_plot = top_risk.copy()
    demo_plot["label"] = ("Rank " + demo_plot["Rank"].astype(str)
                          + " · " + demo_plot["tract_geoid"].astype(str))

    c1, c2 = st.columns(2)
    with c1:
        fig_race = px.bar(
            demo_plot.sort_values("mean_residual"),
            x=["pct_black","pct_white"], y="label", orientation="h", barmode="stack",
            color_discrete_map={"pct_black":"#4e79a7","pct_white":"#f28e2b"},
            labels={"value":"Share","label":"Tract (rank · GEOID)",
                    "variable":"Group"},
            title="Race composition", height=max(280, n_hotspots_found * 32),
        )
        fig_race.update_layout(plot_bgcolor="white", margin=dict(l=5,r=5,t=40,b=5),
                               legend=dict(title=""), xaxis_tickformat=".0%")
        fig_race.for_each_trace(lambda t: t.update(
            name={"pct_black":"% Black","pct_white":"% White"}.get(t.name, t.name)))
        st.plotly_chart(fig_race, use_container_width=True)

    with c2:
        fig_ses = px.bar(
            demo_plot.sort_values("mean_residual"),
            x=["pct_college","unemployment_rate"], y="label", orientation="h", barmode="group",
            color_discrete_map={"pct_college":"#59a14f","unemployment_rate":"#e15759"},
            labels={"value":"Rate","label":"Tract (rank · GEOID)",
                    "variable":"Indicator"},
            title="Education & unemployment", height=max(280, n_hotspots_found * 32),
        )
        fig_ses.update_layout(plot_bgcolor="white", margin=dict(l=5,r=5,t=40,b=5),
                              legend=dict(title=""), xaxis_tickformat=".0%")
        fig_ses.for_each_trace(lambda t: t.update(
            name={"pct_college":"% College+","unemployment_rate":"Unemployment"}.get(t.name, t.name)))
        st.plotly_chart(fig_ses, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LAW ENFORCEMENT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Law Enforcement Dashboard":

    st.title("Law Enforcement Dashboard")
    st.markdown(
        f"We create this dashboard specifically for the law enforcement division. Using historic relations between streetlight outage rates and crime in buffer rates, we predict the crime rates in census tracts. This dashboard highlights tracts where the actual crime rates are exceeding the predicted crime rates for each crime type. This allows law enforcement to identify crime hostpot areas and address the issue effectively.     \n"
        f"Per-tract OLS trained on **2011–2017** for each crime type: "
        f"`type_inside / type_total ~ outage_rate`.  \n"
        f"Coral = top **{le_n_hotspots}** tracts where crime ran furthest above the tract's own "
        f"historical prediction through **{le_selected_label}**."
    )
    st.markdown("---")

    if not le_selected_crimes:
        st.info("Select at least one crime type in the sidebar.")
        st.stop()

    acs_lookup = load_acs().to_dict("index")
    COLOR_HOTSPOT_LE = [255, 100, 60, 240]
    COLOR_OTHER_LE   = [210, 210, 210, 70]
    VIEW = pdk.ViewState(latitude=41.83, longitude=-87.68, zoom=9.5, pitch=0)

    def _build_le_features(risk_lookup, hotspot_geoids):
        feats = []
        for _, row in tract_polys.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            gid      = str(row["tract_geoid"])
            info     = risk_lookup.get(gid, {})
            acs_info = acs_lookup.get(gid, {})
            feats.append({
                "type": "Feature",
                "geometry": shapely.geometry.mapping(row.geometry),
                "properties": {
                    "tract_geoid":    gid,
                    "mean_residual":  round(float(info.get("mean_residual",      0)), 5),
                    "mean_actual":    round(float(info.get("mean_actual",        0)), 5),
                    "mean_predicted": round(float(info.get("mean_predicted",     0)), 5),
                    "outage_rate":    round(float(info.get("latest_outage_rate", 0)), 5),
                    "n_weeks":        int(info.get("n_weeks", 0)),
                    "population":     int(acs_info.get("population", 0)),
                    "pct_black":      round(float(acs_info.get("pct_black",         0)) * 100, 1),
                    "pct_white":      round(float(acs_info.get("pct_white",         0)) * 100, 1),
                    "pct_college":    round(float(acs_info.get("pct_college",       0)) * 100, 1),
                    "unemployment":   round(float(acs_info.get("unemployment_rate", 0)) * 100, 1),
                    "fill_color":     COLOR_HOTSPOT_LE if gid in hotspot_geoids else COLOR_OTHER_LE,
                },
            })
        return feats

    # ── Render 2-column map grid ───────────────────────────────────────────────
    for i in range(0, len(le_selected_crimes), 2):
        col_a, col_b = st.columns(2)
        for col_widget, (crime_col, crime_label) in zip(
            [col_a, col_b], le_selected_crimes[i : i + 2]
        ):
            with col_widget:
                st.markdown(f"**{crime_label}**")
                risk_ct = compute_2018_crime_risk(crime_col, str(le_selected_date))
                if risk_ct.empty:
                    st.caption(f"No data for {crime_label} up to this cutoff.")
                    continue

                risk_sorted_ct  = risk_ct.sort_values("mean_residual", ascending=False)
                hotspot_geoids_ct = set(risk_sorted_ct.head(le_n_hotspots)["tract_geoid"])
                risk_lookup_ct  = risk_ct.set_index("tract_geoid").to_dict("index")

                feats = _build_le_features(risk_lookup_ct, hotspot_geoids_ct)

                st.pydeck_chart(pdk.Deck(
                    layers=[pdk.Layer("GeoJsonLayer",
                                      {"type": "FeatureCollection", "features": feats},
                                      pickable=True, stroked=True, filled=True,
                                      get_fill_color="properties.fill_color",
                                      get_line_color=[120, 120, 120, 50],
                                      line_width_min_pixels=0.3)],
                    initial_view_state=VIEW,
                    tooltip={"html": (
                                f"<b>{crime_label}</b><br/>"
                                "<b>Tract:</b> {properties.tract_geoid}<br/>"
                                "<b>Mean excess:</b> {properties.mean_residual}<br/>"
                                "<b>Actual rate:</b> {properties.mean_actual}<br/>"
                                "<b>Predicted rate:</b> {properties.mean_predicted}<br/>"
                                "<b>Outage rate:</b> {properties.outage_rate}<br/>"
                                "<b>2018 weeks:</b> {properties.n_weeks}<br/>"
                                "<hr style='margin:3px 0;border-color:#555'>"
                                "<b>Population:</b> {properties.population}<br/>"
                                "<b>% Black:</b> {properties.pct_black}%<br/>"
                                "<b>% White:</b> {properties.pct_white}%<br/>"
                                "<b>% College+:</b> {properties.pct_college}%<br/>"
                                "<b>Unemployment:</b> {properties.unemployment}%"
                             ),
                             "style": {"backgroundColor":"#1a1a2e","color":"white",
                                       "fontSize":"12px","padding":"7px","maxWidth":"240px"}},
                    map_style="mapbox://styles/mapbox/light-v9",
                ), height=340)
                top1 = risk_sorted_ct.iloc[0]
                st.caption(
                    f"Top hotspot: tract {top1['tract_geoid']} "
                    f"(excess {top1['mean_residual']:+.4f})"
                )
