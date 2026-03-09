from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import altair as alt

project_dir  = Path(__file__).parent.parent   # code/ → project root
raw_data_dir = project_dir / "data" / "raw_data"
derived_dir  = project_dir / "data" / "derived_data"
output_dir  = project_dir / "_output"
output_dir.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
ev30_tracts      = pd.read_csv(derived_dir / "processed_ev30_tracts.csv")
tc_all           = pd.read_csv(derived_dir / "processed_tc_all.csv")
outage_days_base = pd.read_csv(derived_dir / "processed_outage_days_base.csv")
top_types        = pd.read_csv(derived_dir / "processed_top_types.csv")["crime_type"].tolist()
tracts_gdf       = gpd.read_file(raw_data_dir / "illinois_tract_income/illinois_tract_income.shp")

# ── Helper ─────────────────────────────────────────────────────────────────────
def before_during_tract(ev30_tracts_sub, outage_days_df):
    sym = ev30_tracts_sub[
        ev30_tracts_sub["days_from_outage_request"].abs()
        <= ev30_tracts_sub["time_to_fix"]
    ].copy()

    pre = (
        sym[sym["crime_streetlight_outage"] == 0]
        .groupby("GEOID").size().reset_index(name="pre_n")
    )
    dur = (
        sym[sym["crime_streetlight_outage"] == 1]
        .groupby("GEOID").size().reset_index(name="dur_n")
    )

    result = (
        outage_days_df
        .merge(pre, on="GEOID", how="left")
        .merge(dur, on="GEOID", how="left")
    )
    result["pre_n"]  = result["pre_n"].fillna(0)
    result["dur_n"]  = result["dur_n"].fillna(0)
    result["pre_rate"]    = result["pre_n"] / result["total_outage_days"]
    result["dur_rate"]    = result["dur_n"] / result["total_outage_days"]
    result["rate_change"] = result["dur_rate"] - result["pre_rate"]
    result["crime_increased"] = result["rate_change"] > 0
    return result[result["median_income"] > 0].copy()


# ── Plot 1: Crime-type relative change — lollipop ─────────────────────────────
type_stats = []
for ct in top_types:
    sub   = ev30_tracts[ev30_tracts["primary_type"] == ct]
    tc_ct = before_during_tract(sub, outage_days_base)
    if len(tc_ct) < 5:
        continue
    pre_total = tc_ct["pre_n"].sum()
    dur_total = tc_ct["dur_n"].sum()
    if pre_total <= 0:
        continue
    type_stats.append({
        "crime_type": ct,
        "rel_change": (dur_total - pre_total) / pre_total * 100,
        "is_overall": False,
    })

type_df = pd.DataFrame(type_stats).sort_values("rel_change").reset_index(drop=True)

pre_all = tc_all["pre_n"].sum()
dur_all = tc_all["dur_n"].sum()
if pre_all > 0:
    type_df = pd.concat([
        type_df,
        pd.DataFrame([{
            "crime_type": "ALL CRIMES",
            "rel_change": (dur_all - pre_all) / pre_all * 100,
            "is_overall": True,
        }])
    ], ignore_index=True)

type_df["zero"]       = 0
type_df["color"]      = type_df.apply(
    lambda r: "#333333" if r["is_overall"] else ("#d7191c" if r["rel_change"] > 0 else "#2c7bb6"),
    axis=1,
)
type_df["label"] = type_df["rel_change"].map(
    lambda v: f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"
)

# sort order for y axis (ascending rel_change)
sort_order = type_df.sort_values("rel_change")["crime_type"].tolist()

base = alt.Chart(type_df).encode(
    y=alt.Y("crime_type:N", sort=sort_order, title=None, axis=alt.Axis(labelFontSize=12)),
    color=alt.Color("color:N", scale=None, legend=None),
)

rules = base.mark_rule(strokeWidth=2.2).encode(
    x=alt.X("zero:Q",   title="% change vs. pre-outage baseline",
            axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
    x2=alt.X2("rel_change:Q"),
)
points = base.mark_point(size=90, filled=True).encode(
    x=alt.X("rel_change:Q"),
)
zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
    color="black", strokeDash=[4, 3], strokeWidth=1
).encode(x="x:Q")

labels = base.mark_text(align="left", dx=6, dy=1, fontSize=10).encode(
    x=alt.X("rel_change:Q"),
    text=alt.Text("label:N"),
    color=alt.value("black"),
)

chart1 = (zero_rule + rules + points + labels).properties(
    width=460,
    height=340,
    title=alt.TitleParams(
        "Outages shift the crime mix",
        fontSize=14,
    ),
).configure_view(stroke=None).configure_axis(grid=False)

chart1.save(str(output_dir / "plot_H_crime_type_relative_change.png"))
print("Saved: plot_H_crime_type_relative_change.png")


# ── Plot 2: Bivariate choropleth — income vs crime change ─────────────────────
inc_med = tc_all["median_income"].median()
tc_all["income_group"] = np.where(tc_all["median_income"] < inc_med, "low_inc", "high_inc")
tc_all["crime_group"]  = np.where(tc_all["rate_change"] >= 0, "crime_up", "crime_down")

LABEL_MAP = {
    "low_inc_crime_up":   "Low income · Crime ↑",
    "low_inc_crime_down": "Low income · Crime ↓",
    "high_inc_crime_up":  "High income · Crime ↑",
    "high_inc_crime_down":"High income · Crime ↓",
}
COLOR_MAP = {
    "Low income · Crime ↑":  "#d7191c",
    "Low income · Crime ↓":  "#fdae61",
    "High income · Crime ↑": "#7b3294",
    "High income · Crime ↓": "#2c7bb6",
}

tc_all["GEOID"] = tc_all["GEOID"].astype(str)
tracts_gdf["GEOID"] = tracts_gdf["GEOID"].astype(str)

tracts_map = (
    tracts_gdf[["GEOID", "NAMELSAD", "geometry"]]
    .merge(
        tc_all[["GEOID", "income_group", "crime_group", "median_income", "rate_change"]],
        on="GEOID", how="inner",
    )
    .to_crs(epsg=4326)
)

tracts_map["quad_label"]    = (tracts_map["income_group"] + "_" + tracts_map["crime_group"]).map(LABEL_MAP)
tracts_map["median_income"] = tracts_map["median_income"].round(0).astype(int)
tracts_map["rate_change"]   = tracts_map["rate_change"].round(4)

chart2 = (
    alt.Chart(tracts_map)
    .mark_geoshape(stroke="white", strokeWidth=0.3)
    .encode(
        color=alt.Color(
            "quad_label:N",
            scale=alt.Scale(
                domain=list(COLOR_MAP.keys()),
                range=list(COLOR_MAP.values()),
            ),
            legend=alt.Legend(
                title="Income · Crime change",
                titleFontSize=11,
                labelFontSize=10,
                orient="bottom-left",
            ),
        ),
        tooltip=[
            alt.Tooltip("NAMELSAD:N",       title="Tract"),
            alt.Tooltip("quad_label:N",     title="Group"),
            alt.Tooltip("median_income:Q",  title="Median income ($)", format=","),
            alt.Tooltip("rate_change:Q",    title="Crime rate change", format="+.4f"),
        ],
    )
    .project("mercator")
    .properties(
        width=520,
        height=680,
        title=alt.TitleParams(
            "Income vs change in nighttime crime during streetlight outages",
            fontSize=13,
        ),
    )
    .configure_view(stroke=None)
)

chart2.save(str(output_dir / "plot_E_bivariate_choropleth.png"))
print("Saved: plot_E_bivariate_choropleth.png")
