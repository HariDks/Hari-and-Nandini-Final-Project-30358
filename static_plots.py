from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import contextily as ctx

project_dir = Path(__file__).parent
data_dir = project_dir / "data"

ev30_tracts_csv = data_dir / "processed_ev30_tracts.csv"
tc_all_csv = data_dir / "processed_tc_all.csv"
excess_df_csv = data_dir / "processed_excess_df.csv"
top_types_csv = data_dir / "processed_top_types.csv"
outage_days_base_csv = data_dir / "processed_outage_days_base.csv"

tracts_shp = data_dir / "illinois_tract_income/illinois_tract_income.shp"

ev30_tracts = pd.read_csv(ev30_tracts_csv)
tc_all = pd.read_csv(tc_all_csv)
excess_df = pd.read_csv(excess_df_csv)
outage_days_base = pd.read_csv(outage_days_base_csv)

top_types = pd.read_csv(top_types_csv)["crime_type"].tolist()

tracts_gdf = gpd.read_file(tracts_shp)

fig_dir = data_dir


def before_during_tract(ev30_tracts_sub, outage_days_df):

    sym = ev30_tracts_sub[
        ev30_tracts_sub["days_from_outage_request"].abs()
        <= ev30_tracts_sub["time_to_fix"]
    ].copy()

    pre = (
        sym[sym["crime_streetlight_outage"] == 0]
        .groupby("GEOID")
        .size()
        .reset_index(name="pre_n")
    )

    dur = (
        sym[sym["crime_streetlight_outage"] == 1]
        .groupby("GEOID")
        .size()
        .reset_index(name="dur_n")
    )

    result = (
        outage_days_df
        .merge(pre, on="GEOID", how="left")
        .merge(dur, on="GEOID", how="left")
    )

    result["pre_n"] = result["pre_n"].fillna(0)
    result["dur_n"] = result["dur_n"].fillna(0)

    result["pre_rate"] = result["pre_n"] / result["total_outage_days"]
    result["dur_rate"] = result["dur_n"] / result["total_outage_days"]

    result["rate_change"] = result["dur_rate"] - result["pre_rate"]
    result["crime_increased"] = result["rate_change"] > 0

    return result[result["median_income"] > 0].copy()

type_stats = []

for ct in top_types:

    sub = ev30_tracts[ev30_tracts["primary_type"] == ct]

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
        "is_overall": False
    })

type_df = pd.DataFrame(type_stats).sort_values("rel_change").reset_index(drop=True)

pre_all = tc_all["pre_n"].sum()
dur_all = tc_all["dur_n"].sum()

if pre_all > 0:
    type_df = pd.concat([
        type_df,
        pd.DataFrame([{
            "crime_type":"ALL CRIMES",
            "rel_change":(dur_all-pre_all)/pre_all*100,
            "is_overall":True
        }])
    ],ignore_index=True)

x_range = type_df["rel_change"].abs().max()

fig, ax = plt.subplots(figsize=(9,7))

for i,row in type_df.iterrows():

    color = "#333333" if row["is_overall"] else (
        "#d7191c" if row["rel_change"] > 0 else "#2c7bb6"
    )

    ax.plot([0,row["rel_change"]],[i,i],color=color,linewidth=2.2)
    ax.scatter(row["rel_change"],i,color=color,s=90)

    sign = "+" if row["rel_change"]>0 else ""

    ax.text(
        row["rel_change"] + x_range*0.05*np.sign(row["rel_change"]+1e-6),
        i,
        f"{sign}{row['rel_change']:.1f}%",
        va="center"
    )

ax.axvline(0,color="black",linestyle="--")

ax.set_yticks(type_df.index)
ax.set_yticklabels(type_df["crime_type"])

ax.set_title("Outages shift the crime mix")

plt.tight_layout()
plt.savefig(data_dir/"plot_H_crime_type_relative_change.png",dpi=150)
plt.show()

bins_g = [7,14,30,60]
labels_g = ["1–2 weeks","2–4 weeks","1–2 months"]

po_g = excess_df[excess_df["time_to_fix"] < 60].copy()

po_g["ttf_bin"] = pd.cut(
    po_g["time_to_fix"],
    bins=bins_g,
    labels=labels_g,
    right=False
)

binned = (
    po_g.groupby("ttf_bin", observed=True)["excess"]
    .agg(mean="mean", se=sem)
    .reindex(labels_g)   # keeps original order
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8,5))

colors_g = ["#92c5de","#4393c3","#2166ac"]

for i,row in binned.iterrows():

    ax.bar(
        i,
        row["mean"],
        color=colors_g[i],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85
    )

    ax.errorbar(
        i,
        row["mean"],
        yerr=1.96*row["se"],
        fmt="none",
        color="#333",
        linewidth=1.5,
        capsize=5
    )

ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.5,
    label="Pre-outage baseline"
)

ax.set_xticks(range(len(labels_g)))
ax.set_xticklabels(labels_g,fontsize=13)

ax.set_xlabel("How long the streetlight stayed broken",fontsize=12)

ax.set_ylabel(
    "Extra crimes per outage per day\nabove pre-outage baseline",
    fontsize=12
)

ax.set_title(
    "The longer a streetlight stays broken, the more crime rises above baseline\n"
    "(error bars = 95% CI; symmetric 7-day before/after window)",
    fontsize=11
)

ax.legend(fontsize=10)
ax.grid(axis="y",linestyle="--",alpha=0.35)

plt.tight_layout()

plt.savefig(fig_dir/"policy_G_binned_dose_response.png",dpi=150)

plt.show()

inc_med = tc_all["median_income"].median()

tc_all["income_group"] = np.where(tc_all["median_income"] < inc_med,"low_inc","high_inc")
tc_all["crime_group"] = np.where(tc_all["rate_change"] >= 0,"crime_up","crime_down")

QUAD_COLORS = {
("low_inc","crime_up"): "#d7191c",
("low_inc","crime_down"): "#fdae61",
("high_inc","crime_down"): "#2c7bb6",
("high_inc","crime_up"): "#7b3294"
}

tracts_gdf["GEOID"] = tracts_gdf["GEOID"].astype(str)
tc_all["GEOID"] = tc_all["GEOID"].astype(str)
tracts = tracts_gdf.merge(tc_all,on="GEOID",how="inner")

tracts["quad_color"] = tracts.apply(
    lambda r: QUAD_COLORS[(r["income_group"],r["crime_group"])],
    axis=1
)

tracts = tracts.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(11,14))

tracts.plot(ax=ax,color=tracts["quad_color"],edgecolor="white",linewidth=0.1)

ctx.add_basemap(ax,source=ctx.providers.CartoDB.Positron)

ax.set_axis_off()

ax.set_title("Income vs change in nighttime crime during outages")

plt.tight_layout()
plt.savefig(data_dir/"plot_E_bivariate_choropleth.png",dpi=150)
plt.show()

ev30_tracts["crime_streetlight_outage"].value_counts()
ev30_tracts.groupby("crime_streetlight_outage")["days_from_outage_request"].describe()
