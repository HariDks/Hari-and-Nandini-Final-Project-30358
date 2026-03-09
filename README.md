# Chicago Streetlight Outages & Crime

This project links **Chicago 311 streetlight outage complaints (2011–2018)** with **Chicago Police Department crime reports** to test whether crime rates rise during streetlight outages, and whether that effect is concentrated in lower-income census tracts.

---

## Project Structure

```
project/
├── code/                          # Data processing and static plots
│   ├── pull_data.py               # Download raw data from Chicago Open Data- crime and streetlight requests
│   ├── preprocessing.py           # All data cleaning and feature engineering
│   ├── preprocessing.qmd          # Quarto doc version of preprocessing 
│   └── static_plots.py            # Generate static plots
│
├── streamlit_app/
│   └── dashboard.py               # Interactive Streamlit dashboard
│
├── data/
│   ├── raw_data/                  # Downloaded source files (git-ignored)
│   └── derived_data/              # Processed outputs (large files git-ignored)
│
├── _output/                       # Static plot PNGs
├── requirements.txt
└── README.md
```

---

## Running Order

### Step 1 — Download raw data
**Note this step is very processing intensive, if you want to skip this step, kindly download the data folder from the below link. Replace the data folder in the cloned repo with the downloaded data folder. Ensure the folder structure resembles the above explained tree** 

**Data folder link- https://drive.google.com/drive/folders/1--GsDpZUq5XKxQVL13xDdD7X1wy1PyPS?usp=drive_link**

```bash
python code/pull_data.py
```

Downloads two datasets from the Chicago Open Data portal via Socrata API:
- `data/raw_data/streetlight_chicago.csv` — all 311 streetlight outage service requests
- `data/raw_data/crimes_2011_2018.csv` — CPD crime incidents 2011–2018

You will also need to manually place the following in `data/raw_data/`:- download from the drive folder-> data-> raw_data
- `illinois_tract_income/illinois_tract_income.shp` — Illinois census tract shapefile with ACS income data (from NHGIS)
- `nhgis0001_csv/nhgis0001_ds215_20155_tract.csv` — ACS 2011–2015 5-year estimates (from NHGIS IPUMS)
- `transportation_*.geojson` — Chicago transportation network file (from Chicago Open Data)

### Step 2 — Preprocess data

```bash
python code/preprocessing.py
```

Reads from `data/raw_data/`, writes all outputs to `data/derived_data/`. Key steps:
1. Clean and parse streetlight outage records; compute outage duration
2. Build 30 m spatial buffers around each outage complaint
3. Spatially join crimes to buffers; compute days-from-outage-request for each crime
4. Assign census tract GEOIDs to all events via spatial join with Illinois tract shapefile
5. Aggregate to tract-week panel with outage rates and crime counts
6. Compute pre/during crime rate metrics using symmetric windows

See `code/preprocessing.qmd` for a fully annotated walkthrough of each step.

### Step 3 — Generate static plots

```bash
python code/static_plots.py
```

Reads from `data/derived_data/`, saves two PNG figures to `_output/`:
- `plot_H_crime_type_relative_change.png` — lollipop chart of % crime change by type during outages
- `plot_E_bivariate_choropleth.png` — tract-level choropleth: income vs crime rate change

Requires `vl-convert-python` for PNG export from Altair.

### Step 4 — Run the dashboard

```bash
streamlit run streamlit_app/dashboard.py
```

Interactive dashboard with three pages:
- **Overview** — day-level map of active outages and crimes
- **Crime Impact** — pre/during crime rate comparison by tract; scatter plots vs census variables
- **Hotspot Analysis** — per-tract OLS model; tracts where 2018 crime exceeds historical prediction
- **Law Enforcement Dashboard** — same hotspot analysis broken down by crime type

**Please note:** Streamlit apps need to be “woken up”
if they have not been run in the last 24 hours. This should be included because a
potential employer who is reviewing your project should know that the app needing
to be woken up is normal behavior rather than evidence of a bug
---

## Key Outputs

| File | Size | Description |
|------|------|-------------|
| `data/derived_data/processed_ev30_tracts.csv` | 421 KB | Event-level crime-outage pairs (30 m buffer), tract-assigned |
| `data/derived_data/processed_tc_all.csv` | 45 KB | Tract-level pre/during crime rate summary (all crimes) |
| `data/derived_data/processed_outage_days_base.csv` | 16 KB | Per-tract total outage days |
| `data/derived_data/processed_top_types.csv` | 1 KB | Top crime types by frequency |
| `data/derived_data/tract_week_crime_outage_panel.csv` | 47 MB | Weekly panel: outage rates × crime counts per tract |
| `data/derived_data/chicago_streetlights_tract_data.csv` | 152 KB | Tract-level streetlight counts and ACS demographics |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `geopandas`, `pandas`, `altair`, `vl-convert-python`, `streamlit`, `pydeck`, `plotly`, `sodapy`, `scipy`

---

## Data Sources

- [Chicago 311 Streetlight Outage Requests](https://data.cityofchicago.org/d/zuxi-7xem) — City of Chicago Open Data
- [Chicago Crimes 2001–Present](https://data.cityofchicago.org/d/ijzp-q8t2) — Chicago Police Department
- [NHGIS ACS 2011–2015 5-year Tract Estimates](https://www.nhgis.org/) — Minnesota Population Center
