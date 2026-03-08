from pathlib import Path
import time
import pandas as pd
from sodapy import Socrata

project_dir = Path(__file__).parent
data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True)

def download_streetlights():
    output_csv = data_dir / "streetlight_chicago.csv"

    client = Socrata(
        "data.cityofchicago.org",
        "Lg7zqHMbACUvgMjA2CcB3iPyH"
    )

    all_results = []
    limit = 50000
    offset = 0

    while True:
        print(f"Requesting rows {offset} to {offset + limit}...")

        batch = client.get(
            "zuxi-7xem",
            limit=limit,
            offset=offset
        )

        if len(batch) == 0:
            print("Done downloading streetlights.")
            break

        all_results.extend(batch)
        offset += limit
        print(f"Total rows so far: {len(all_results)}")

    df = pd.DataFrame.from_records(all_results)
    df.to_csv(output_csv, index=False)

    print("Saved to", output_csv)
    print("Final shape:", df.shape)

    return df


def download_crimes():
    output_csv = data_dir / "crimes_2011_2018.csv"

    client = Socrata(
        "data.cityofchicago.org",
        "Lg7zqHMbACUvgMjA2CcB3iPyH",
        timeout=120
    )

    dataset_id = "ijzp-q8t2"
    select_cols = "id,date,year,primary_type,latitude,longitude,community_area,beat,district,ward"
    where_clause = "date between '2011-01-01T00:00:00.000' and '2018-12-31T23:59:59.999'"

    limit = 10000
    offset = 0
    rows = []

    while True:
        batch = client.get(
            dataset_id,
            select=select_cols,
            where=where_clause,
            limit=limit,
            offset=offset
        )

        if not batch:
            print("Done downloading crimes.")
            break

        rows.extend(batch)
        offset += limit
        print("rows:", len(rows))
        time.sleep(0.2)

    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_csv, index=False)

    print("Saved to", output_csv)
    print("Final shape:", df.shape)

    return df


streetlights_df = download_streetlights()
crimes_df = download_crimes()
