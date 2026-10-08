# Brooklyn Bus Equity Explorer

An interactive Streamlit work sample for the Brooklyn Bus Network Redesign. It combines daily bus stop ridership, GTFS schedule data, Brooklyn census geography, ACS equity measures, and a local light basemap.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app1.py
```

The app uses the daily ridership and schedule Parquet files in the repository. The basemap tiles are local PNG tiles under `static/tiles`; no external map-tile API key is required.

## Deploy with Streamlit Community Cloud

Create a Streamlit app from this repository, use the `main` branch, and set the main file to `app1.py`. Use Python 3.11 or 3.12.

## Data notes

The repository contains processed application data only. Raw PBF, GeoPackage, ZIP, and source GTFS archives are intentionally excluded.
