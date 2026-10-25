# Urban Traffic Intelligence

Interactive Big Data Analytics mini-project for accident hotspot detection and temporal risk analysis.

## Features
- Dark, animated Streamlit interface with hover transitions and gradient hero header
- CSV upload or local dataset loading
- PySpark local-mode CSV ingestion (optional) and Pandas preview mode
- Filters by state, severity, hour, and city
- Hourly, weekday, severity, and city analytics
- Interactive Folium heatmap
- Grid-based hotspot ranking and CSV export
- Dataset quality and record preview

## 1. Recommended setup (Mac)

Open Terminal and go to the extracted project folder:

```bash
cd urban_traffic_intelligence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PySpark requires a compatible Java installation. Check:

```bash
java -version
```

If Java is not installed, install a supported JDK (for example, Temurin/OpenJDK 17) and reopen Terminal. If Spark setup causes issues, select **Pandas** in the sidebar to test the dashboard first.

## 2. Add your downloaded dataset

For a large CSV, put it in the `data` folder and rename it to:

`US_Accidents_March23.csv`

Expected common columns include:
- `Start_Time`
- `Start_Lat`
- `Start_Lng`
- `Severity`
- `City`
- `State`
- `Weather_Condition`

Alternatively, launch the app and upload a CSV using the sidebar. Uploading a multi-gigabyte CSV through a browser is not recommended; use the `data` folder for large files.

## 3. Run

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in your browser.

## 4. First run settings

1. Select `Pandas` initially if Spark is not configured.
2. Choose `100,000 (quick preview)` and check that the charts and map render.
3. Switch to `PySpark` once Java and PySpark are working.
4. Increase the row count to 500,000 or 1,000,000.
5. Use `All available records` only when your machine has enough memory and you have tested the smaller samples.

## 5. Notes about data and interpretation

- This starter project analyzes reported accidents, not direct traffic speed or congestion measurements.
- Hotspot map density shows the concentration of recorded accident points. It is not a normalized collision probability because road usage/exposure data is not included.
- The coordinate grid rounds latitude/longitude to two decimal places. Treat it as a broad hotspot approximation, not an exact road-segment analysis.
- The `Severity` score is the dataset's recorded category (usually 1–4); it is not a prediction.
- If your dataset uses different column names, adjust `REQUIRED` and the field mapping near the top of `app.py`.

## 6. Suggested viva points

- Why PySpark? It provides distributed DataFrame processing and can scale beyond a small in-memory workflow.
- What is a hotspot? A geographic area with a high concentration of records under a chosen spatial aggregation method.
- Why clean data? Invalid timestamps, coordinates, and missing fields can distort results.
- What is the limitation? Accident counts are not the same as accident risk; traffic exposure, road length, and vehicle counts would improve the analysis.
