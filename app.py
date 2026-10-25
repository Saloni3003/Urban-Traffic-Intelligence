import os
import time
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional geographic map dependencies
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# Optional Spark engine
try:
    from pyspark.sql import SparkSession, functions as F
    SPARK_AVAILABLE = True
except Exception:
    SPARK_AVAILABLE = False

st.set_page_config(
    page_title="Urban Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Aesthetic UI ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
  --ink: #edf4ff; --muted: #93a4bd; --panel: rgba(15, 25, 45, .78);
  --line: rgba(148, 163, 184, .18); --cyan: #62e6dc; --blue: #7aa7ff;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 5% 0%, rgba(42, 104, 150, .24), transparent 35%),
    radial-gradient(ellipse at 95% 10%, rgba(112, 70, 178, .18), transparent 30%),
    linear-gradient(135deg, #07111f 0%, #0b1424 50%, #10172a 100%);
  color: var(--ink);
}
[data-testid="stHeader"] { background: rgba(7,17,31,.65); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c1728 0%, #101a2e 100%);
  border-right: 1px solid var(--line);
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
.hero {
  position: relative; overflow: hidden; border: 1px solid rgba(98,230,220,.22);
  background: linear-gradient(115deg, rgba(22,48,72,.92), rgba(24,27,53,.86));
  border-radius: 26px; padding: 30px 32px; margin-bottom: 22px;
  box-shadow: 0 20px 70px rgba(0,0,0,.20);
  animation: riseIn .7s ease-out both;
}
.hero:after {
  content:""; position:absolute; width:300px; height:300px; right:-75px; top:-170px;
  border-radius:50%; border:1px solid rgba(98,230,220,.22);
  box-shadow: 0 0 0 28px rgba(98,230,220,.035), 0 0 0 58px rgba(98,230,220,.025);
  animation: orbit 18s linear infinite;
}
.eyebrow { color: #62e6dc; font-size: .75rem; letter-spacing: .18em; font-weight: 700; text-transform: uppercase; }
.hero h1 { font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.05; margin: 10px 0; color: #f4f8ff; }
.hero p { color: #a9bad1; font-size: 1rem; max-width: 760px; margin-bottom: 0; }
.pill { display:inline-block; border:1px solid rgba(98,230,220,.28); color:#9cf4eb; border-radius:99px; padding:5px 10px; font-size:.75rem; margin: 16px 6px 0 0; }
.section-label { color:#8fa5c2; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; margin: 18px 0 10px; }
.metric-card {
  background: linear-gradient(145deg, rgba(22,36,58,.94), rgba(14,25,43,.94));
  border:1px solid var(--line); border-radius:18px; padding:18px 20px;
  min-height:118px; transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
  animation: riseIn .55s ease both;
}
.metric-card:hover { transform: translateY(-5px); border-color:rgba(98,230,220,.45); box-shadow:0 14px 34px rgba(0,0,0,.22); }
.metric-label { color:#94a8c2; font-size:.82rem; }
.metric-value { color:#f4f8ff; font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; margin-top:7px; }
.metric-note { color:#62e6dc; font-size:.75rem; margin-top:3px; }
.panel {
  background:rgba(15,25,45,.72); border:1px solid var(--line); border-radius:20px;
  padding:18px 20px; margin-bottom:16px; transition:border-color .25s ease;
  animation: riseIn .6s ease both;
}
.panel:hover { border-color:rgba(122,167,255,.35); }
div[data-testid="stPlotlyChart"] { border-radius:16px; overflow:hidden; }
.stButton > button, .stDownloadButton > button {
  border:1px solid rgba(98,230,220,.35); border-radius:12px; color:#eafcff;
  background:linear-gradient(100deg, rgba(38,112,126,.85), rgba(53,73,130,.85));
  transition:all .2s ease; font-weight:600;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px); border-color:#62e6dc; box-shadow:0 8px 22px rgba(41,160,169,.18); }
div[data-testid="stSelectbox"], div[data-testid="stMultiSelect"], div[data-testid="stDateInput"] { border-radius:12px; }
.small-muted { color:#8fa3bd; font-size:.84rem; }
@keyframes riseIn { from {opacity:0; transform:translateY(12px)} to {opacity:1; transform:translateY(0)} }
@keyframes orbit { from {transform:rotate(0)} to {transform:rotate(360deg)} }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CANDIDATES = [
    DATA_DIR / "US_Accidents_March23.csv",
    DATA_DIR / "US_Accidents.csv",
    Path(__file__).parent / "US_Accidents_March23.csv",
    Path(__file__).parent / "US_Accidents.csv",
]

# Hadoop MapReduce output generated from the full dataset.
HADOOP_SUMMARY_PATH = Path(__file__).parent / "output" / "traffic_summary.txt"


@st.cache_data(show_spinner=False)
def load_hadoop_summary(path, mtime):
    """Read key/value aggregates produced by UrbanTrafficAnalysis.

    Hadoop normally separates key and count with a tab. split() also accepts
    spaces and other whitespace, making the parser robust to copied files.
    """
    records = []
    with open(path, "r", encoding="utf-8-sig") as summary_file:
        for raw_line in summary_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                key, value = line.rsplit(None, 1)
                category, label = key.split("|", 1)
                records.append({
                    "Category": category.strip(),
                    "Label": label.strip(),
                    "Records": int(value.strip()),
                })
            except (ValueError, TypeError):
                # Ignore malformed lines rather than breaking the dashboard.
                continue
    return pd.DataFrame(records)


def render_hadoop_dashboard():
    """Render dashboard panels directly from the completed Hadoop job output."""
    if not HADOOP_SUMMARY_PATH.exists():
        st.error(f"Hadoop summary file not found: {HADOOP_SUMMARY_PATH}")
        st.info(
            "Copy the MapReduce result into the project folder with:\\n\\n"
            "`hdfs dfs -get -f /urban-traffic/output/part-r-00000 "
            "output/traffic_summary.txt`"
        )
        st.stop()

    summary = load_hadoop_summary(
        str(HADOOP_SUMMARY_PATH), HADOOP_SUMMARY_PATH.stat().st_mtime
    )
    if summary.empty:
        st.error("The Hadoop summary file exists, but no valid key/value rows were read.")
        st.stop()

    def category_frame(category):
        return summary.loc[summary["Category"] == category, ["Label", "Records"]].copy()

    states = category_frame("STATE")
    severity = category_frame("SEVERITY")
    weather = category_frame("WEATHER")
    day_night = category_frame("DAY_NIGHT")
    hourly = category_frame("HOUR")

    total_records = int(severity["Records"].sum())
    high_severity = int(
        severity.loc[severity["Label"].isin(["3", "4"]), "Records"].sum()
    )
    state_count = len(states)
    day_count = int(day_night.loc[day_night["Label"] == "DAY", "Records"].sum())
    night_count = int(day_night.loc[day_night["Label"] == "NIGHT", "Records"].sum())
    peak_row = hourly.loc[hourly["Records"].idxmax()] if not hourly.empty else None
    top_state = states.sort_values("Records", ascending=False).iloc[0] if not states.empty else None

    st.markdown('<div class="section-label">Hadoop MapReduce · full dataset</div>', unsafe_allow_html=True)
    st.caption(
        "These panels read the aggregated output of your completed Hadoop MapReduce job. "
        "They do not rescan the 3 GB CSV file."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric("VALID ACCIDENT RECORDS", fmt_num(total_records), "aggregated by MapReduce")
    with m2:
        metric("SEVERITY 3–4", fmt_num(high_severity), "higher-severity records")
    with m3:
        metric("STATES REPRESENTED", fmt_num(state_count), "states / districts in output")
    with m4:
        peak_label = f"{int(peak_row['Label']):02d}:00" if peak_row is not None else "N/A"
        metric("PEAK ACCIDENT HOUR", peak_label, fmt_num(peak_row["Records"]) + " records" if peak_row is not None else "")

    if top_state is not None:
        st.caption(
            f"Highest-count state: {top_state['Label']} ({fmt_num(top_state['Records'])} records) · "
            f"Day: {fmt_num(day_count)} · Night: {fmt_num(night_count)}"
        )

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Accident activity · by hour")
        if not hourly.empty:
            hourly["Hour"] = pd.to_numeric(hourly["Label"], errors="coerce")
            hourly = hourly.dropna(subset=["Hour"]).sort_values("Hour")
            hourly["Hour"] = hourly["Hour"].astype(int)
            hourly["Time"] = hourly["Hour"].map(lambda h: f"{h:02d}:00")
            fig = px.area(
                hourly, x="Time", y="Records", markers=True,
                color_discrete_sequence=["#62e6dc"]
            )
            fig.update_traces(
                line=dict(width=3),
                fillcolor="rgba(98,230,220,.12)",
                marker=dict(size=6),
            )
            st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
        else:
            st.info("No hourly aggregation was found in the summary.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Severity distribution")
        if not severity.empty:
            severity["Label"] = severity["Label"].map(lambda x: f"Level {x}")
            fig = px.pie(
                severity, names="Label", values="Records", hole=.68,
                color_discrete_sequence=["#62e6dc", "#7aa7ff", "#b49aff", "#ffb86b"]
            )
            fig.update_traces(
                textposition="outside", textinfo="percent+label",
                marker=dict(line=dict(color="#101a2c", width=3))
            )
            st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
        else:
            st.info("No severity aggregation was found in the summary.")
        st.markdown('</div>', unsafe_allow_html=True)

    left2, right2 = st.columns(2)
    with left2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Day vs night")
        if not day_night.empty:
            fig = px.bar(
                day_night, x="Label", y="Records", color="Records",
                color_continuous_scale=["#23465e", "#62e6dc"],
                labels={"Label": "Period", "Records": "Accident records"}
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
        else:
            st.info("No day/night aggregation was found in the summary.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Top states by reported accidents")
        if not states.empty:
            top_states = states.sort_values("Records", ascending=False).head(12).sort_values("Records")
            fig = px.bar(
                top_states, x="Records", y="Label", orientation="h",
                color="Records", color_continuous_scale=["#283a68", "#7aa7ff"],
                labels={"Label": "State", "Records": "Accident records"}
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
        else:
            st.info("No state aggregation was found in the summary.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Weather conditions")
    st.caption("Showing the 15 most frequent weather labels in the MapReduce output.")
    if not weather.empty:
        top_weather = weather.sort_values("Records", ascending=False).head(15).sort_values("Records")
        fig = px.bar(
            top_weather, x="Records", y="Label", orientation="h",
            color="Records", color_continuous_scale=["#23465e", "#62e6dc"],
            labels={"Label": "Weather condition", "Records": "Accident records"}
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(dark_layout(fig, 430), use_container_width=True)
    else:
        st.info("No weather aggregation was found in the summary.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Hadoop output explorer")
    category_options = ["All categories"] + sorted(summary["Category"].unique().tolist())
    selected_category = st.selectbox(
        "Filter aggregated output", category_options, key="hadoop_category_filter"
    )
    displayed = summary if selected_category == "All categories" else summary[summary["Category"] == selected_category]
    displayed = displayed.sort_values(["Category", "Records"], ascending=[True, False])
    st.dataframe(displayed, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Hadoop summary (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="hadoop_traffic_summary.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;color:#7186a3;padding:24px 0 4px;font-size:.8rem">
    URBAN TRAFFIC INTELLIGENCE · Hadoop MapReduce · Full-dataset aggregates
    </div>
    """, unsafe_allow_html=True)


REQUIRED = ["Start_Time", "Start_Lat", "Start_Lng", "Severity"]

def fmt_num(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

@st.cache_resource(show_spinner=False)
def get_spark():
    if not SPARK_AVAILABLE:
        return None
    return (SparkSession.builder
            .appName("UrbanTrafficIntelligence")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())

@st.cache_data(show_spinner="Reading and preparing accident records…")
def load_with_pandas(path, mtime, row_limit):
    # mtime is included in cache key so changing the file refreshes the data.
    df = pd.read_csv(path, low_memory=False, nrows=row_limit if row_limit else None)
    return clean_data(df)

def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["Start_Time", "End_Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ["Start_Lat", "Start_Lng", "Severity", "Distance(mi)", "Temperature(F)",
                "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Start_Time" in df.columns:
        df["Hour"] = df["Start_Time"].dt.hour
        df["Weekday"] = df["Start_Time"].dt.day_name()
        df["Month"] = df["Start_Time"].dt.month_name()
        df["Date"] = df["Start_Time"].dt.date
    if "Start_Lat" in df.columns and "Start_Lng" in df.columns:
        df = df.dropna(subset=["Start_Lat", "Start_Lng"])
        df = df[df["Start_Lat"].between(-90, 90) & df["Start_Lng"].between(-180, 180)]
    if "Severity" in df.columns:
        df["Severity"] = df["Severity"].fillna(0).astype("int64")
    if "City" not in df.columns:
        df["City"] = "Unknown"
    if "Weather_Condition" not in df.columns:
        df["Weather_Condition"] = "Unknown"
    if "State" not in df.columns:
        df["State"] = "Unknown"
    return df

def load_data_source(uploaded, selected_path, row_limit, engine):
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_path = tmp.name
        path = temp_path
    else:
        path = str(selected_path) if selected_path else None
        temp_path = None
    if not path or not os.path.exists(path):
        return None, None, "No dataset selected"
    started = time.perf_counter()
    try:
        # Spark engine is used for CSV ingestion when available; pandas prepares
        # a local dashboard dataframe for map and chart rendering.
        spark_status = "Pandas"
        if engine == "PySpark" and SPARK_AVAILABLE:
            spark = get_spark()
            sdf = (spark.read.option("header", True).option("inferSchema", True).csv("file://" + os.path.abspath(path)))
            if row_limit:
                sdf = sdf.limit(int(row_limit))
            count = sdf.count()
            # Convert only selected sample to pandas for UI rendering.
            pdf = sdf.toPandas()
            spark_status = f"PySpark local mode · {count:,} rows read"
            df = clean_data(pdf)
        else:
            mtime = os.path.getmtime(path)
            df = load_with_pandas(path, mtime, int(row_limit) if row_limit else 0)
            spark_status = "Pandas preview mode"
        elapsed = time.perf_counter() - started
        return df, {"path": path, "elapsed": elapsed, "engine": spark_status, "temp_path": temp_path}, None
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception: pass
        return None, None, f"{type(e).__name__}: {e}"

def dark_layout(fig, height=360):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=8,r=8,t=48,b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#dbe8f8"),
        title_font=dict(family="Space Grotesk", size=17, color="#f1f7ff"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.12)")
    return fig

def metric(label, value, note=""):
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">{label}</div><div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div></div>""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">BIG DATA · URBAN MOBILITY · ROAD SAFETY</div>
  <h1>Urban Traffic<br>Intelligence.</h1>
  <p>Explore accident patterns, discover geographic hotspots, and turn historical road-safety data into clear, actionable insights.</p>
  <span class="pill">✦ PySpark analytics</span><span class="pill">⌖ Hotspot discovery</span><span class="pill">↗ Interactive insights</span>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar controls ----------
st.sidebar.markdown("## 🚦 Traffic Intelligence")
st.sidebar.caption("A road-safety analytics workspace")
st.sidebar.markdown("---")
dashboard_mode = st.sidebar.radio(
    "Dashboard data source",
    ["Hadoop MapReduce summary", "CSV exploration"],
    index=0,
    help="Use the completed Hadoop aggregates for full-dataset charts, or choose CSV exploration for maps and interactive record-level filters.",
)

if dashboard_mode == "Hadoop MapReduce summary":
    render_hadoop_dashboard()
    st.stop()

uploaded = st.sidebar.file_uploader("Upload accident CSV", type=["csv"], help="For large datasets, place the CSV in the project's data folder instead.")
local_files = []
for p in DEFAULT_CANDIDATES:
    if p.exists() and str(p) not in local_files:
        local_files.append(str(p))
data_folder_files = sorted(str(p) for p in DATA_DIR.glob("*.csv"))
for p in data_folder_files:
    if p not in local_files: local_files.append(p)
selected_path = None
if not uploaded:
    if local_files:
        selected_path = st.sidebar.selectbox("Or choose a local dataset", options=local_files, format_func=lambda x: Path(x).name)
    else:
        st.sidebar.info("Add your CSV to the project's data/ folder or upload it above.")
engine = st.sidebar.radio("Processing engine", ["PySpark", "Pandas"], index=0 if SPARK_AVAILABLE else 1,
                          help="PySpark is recommended for the Big Data Analytics requirement. Install Java and pyspark if needed.")
row_choice = st.sidebar.selectbox("Records to analyze", ["100,000 (quick preview)", "500,000", "1,000,000", "All available records"], index=0)
row_limit = None if row_choice.startswith("All") else int(row_choice.split()[0].replace(",", ""))
st.sidebar.markdown("---")
st.sidebar.caption("Tip: start with 100k records, verify the charts, then scale up.")

if uploaded is None and selected_path is None:
    st.markdown('<div class="panel"><h3>Start with your dataset</h3><p class="small-muted">Upload your accident CSV in the sidebar, or place it in the <code>data/</code> folder beside <code>app.py</code>. The app supports the common US Accidents column names.</p></div>', unsafe_allow_html=True)
    st.stop()

with st.spinner("Preparing your traffic intelligence workspace…"):
    df, load_info, error = load_data_source(uploaded, selected_path, row_limit, engine)
if error:
    st.error(f"Could not load the dataset: {error}")
    st.info("Check that the file is a valid CSV and includes Start_Time, Start_Lat, Start_Lng and Severity columns.")
    st.stop()
if df is None or df.empty:
    st.warning("No rows found in this dataset selection.")
    st.stop()

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("The dataset is missing required columns: " + ", ".join(missing))
    st.markdown("Expected columns include `Start_Time`, `Start_Lat`, `Start_Lng`, and `Severity`. Rename your columns or use the US Accidents dataset.")
    st.stop()

# Filters
st.markdown('<div class="section-label">Explore your data</div>', unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns([1.15, 1.15, 1.1, 1.4])
with f1:
    states = sorted([str(x) for x in df["State"].dropna().unique()])
    state_sel = st.selectbox("State", ["All states"] + states)
with f2:
    severities = sorted([int(x) for x in df["Severity"].dropna().unique()])
    severity_sel = st.selectbox("Severity", ["All levels"] + severities)
with f3:
    hours = sorted([int(x) for x in df["Hour"].dropna().unique()]) if "Hour" in df else []
    hour_sel = st.selectbox("Hour of day", ["All hours"] + hours)
with f4:
    city_query = st.text_input("Search city", placeholder="e.g. Miami, Houston…")

filtered = df.copy()
if state_sel != "All states": filtered = filtered[filtered["State"].astype(str) == state_sel]
if severity_sel != "All levels": filtered = filtered[filtered["Severity"] == severity_sel]
if hour_sel != "All hours" and "Hour" in filtered: filtered = filtered[filtered["Hour"] == hour_sel]
if city_query.strip(): filtered = filtered[filtered["City"].astype(str).str.contains(city_query.strip(), case=False, na=False)]

if filtered.empty:
    st.warning("No records match these filters. Try a wider selection.")
    st.stop()

# Metrics
st.markdown('<div class="section-label">Live overview</div>', unsafe_allow_html=True)
total = len(filtered)
avg_severity = filtered["Severity"].mean()
cities = filtered["City"].nunique()
high_sev = int((filtered["Severity"] >= 3).sum())
m1,m2,m3,m4 = st.columns(4)
with m1: metric("ACCIDENT RECORDS", fmt_num(total), "records in current view")
with m2: metric("HIGHER SEVERITY", fmt_num(high_sev), "severity level 3–4")
with m3: metric("LOCATIONS", fmt_num(cities), "distinct cities represented")
with m4: metric("AVG. SEVERITY", f"{avg_severity:.2f}", "dataset scale: 1–4")

st.caption(f"Loaded in {load_info['elapsed']:.2f}s · Engine: {load_info['engine']} · Source: {Path(load_info['path']).name if load_info.get('path') else uploaded.name}")

# Main charts
left, right = st.columns([1.35, 1])
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Accident activity · by hour")
    if "Hour" in filtered and filtered["Hour"].notna().any():
        hourly = filtered.groupby("Hour").size().reindex(range(24), fill_value=0).rename("Accidents").reset_index()
        fig = px.area(hourly, x="Hour", y="Accidents", markers=True,
                      color_discrete_sequence=["#62e6dc"])
        fig.update_traces(line=dict(width=3), fillcolor="rgba(98,230,220,.12)", marker=dict(size=6))
        st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
    else:
        st.info("Time data is not available.")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Severity distribution")
    sev = filtered["Severity"].value_counts().sort_index().rename_axis("Severity").reset_index(name="Records")
    fig = px.pie(sev, names="Severity", values="Records", hole=.68,
                 color_discrete_sequence=["#62e6dc", "#7aa7ff", "#b49aff", "#ffb86b"])
    fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="#101a2c", width=3)))
    st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

left2, right2 = st.columns(2)
with left2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Accidents by weekday")
    if "Weekday" in filtered:
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekdays = filtered["Weekday"].value_counts().reindex(order, fill_value=0).rename_axis("Weekday").reset_index(name="Records")
        fig = px.bar(weekdays, x="Weekday", y="Records", color="Records",
                     color_continuous_scale=["#23465e","#62e6dc"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with right2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Top cities by reported accidents")
    city_counts = filtered.groupby("City").size().sort_values(ascending=False).head(12).sort_values().rename("Records").reset_index()
    fig = px.bar(city_counts, x="Records", y="City", orientation="h",
                 color="Records", color_continuous_scale=["#283a68","#7aa7ff"])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Hotspot map
st.markdown('<div class="section-label">Geospatial intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Accident hotspot explorer")
st.caption("Map markers show recorded accident locations in the filtered sample. Dense areas indicate more records, not a normalized collision probability.")
map_sample_max = min(5000, len(filtered))
map_n = st.slider("Map points to render", min_value=min(100, map_sample_max), max_value=map_sample_max, value=min(1500, map_sample_max), step=max(1, min(100, map_sample_max)//10))
map_df = filtered.dropna(subset=["Start_Lat", "Start_Lng"])
if len(map_df) > map_n:
    map_df = map_df.sample(map_n, random_state=42)
if FOLIUM_AVAILABLE and not map_df.empty:
    center = [float(map_df["Start_Lat"].median()), float(map_df["Start_Lng"].median())]
    fmap = folium.Map(location=center,zoom_start=5,tiles="OpenStreetMap",control_scale=True)
    from folium.plugins import HeatMap
    heat_points = map_df[["Start_Lat", "Start_Lng"]].dropna().values.tolist()
    HeatMap(heat_points, radius=10, blur=14, min_opacity=.22).add_to(fmap)
    st_folium(fmap, height=470, use_container_width=True, returned_objects=[])
else:
    st.warning("Install folium and streamlit-folium to enable the interactive map.")
    st.map(map_df.rename(columns={"Start_Lat":"lat","Start_Lng":"lon"})[["lat","lon"]].head(3000) if not map_df.empty else pd.DataFrame())
st.markdown('</div>', unsafe_allow_html=True)

# Hotspot grid table
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Top geographic hotspot cells")
st.caption("A simple grid-based ranking that is fast to calculate and easy to explain in a viva.")
grid_df = filtered.dropna(subset=["Start_Lat", "Start_Lng"]).copy()
if not grid_df.empty:
    # Rounded coordinates approximate local geographic cells; cell size is not a precise road segment.
    grid_df["Lat_Cell"] = grid_df["Start_Lat"].round(2)
    grid_df["Lng_Cell"] = grid_df["Start_Lng"].round(2)
    group_cols = ["Lat_Cell", "Lng_Cell"]
    if "City" in grid_df: group_cols.append("City")
    hotspot = (grid_df.groupby(group_cols)
               .agg(Accident_Records=("Severity","size"),
                    Avg_Severity=("Severity","mean"))
               .reset_index()
               .sort_values(["Accident_Records","Avg_Severity"], ascending=False)
               .head(20))
    hotspot["Avg_Severity"] = hotspot["Avg_Severity"].round(2)
    st.dataframe(hotspot, use_container_width=True, hide_index=True)
    st.download_button("Download hotspot report (CSV)", hotspot.to_csv(index=False).encode("utf-8"),
                       file_name="traffic_hotspot_report.csv", mime="text/csv")
else:
    st.info("No valid coordinates found.")
st.markdown('</div>', unsafe_allow_html=True)

# Data preview / quality
with st.expander("Dataset quality & sample records"):
    q1,q2,q3 = st.columns(3)
    q1.metric("Rows currently analyzed", fmt_num(len(df)))
    q2.metric("Columns", fmt_num(len(df.columns)))
    q3.metric("Missing cells", fmt_num(int(df.isna().sum().sum())))
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

st.markdown("""
<div style="text-align:center;color:#7186a3;padding:24px 0 4px;font-size:.8rem">
URBAN TRAFFIC INTELLIGENCE · Big Data Analytics Laboratory · Built with PySpark & Streamlit
</div>
""", unsafe_allow_html=True)
