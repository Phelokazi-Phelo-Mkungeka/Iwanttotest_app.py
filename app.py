
# app.py
# Streamlit app: Misinformation Network with Time-of-Day Taxi Violence Dynamics
# Author: Phelokazi Mkungeka & M365 Copilot
# Description:
# - Upload a CSV (or load sample data) to visualize a co-occurrence network (hashtags/keywords/accounts).
# - Applies time-of-day weights to taxi-violence misinformation (morning boost, evening drop).
# - Interactive controls for time buckets, keyword rules, and scaling factors.
# - Exports nodes/edges as CSV.

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import re
from datetime import datetime, time as dtime
import pytz
from io import StringIO
from streamlit.components.v1 import html

# ------------------------------
# App config
# ------------------------------
st.set_page_config(
    page_title="Misinformation Network (Taxi Violence Dynamics)",
    page_icon="🕸️",
    layout="wide",
)

DEFAULT_TZ = "Africa/Johannesburg"  # UTC+2
tz = pytz.timezone(DEFAULT_TZ)

# ------------------------------
# Helpers
# ------------------------------
def parse_time_bucket(start: dtime, end: dtime, hour: int) -> bool:
    """Return True if hour (0-23) falls in [start, end], inclusive, handling ranges that cross midnight."""
    start_h = start.hour
    end_h = end.hour
    if start_h <= end_h:
        return start_h <= hour <= end_h
    # crosses midnight (e.g., 22:00–04:00)
    return hour >= start_h or hour <= end_h

def normalize_list_from_textarea(text: str) -> list:
    """Comma/whitespace-separated -> lowercase, trimmed, unique list."""
    raw = re.split(r"[,\n;]+", text.strip()) if text else []
    return sorted({s.strip().lower() for s in raw if s.strip()})

def extract_hashtags(text: str) -> list:
    if not isinstance(text, str):
        return []
    tags = re.findall(r"#(\w+)", text)
    return [t.lower() for t in tags]

def in_any_keyword(s: str, keywords: list) -> bool:
    s_low = (s or "").lower()
    return any(k in s_low for k in keywords)

def is_taxi_related(text: str, hashtags: list, taxi_keywords: list) -> bool:
    if in_any_keyword(text, taxi_keywords):
        return True
    for h in hashtags:
        if any(k in h for k in taxi_keywords):
            return True
    return False

def is_misinfo(text: str, misinfo_keywords: list) -> bool:
    return in_any_keyword(text, misinfo_keywords)

def hour_of(ts) -> int:
    try:
        return pd.to_datetime(ts).tz_localize(tz=None).tz_localize("UTC").tz_convert(DEFAULT_TZ).hour
    except Exception:
        # Try naive parse then localize to DEFAULT_TZ
        try:
            return pd.to_datetime(ts).tz_localize(DEFAULT_TZ).hour
        except Exception:
            # Fallback: assume already local
            return pd.to_datetime(ts).hour

def weight_for_post(hour: int, is_taxi: bool, is_mis: bool,
                    morning_start: dtime, morning_end: dtime, morning_boost: float,
                    evening_start: dtime, evening_end: dtime, evening_drop: float,
                    baseline: float = 1.0) -> float:
    """Apply weights only if both taxi-related AND misinformation."""
    if not (is_taxi and is_mis):
        return baseline
    if parse_time_bucket(morning_start, morning_end, hour):
        return baseline * morning_boost
    if parse_time_bucket(evening_start, evening_end, hour):
        return baseline * evening_drop
    return baseline

def build_items(row, node_mode, topic_keywords):
    """
    Return list of items for nodes:
    - 'hashtags': from 'hashtags' column or extracted from text.
    - 'keywords': from topic_keywords found in text.
    - 'accounts': from 'account' column (or 'account_id').
    """
    text = row.get("text", "")
    hashtags_col = row.get("hashtags", None)
    if isinstance(hashtags_col, str) and hashtags_col.strip():
        hashtags = [h.strip().lower() for h in re.split(r"[,\s]+", hashtags_col) if h.strip()]
    else:
        hashtags = extract_hashtags(text)

    if node_mode == "hashtags":
        return hashtags

    if node_mode == "keywords":
        found = []
        s_low = (text or "").lower()
        for kw in topic_keywords:
            if kw in s_low:
                found.append(kw)
        # Also include any taxi-related tokens present in hashtags if they match keywords
        for h in hashtags:
            for kw in topic_keywords:
                if kw in h:
                    found.append(kw)
        return sorted({x for x in found})

    # accounts
    acct = row.get("account", None) or row.get("account_id", None) or row.get("username", None)
    if isinstance(acct, str) and acct.strip():
        return [acct.strip().lower()]
    return []

def add_cooccurrence_edges(G: nx.Graph, items: list, w: float):
    """Add edges for all unique pairs within items with weight accumulation."""
    if len(items) < 2:
        # Add single node with weight on degree proxy via self (store as node weight)
        for it in items:
            G.add_node(it)
            G.nodes[it]["node_weight"] = G.nodes[it].get("node_weight", 0.0) + w
        return
    for i in range(len(items)):
        G.add_node(items[i])
        G.nodes[items[i]]["node_weight"] = G.nodes[items[i]].get("node_weight", 0.0) + w
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
                G[a][b]["count"] += 1
            else:
                G.add_edge(a, b, weight=w, count=1)

def color_for_node(label: str, node_stats: dict, taxi_keywords: list) -> str:
    info = node_stats.get(label, {"mis_weight": 0.0, "total_weight": 0.0})
    mis_ratio = (info["mis_weight"] / info["total_weight"]) if info["total_weight"] > 0 else 0.0
    is_taxi_node = any(k in label for k in taxi_keywords)
    # Coloring rules:
    # - Red: Taxi-related & majority misinfo
    # - Orange: Majority misinfo
    # - SteelBlue: Otherwise
    if is_taxi_node and mis_ratio >= 0.5:
        return "#d62728"  # red
    if mis_ratio >= 0.5:
        return "#ff7f0e"  # orange
    return "#4682b4"      # steelblue

def build_pyvis(G: nx.Graph, node_stats: dict, taxi_keywords: list, height="700px", width="100%"):
    net = Network(height=height, width=width, notebook=False, directed=False)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.03)
    # Scale node sizes
    node_weights = [G.nodes[n].get("node_weight", 0.0) for n in G.nodes()]
    max_node_w = max(node_weights) if node_weights else 1.0

    for n in G.nodes():
        w = G.nodes[n].get("node_weight", 0.0)
        size = 10 + 30 * (w / max_node_w) if max_node_w > 0 else 10
        color = color_for_node(n, node_stats, taxi_keywords)
        title = f"{n}<br>Weighted freq: {w:.2f}"
        net.add_node(n, label=n, title=title, size=size, color=color)

    # Scale edge widths
    edge_weights = [G[e[0]][e[1]].get("weight", 0.0) for e in G.edges()]
    max_edge_w = max(edge_weights) if edge_weights else 1.0

    for a, b in G.edges():
        w = G[a][b].get("weight", 0.0)
        count = G[a][b].get("count", 0)
        width = 1 + 7 * (w / max_edge_w) if max_edge_w > 0 else 1
        title = f"{a} — {b}<br>Weight: {w:.2f} | Co-occurrences: {count}"
        net.add_edge(a, b, value=w, title=title, width=width)

    return net

@st.cache_data(show_spinner=False)
def generate_sample_data(n=500, seed=42):
    """
    Create synthetic posts across hours with taxi-violence + misinfo patterns
    so the morning shows higher weighted frequencies and evening shows lower.
    """
    rng = np.random.default_rng(seed)
    base_hours = rng.integers(0, 24, size=n)

    text_templates = [
        "General update on commuting in Bloemfontein.",
        "HPV vaccine rollout news and school health communication.",
        "Crime alert near taxi rank reported by locals.",
        "Rumor of taxi route changes circulating without verification.",
        "Discussion on minibus taxi protests and alleged violence.",
        "Unverified claim about taxi strike affecting schools.",
        "Community report: safer evenings with fewer taxis operating.",
        "Debate on misinformation spreading on social platforms.",
    ]

    rows = []
    today = pd.Timestamp.now(tz=DEFAULT_TZ).normalize()
    for i in range(n):
        hour = int(base_hours[i])
        ts = today + pd.Timedelta(hours=hour) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
        text = rng.choice(text_templates)
        # Randomly add hashtags
        hash_pool = ["#taxi", "#minibus", "#violence", "#HPV", "#vaccine", "#crime", "#education", "#rumor", "#strike"]
        hashtags = ",".join(sorted(set(rng.choice(hash_pool, size=rng.integers(0, 4), replace=False))))
        # Misinfo flag heuristic
        mis = any(k in text.lower() for k in ["rumor", "unverified", "alleged"])
        # Taxi-related heuristic
        taxi = any(k in (text.lower() + " " + hashtags.lower()) for k in ["taxi", "minibus", "route", "violence", "strike"])
        rows.append({
            "timestamp": ts.isoformat(),
            "text": text,
            "hashtags": hashtags,
            "is_misinfo": mis,
            "topic": "taxi_violence" if taxi else "other",
            "account": f"user_{rng.integers(1, 50)}",
        })
    return pd.DataFrame(rows)

def compute_metrics(df, morning_start, morning_end, evening_start, evening_end,
                    taxi_keywords, misinfo_keywords):
    # Classify & weight per post
    def classify_row(row):
        text = row.get("text", "")
        hashtags = [h.strip().lower() for h in re.split(r"[,\s]+", str(row.get("hashtags", ""))) if h.strip()]
        hour = hour_of(row.get("timestamp", pd.Timestamp.now()))
        is_taxi = is_taxi_related(text, hashtags, taxi_keywords) or (str(row.get("topic", "")).lower() == "taxi_violence")
        is_mis = bool(row.get("is_misinfo", False)) or is_misinfo(text, misinfo_keywords)
        return hour, is_taxi, is_mis

    hours = []
    taxi_flags = []
    mis_flags = []
    for _, r in df.iterrows():
        h, t, m = classify_row(r)
        hours.append(h)
        taxi_flags.append(t)
        mis_flags.append(m)

    df2 = df.copy()
    df2["hour"] = hours
    df2["is_taxi"] = taxi_flags
    df2["is_misinfo_final"] = mis_flags

    # Dummy baseline; actual weights applied later in graph building, but we compute counts per bucket here.
    def in_morning(h): return parse_time_bucket(morning_start, morning_end, h)
    def in_evening(h): return parse_time_bucket(evening_start, evening_end, h)

    morning_count = int(((df2["is_taxi"] & df2["is_misinfo_final"]) & df2["hour"].apply(in_morning)).sum())
    evening_count = int(((df2["is_taxi"] & df2["is_misinfo_final"]) & df2["hour"].apply(in_evening)).sum())

    return df2, morning_count, evening_count

# ------------------------------
# UI: Sidebar Controls
# ------------------------------
st.sidebar.title("⚙️ Controls")

# Time buckets
st.sidebar.subheader("Time-of-day buckets")
morning_start = st.sidebar.time_input("Morning start", dtime(5, 0))
morning_end = st.sidebar.time_input("Morning end", dtime(9, 0))
evening_start = st.sidebar.time_input("Evening start", dtime(17, 0))
evening_end = st.sidebar.time_input("Evening end", dtime(21, 0))

st.sidebar.subheader("Scaling factors (applied only to taxi-violence misinformation)")
morning_boost = st.sidebar.slider("Morning boost ×", min_value=1.0, max_value=3.0, value=1.8, step=0.1)
evening_drop = st.sidebar.slider("Evening drop ×", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
baseline_weight = 1.0

st.sidebar.subheader("Keywords")
default_taxi_keywords = "taxi,minibus,route,taxi violence,strike,rank,association,operator"
default_misinfo_keywords = "rumor,hoax,fake,unverified,alleged,false claim,disinformation,misinformation"

taxi_keywords = normalize_list_from_textarea(
    st.sidebar.text_area("Taxi-violence keywords", value=default_taxi_keywords, height=80)
)
misinfo_keywords = normalize_list_from_textarea(
    st.sidebar.text_area("Misinformation keywords", value=default_misinfo_keywords, height=80)
)

st.sidebar.subheader("Node type")
node_mode = st.sidebar.radio("Build nodes from", options=["hashtags", "keywords", "accounts"], index=0)

topic_keywords_input = st.sidebar.text_area(
    "Topic keywords (used only if node type = keywords)",
    value="taxi violence,taxi,minibus,route,rank,strike,HPV,vaccine,crime,education,health communication",
    height=80
)
topic_keywords = normalize_list_from_textarea(topic_keywords_input)

st.sidebar.subheader("Data source")
uploaded = st.sidebar.file_uploader("Upload CSV (timestamp,text,hashtags,is_misinfo,topic,account)", type=["csv"])
use_sample = st.sidebar.checkbox("Load sample data", value=(uploaded is None))

st.sidebar.subheader("Time selection")
time_mode = st.sidebar.radio("Apply weights using", options=["Server time (Africa/Johannesburg)", "Use each post's timestamp bucket"], index=1)

# ------------------------------
# Main
# ------------------------------
st.title("🕸️ Misinformation Network — Taxi Violence Time Dynamics")

st.markdown("""
This app visualizes a co-occurrence network and **weights taxi-violence misinformation** by time-of-day:

- **Morning** (default 05:00–09:00): **boosted**
- **Evening** (default 17:00–21:00): **downweighted**

Only posts that are **both** taxi-related **and** misinformation are affected.
""")

# Load data
if use_sample and not uploaded:
    df = generate_sample_data(n=600)
    st.info("Using synthetic sample data. Upload your CSV to use real data.")
else:
    if uploaded is None:
        st.warning("Please upload a CSV or check 'Load sample data'.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# Basic schema check
required_cols = ["timestamp", "text"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Your CSV must include at least these columns: {required_cols}. Missing: '{col}'")
        st.stop()

# Compute classification & simple counts
df2, morning_count, evening_count = compute_metrics(df, morning_start, morning_end, evening_start, evening_end, taxi_keywords, misinfo_keywords)

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Taxi-violence misinfo (morning bucket)", morning_count)
with colB:
    st.metric("Taxi-violence misinfo (evening bucket)", evening_count)
with colC:
    perc = 0.0
    if evening_count > 0:
        perc = ((morning_count - evening_count) / evening_count) * 100.0
    elif morning_count > 0:
        perc = 100.0
    st.metric("Δ Morning vs Evening (%)", f"{perc:.1f}%")

# Hourly series (unweighted counts to inspect distribution)
series = df2[df2["is_taxi"] & df2["is_misinfo_final"]]["hour"].value_counts().sort_index()
if not series.empty:
    fig = px.bar(
        x=series.index,
        y=series.values,
        labels={"x": "Hour of day", "y": "Count (taxi-violence misinfo)"},
        title="Hourly distribution (unweighted counts)",
    )
    st.plotly_chart(fig, use_container_width=True)

# Build network
G = nx.Graph()
node_stats = {}  # label -> {"mis_weight": float, "total_weight": float}

# Determine applied hour (time_mode)
current_hour_server = datetime.now(tz=tz).hour

for _, row in df2.iterrows():
    text = row.get("text", "")
    hashtags = [h.strip().lower() for h in re.split(r"[,\s]+", str(row.get("hashtags", ""))) if h.strip()]
    is_taxi = bool(row.get("is_taxi", False))
    is_mis = bool(row.get("is_misinfo_final", False))

    # Choose hour context
    h = current_hour_server if time_mode.startswith("Server") else int(row.get("hour", current_hour_server))

    w = weight_for_post(
        hour=h,
        is_taxi=is_taxi,
        is_mis=is_mis,
        morning_start=morning_start,
        morning_end=morning_end,
        morning_boost=morning_boost,
        evening_start=evening_start,
        evening_end=evening_end,
        evening_drop=evening_drop,
        baseline=baseline_weight,
    )

    items = build_items(row, node_mode=node_mode, topic_keywords=topic_keywords)

    # Track node stats per item
    for it in items:
        stats = node_stats.get(it, {"mis_weight": 0.0, "total_weight": 0.0})
        stats["total_weight"] += w
        if is_mis:
            stats["mis_weight"] += w
        node_stats[it] = stats

    # Add co-occurrence edges
    add_cooccurrence_edges(G, items, w)

# PyVis network render
net = build_pyvis(G, node_stats, taxi_keywords, height="700px", width="100%")
net.show("network.html")
with open("network.html", "r", encoding="utf-8") as f:
    html_content = f.read()
html(html_content, height=720)

# Export nodes/edges
nodes_export = []
for n in G.nodes():
    stats = node_stats.get(n, {"mis_weight": 0.0, "total_weight": 0.0})
    nodes_export.append({
        "node": n,
        "weighted_frequency": G.nodes[n].get("node_weight", 0.0),
        "misinfo_weight": stats["mis_weight"],
        "total_weight": stats["total_weight"],
        "misinfo_ratio": (stats["mis_weight"] / stats["total_weight"]) if stats["total_weight"] > 0 else 0.0,
    })

edges_export = []
for a, b, data in G.edges(data=True):
    edges_export.append({
        "source": a,
        "target": b,
        "weight": data.get("weight", 0.0),
        "cooccurrence_count": data.get("count", 0),
    })

nodes_df = pd.DataFrame(nodes_export)
edges_df = pd.DataFrame(edges_export)

st.subheader("📤 Download network data")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download nodes.csv",
        data=nodes_df.to_csv(index=False).encode("utf-8"),
        file_name="nodes.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        "Download edges.csv",
        data=edges_df.to_csv(index=False).encode("utf-8"),
        file_name="edges.csv",
        mime="text/csv",
    )

st.caption("Tip: The node color indicates misinfo proportion; red nodes are taxi-related with majority misinfo.")

