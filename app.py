
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ALIASES = {
    FEATURES[0]: ["airtemperaturek", "airtemperature", "airtempk", "airtemp"],
    FEATURES[1]: ["processtemperaturek", "processtemperature", "processtempk", "processtemp"],
    FEATURES[2]: ["rotationalspeedrpm", "rotationalspeed", "rpm", "speed"],
    FEATURES[3]: ["torquenm", "torque"],
    FEATURES[4]: ["toolwearmin", "toolwear", "wear"],
}

ID_CANDIDATES = [
    "machineid", "productid", "udi",
    "assetid", "equipmentid", "machine", "id"
]

FEATURE_RANGES = {
    FEATURES[0]: (290.0, 310.0, 0.1),
    FEATURES[1]: (300.0, 320.0, 0.1),
    FEATURES[2]: (1000.0, 3000.0, 10.0),
    FEATURES[3]: (3.0, 80.0, 0.1),
    FEATURES[4]: (0.0, 260.0, 1.0),
}

STATUS_ORDER = ["Healthy", "Monitor", "Warning", "Critical"]
PRIORITY_ORDER = ["Critical", "High", "Warning", "Review"]

ACTIONS = {
    "Critical": "Stop and inspect with an engineer",
    "High": "Schedule an inspection this shift",
    "Warning": "Check tool wear and loading",
    "Review": "Review the unusual sensor pattern",
    "OK": "No action needed",
}


def _st_version():
    try:
        return tuple(
            int(p) for p in st.__version__.split(".")[:2]
        )
    except ValueError:
        return (0, 0)


# Compatibility with different Streamlit versions
STRETCH = (
    {"width": "stretch"}
    if _st_version() >= (1, 50)
    else {"use_container_width": True}
)

PAGES = [
    "Upload & Analyse",
    "Dashboard",
    "Machine Risk",
    "Failure Diagnosis",
    "Anomaly Detection",
    "Health Analytics",
    "Explainable AI",
    "Maintenance Planner",
    "What-if Simulator",
    "Reports & Export",
]


# ============================================================
# 2. THEME
# ============================================================

LIGHT = bool(st.session_state.get("light_theme", False))

CHART_TXT = "#17212b" if LIGHT else "#f1f5fa"

STATUS_COLORS = (
    {
        "Healthy": "#1a9d68",
        "Monitor": "#1783a8",
        "Warning": "#c98200",
        "Critical": "#d03d3d",
    }
    if LIGHT
    else {
        "Healthy": "#3fc48f",
        "Monitor": "#4cb8d6",
        "Warning": "#f0b441",
        "Critical": "#ee6b6b",
    }
)

PRIORITY_COLORS = {
    "Critical": STATUS_COLORS["Critical"],
    "High": "#f08a4b",
    "Warning": STATUS_COLORS["Warning"],
    "Review": STATUS_COLORS["Monitor"],
    "OK": STATUS_COLORS["Healthy"],
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700;800&display=swap');

:root {
    --bg: #0d1218;
    --panel: #151c25;
    --panel-2: #1a232e;
    --line: #29343f;
    --text: #e9eef4;
    --muted: #8d9bab;
    --accent: #f0b441;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(13,18,24,.92);
}

[data-testid="stSidebar"] {
    background: #10161d;
    border-right: 1px solid var(--line);
}

.block-container {
    padding-top: 1.6rem;
    max-width: 1500px;
}

h1, h2, h3 {
    letter-spacing: -.02em;
    font-weight: 650;
}

h1 {
    font-size: 1.9rem !important;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    padding: 14px 16px;
    border-radius: 6px;
}

div[data-testid="stMetricLabel"] p {
    color: #aebccc;
    font-size: .85rem;
    font-weight: 500;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1.85rem;
    color: #ffffff !important;
    letter-spacing: -.02em;
    font-variant-numeric: tabular-nums;
    text-shadow: 0 0 16px rgba(255,255,255,.18);
}

div[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

.kpi {
    background: var(--panel);
    border: 1px solid var(--line);
    border-top: 3px solid var(--kpi, #f0b441);
    border-radius: 6px;
    padding: 14px 16px 12px;
    height: 100%;
}

.kpi .k-label {
    color: #aebccc;
    font-size: .82rem;
    font-weight: 500;
}

.kpi .k-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 2.15rem;
    line-height: 1.15;
    color: var(--kpi, #ffffff);
    letter-spacing: -.03em;
    font-variant-numeric: tabular-nums;
    text-shadow: 0 0 22px color-mix(in srgb, var(--kpi, #ffffff) 45%, transparent);
}

.kpi .k-sub {
    color: var(--muted);
    font-size: .78rem;
    margin-top: 2px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 16px 18px;
}

.panel h3 {
    color: #ffffff;
}

.status-banner {
    border: 1px solid var(--line);
    border-left-width: 5px;
    border-radius: 6px;
    background: var(--panel);
    padding: 14px 18px;
    margin-bottom: 8px;
}

.status-banner .label {
    color: var(--muted);
    font-size: .8rem;
}

.status-banner .value {
    font-size: 1.5rem;
    font-weight: 650;
}

.brand {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -.01em;
}

.brand-sub, .small-muted {
    color: var(--muted);
    font-size: .82rem;
}

.stButton button, .stDownloadButton button {
    border-radius: 6px;
    font-weight: 600;
}

.stButton button[kind="primary"] {
    background: var(--accent);
    color: #11161c;
    border: 0;
}

.stButton button[kind="primary"]:hover {
    background: #f5c566;
    color: #11161c;
}

button[data-baseweb="tab"] {
    font-weight: 600;
}

hr {
    border-color: var(--line);
}

[data-testid="stAppDeployButton"],
[data-testid="stDeployButton"],
.stAppDeployButton,
.stDeployButton {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

if LIGHT:
    st.markdown(
        """
<style>
:root {
    --bg: #f3f5f8;
    --panel: #ffffff;
    --panel-2: #eef1f5;
    --line: #d3dae3;
    --text: #17212b;
    --muted: #566475;
    --accent: #c98200;
}

.stApp, .stApp p, .stApp li, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(243,245,248,.95);
}

[data-testid="stHeader"] * {
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

div[data-testid="stMetric"],
.kpi, .panel, .status-banner {
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(20,30,45,.06);
}

div[data-testid="stMetricLabel"] p,
.kpi .k-label, .kpi .k-sub {
    color: #566475;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #0b1520 !important;
    text-shadow: none;
}

.kpi .k-value {
    text-shadow: none;
}

.panel h3 {
    color: #0b1520;
}

.status-banner .label,
.small-muted, .brand-sub {
    color: #566475;
}

.stButton button, .stDownloadButton button {
    background: #ffffff;
    color: var(--text);
    border: 1px solid var(--line);
}

.stButton button[kind="primary"] {
    background: #c98200;
    color: #ffffff;
    border: 0;
}

.stButton button[kind="primary"]:hover {
    background: #a96c00;
    color: #ffffff;
}

[data-baseweb="input"] > div,
[data-baseweb="base-input"],
[data-baseweb="select"] > div,
textarea {
    background: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

input {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: #ffffff !important;
}

[data-baseweb="popover"] li,
[data-baseweb="popover"] * {
    color: var(--text) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #ffffff;
    border: 1px dashed var(--line);
}

[data-testid="stFileUploaderDropzone"] * {
    color: var(--text);
}

[data-testid="stAlert"] {
    background: #ffffff;
    border: 1px solid var(--line);
}

[data-testid="stAlert"] * {
    color: var(--text);
}

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 6px;
}

[data-testid="stExpander"] * {
    color: var(--text);
}

button[data-baseweb="tab"] {
    color: var(--text);
}
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    files = {
        "failure": "optimized_failure_model.pkl",
        "metadata": "optimized_model_metadata.pkl",
        "anomaly": "isolation_forest_model.pkl",
        "diagnosis": "failure_diagnosis_model.pkl",
        "health": "health_scoring_config.pkl",
    }

    paths = {
        key: os.path.join(MODEL_DIR, filename)
        for key, filename in files.items()
    }

    missing = [
        path for path in paths.values()
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing)
        )

    return {
        key: joblib.load(path)
        for key, path in paths.items()
    }


# ============================================================
# 4. ANALYSIS ENGINE
# ============================================================

def assess(frame, assets):
    """Score every row: failure risk, anomaly, health and failure modes."""

    metadata = assets["metadata"]
    health_cfg = assets["health"]

    frame = frame[FEATURES].astype(float).copy()

    threshold = float(metadata["threshold"])

    # XGBoost failure prediction
    failure_probability = (
        assets["failure"].predict_proba(frame)[:, 1]
    )

    # Isolation Forest anomaly detection
    anomaly_prediction = assets["anomaly"].predict(frame)

    anomaly_raw = (
        -assets["anomaly"].decision_function(frame)
    )

    reference = np.asarray(
        health_cfg["anomaly_reference_scores"]
    )

    anomaly_percentile = (
        np.searchsorted(
            reference,
            anomaly_raw,
            side="right"
        )
        / max(len(reference), 1) * 100
    )

    # Multi-output Random Forest diagnosis
    diag_model = assets["diagnosis"]["model"]
    diag_targets = assets["diagnosis"]["targets"]

    diag_probs = diag_model.predict_proba(frame)

    # Combined machine risk and health score
    combined_risk = (
        float(health_cfg["failure_weight"])
        * failure_probability * 100
        + float(health_cfg["anomaly_weight"])
        * anomaly_percentile
    )

    health_score = np.clip(
        100 - combined_risk,
        0,
        100
    )

    thresholds = health_cfg["health_thresholds"]

    def status(score):

        if score >= thresholds["healthy"]:
            return "Healthy"

        if score >= thresholds["monitor"]:
            return "Monitor"

        if score >= thresholds["warning"]:
            return "Warning"

        return "Critical"

    out = frame.copy()

    out["Failure probability (%)"] = (
        failure_probability * 100
    )

    out["Failure prediction"] = np.where(
        failure_probability >= threshold,
        "Failure risk",
        "No failure predicted"
    )

    out["Anomaly"] = np.where(
        anomaly_prediction == -1,
        "Anomaly",
        "Normal pattern"
    )

    out["Anomaly score"] = anomaly_raw

    out["Anomaly percentile (%)"] = (
        anomaly_percentile
    )

    out["Health score"] = health_score

    out["Health status"] = [
        status(score) for score in health_score
    ]

    for i, target in enumerate(diag_targets):

        out[f"{target} probability (%)"] = (
            diag_probs[i][:, 1] * 100
        )

    return out


def enrich(res):
    """Add risk flag, priority and suggested action."""

    res = res.copy()

    predicted = (
        res["Failure prediction"] == "Failure risk"
    )

    res["At risk"] = (
        predicted
        | res["Health status"].isin(["Warning", "Critical"])
    )

    res["Priority"] = np.select(
        [
            res["Health status"] == "Critical",
            predicted,
            res["Health status"] == "Warning",
            res["Anomaly"] == "Anomaly",
        ],
        [
            "Critical",
            "High",
            "Warning",
            "Review",
        ],
        default="OK",
    )

    res["Suggested action"] = (
        res["Priority"].map(ACTIONS)
    )

    return res


def norm(text):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower()
    )


def auto_map(columns):

    lookup = {
        norm(column): column
        for column in columns
    }

    mapping = {}

    for feature in FEATURES:

        for alias in ALIASES[feature]:

            if alias in lookup:
                mapping[feature] = lookup[alias]
                break

    return mapping


def detect_id(columns):

    lookup = {
        norm(column): column
        for column in columns
    }

    for candidate in ID_CANDIDATES:

        if candidate in lookup:
            return lookup[candidate]

    return None


def analyse_frame(raw, mapping, id_col):
    """Analyse uploaded CSV and return results and dropped-row count."""

    clean = pd.DataFrame({
        feature: pd.to_numeric(
            raw[mapping[feature]],
            errors="coerce"
        )
        for feature in FEATURES
    })

    valid = clean.dropna()

    if valid.empty:
        return None, len(raw)

    valid = valid[
        np.isfinite(valid).all(axis=1)
    ]

    if valid.empty:
        return None, len(raw)

    res = assess(valid, ASSETS).reset_index(drop=True)

    ids = (
        raw.loc[valid.index, id_col].astype(str).values
        if id_col
        else (valid.index + 1).astype(str).values
    )

    res.insert(0, "Machine ID", ids)

    used = (
        set(mapping.values())
        | ({id_col} if id_col else set())
    )

    extras = (
        raw.loc[valid.index]
        .drop(columns=list(used & set(raw.columns)))
        .reset_index(drop=True)
    )

    extras = extras[
        [column for column in extras.columns if column not in res.columns]
    ]

    res = pd.concat(
        [res, extras],
        axis=1
    )

    return enrich(res), len(raw) - len(valid)


# ============================================================
# 5. UI HELPERS
# ============================================================

def style_chart(fig, height=340):

    fig.update_layout(
        height=height,
        template="plotly_white" if LIGHT else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="IBM Plex Sans",
            color=CHART_TXT,
            size=13
        ),
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10
        ),
        legend=dict(
            orientation="h",
            y=-0.2
        ),
        hoverlabel=dict(
            font=dict(
                family="JetBrains Mono",
                color=CHART_TXT
            ),
            bgcolor="#ffffff" if LIGHT else "#1a232e",
        ),
    )

    tick = dict(
        family="JetBrains Mono",
        color=CHART_TXT,
        size=12
    )

    grid = "#e1e6ed" if LIGHT else "#242f3a"

    fig.update_xaxes(
        gridcolor=grid,
        tickfont=tick
    )

    fig.update_yaxes(
        gridcolor=grid,
        tickfont=tick
    )

    return fig


def show(fig, height=340):

    st.plotly_chart(
        style_chart(fig, height),
        **STRETCH
    )


def gauge(value, title, invert=False, height=250):

    good, mid, bad = (
        ("#cdeede", "#faeccb", "#f6d3d3")
        if LIGHT
        else ("#25483a", "#4d4326", "#512d2d")
    )

    steps = (
        [(0, 33, good), (33, 66, mid), (66, 100, bad)]
        if invert
        else [(0, 33, bad), (33, 66, mid), (66, 100, good)]
    )

    if invert:

        bar = (
            STATUS_COLORS["Critical"]
            if value > 66
            else STATUS_COLORS["Warning"]
            if value > 33
            else STATUS_COLORS["Healthy"]
        )

    else:

        bar = (
            STATUS_COLORS["Critical"]
            if value < 33
            else STATUS_COLORS["Warning"]
            if value < 66
            else STATUS_COLORS["Healthy"]
        )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number=dict(
                suffix="%" if invert else "",
                valueformat=".1f",
                font=dict(
                    family="JetBrains Mono",
                    size=46,
                    color=CHART_TXT
                )
            ),
            title=dict(
                text=title,
                font=dict(
                    size=15,
                    color="#566475" if LIGHT else "#c5d2e0"
                )
            ),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(
                    color=bar,
                    thickness=0.28
                ),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[a, b], color=c)
                    for a, b, c in steps
                ],
            ),
        )
    )

    return style_chart(fig, height)


def kpi(col, label, value, sub="", color="#ffffff"):

    col.markdown(
        f"""
        <div class="kpi" style="--kpi:{color}">
            <div class="k-label">{label}</div>
            <div class="k-value">{value}</div>
            <div class="k-sub">{sub}&nbsp;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_banner(status, detail=""):

    color = STATUS_COLORS.get(
        status,
        "#8d9bab"
    )

    st.markdown(
        f"""
        <div class="status-banner" style="border-left-color:{color}">
            <div class="label">Machine condition</div>
            <div class="value" style="color:{color}">{status}</div>
            <div class="small-muted">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def results_table(df, cols=None, height=None):

    cols = [
        column
        for column in (cols or df.columns)
        if column in df.columns
    ]

    pct = lambda label: st.column_config.ProgressColumn(
        label,
        min_value=0,
        max_value=100,
        format="%.1f"
    )

    extra = (
        {"height": height}
        if height
        else {}
    )

    st.dataframe(
        df[cols],
        **STRETCH,
        hide_index=True,
        **extra,
        column_config={
            "Health score": pct("Health score"),
            "Failure probability (%)": pct("Failure probability (%)"),
            "Anomaly percentile (%)": pct("Anomaly percentile (%)"),
        },
    )


# FIXED NAVIGATION:
# Do not directly modify the state of a widget after it is created.
# Queue navigation and apply it before the radio widget is instantiated.

def goto(page):

    st.session_state["next_page"] = page


def has_data():

    if st.session_state.get("data") is None:

        st.markdown(
            """
            <div class="panel">
                <h3>No dataset yet</h3>
                <div class="small-muted">
                    Upload a machine sensor CSV and run the analysis.
                    This page then shows results for your own data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.button(
            "Go to upload",
            type="primary",
            on_click=goto,
            args=("Upload & Analyse",)
        )

        return False

    return True


def diag_cols(df):

    return [
        column
        for column in df.columns
        if column.endswith(" probability (%)")
        and column != "Failure probability (%)"
    ]


def mode_name(col):

    return col.replace(
        " probability (%)",
        ""
    )


def slider_range(feat):

    lo, hi, step = FEATURE_RANGES[feat]

    data = st.session_state.get("data")

    if data is not None and feat in data.columns:

        lo = min(
            lo,
            float(np.floor(data[feat].min()))
        )

        hi = max(
            hi,
            float(np.ceil(data[feat].max()))
        )

    return lo, hi, step


# ============================================================
# 6. PAGE: UPLOAD AND ANALYSE
# ============================================================

def page_upload():

    if st.session_state.get("flash"):
        st.success(
            st.session_state.pop("flash")
        )

    st.markdown(
        "Upload a CSV of machine sensor readings. "
        "The app scores every row for failure risk, anomalies "
        "and health condition, then builds the dashboard from your file."
    )

    left, right = st.columns([2, 1])

    with right:

        template = pd.DataFrame(
            [[300.0, 310.0, 1500, 40.0, 100]],
            columns=FEATURES
        )

        template.insert(
            0,
            "Machine ID",
            "M-001"
        )

        st.download_button(
            "Download CSV template",
            template.to_csv(index=False),
            "sensor_template.csv",
            "text/csv",
            **STRETCH
        )

        st.caption(
            "Required: "
            + ", ".join(FEATURES)
            + ". An ID column is optional."
        )

    with left:

        uploaded = st.file_uploader(
            "Machine sensor CSV",
            type=["csv"]
        )

    if uploaded is None:

        if st.session_state.get("data") is not None:

            meta = st.session_state["meta"]

            st.info(
                f"Current dataset: {meta['name']} "
                f"({meta['analysed']:,} machines). "
                "Upload another file to replace it."
            )

            st.button(
                "Open dashboard",
                type="primary",
                on_click=goto,
                args=("Dashboard",)
            )

        return

    try:

        raw = pd.read_csv(
            uploaded,
            sep=None,
            engine="python"
        )

    except Exception as err:

        st.error(
            f"Could not read this file as CSV: {err}"
        )

        return

    st.caption(
        f"{len(raw):,} rows · {raw.shape[1]} columns"
    )

    st.dataframe(
        raw.head(10),
        **STRETCH,
        hide_index=True
    )

    mapping = auto_map(raw.columns)

    missing = [
        feature
        for feature in FEATURES
        if feature not in mapping
    ]

    options = list(raw.columns)

    with st.expander(
        "Match columns",
        expanded=bool(missing)
    ):

        if missing:

            st.warning(
                "Some sensor columns were not found. "
                "Choose the matching column for each."
            )

        cols = st.columns(len(FEATURES))

        for col, feat in zip(cols, FEATURES):

            default = (
                options.index(mapping[feat])
                if feat in mapping
                else None
            )

            pick = col.selectbox(
                feat,
                options,
                index=default,
                placeholder="Select column",
                key=f"map_{feat}_{uploaded.name}"
            )

            if pick:
                mapping[feat] = pick

        guess = detect_id(raw.columns)

        id_choice = st.selectbox(
            "Machine ID column",
            ["(use row number)"] + options,
            index=(options.index(guess) + 1) if guess else 0,
        )

    id_col = (
        None
        if id_choice == "(use row number)"
        else id_choice
    )

    if any(
        feature not in mapping
        for feature in FEATURES
    ):

        st.error(
            "Match all five sensor columns to continue."
        )

        return

    if st.button(
        "Analyse dataset",
        type="primary"
    ):

        with st.spinner(
            "Running diagnostics on every machine..."
        ):

            res, dropped = analyse_frame(
                raw,
                mapping,
                id_col
            )

        if res is None:

            st.error(
                "No valid numeric rows were found in the matched columns."
            )

            return

        st.session_state["data"] = res

        st.session_state["meta"] = dict(
            name=uploaded.name,
            rows=len(raw),
            analysed=len(res),
            dropped=dropped
        )

        st.session_state["wi_seed"] = None

        msg = (
            f"Analysed {len(res):,} machines from {uploaded.name}."
        )

        if dropped:

            msg += (
                f" Skipped {dropped:,} rows with missing "
                "or non-numeric sensor values."
            )

        st.session_state["flash"] = msg

        # Queue the page change; do not directly edit the radio state.
        goto("Dashboard")

        st.rerun()


# ============================================================
# 7. PAGE: DASHBOARD
# ============================================================

def page_dashboard():

    if not has_data():
        return

    d = st.session_state["data"]
    meta = st.session_state["meta"]

    n = len(d)

    at_risk = int(
        d["At risk"].sum()
    )

    predicted = int(
        (d["Failure prediction"] == "Failure risk").sum()
    )

    anomalies = int(
        (d["Anomaly"] == "Anomaly").sum()
    )

    avg = float(
        d["Health score"].mean()
    )

    cond = (
        d["Health status"]
        .value_counts()
        .reindex(STATUS_ORDER)
        .fillna(0)
        .astype(int)
    )

    if st.session_state.get("flash"):

        st.success(
            st.session_state.pop("flash")
        )

    st.caption(
        f"Dataset: {meta['name']} · {n:,} machines analysed"
    )

    risk_color = (
        STATUS_COLORS["Critical"]
        if at_risk / n > 0.1
        else STATUS_COLORS["Warning"]
        if at_risk
        else STATUS_COLORS["Healthy"]
    )

    t = ASSETS["health"]["health_thresholds"]

    avg_status = (
        "Healthy"
        if avg >= t["healthy"]
        else "Monitor"
        if avg >= t["monitor"]
        else "Warning"
        if avg >= t["warning"]
        else "Critical"
    )

    cols = st.columns(5)

    kpi(
        cols[0],
        "Machines analysed",
        f"{n:,}",
        meta["name"]
    )

    kpi(
        cols[1],
        "Machines at risk",
        f"{at_risk:,}",
        f"{at_risk / n * 100:.1f}% of machines",
        risk_color
    )

    kpi(
        cols[2],
        "Predicted failures",
        f"{predicted:,}",
        f"{predicted / n * 100:.1f}% of machines",
        STATUS_COLORS["Critical"]
        if predicted
        else STATUS_COLORS["Healthy"]
    )

    kpi(
        cols[3],
        "Anomalies",
        f"{anomalies:,}",
        "unusual sensor patterns",
        STATUS_COLORS["Warning"]
        if anomalies
        else STATUS_COLORS["Healthy"]
    )

    kpi(
        cols[4],
        "Average health",
        f"{avg:.1f}",
        "out of 100",
        STATUS_COLORS[avg_status]
    )

    st.write("")

    st.markdown(
        "**Health condition of machines**"
    )

    cols = st.columns(4)

    for col, status in zip(cols, STATUS_ORDER):

        kpi(
            col,
            status,
            f"{int(cond[status]):,}",
            f"{cond[status] / n * 100:.1f}% of machines",
            STATUS_COLORS[status]
        )

    st.write("")

    a, b, c = st.columns([1, 1.2, 1.2])

    with a:

        st.plotly_chart(
            gauge(
                avg,
                "Average health score"
            ),
            **STRETCH
        )

    with b:

        cdf = cond.reset_index()

        cdf.columns = [
            "Condition",
            "Machines"
        ]

        fig = px.pie(
            cdf,
            names="Condition",
            values="Machines",
            hole=0.6,
            color="Condition",
            color_discrete_map=STATUS_COLORS,
            category_orders={
                "Condition": STATUS_ORDER
            },
            title="Condition split"
        )

        fig.update_traces(
            textinfo="value+percent",
            sort=False,
            textfont=dict(
                family="JetBrains Mono",
                size=13
            )
        )

        show(fig, 250)

    with c:

        pr = (
            d["Priority"]
            .value_counts()
            .reindex(PRIORITY_ORDER)
            .fillna(0)
            .reset_index()
        )

        pr.columns = [
            "Priority",
            "Machines"
        ]

        fig = px.bar(
            pr,
            x="Priority",
            y="Machines",
            color="Priority",
            text="Machines",
            color_discrete_map=PRIORITY_COLORS,
            title="Machines needing action"
        )

        fig.update_layout(
            showlegend=False
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(
                family="JetBrains Mono",
                color=CHART_TXT
            )
        )

        show(fig, 250)

    tabs = [
        "Risk map",
        "Health distribution",
        "Sensor profile",
        "Failure modes",
        "Highest risk"
    ]

    has_actual = (
        "Machine failure" in d.columns
    )

    if has_actual:
        tabs.append("Model vs actual")

    tab_objs = st.tabs(tabs)

    with tab_objs[0]:

        fig = px.scatter(
            d,
            x="Anomaly percentile (%)",
            y="Failure probability (%)",
            color="Health status",
            color_discrete_map=STATUS_COLORS,
            category_orders={
                "Health status": STATUS_ORDER
            },
            hover_data=["Machine ID"] + FEATURES,
            opacity=0.8,
        )

        fig.add_hline(
            y=float(ASSETS["metadata"]["threshold"]) * 100,
            line_dash="dash",
            line_color="#8d9bab",
            annotation_text="decision threshold"
        )

        show(fig, 400)

    with tab_objs[1]:

        fig = px.histogram(
            d,
            x="Health score",
            color="Health status",
            nbins=30,
            color_discrete_map=STATUS_COLORS,
            category_orders={
                "Health status": STATUS_ORDER
            }
        )

        show(fig, 360)

    with tab_objs[2]:

        sensor = st.selectbox(
            "Sensor",
            FEATURES,
            index=3,
            key="dash_sensor"
        )

        fig = px.box(
            d,
            x="Health status",
            y=sensor,
            color="Health status",
            color_discrete_map=STATUS_COLORS,
            category_orders={
                "Health status": STATUS_ORDER
            }
        )

        fig.update_layout(
            showlegend=False
        )

        show(fig, 360)

    with tab_objs[3]:

        modes = diag_cols(d)

        if modes:

            flagged = pd.DataFrame({
                "Failure mode": [
                    mode_name(m) for m in modes
                ],
                "Machines flagged": [
                    int((d[m] >= 50).sum())
                    for m in modes
                ],
            })

            fig = px.bar(
                flagged,
                x="Failure mode",
                y="Machines flagged",
                text="Machines flagged",
                title=(
                    "Machines with a failure-mode "
                    "probability of 50% or more"
                )
            )

            fig.update_traces(
                marker_color=STATUS_COLORS["Warning"],
                textposition="outside"
            )

            show(fig, 340)

        else:

            st.info(
                "No failure-mode outputs available."
            )

    with tab_objs[4]:

        top = (
            d.sort_values(
                "Failure probability (%)",
                ascending=False
            ).head(15)
        )

        results_table(
            top,
            [
                "Machine ID",
                "Priority",
                "Health status",
                "Health score",
                "Failure probability (%)",
                "Anomaly"
            ] + FEATURES
        )

    if has_actual:

        with tab_objs[5]:

            actual = pd.to_numeric(
                d["Machine failure"],
                errors="coerce"
            )

            ok = actual.notna()

            y = actual[ok].astype(int)

            p = (
                d.loc[ok, "Failure prediction"]
                == "Failure risk"
            ).astype(int)

            tp = int(((y == 1) & (p == 1)).sum())
            fp = int(((y == 0) & (p == 1)).sum())
            fn = int(((y == 1) & (p == 0)).sum())
            tn = int(((y == 0) & (p == 0)).sum())

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Recall",
                f"{tp / max(tp + fn, 1) * 100:.1f}%"
            )

            m2.metric(
                "Precision",
                f"{tp / max(tp + fp, 1) * 100:.1f}%"
            )

            m3.metric(
                "Accuracy",
                f"{(tp + tn) / max(len(y), 1) * 100:.1f}%"
            )

            fig = px.imshow(
                [[tn, fp], [fn, tp]],
                text_auto=True,
                x=[
                    "Predicted OK",
                    "Predicted failure"
                ],
                y=[
                    "Actually OK",
                    "Actually failed"
                ],
                color_continuous_scale="Blues",
                title="Confusion matrix on your file"
            )

            show(fig, 320)


# ============================================================
# 8. PAGE: MACHINE RISK
# ============================================================

def page_machine_risk():

    if not has_data():
        return

    d = st.session_state["data"]

    f1, f2, f3 = st.columns([2, 1.5, 1])

    pick = f1.multiselect(
        "Health condition",
        STATUS_ORDER,
        default=STATUS_ORDER
    )

    query = f2.text_input(
        "Search machine ID",
        placeholder="e.g. M-014"
    )

    only_risk = f3.toggle(
        "At risk only",
        value=True
    )

    view = d[
        d["Health status"].isin(pick)
    ]

    if only_risk:
        view = view[view["At risk"]]

    if query:

        view = view[
            view["Machine ID"].str.contains(
                query,
                case=False,
                regex=False
            )
        ]

    view = view.sort_values(
        "Failure probability (%)",
        ascending=False
    )

    c = st.columns(4)

    kpi(
        c[0],
        "Machines shown",
        f"{len(view):,}",
        f"of {len(d):,}"
    )

    kpi(
        c[1],
        "Avg failure probability",
        f"{view['Failure probability (%)'].mean():.1f}%"
        if len(view) else "-",
        "",
        STATUS_COLORS["Warning"]
    )

    kpi(
        c[2],
        "Avg health",
        f"{view['Health score'].mean():.1f}"
        if len(view) else "-",
        "out of 100"
    )

    kpi(
        c[3],
        "Anomalies",
        f"{int((view['Anomaly'] == 'Anomaly').sum()):,}"
        if len(view) else "-",
        "",
        STATUS_COLORS["Monitor"]
    )

    st.write("")

    if view.empty:

        st.info(
            "No machines match these filters."
        )

        return

    results_table(
        view,
        [
            "Machine ID",
            "Priority",
            "Health status",
            "Health score",
            "Failure probability (%)",
            "Failure prediction",
            "Anomaly",
            "Anomaly percentile (%)"
        ] + FEATURES,
        height=360
    )

    st.subheader("Machine detail")

    ids = list(
        view.index[:500]
    )

    idx = st.selectbox(
        "Select a machine",
        ids,
        format_func=lambda i:
            f"{d.loc[i, 'Machine ID']} · "
            f"{d.loc[i, 'Health status']} · "
            f"{d.loc[i, 'Failure probability (%)']:.1f}% risk",
    )

    r = d.loc[idx]

    g1, g2, g3 = st.columns(3)

    g1.plotly_chart(
        gauge(
            r["Health score"],
            "Health score"
        ),
        **STRETCH
    )

    g2.plotly_chart(
        gauge(
            r["Failure probability (%)"],
            "Failure probability",
            True
        ),
        **STRETCH
    )

    g3.plotly_chart(
        gauge(
            r["Anomaly percentile (%)"],
            "Anomaly percentile",
            True
        ),
        **STRETCH
    )

    status_banner(
        r["Health status"],
        f"{r['Failure prediction']} · "
        f"{r['Anomaly']} · "
        f"{r['Suggested action']}"
    )

    a, b = st.columns(2)

    with a:

        pct = [
            (d[feature] < r[feature]).mean() * 100
            for feature in FEATURES
        ]

        fig = go.Figure(
            go.Bar(
                x=pct,
                y=[
                    f.split(" [")[0]
                    for f in FEATURES
                ],
                orientation="h",
                marker_color=STATUS_COLORS["Monitor"],
                text=[
                    f"{r[f]:.1f}"
                    for f in FEATURES
                ],
                textposition="inside",
                textfont=dict(
                    family="JetBrains Mono",
                    color="#ffffff"
                ),
            )
        )

        fig.update_layout(
            title="Reading vs rest of the dataset (percentile)"
        )

        fig.update_xaxes(
            range=[0, 100]
        )

        show(fig, 300)

    with b:

        modes = diag_cols(d)

        if modes:

            mdf = pd.DataFrame({
                "Failure mode": [
                    mode_name(m) for m in modes
                ],
                "Probability (%)": [
                    float(r[m]) for m in modes
                ]
            })

            fig = px.bar(
                mdf.sort_values("Probability (%)"),
                x="Probability (%)",
                y="Failure mode",
                orientation="h",
                range_x=[0, 100],
                title="Failure mode probabilities"
            )

            fig.update_traces(
                marker_color=STATUS_COLORS["Warning"]
            )

            show(fig, 300)


# ============================================================
# 9. PAGE: FAILURE DIAGNOSIS
# ============================================================

def page_diagnosis():

    if not has_data():
        return

    d = st.session_state["data"]

    modes = diag_cols(d)

    if not modes:

        st.info(
            "No failure-mode outputs available."
        )

        return

    thr = st.slider(
        "Flag a failure mode when its probability is at least (%)",
        5,
        95,
        50,
        5
    )

    counts = pd.DataFrame({
        "Failure mode": [
            mode_name(m) for m in modes
        ],
        "Machines flagged": [
            int((d[m] >= thr).sum())
            for m in modes
        ],
        "Average probability (%)": [
            float(d[m].mean())
            for m in modes
        ],
        "Highest probability (%)": [
            float(d[m].max())
            for m in modes
        ],
    })

    cols = st.columns(len(modes))

    for col, (_, row) in zip(
        cols,
        counts.iterrows()
    ):

        kpi(
            col,
            row["Failure mode"],
            f"{int(row['Machines flagged']):,}",
            "machines flagged",
            STATUS_COLORS["Critical"]
            if row["Machines flagged"]
            else STATUS_COLORS["Healthy"]
        )

    st.write("")

    a, b = st.columns(2)

    with a:

        fig = px.bar(
            counts,
            x="Failure mode",
            y="Machines flagged",
            text="Machines flagged",
            title="Machines flagged per failure mode"
        )

        fig.update_traces(
            marker_color=STATUS_COLORS["Critical"],
            textposition="outside"
        )

        show(fig, 320)

    with b:

        fig = px.bar(
            counts,
            x="Failure mode",
            y="Average probability (%)",
            title="Average probability across all machines"
        )

        fig.update_traces(
            marker_color=STATUS_COLORS["Warning"]
        )

        show(fig, 320)

    top = (
        d.sort_values(
            "Failure probability (%)",
            ascending=False
        ).head(25)
    )

    heat = top.set_index("Machine ID")[modes]

    heat.columns = [
        mode_name(m) for m in modes
    ]

    fig = px.imshow(
        heat,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=100,
        title=(
            "Failure-mode probabilities for "
            "the 25 highest-risk machines"
        )
    )

    show(fig, 560)

    st.caption(
        "Rare failure modes such as TWF and RNF have few "
        "training examples and may be unreliable."
    )


# ============================================================
# 10. PAGE: ANOMALY DETECTION
# ============================================================

def page_anomaly():

    if not has_data():
        return

    d = st.session_state["data"]

    flagged = (
        d["Anomaly"] == "Anomaly"
    )

    n_anom = int(
        flagged.sum()
    )

    c = st.columns(3)

    kpi(
        c[0],
        "Anomalies",
        f"{n_anom:,}",
        "unusual patterns",
        STATUS_COLORS["Warning"]
        if n_anom
        else STATUS_COLORS["Healthy"]
    )

    kpi(
        c[1],
        "Normal patterns",
        f"{len(d) - n_anom:,}",
        "",
        STATUS_COLORS["Healthy"]
    )

    kpi(
        c[2],
        "Anomaly rate",
        f"{n_anom / len(d) * 100:.1f}%",
        "of machines"
    )

    st.write("")

    x1, x2 = st.columns(2)

    xf = x1.selectbox(
        "X axis",
        FEATURES,
        index=3
    )

    yf = x2.selectbox(
        "Y axis",
        FEATURES,
        index=2
    )

    plot = d.assign(
        Pattern=np.where(
            flagged,
            "Anomaly",
            "Normal"
        )
    )

    cmap = {
        "Normal": STATUS_COLORS["Monitor"],
        "Anomaly": STATUS_COLORS["Critical"]
    }

    fig = px.scatter(
        plot,
        x=xf,
        y=yf,
        color="Pattern",
        color_discrete_map=cmap,
        hover_data=[
            "Machine ID",
            "Health score"
        ],
        opacity=0.8
    )

    show(fig, 420)

    fig = px.histogram(
        plot,
        x="Anomaly percentile (%)",
        color="Pattern",
        nbins=30,
        color_discrete_map=cmap,
        title="Anomaly percentile distribution"
    )

    show(fig, 280)

    if n_anom:

        with st.expander(
            f"Machines with anomalies ({n_anom:,})"
        ):

            results_table(
                d[flagged].sort_values(
                    "Anomaly percentile (%)",
                    ascending=False
                ),
                [
                    "Machine ID",
                    "Health status",
                    "Health score",
                    "Anomaly percentile (%)",
                    "Failure probability (%)"
                ] + FEATURES
            )

    st.caption(
        "An anomaly is not automatically a machine failure."
    )


# ============================================================
# 11. PAGE: HEALTH ANALYTICS
# ============================================================

def page_health():

    if not has_data():
        return

    d = st.session_state["data"]

    a, b = st.columns(2)

    with a:

        fig = px.histogram(
            d,
            x="Health score",
            color="Health status",
            nbins=30,
            color_discrete_map=STATUS_COLORS,
            category_orders={
                "Health status": STATUS_ORDER
            },
            title="Health score distribution"
        )

        show(fig)

    with b:

        fig = px.box(
            d,
            x="Health status",
            y="Health score",
            color="Health status",
            color_discrete_map=STATUS_COLORS,
            category_orders={
                "Health status": STATUS_ORDER
            },
            title="Score range by condition"
        )

        fig.update_layout(
            showlegend=False
        )

        show(fig)

    sensor = st.selectbox(
        "Sensor vs health score",
        FEATURES,
        index=4
    )

    fig = px.scatter(
        d,
        x=sensor,
        y="Health score",
        color="Health status",
        color_discrete_map=STATUS_COLORS,
        category_orders={
            "Health status": STATUS_ORDER
        },
        hover_data=["Machine ID"],
        opacity=0.7
    )

    show(fig, 360)

    group_cols = [
        c for c in d.columns
        if c.lower() in (
            "type",
            "product type",
            "machine type"
        )
    ]

    if group_cols:

        group = group_cols[0]

        grp = (
            d.groupby(
                [group, "Health status"]
            )
            .size()
            .reset_index(name="Machines")
        )

        fig = px.bar(
            grp,
            x=group,
            y="Machines",
            color="Health status",
            color_discrete_map=STATUS_COLORS,
            category_orders={
                "Health status": STATUS_ORDER
            },
            title=f"Health condition by {group}"
        )

        show(fig, 320)

    corr_cols = FEATURES + [
        "Health score",
        "Failure probability (%)",
        "Anomaly percentile (%)"
    ]

    corr = d[corr_cols].corr()

    corr.index = corr.columns = [
        c.split(" [")[0]
        for c in corr_cols
    ]

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation between sensors and outcomes"
    )

    show(fig, 460)

    st.caption(
        "The health score blends failure risk and anomaly percentile "
        "with prototype weights. It is not a validated industrial "
        "standard or a remaining-useful-life estimate."
    )


# ============================================================
# 12. PAGE: EXPLAINABLE AI
# ============================================================

def page_xai():

    d = st.session_state.get("data")

    if d is not None:

        st.subheader(
            "Sensor influence in your dataset"
        )

        corr = (
            d[FEATURES + ["Failure probability (%)"]]
            .corr()["Failure probability (%)"]
            .drop("Failure probability (%)")
        )

        cdf = pd.DataFrame({
            "Sensor": [
                f.split(" [")[0]
                for f in corr.index
            ],
            "Correlation": corr.values
        })

        cdf = cdf.reindex(
            cdf["Correlation"]
            .abs()
            .sort_values()
            .index
        )

        fig = px.bar(
            cdf,
            x="Correlation",
            y="Sensor",
            orientation="h",
            range_x=[-1, 1],
            title=(
                "Correlation of each sensor with "
                "predicted failure probability"
            )
        )

        fig.update_traces(
            marker_color=STATUS_COLORS["Warning"]
        )

        show(fig, 300)

        st.caption(
            "This describes your uploaded data. "
            "It is a simple correlation, not a SHAP value."
        )

    else:

        has_data()

    st.subheader(
        "Model explainability from training (SHAP)"
    )

    st.caption(
        "These figures were produced when the model was trained "
        "and describe the model, not this upload."
    )

    images = [
        ("shap_summary.png", "Global summary"),
        ("shap_beeswarm.png", "Feature impact distribution"),
        ("shap_local_explanation.png", "Local prediction explanation")
    ]

    found = [
        (filename, title)
        for filename, title in images
        if os.path.exists(
            os.path.join(REPORT_DIR, filename)
        )
    ]

    if found:

        tabs = st.tabs([
            title for _, title in found
        ])

        for tab, (filename, _) in zip(tabs, found):

            with tab:

                st.image(
                    os.path.join(REPORT_DIR, filename),
                    **STRETCH
                )

    else:

        st.info(
            "No SHAP images found in the reports folder."
        )

    st.caption(
        "SHAP explains learned model associations, not physical causation."
    )


# ============================================================
# 13. PAGE: MAINTENANCE PLANNER
# ============================================================

def page_planner():

    if not has_data():
        return

    d = st.session_state["data"]

    tasks = d[
        d["Priority"] != "OK"
    ].copy()

    counts = tasks["Priority"].value_counts()

    cols = st.columns(4)

    for col, priority in zip(
        cols,
        PRIORITY_ORDER
    ):

        kpi(
            col,
            priority,
            f"{int(counts.get(priority, 0)):,}",
            ACTIONS[priority],
            PRIORITY_COLORS[priority]
        )

    st.write("")

    if tasks.empty:

        st.success(
            "No maintenance tasks. Every machine is in good condition."
        )

        return

    pick = st.multiselect(
        "Priority",
        PRIORITY_ORDER,
        default=PRIORITY_ORDER
    )

    tasks = tasks[
        tasks["Priority"].isin(pick)
    ]

    tasks["rank"] = (
        tasks["Priority"].map({
            p: i
            for i, p in enumerate(PRIORITY_ORDER)
        })
    )

    tasks = tasks.sort_values(
        ["rank", "Failure probability (%)"],
        ascending=[True, False]
    )

    show_cols = [
        "Machine ID",
        "Priority",
        "Health status",
        "Failure probability (%)",
        "Suggested action"
    ]

    table = (
        tasks[show_cols]
        .head(300)
        .copy()
    )

    table["Done"] = False

    edited = st.data_editor(
        table,
        hide_index=True,
        **STRETCH,
        disabled=show_cols,
        column_config={
            "Failure probability (%)":
                st.column_config.NumberColumn(
                    format="%.2f"
                )
        },
    )

    st.caption(
        f"{int(edited['Done'].sum())} of "
        f"{len(edited)} tasks ticked off in this session."
    )

    st.download_button(
        "Export open tasks",
        edited[
            ~edited["Done"]
        ].drop(columns="Done").to_csv(index=False),
        "maintenance_plan.csv",
        "text/csv"
    )

    st.caption(
        "Alerts are advisory. An engineer must verify "
        "the condition before maintenance action."
    )


# ============================================================
# 14. PAGE: WHAT-IF SIMULATOR
# ============================================================

def page_whatif():

    st.warning(
        "This is a model-based scenario comparison, "
        "not a physical simulation or causal experiment."
    )

    d = st.session_state.get("data")

    seeds = {
        "Manual values": None
    }

    if d is not None:

        for i, r in (
            d.sort_values(
                "Failure probability (%)",
                ascending=False
            ).head(100).iterrows()
        ):

            seeds[
                f"{r['Machine ID']} · "
                f"{r['Health status']} · "
                f"{r['Failure probability (%)']:.1f}% risk "
                f"(row {i})"
            ] = i

    choice = st.selectbox(
        "Start from",
        list(seeds)
    )

    if st.session_state.get("wi_seed") != choice:

        for feat in FEATURES:

            lo, hi, _ = slider_range(feat)

            default = (
                (lo + hi) / 2
                if seeds[choice] is None
                else float(d.loc[seeds[choice], feat])
            )

            if seeds[choice] is None:

                default = {
                    FEATURES[0]: 300.0,
                    FEATURES[1]: 310.0,
                    FEATURES[2]: 1500.0,
                    FEATURES[3]: 40.0,
                    FEATURES[4]: 100.0
                }[feat]

            val = float(
                min(max(default, lo), hi)
            )

            st.session_state[f"wb_{feat}"] = val
            st.session_state[f"ws_{feat}"] = val

        st.session_state["wi_seed"] = choice

    def reset_scenario():

        for feat in FEATURES:

            st.session_state[f"ws_{feat}"] = (
                st.session_state[f"wb_{feat}"]
            )

    left, right = st.columns(2)

    base_vals = []
    scen_vals = []

    with left:

        st.subheader("Baseline")

        for feat in FEATURES:

            lo, hi, step = slider_range(feat)

            base_vals.append(
                st.slider(
                    feat,
                    lo,
                    hi,
                    step=step,
                    key=f"wb_{feat}"
                )
            )

    with right:

        st.subheader("Scenario")

        for feat in FEATURES:

            lo, hi, step = slider_range(feat)

            scen_vals.append(
                st.slider(
                    feat,
                    lo,
                    hi,
                    step=step,
                    key=f"ws_{feat}"
                )
            )

        st.button(
            "Reset scenario to baseline",
            on_click=reset_scenario
        )

    base = assess(
        pd.DataFrame(
            [base_vals],
            columns=FEATURES
        ),
        ASSETS
    ).iloc[0]

    scen = assess(
        pd.DataFrame(
            [scen_vals],
            columns=FEATURES
        ),
        ASSETS
    ).iloc[0]

    st.divider()

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Failure probability",
        f"{scen['Failure probability (%)']:.2f}%",
        f"{scen['Failure probability (%)'] - base['Failure probability (%)']:+.2f} pts",
        delta_color="inverse"
    )

    m2.metric(
        "Health score",
        f"{scen['Health score']:.1f}",
        f"{scen['Health score'] - base['Health score']:+.1f}"
    )

    m3.metric(
        "Anomaly percentile",
        f"{scen['Anomaly percentile (%)']:.1f}%",
        f"{scen['Anomaly percentile (%)'] - base['Anomaly percentile (%)']:+.1f} pts",
        delta_color="inverse"
    )

    measures = [
        "Failure probability (%)",
        "Health score",
        "Anomaly percentile (%)"
    ]

    fig = go.Figure()

    fig.add_bar(
        name="Baseline",
        x=measures,
        y=[base[m] for m in measures],
        marker_color="#8d9bab"
    )

    fig.add_bar(
        name="Scenario",
        x=measures,
        y=[scen[m] for m in measures],
        marker_color=STATUS_COLORS["Warning"]
    )

    fig.update_layout(
        barmode="group"
    )

    show(fig, 300)

    st.subheader("Sensitivity sweep")

    sweep_feat = st.selectbox(
        "Sensor to sweep",
        FEATURES,
        index=3
    )

    lo, hi, _ = slider_range(sweep_feat)

    grid = np.linspace(
        lo,
        hi,
        60
    )

    sweep_df = pd.DataFrame(
        [scen_vals] * len(grid),
        columns=FEATURES
    )

    sweep_df[sweep_feat] = grid

    sweep = assess(
        sweep_df,
        ASSETS
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=sweep["Failure probability (%)"],
            name="Failure probability (%)",
            line=dict(
                color=STATUS_COLORS["Critical"]
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=sweep["Health score"],
            name="Health score",
            line=dict(
                color=STATUS_COLORS["Healthy"]
            )
        )
    )

    fig.add_vline(
        x=scen_vals[FEATURES.index(sweep_feat)],
        line_dash="dash",
        line_color="#8d9bab",
        annotation_text="scenario"
    )

    fig.update_xaxes(
        title=sweep_feat
    )

    show(fig, 360)


# ============================================================
# 15. PAGE: REPORTS AND EXPORT
# ============================================================

def page_reports():

    if has_data():

        d = st.session_state["data"]
        meta = st.session_state["meta"]

        st.subheader(
            "Results for your dataset"
        )

        out_cols = [
            c for c in d.columns
            if c != "At risk"
        ]

        summary = pd.DataFrame({

            "Measure": [
                "Machines analysed",
                "Machines at risk",
                "Predicted failures",
                "Anomalies",
                "Average health score"
            ] + [
                f"{s} machines"
                for s in STATUS_ORDER
            ],

            "Value": [
                len(d),
                int(d["At risk"].sum()),
                int(
                    (d["Failure prediction"] == "Failure risk").sum()
                ),
                int(
                    (d["Anomaly"] == "Anomaly").sum()
                ),
                round(
                    float(d["Health score"].mean()),
                    2
                )
            ] + [
                int(
                    (d["Health status"] == s).sum()
                )
                for s in STATUS_ORDER
            ],

        })

        c1, c2, c3 = st.columns(3)

        c1.download_button(
            "Full analysis (CSV)",
            d[out_cols].to_csv(index=False),
            f"analysis_{meta['name']}",
            "text/csv",
            **STRETCH
        )

        c2.download_button(
            "At-risk machines (CSV)",
            d[d["At risk"]][out_cols].to_csv(index=False),
            "at_risk_machines.csv",
            "text/csv",
            **STRETCH
        )

        c3.download_button(
            "Summary (CSV)",
            summary.to_csv(index=False),
            "analysis_summary.csv",
            "text/csv",
            **STRETCH
        )

        st.dataframe(
            summary,
            hide_index=True,
            **STRETCH
        )

    reports = [
        "optimized_model_comparison.csv",
        "optimized_test_predictions.csv",
        "optimized_confusion_matrix.csv",
        "failure_diagnosis_results.csv",
        "anomaly_detection_results.csv",
        "machine_health_scores.csv",
        "shap_feature_importance.csv",
    ]

    available = [
        f for f in reports
        if os.path.exists(
            os.path.join(REPORT_DIR, f)
        )
    ]

    if available:

        with st.expander("Model training reports"):

            for filename in available:

                with open(
                    os.path.join(REPORT_DIR, filename),
                    "rb"
                ) as fh:

                    st.download_button(
                        f"Download {filename}",
                        fh.read(),
                        filename,
                        key=filename
                    )


# ============================================================
# 16. PAGE ROUTING
# ============================================================

PAGE_FUNCS = {
    "Upload & Analyse": page_upload,
    "Dashboard": page_dashboard,
    "Machine Risk": page_machine_risk,
    "Failure Diagnosis": page_diagnosis,
    "Anomaly Detection": page_anomaly,
    "Health Analytics": page_health,
    "Explainable AI": page_xai,
    "Maintenance Planner": page_planner,
    "What-if Simulator": page_whatif,
    "Reports & Export": page_reports,
}


# ============================================================
# 17. INITIALISE MODELS
# ============================================================

try:

    ASSETS = load_models()

except Exception as error:

    st.error(
        f"Model loading error: {error}"
    )

    st.info(
        "Check that all five trained model files "
        "are in the models folder."
    )

    st.stop()


# ============================================================
# 18. FIXED STREAMLIT NAVIGATION
# ============================================================

# IMPORTANT:
# Apply queued navigation BEFORE creating the radio widget.
# This prevents:
# StreamlitAPIException:
# st.session_state.page cannot be modified after widget creation.

if "next_page" in st.session_state:

    st.session_state["page"] = (
        st.session_state.pop("next_page")
    )


# ============================================================
# 19. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Predictive Maintenance</div>'
        '<div class="brand-sub">'
        'Upload a dataset, find risky machines'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Workspace",
        PAGES,
        key="page",
        label_visibility="collapsed"
    )

    st.divider()

    st.toggle(
        "☀️ Light theme",
        key="light_theme",
        help="Switch between dark and light appearance"
    )

    st.divider()

    data = st.session_state.get("data")

    st.markdown(
        "**Current dataset**"
    )

    if data is None:

        st.caption(
            "No dataset uploaded yet."
        )

    else:

        s1, s2 = st.columns(2)

        s1.metric(
            "Machines",
            f"{len(data):,}"
        )

        s2.metric(
            "At risk",
            f"{int(data['At risk'].sum()):,}"
        )

        st.progress(
            float(
                np.clip(
                    data["Health score"].mean() / 100,
                    0,
                    1
                )
            ),
            text=(
                f"Average health "
                f"{data['Health score'].mean():.0f}/100"
            )
        )

        st.caption(
            st.session_state["meta"]["name"]
        )

        if st.button(
            "Clear dataset",
            **STRETCH
        ):

            for key in (
                "data",
                "meta",
                "wi_seed"
            ):

                st.session_state.pop(
                    key,
                    None
                )

            st.rerun()

    st.divider()

    st.caption(
        "Dataset reference: AI4I 2020 · v3.0"
    )


# ============================================================
# 20. DISPLAY SELECTED PAGE
# ============================================================

st.title(page)

st.caption(
    "Machine condition monitoring and maintenance decision support."
)

PAGE_FUNCS[page]()


# ============================================================
# 21. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#8d9bab;font-size:12px;">
        Predictive Maintenance Assistant ·
        Model predictions are advisory and require human validation.
    </div>
    """,
    unsafe_allow_html=True,
)

