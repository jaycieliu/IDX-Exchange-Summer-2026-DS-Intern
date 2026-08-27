from pathlib import Path
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "week6_feature_engineering" / "week6_deployment_pipeline_refit_through_june.joblib"
METRICS_PATH = ROOT / "outputs" / "week8_evaluation" / "metrics_summary.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "week8_evaluation" / "week8_xgboost_june_predictions.csv"
CLEANED_DATA_PATH = ROOT / "outputs" / "week3_preprocessing" / "crmls_sfr_quality_cleaned_202501_202606.csv"

# Required so joblib can load the custom pipeline classes saved with the model.
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "week6"))


st.set_page_config(
    page_title="California Home Price Review",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #0f172a;
        --muted: #64748b;
        --panel: #ffffff;
        --line: #e6e9f2;
        --bg: #f4f6fb;
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --primary-soft: #eff4ff;
        --teal: #0f766e;
        --amber: #b45309;
        --violet: #7c3aed;
        --radius-lg: 14px;
        --radius-md: 10px;
        --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
        --shadow-md: 0 10px 26px rgba(15, 23, 42, 0.07);
        --shadow-lg: 0 20px 44px rgba(15, 23, 42, 0.14);
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* ---------- App shell ---------- */
    .stApp {
        background:
            radial-gradient(1200px 480px at 12% -10%, rgba(37, 99, 235, 0.06), transparent 60%),
            var(--bg);
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }
    ::selection {
        background: rgba(37, 99, 235, 0.18);
    }
    *:focus-visible {
        outline: 2px solid var(--primary) !important;
        outline-offset: 2px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 45%, #132033 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0 0.15rem 1.1rem 0.15rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-brand .mark {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, #2563eb, #0f766e);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
        color: #fff;
        flex-shrink: 0;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.35);
    }
    .sidebar-brand .title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .sidebar-brand .subtitle {
        font-size: 0.72rem;
        color: #94a3b8;
        letter-spacing: 0.03em;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7c8aa5;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.2rem;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.6rem 0.75rem;
        margin: 0.1rem 0;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.07);
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(37, 99, 235, 0.22);
        border-color: rgba(96, 165, 250, 0.35);
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: #ffffff;
        font-weight: 650;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 {
        letter-spacing: -0.01em;
        color: var(--ink);
    }
    h3 {
        font-weight: 700;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, #0b1220 0%, #1f2937 55%, #0f766e 130%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius-lg);
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.3rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(420px 220px at 92% -20%, rgba(96, 165, 250, 0.25), transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 1.95rem;
        font-weight: 750;
        color: #ffffff;
        line-height: 1.14;
    }
    .hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 1.02rem;
        max-width: 760px;
        position: relative;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #93c5fd;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .eyebrow:before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #60a5fa;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.25);
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        padding: 1.05rem 1.1rem;
        min-height: 118px;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .metric-card:before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 3px;
        background: var(--accent);
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: var(--ink);
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 760;
        margin-top: 0.45rem;
        font-variant-numeric: tabular-nums;
    }
    .metric-caption {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.5rem;
    }

    /* ---------- Content cards / callouts ---------- */
    .content-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        padding: 1.2rem 1.35rem;
        box-shadow: var(--shadow-sm);
        margin: 0.75rem 0 1.1rem 0;
    }
    .content-card h3 {
        margin-top: 0;
    }
    .note, .success-note {
        border-radius: var(--radius-md);
        padding: 0.9rem 1.05rem;
        border-left-width: 3px;
        border-left-style: solid;
        margin: 0.75rem 0 1.1rem 0;
    }
    .note {
        background: #fffaf0;
        border: 1px solid #fde7c7;
        border-left-color: var(--amber);
        color: #7c2d12;
    }
    .success-note {
        background: #ecfdf5;
        border: 1px solid #b8f0d6;
        border-left-color: #059669;
        color: #064e3b;
    }

    /* ---------- Native Streamlit widgets ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        padding: 0.85rem;
        box-shadow: var(--shadow-sm);
    }
    .stButton > button, .stFormSubmitButton > button {
        background: var(--primary);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 620;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.28);
        transition: background 0.15s ease, transform 0.1s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: var(--primary-dark);
        transform: translateY(-1px);
        color: #ffffff;
    }
    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.25rem 0.4rem 1.25rem;
        box-shadow: var(--shadow-sm);
    }
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        border-color: var(--line) !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    .stAlert {
        border-radius: var(--radius-md);
    }
    hr {
        border-color: var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def metric_card(label, value, caption="", accent="#2563eb"):
    return f"""
    <div class="metric-card" style="--accent: {accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """


def render_metric_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(
            metric_card(
                item["label"],
                item["value"],
                item.get("caption", ""),
                item.get("accent", "#2563eb"),
            ),
            unsafe_allow_html=True,
        )


def plain_segment_name(segment):
    names = {
        "Q1_lowest": "Lowest-price group",
        "Q2": "Lower-middle group",
        "Q3": "Middle group",
        "Q4": "Upper-middle group",
        "Q5_highest": "Highest-price group",
    }
    return names.get(str(segment), str(segment))


@st.cache_resource
def load_prediction_model():
    if not MODEL_PATH.exists():
        return None
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"]
    return artifact


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(PREDICTIONS_PATH)
    frame["CloseDate"] = pd.to_datetime(frame["CloseDate"], errors="coerce")
    frame["absolute_percentage_error"] = pd.to_numeric(
        frame["absolute_percentage_error"], errors="coerce"
    )
    frame["ClosePrice"] = pd.to_numeric(frame["ClosePrice"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame["absolute_error"] = pd.to_numeric(frame["absolute_error"], errors="coerce")
    return frame


@st.cache_data
def load_market_history():
    if not CLEANED_DATA_PATH.exists():
        return pd.DataFrame()
    columns = [
        "CloseDate",
        "close_month",
        "ClosePrice",
        "ListPrice",
        "ClosePrice_to_ListPrice_ratio",
        "DaysOnMarket",
        "City",
        "CountyOrParish",
        "PostalCode",
        "Latitude",
        "Longitude",
        "LivingArea",
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "LotSizeSquareFeet",
        "YearBuilt",
        "PropertyType",
        "PropertySubType",
    ]
    frame = pd.read_csv(CLEANED_DATA_PATH, usecols=columns, low_memory=False)
    frame = frame[
        frame["PropertyType"].astype(str).str.strip().eq("Residential")
        & frame["PropertySubType"].astype(str).str.strip().eq("SingleFamilyResidence")
    ].copy()
    frame["CloseDate"] = pd.to_datetime(frame["CloseDate"], errors="coerce")
    for column in [
        "ClosePrice",
        "ListPrice",
        "ClosePrice_to_ListPrice_ratio",
        "DaysOnMarket",
        "LivingArea",
        "Latitude",
        "Longitude",
        "PostalCode",
        "YearBuilt",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # LivingArea has zero-valued records; dividing by them yields inf and breaks the price scales.
    frame["price_per_sqft"] = frame["ClosePrice"] / frame["LivingArea"].replace(0, np.nan)
    frame = frame.dropna(subset=["CloseDate", "ClosePrice"])
    return frame


def build_property_row(living_area, bedrooms, bathrooms, lot_size, city, county, postal_code):
    return pd.DataFrame(
        [
            {
                "LivingArea": living_area,
                "BedroomsTotal": bedrooms,
                "BathroomsTotalInteger": bathrooms,
                "LotSizeSquareFeet": lot_size,
                "City": city or pd.NA,
                "CountyOrParish": county or pd.NA,
                "PostalCode": postal_code or pd.NA,
                "PropertyType": "Residential",
                "PropertySubType": "SingleFamilyResidence",
            }
        ]
    )


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">CRMLS pricing intelligence</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_page(metrics, predictions):
    page_header(
        "California Home Price Review",
        "A decision-support demo for reviewing residential single-family sale prices.",
    )

    main = metrics.loc[metrics["section"].eq("main")]
    robust = metrics.loc[metrics["section"].eq("robustness")]
    main_row = main.iloc[0] if len(main) else None
    robust_row = robust.iloc[0] if len(robust) else None

    render_metric_row(
        [
            {
                "label": "Homes reviewed",
                "value": f"{int(main_row['rows']):,}" if main_row is not None else "N/A",
                "caption": "June test set",
                "accent": "#2563eb",
            },
            {
                "label": "Overall fit",
                "value": f"{main_row['r2']:.3f}" if main_row is not None else "N/A",
                "caption": "Captures most pricing variation",
                "accent": "#0f766e",
            },
            {
                "label": "Average error",
                "value": pct(main_row["mape"]) if main_row is not None else "N/A",
                "caption": "Affected by harder cases",
                "accent": "#b45309",
            },
            {
                "label": "Typical error",
                "value": pct(main_row["mdape"]) if main_row is not None else "N/A",
                "caption": "Normal prediction miss",
                "accent": "#7c3aed",
            },
        ]
    )

    st.markdown(
        """
        <div class="content-card">
            <h3 style="margin-top:0;">What this means</h3>
            <p>The model is strong enough to support pricing review. It can help flag homes where the expected price and actual or proposed price look far apart.</p>
            <p style="margin-bottom:0;">It should not set final listing prices by itself. Luxury homes, unusual homes, and very low-price homes still need comparable-sale review and local market judgment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if robust_row is not None:
        st.markdown(
            f"""
            <div class="note">
            <b>Robustness check:</b> when all June records are included, average error is {pct(robust_row["mape"])}
            and typical error is {pct(robust_row["mdape"])}. The lower overall fit shows why unusual records should be reviewed manually.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not predictions.empty:
        st.subheader("Main business use")
        use_cases = pd.DataFrame(
            {
                "Use case": [
                    "Price reasonableness check",
                    "Manual-review prioritization",
                    "Segment risk comparison",
                ],
                "How to use it": [
                    "Compare predicted price against listed or closed price.",
                    "Flag homes with large differences for analyst review.",
                    "Focus extra review on price ranges where errors are larger.",
                ],
            }
        )
        st.dataframe(use_cases, hide_index=True, width="stretch")


def prediction_page():
    page_header(
        "Predict A Home Price",
        "Enter property details to estimate a likely close price for pricing review.",
    )

    model = load_prediction_model()
    if model is None:
        st.error("The saved model artifact was not found. Rebuild the model output before using this page.")
        return

    st.markdown(
        """
        <div class="note">
        This page uses the saved Random Forest pipeline artifact because that is the deployable joblib model available in the repository.
        The project report still presents XGBoost as the final evaluation winner.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        with st.form("prediction_form"):
            st.subheader("Required inputs")
            living_area = st.number_input("Living area (sq ft)", 300, 20000, 1800, 50)
            bedrooms = st.number_input("Bedrooms", 0.0, 20.0, 3.0, 1.0)
            bathrooms = st.number_input("Bathrooms", 0.5, 20.0, 2.0, 0.5)
            lot_size = st.number_input("Lot size (sq ft)", 500, 500000, 6000, 100)

            st.subheader("Optional location context")
            city = st.text_input("City", "")
            county = st.text_input("County", "")
            postal_code = st.text_input("ZIP code", "")

            submitted = st.form_submit_button("Predict price")

    with right:
        st.subheader("How to read the estimate")
        st.write(
            "Treat the output as a screening number. A large gap between this estimate and a proposed price should trigger review, not an automatic decision."
        )
        st.write("The estimate is less reliable for unusual condition, luxury finishes, large acreage, or special location factors.")

    if submitted:
        row = build_property_row(living_area, bedrooms, bathrooms, lot_size, city, county, postal_code)
        prediction = float(model.predict(row)[0])
        price_per_sqft = prediction / living_area if living_area else 0

        st.markdown("---")
        render_metric_row(
            [
                {
                    "label": "Estimated close price",
                    "value": money(prediction),
                    "caption": "Model-generated review estimate",
                    "accent": "#2563eb",
                },
                {
                    "label": "Estimated price / sq ft",
                    "value": money(price_per_sqft),
                    "caption": "Derived from predicted price",
                    "accent": "#0f766e",
                },
                {
                    "label": "Review guidance",
                    "value": "Manual check",
                    "caption": "Use comparable sales before action",
                    "accent": "#b45309",
                },
            ]
        )

        st.markdown(
            """
            <div class="success-note">
            Use this result to decide whether a property needs deeper comparable-sale review.
            The app does not replace an agent, appraiser, or local market specialist.
            </div>
            """,
            unsafe_allow_html=True,
        )


PRICE_RAMP = [
    (37, 99, 235),
    (6, 148, 162),
    (15, 118, 110),
    (217, 119, 6),
    (190, 24, 93),
]

CA_BOUNDS = {"lat": (32.5, 42.1), "lon": (-124.5, -114.1)}
CA_ZIP_RANGE = (90001, 96162)


def days(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f} days"


MAP_METRICS = {
    "Median price per sq ft": ("median_ppsf", money),
    "Median sale price": ("median_price", money),
    "Median days on market": ("median_dom", days),
}


def build_zip_summary(frame):
    summary = (
        frame.dropna(subset=["PostalCode", "Latitude", "Longitude"])
        .groupby("PostalCode")
        .agg(
            city=("City", "first"),
            county=("CountyOrParish", "first"),
            homes=("ClosePrice", "size"),
            median_price=("ClosePrice", "median"),
            median_ppsf=("price_per_sqft", "median"),
            median_dom=("DaysOnMarket", "median"),
            lat=("Latitude", "median"),
            lon=("Longitude", "median"),
        )
        .reset_index()
    )
    # Thin markets give unstable medians, and a few records carry mistyped ZIPs or bad geocodes.
    summary = summary[summary["homes"] >= 5]
    summary = summary[summary["PostalCode"].between(*CA_ZIP_RANGE)]
    summary = summary[
        summary["lat"].between(*CA_BOUNDS["lat"]) & summary["lon"].between(*CA_BOUNDS["lon"])
    ]
    return summary


def price_distribution_chart(series, axis_title, color):
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean > 0]
    if len(clean) < 20:
        return None

    low, high = clean.quantile(0.01), clean.quantile(0.99)
    trimmed = clean[(clean >= low) & (clean <= high)]
    counts, edges = np.histogram(trimmed, bins=40)
    bins = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "homes": counts})
    bins["range_label"] = bins["start"].map(money) + " - " + bins["end"].map(money)

    bars = (
        alt.Chart(bins)
        .mark_bar(color=color, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("start:Q", title=axis_title, axis=alt.Axis(format="~s")),
            x2="end:Q",
            y=alt.Y("homes:Q", title="Homes sold"),
            tooltip=[
                alt.Tooltip("range_label:N", title="Range"),
                alt.Tooltip("homes:Q", title="Homes sold", format=","),
            ],
        )
    )
    median_rule = (
        alt.Chart(pd.DataFrame({"median": [clean.median()]}))
        .mark_rule(color="#0f172a", strokeDash=[5, 4], size=1.5)
        .encode(x="median:Q", tooltip=[alt.Tooltip("median:Q", title="Median", format=",.0f")])
    )
    return (bars + median_rule).properties(height=270)


def map_legend(summary, column, formatter):
    # Must stay on a single line: st.markdown runs Markdown first, and indented
    # lines would be parsed as code blocks instead of HTML.
    edges = summary[column].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    swatches = []
    for index, rgb in enumerate(PRICE_RAMP):
        label = f"{formatter(edges[index])} - {formatter(edges[index + 1])}"
        swatches.append(
            '<span style="display:inline-flex;align-items:center;gap:0.45rem;">'
            f'<span style="width:14px;height:14px;border-radius:3px;background:rgb{rgb};"></span>'
            f'<span style="font-size:0.76rem;color:#475569;">{label}</span>'
            "</span>"
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:1.15rem;padding:0.5rem 0.2rem 0.1rem 0.2rem;">'
        + "".join(swatches)
        + "</div>"
    )


def market_geography_page(history, predictions):
    page_header(
        "Market Geography",
        "Where price, activity, and days on market concentrate across California single-family sales.",
    )

    if history.empty:
        st.warning("Cleaned historical data file is not available.")
        return

    counties = sorted(history["CountyOrParish"].dropna().astype(str).unique().tolist())
    filter_col, metric_col = st.columns([2, 1])
    with filter_col:
        selected_county = st.selectbox("County", ["All counties"] + counties)
    with metric_col:
        metric_choice = st.selectbox("Color the map by", list(MAP_METRICS))

    frame = history
    if selected_county != "All counties":
        frame = history[history["CountyOrParish"].astype(str).eq(selected_county)]

    if frame.empty:
        st.warning("No closed sales available for that county.")
        return

    sale_to_list = frame["ClosePrice_to_ListPrice_ratio"].replace([np.inf, -np.inf], np.nan).median()
    render_metric_row(
        [
            {
                "label": "Homes sold",
                "value": f"{len(frame):,}",
                "caption": selected_county,
                "accent": "#2563eb",
            },
            {
                "label": "Median sale price",
                "value": money(frame["ClosePrice"].median()),
                "caption": "Closed transactions",
                "accent": "#0f766e",
            },
            {
                "label": "Median price / sq ft",
                "value": money(frame["price_per_sqft"].median()),
                "caption": "Living area basis",
                "accent": "#7c3aed",
            },
            {
                "label": "Median days on market",
                "value": days(frame["DaysOnMarket"].median()),
                "caption": "List to contract speed",
                "accent": "#b45309",
            },
            {
                "label": "Sale to list",
                "value": f"{sale_to_list * 100:.1f}%" if pd.notna(sale_to_list) else "N/A",
                "caption": "Above 100% means over asking",
                "accent": "#be185d",
            },
        ]
    )

    summary = build_zip_summary(frame)
    column, formatter = MAP_METRICS[metric_choice]
    summary = summary.dropna(subset=[column])

    st.subheader(f"{metric_choice} by ZIP code")
    if summary.empty:
        st.info("Not enough ZIP-level volume to draw a reliable map for this selection.")
    else:
        ranks = summary[column].rank(pct=True, method="average")
        buckets = np.clip(np.ceil(ranks * len(PRICE_RAMP)).astype(int) - 1, 0, len(PRICE_RAMP) - 1)
        summary["color"] = [list(PRICE_RAMP[bucket]) + [190] for bucket in buckets]
        summary["radius"] = np.sqrt(summary["homes"]) * 430

        summary["zip_label"] = summary["PostalCode"].astype(int).astype(str).str.zfill(5)
        summary["price_label"] = summary["median_price"].map(money)
        summary["ppsf_label"] = summary["median_ppsf"].map(money)
        summary["dom_label"] = summary["median_dom"].map(days)
        summary["homes_label"] = summary["homes"].map(lambda value: f"{value:,}")

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=summary[
                [
                    "lat",
                    "lon",
                    "color",
                    "radius",
                    "zip_label",
                    "city",
                    "county",
                    "price_label",
                    "ppsf_label",
                    "dom_label",
                    "homes_label",
                ]
            ],
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=3,
            radius_max_pixels=36,
            stroked=True,
            get_line_color=[255, 255, 255, 140],
            line_width_min_pixels=0.5,
            pickable=True,
        )
        view_state = pdk.ViewState(
            latitude=float(summary["lat"].median()),
            longitude=float(summary["lon"].median()),
            zoom=5.1 if selected_county == "All counties" else 8.2,
        )
        tooltip = {
            "html": (
                "<b>ZIP {zip_label}</b> &middot; {city}<br/>"
                "Median price: <b>{price_label}</b><br/>"
                "Median $/sq ft: {ppsf_label}<br/>"
                "Median days on market: {dom_label}<br/>"
                "Homes sold: {homes_label}"
            ),
            "style": {
                "backgroundColor": "#0f172a",
                "color": "#f8fafc",
                "fontSize": "12px",
                "borderRadius": "8px",
                "padding": "8px",
            },
        }
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="light",
                tooltip=tooltip,
                height=520,
            )
        )
        st.markdown(map_legend(summary, column, formatter), unsafe_allow_html=True)
        st.caption(
            f"Each circle is a ZIP code with at least 5 closed sales. Circle size shows sales volume; "
            f"color shows {metric_choice.lower()}. {len(summary):,} ZIP codes shown."
        )

    st.subheader("Price distribution")
    price_col, ppsf_col = st.columns(2)
    with price_col:
        chart = price_distribution_chart(frame["ClosePrice"], "Close price", "#2563eb")
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        st.caption("Dashed line marks the median. Top and bottom 1% trimmed for readability.")
    with ppsf_col:
        chart = price_distribution_chart(frame["price_per_sqft"], "Price per sq ft", "#0f766e")
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        st.caption("Price per square foot normalizes for home size across markets.")

    st.subheader("Largest local markets")
    city_summary = (
        frame.dropna(subset=["City"])
        .groupby("City")
        .agg(
            homes=("ClosePrice", "size"),
            median_price=("ClosePrice", "median"),
            median_ppsf=("price_per_sqft", "median"),
            median_dom=("DaysOnMarket", "median"),
            sale_to_list=("ClosePrice_to_ListPrice_ratio", "median"),
        )
        .query("homes >= 25")
        .sort_values("homes", ascending=False)
        .head(15)
        .reset_index()
    )
    city_display = city_summary.copy()
    city_display["median_price"] = city_display["median_price"].map(money)
    city_display["median_ppsf"] = city_display["median_ppsf"].map(money)
    city_display["median_dom"] = city_display["median_dom"].map(days)
    city_display["sale_to_list"] = city_display["sale_to_list"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value * 100:.1f}%"
    )
    city_display = city_display.rename(
        columns={
            "City": "City",
            "homes": "Homes sold",
            "median_price": "Median price",
            "median_ppsf": "Median $/sq ft",
            "median_dom": "Median days on market",
            "sale_to_list": "Sale to list",
        }
    )
    st.dataframe(city_display, hide_index=True, width="stretch")

    with st.expander("Model error by county (review support)"):
        if predictions.empty:
            st.info("Prediction output file is not available.")
        else:
            county_error = (
                predictions.dropna(subset=["CountyOrParish"])
                .groupby("CountyOrParish")
                .agg(
                    homes=("ClosePrice", "size"),
                    median_sale_price=("ClosePrice", "median"),
                    typical_error=("absolute_percentage_error", "median"),
                )
                .query("homes >= 25")
                .sort_values("homes", ascending=False)
                .head(12)
                .reset_index()
            )
            county_error["median_sale_price"] = county_error["median_sale_price"].map(money)
            county_error["typical_error"] = county_error["typical_error"].map(pct)
            st.dataframe(county_error, hide_index=True, width="stretch")
            st.caption("Higher error means those records need more careful human review, not that the market is unhealthy.")


def trend_page(history):
    page_header(
        "Market Trend Analysis",
        "Review monthly sale volume, median sale price, and price per square foot from the cleaned CRMLS data.",
    )

    if history.empty:
        st.warning("Cleaned historical data file is not available.")
        return

    county_options = ["All counties"] + sorted(history["CountyOrParish"].dropna().astype(str).unique().tolist())
    selected_county = st.selectbox("County filter", county_options)
    filtered = history.copy()
    if selected_county != "All counties":
        filtered = filtered[filtered["CountyOrParish"].astype(str).eq(selected_county)]

    monthly = (
        filtered.groupby("close_month")
        .agg(
            homes_sold=("ClosePrice", "size"),
            median_sale_price=("ClosePrice", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
        )
        .reset_index()
        .sort_values("close_month")
    )

    render_metric_row(
        [
            {
                "label": "Homes in view",
                "value": f"{len(filtered):,}",
                "caption": selected_county,
                "accent": "#2563eb",
            },
            {
                "label": "Median sale price",
                "value": money(filtered["ClosePrice"].median()),
                "caption": "Closed transactions",
                "accent": "#0f766e",
            },
            {
                "label": "Median price / sq ft",
                "value": money(filtered["price_per_sqft"].median()),
                "caption": "Living area basis",
                "accent": "#7c3aed",
            },
        ]
    )

    st.subheader("Monthly sale volume")
    st.line_chart(monthly.set_index("close_month")["homes_sold"], color="#7c3aed")

    st.subheader("Monthly median sale price")
    st.line_chart(monthly.set_index("close_month")["median_sale_price"], color="#2563eb")

    st.subheader("Monthly median price per square foot")
    st.line_chart(monthly.set_index("close_month")["median_price_per_sqft"], color="#0f766e")

    st.caption("Trend charts describe historical closed sales. They do not prove causal market drivers.")


def performance_page(metrics, predictions):
    page_header(
        "Model Performance",
        "Understand where the model is strong and where manual review is still needed.",
    )

    if metrics.empty:
        st.warning("Metrics output file is not available.")
        return

    main = metrics[metrics["section"].eq("main")].copy()
    robustness = metrics[metrics["section"].eq("robustness")].copy()
    segments = metrics[metrics["section"].eq("main_price_band")].copy()

    if not main.empty:
        row = main.iloc[0]
        render_metric_row(
            [
                {
                    "label": "Homes reviewed",
                    "value": f"{int(row['rows']):,}",
                    "caption": "June test set",
                    "accent": "#2563eb",
                },
                {
                    "label": "Overall fit",
                    "value": f"{row['r2']:.3f}",
                    "caption": "Main test population",
                    "accent": "#0f766e",
                },
                {
                    "label": "Average error",
                    "value": pct(row["mape"]),
                    "caption": "Mean percentage miss",
                    "accent": "#b45309",
                },
                {
                    "label": "Typical error",
                    "value": pct(row["mdape"]),
                    "caption": "Median percentage miss",
                    "accent": "#7c3aed",
                },
            ]
        )

    st.subheader("Error by price range")
    segment_display = segments.copy()
    segment_display["price_segment"] = segment_display["price_segment"].map(plain_segment_name)
    segment_display = segment_display.rename(
        columns={
            "price_segment": "Price range",
            "rows": "Homes",
            "median_close_price": "Median sale price",
            "mape": "Average error",
            "mdape": "Typical error",
            "p90_ape": "High-end error risk",
        }
    )
    segment_display["Median sale price"] = segment_display["Median sale price"].map(money)
    for column in ["Average error", "Typical error", "High-end error risk"]:
        segment_display[column] = segment_display[column].map(pct)
    st.dataframe(
        segment_display[
            ["Price range", "Homes", "Median sale price", "Average error", "Typical error", "High-end error risk"]
        ],
        hide_index=True,
        width="stretch",
    )

    if not segments.empty:
        chart = segments.copy()
        chart["price_segment"] = chart["price_segment"].map(plain_segment_name)
        st.subheader("Typical error by price range")
        st.bar_chart(chart.set_index("price_segment")["mdape"] * 100, color="#b45309")
        st.caption("Higher-price and lowest-price groups typically carry more error and need closer review.")

    if not robustness.empty:
        row = robustness.iloc[0]
        st.markdown(
            f"""
            <div class="note">
            <b>Robustness check:</b> full June includes {int(row["rows"]):,} homes.
            Typical error is {pct(row["mdape"])} and average error is {pct(row["mape"])}.
            This confirms that unusual records need manual review.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not predictions.empty:
        st.subheader("Actual vs predicted sample")
        sample = predictions[
            ["CloseDate", "City", "CountyOrParish", "ClosePrice", "prediction", "absolute_percentage_error"]
        ].head(20).copy()
        sample["ClosePrice"] = sample["ClosePrice"].map(money)
        sample["prediction"] = sample["prediction"].map(money)
        sample["absolute_percentage_error"] = sample["absolute_percentage_error"].map(pct)
        st.dataframe(sample, hide_index=True, width="stretch")


def handoff_page():
    page_header(
        "Handoff Notes",
        "What is included, how to run it, and what should be improved before production use.",
    )

    st.markdown(
        """
        <div class="content-card">
            <h3>Included in this app</h3>
            <p style="margin-bottom:0;">The app includes an executive overview, a prediction demo, geographic error
            summaries, market trend views, and model performance diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Run locally")
    st.code(
        """cd "/Users/amyliu/Desktop/summer intern"
source .venv/bin/activate
streamlit run app.py""",
        language="bash",
    )

    st.markdown(
        """
        <div class="note">
        <b>Production cautions:</b> before production use, save and deploy the final XGBoost artifact,
        add more future-month validation, and improve location handling with a user-friendly address workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )


metrics = load_metrics()
predictions = load_predictions()
history = load_market_history()

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="mark">CA</div>
        <div>
            <div class="title">Home Price Review</div>
            <div class="subtitle">CRMLS pricing intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Predict",
        "Market Geography",
        "Market Trends",
        "Model Performance",
        "Handoff",
    ],
)

if page == "Overview":
    overview_page(metrics, predictions)
elif page == "Predict":
    prediction_page()
elif page == "Market Geography":
    market_geography_page(history, predictions)
elif page == "Market Trends":
    trend_page(history)
elif page == "Model Performance":
    performance_page(metrics, predictions)
else:
    handoff_page()
