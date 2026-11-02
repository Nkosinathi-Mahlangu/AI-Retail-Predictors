


import os
import sys
import warnings
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sales Demand Prediction",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# THEME / CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #1a2744 60%, #0d1f3c 100%);
    border-right: 1px solid #2a3f5f;
}
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stSlider label { color: #a0b4d0 !important; font-size:0.82rem; }
[data-testid="stSidebar"] hr { border-color: #2a3f5f; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #16304f 100%);
    border: 1px solid #2e5080;
    border-radius: 12px;
    padding: 18px 20px 14px 20px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    margin-bottom: 8px;
}
.kpi-value   { font-size: 2rem;  font-weight: 700; color: #5bc8f5; line-height: 1.2; }
.kpi-label   { font-size: 0.78rem; color: #8ab4d4; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 4px; }
.kpi-delta   { font-size: 0.75rem; color: #7ecfa0; margin-top: 2px; }

/* ── Risk badges ── */
.risk-high   { background:#c0392b; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.8rem; }
.risk-medium { background:#d68910; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.8rem; }
.risk-low    { background:#1e8449; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.8rem; }

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #3a8fcc;
    border-left: 4px solid #3a8fcc;
    padding-left: 10px;
    margin: 18px 0 10px 0;
}

/* ── Chat bubbles ── */
.chat-user {
    background: #1a3a5c; border-radius:12px 12px 4px 12px;
    padding:10px 14px; margin:6px 0; max-width:80%; float:right; clear:both;
    color:#e8eaf0; font-size:0.9rem;
}
.chat-ai {
    background: #16304f; border: 1px solid #2e5080; border-radius:12px 12px 12px 4px;
    padding:10px 14px; margin:6px 0; max-width:88%; float:left; clear:both;
    color:#d0e8f8; font-size:0.9rem;
}
.chat-clear { clear:both; }

/* ── Pipeline step boxes ── */
.pipeline-step {
    background:#162a42; border:1px solid #2e5080; border-radius:8px;
    padding:12px 16px; margin:6px 0; color:#c8ddf0;
}
.pipeline-arrow { text-align:center; color:#5bc8f5; font-size:1.4rem; margin:2px 0; }

/* ── Table styling ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Page title strip ── */
.page-title-strip {
    background: linear-gradient(90deg, #0d2137, #1a3f6b);
    border-radius: 10px;
    padding: 18px 24px 14px 24px;
    margin-bottom: 18px;
    border-left: 5px solid #3a8fcc;
}
.page-title-strip h1 { color: #5bc8f5; margin:0; font-size:1.6rem; }
.page-title-strip p  { color: #8ab4d4; margin:4px 0 0 0; font-size:0.88rem; }

/* ── Quick-question buttons ── */
.stButton > button {
    background: #1a3a5c !important;
    border: 1px solid #3a8fcc !important;
    color: #d0e8f8 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 6px 12px !important;
}
.stButton > button:hover {
    background: #254e7a !important;
    border-color: #5bc8f5 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached)
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR      = os.path.join(ROOT, "models")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "filling_station_sales.csv")
RAW_PATH       = os.path.join(ROOT, "data", "raw", "sales_data.csv")
MAPPING_PATH   = os.path.join(ROOT, "data", "product_mapping.csv")


@st.cache_data(show_spinner=False)
def load_processed() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_PATH):
        return pd.DataFrame()
    df = pd.read_csv(PROCESSED_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_forecast_data() -> pd.DataFrame:
    p = os.path.join(MODEL_DIR, "forecast.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_daily_forecast() -> pd.DataFrame:
    p = os.path.join(MODEL_DIR, "forecast_daily.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "Forecast Date" in df.columns:
        df["Forecast Date"] = pd.to_datetime(df["Forecast Date"])
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    p = os.path.join(MODEL_DIR, "metrics.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_test_preds() -> pd.DataFrame:
    p = os.path.join(MODEL_DIR, "test_predictions.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_mapping() -> pd.DataFrame:
    if not os.path.exists(MAPPING_PATH):
        return pd.DataFrame()
    return pd.read_csv(MAPPING_PATH)


# ── Load everything ────────────────────────────────────────────────────────
df_proc    = load_processed()
df_fc      = load_forecast_data()
df_daily   = load_daily_forecast()
df_metrics = load_metrics()
df_preds   = load_test_preds()
df_mapping = load_mapping()

PIPELINE_READY = not df_proc.empty and not df_fc.empty

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
        <div style='font-size:2.2rem;'></div>
        <div style='font-size:1.3rem; font-weight:800; color:#5bc8f5; letter-spacing:0.05em;'>AI Sales Demand Prediction</div>
        <div style='font-size:0.72rem; color:#8ab4d4; margin-top:2px;'>Filling Station Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        [" Dashboard",
         " Sales Analytics",
         " Demand Forecast",
         " Inventory",
         " Products",
         " AI Assistant",
         " Data Quality",
         " Model Performance",
         "ℹ About"],
        label_visibility="collapsed",
    )
    st.divider()

    # Global filters
    st.markdown("**Filters**")

    stores_all = ["All Branches"]
    categories_all = ["All Categories"]
    date_min = datetime.date(2022, 1, 1)
    date_max = datetime.date(2024, 1, 30)

    if PIPELINE_READY:
        stores_all    += sorted(df_proc["Store ID"].dropna().unique().tolist())
        categories_all += sorted(df_proc["Category"].dropna().unique().tolist())
        date_min = df_proc["Date"].min().date()
        date_max = df_proc["Date"].max().date()

    sel_store    = st.selectbox("Branch", stores_all)
    sel_category = st.selectbox("Category", categories_all)
    sel_dates    = st.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    # Safety stock slider
    st.divider()
    safety_pct = st.slider(
        "Safety Stock %",
        min_value=10, max_value=50, value=20, step=5,
        help="Safety buffer as a percentage of forecast demand"
    ) / 100.0

    st.divider()
    horizon = st.select_slider("Forecast Horizon", options=[7, 14, 30], value=7)

    if not PIPELINE_READY:
        st.warning(" Models not trained yet.")
        if st.button(" Train Now", use_container_width=True):
            with st.spinner("Training pipeline running…"):
                try:
                    os.system(f'python "{os.path.join(ROOT, "train_model.py")}"')
                    st.cache_data.clear()
                    st.success("Done! Refresh the page.")
                except Exception as e:
                    st.error(f"Training failed: {e}")


# ── Apply filters to dataframes ────────────────────────────────────────────
# ── Helper: draw a small supporting chart for the AI assistant ────────────
def _draw_assistant_chart(hint: str, df: pd.DataFrame, fc: pd.DataFrame) -> None:
    """Render a small supporting chart based on the assistant's chart hint."""
    try:
        if hint == "top_sellers_bar" and "Product Name" in df.columns:
            agg = df.groupby("Product Name")["Demand"].sum().nlargest(8).reset_index()
            fig = px.bar(agg.sort_values("Demand"), x="Demand", y="Product Name",
                         orientation="h", template="plotly_dark",
                         color_discrete_sequence=["#3a8fcc"], height=220)
            fig.update_layout(margin=dict(t=5, b=5), plot_bgcolor="#0d1f3c",
                              paper_bgcolor="#0d1f3c", font_color="#c8ddf0",
                              yaxis_title="", xaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig, key=f"asst_{hint}_{id(fig)}")
        elif hint == "category_bar" and "Category" in df.columns:
            agg = df.groupby("Category")["Demand"].sum().reset_index()
            fig = px.bar(agg, x="Category", y="Demand", template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Set2, height=220)
            fig.update_layout(margin=dict(t=5, b=5), plot_bgcolor="#0d1f3c",
                              paper_bgcolor="#0d1f3c", font_color="#c8ddf0",
                              yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig, key=f"asst_{hint}_{id(fig)}")
        elif hint == "trend_line" and not df.empty:
            trend = df.groupby(df["Date"].dt.to_period("W").dt.start_time)["Demand"].sum().reset_index()
            trend.columns = ["Week", "Demand"]
            fig = px.line(trend, x="Week", y="Demand", template="plotly_dark",
                          color_discrete_sequence=["#5bc8f5"], height=200)
            fig.update_layout(margin=dict(t=5, b=5), plot_bgcolor="#0d1f3c",
                              paper_bgcolor="#0d1f3c", font_color="#c8ddf0")
            st.plotly_chart(fig, key=f"asst_{hint}_{id(fig)}")
    except Exception:
        pass  # Chart rendering is non-critical; suppress errors silently


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if sel_store != "All Branches" and "Store ID" in out.columns:
        out = out[out["Store ID"] == sel_store]
    if sel_category != "All Categories" and "Category" in out.columns:
        out = out[out["Category"] == sel_category]
    if "Date" in out.columns and len(sel_dates) == 2:
        s, e = pd.Timestamp(sel_dates[0]), pd.Timestamp(sel_dates[1])
        out = out[(out["Date"] >= s) & (out["Date"] <= e)]
    return out


def filter_fc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if sel_store != "All Branches" and "Store ID" in out.columns:
        out = out[out["Store ID"] == sel_store]
    if sel_category != "All Categories" and "Category" in out.columns:
        out = out[out["Category"] == sel_category]
    # Recalculate inventory with current safety_pct
    if "Total Predicted Demand" in out.columns or "Predicted Demand" in out.columns:
        demand_col = "Total Predicted Demand" if "Total Predicted Demand" in out.columns else "Predicted Demand"
        out["Safety Stock"]      = np.ceil(out[demand_col] * safety_pct).astype(int)
        out["Reorder Point"]     = np.ceil(out[demand_col] + out["Safety Stock"]).astype(int)
        if "Inventory Level" in out.columns:
            out["Recommended Order"] = np.maximum(
                np.ceil(out["Reorder Point"] - out["Inventory Level"]), 0
            ).astype(int)
            out["Stockout Risk"] = np.where(
                out["Inventory Level"] < out[demand_col], "HIGH",
                np.where(out["Inventory Level"] < out["Reorder Point"], "MEDIUM", "LOW")
            )
    return out


df_view = filter_df(df_proc)
fc_view = filter_fc(df_fc)


# ── Helper: KPI card ───────────────────────────────────────────────────────
def kpi_card(value: str, label: str, delta: str = "") -> str:
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    return f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-label'>{label}</div>
        {delta_html}
    </div>"""


def section_header(title: str) -> None:
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GATES — show a friendly message if pipeline hasn't been run
# ══════════════════════════════════════════════════════════════════════════════
def needs_pipeline(page_name: str) -> bool:
    """Return True and show a warning if models are not yet trained."""
    if not PIPELINE_READY:
        st.markdown(f"""
        <div class='page-title-strip'>
            <h1> SmartFuel AI — {page_name}</h1>
        </div>
        """, unsafe_allow_html=True)
        st.warning(
            " The model has not been trained yet.  \n"
            "Run the training pipeline first:  \n"
            "```\npython train_model.py\n```\n"
            "Then refresh this page."
        )
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == " Dashboard":
    # Header
    hour = datetime.datetime.now().hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    st.markdown(f"""
    <div class='page-title-strip'>
        <h1> AI Sales Demand Prediction</h1>
        <p>Intelligent Sales Forecasting &amp; Inventory Management &nbsp;|&nbsp;
           Academic Prototype — South African Filling-Station Convenience Store</p>
    </div>
    """, unsafe_allow_html=True)

    if not PIPELINE_READY:
        st.info(" Run `python train_model.py` to generate forecasts, then refresh.")
        st.markdown("""
        ### Welcome to SmartFuel AI
        This system will help you:
        - Understand historical sales and demand patterns
        - Forecast future product demand
        - Identify stockout risks and reorder priorities
        - Analyse best-selling and slow-moving products
        - Ask natural-language questions about your store data

        **To get started:** open a terminal and run `python train_model.py`
        """)
        st.stop()

    st.markdown(f"*{greeting}. Here's your store intelligence.*")
    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────
    total_demand  = int(df_view["Demand"].sum()) if not df_view.empty else 0
    total_units   = int(df_view["Units Sold"].sum()) if "Units Sold" in df_view.columns and not df_view.empty else total_demand
    pred_demand   = int(fc_view["Total Predicted Demand"].sum()) if not fc_view.empty and "Total Predicted Demand" in fc_view.columns else 0
    high_risk     = int((fc_view["Stockout Risk"] == "HIGH").sum()) if not fc_view.empty and "Stockout Risk" in fc_view.columns else 0
    to_reorder    = int((fc_view["Recommended Order"] > 0).sum()) if not fc_view.empty and "Recommended Order" in fc_view.columns else 0
    best_cat      = df_view.groupby("Category")["Demand"].sum().idxmax() if not df_view.empty and "Category" in df_view.columns else "N/A"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(kpi_card(f"{total_units:,}", "Total Units Sold"), unsafe_allow_html=True)
    k2.markdown(kpi_card(f"{total_demand:,}", "Total Demand"),    unsafe_allow_html=True)
    k3.markdown(kpi_card(f"{pred_demand:,}", f"{horizon}-Day Forecast"), unsafe_allow_html=True)
    k4.markdown(kpi_card(str(high_risk),  "High-Risk Items",  "⚠️ Needs attention" if high_risk > 0 else "✅ All good"), unsafe_allow_html=True)
    k5.markdown(kpi_card(str(to_reorder), "Items to Reorder"), unsafe_allow_html=True)
    k6.markdown(kpi_card(best_cat,        "Top Category"),     unsafe_allow_html=True)

    st.divider()
    col_left, col_right = st.columns([2, 1])

    # ── Sales trend chart ─────────────────────────────────────────────────
    with col_left:
        section_header("Sales Trend")
        if not df_view.empty:
            trend = (df_view.groupby(df_view["Date"].dt.to_period("W").dt.start_time)["Demand"]
                     .sum().reset_index())
            trend.columns = ["Week", "Demand"]
            fig = px.area(trend, x="Week", y="Demand",
                          color_discrete_sequence=["#3a8fcc"],
                          template="plotly_dark",
                          labels={"Demand": "Weekly Demand", "Week": ""})
            fig.update_layout(margin=dict(t=20, b=20), height=280,
                              plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                              font_color="#c8ddf0",
                              xaxis=dict(gridcolor="#1e3a5f"),
                              yaxis=dict(gridcolor="#1e3a5f"))
            fig.update_traces(fillcolor="rgba(58,143,204,0.18)", line_width=2)
            st.plotly_chart(fig, key="dash_trend")

    # ── Top products donut ────────────────────────────────────────────────
    with col_right:
        section_header("Category Share")
        if not df_view.empty and "Category" in df_view.columns:
            cat_agg = df_view.groupby("Category")["Demand"].sum().reset_index()
            fig2 = px.pie(cat_agg, values="Demand", names="Category",
                          hole=0.55, template="plotly_dark",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(margin=dict(t=20, b=10), height=280,
                               paper_bgcolor="#0d1f3c", font_color="#c8ddf0",
                               legend=dict(font=dict(size=10), orientation="v"))
            fig2.update_traces(textposition="inside", textinfo="percent")
            st.plotly_chart(fig2, key="dash_donut")

    st.divider()
    col_a, col_b = st.columns(2)

    # ── Top 10 products bar ───────────────────────────────────────────────
    with col_a:
        section_header("Top 10 Products by Demand")
        if not df_view.empty and "Product Name" in df_view.columns:
            top10 = (df_view.groupby(["Product Name", "Category"])["Demand"]
                     .sum().reset_index().sort_values("Demand", ascending=False).head(10))
            fig3 = px.bar(top10.sort_values("Demand"), x="Demand", y="Product Name",
                          orientation="h", color="Category",
                          template="plotly_dark",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig3.update_layout(margin=dict(t=10, b=10), height=340,
                               plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                               font_color="#c8ddf0", yaxis_title="",
                               xaxis=dict(gridcolor="#1e3a5f"),
                               legend=dict(font=dict(size=9)))
            st.plotly_chart(fig3, key="dash_top10")

    # ── Reorder priority table ────────────────────────────────────────────
    with col_b:
        section_header(" Reorder Priorities")
        if not fc_view.empty and "Recommended Order" in fc_view.columns:
            priority = fc_view[fc_view["Recommended Order"] > 0].copy()
            from src.inventory import RISK_ORDER
            priority["_r"] = priority["Stockout Risk"].map(RISK_ORDER).fillna(99)
            priority = priority.sort_values(["_r", "Recommended Order"],
                                            ascending=[True, False]).drop(columns=["_r"])
            display_cols = [c for c in ["Store ID", "Product Name", "Category",
                                        "Inventory Level", "Total Predicted Demand",
                                        "Recommended Order", "Stockout Risk"]
                            if c in priority.columns]
            st.dataframe(
                priority[display_cols].head(12),
                use_container_width=True,
                height=340,
            )
        else:
            st.info("Run the training pipeline to see reorder priorities.")

    st.divider()
    # ── Quick question buttons ────────────────────────────────────────────
    section_header(" Quick Questions — AI Manager Assistant")
    q_cols = st.columns(4)
    quick_questions = [
        "Which products need reordering?",
        "Show top selling products",
        "Which items are at stockout risk?",
        "Show demand forecast summary",
    ]
    clicked_q = None
    for i, qq in enumerate(quick_questions):
        if q_cols[i].button(qq, key=f"qq_{i}"):
            clicked_q = qq

    if clicked_q:
        with st.spinner("Thinking…"):
            try:
                from src.assistant import ManagerAssistant
                from src.feature_engineering import build_features
                assistant = ManagerAssistant(df_view if not df_view.empty else df_proc, fc_view)
                resp = assistant.ask(clicked_q)
                with st.container():
                    st.info(resp["text"])
                    if resp["table"] is not None and not resp["table"].empty:
                        st.dataframe(resp["table"].head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Assistant error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SALES ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Sales Analytics":
    if needs_pipeline("Sales Analytics"):
        st.stop()

    st.markdown("""
    <div class='page-title-strip'>
        <h1> Sales Analytics</h1>
        <p>Explore historical sales trends, category performance and product insights</p>
    </div>""", unsafe_allow_html=True)

    if df_view.empty:
        st.warning("No data matches the current filters. Adjust the sidebar.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(kpi_card(f"{int(df_view['Demand'].sum()):,}",  "Total Demand"),       unsafe_allow_html=True)
    col2.markdown(kpi_card(f"{df_view['Demand'].mean():.1f}",    "Avg Daily Demand"),   unsafe_allow_html=True)
    col3.markdown(kpi_card(f"{df_view['Product ID'].nunique()}", "Products Tracked"),   unsafe_allow_html=True)
    col4.markdown(kpi_card(f"{df_view['Store ID'].nunique()}",   "Branches Monitored"), unsafe_allow_html=True)

    st.divider()

    # ── Sales over time ───────────────────────────────────────────────────
    section_header("Sales Over Time")
    freq = st.radio("Frequency", ["Daily", "Weekly", "Monthly"], horizontal=True, key="freq_analytics")
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    agg_time = (df_view.groupby(df_view["Date"].dt.to_period(freq_map[freq]).dt.start_time)["Demand"]
                .sum().reset_index())
    agg_time.columns = ["Period", "Demand"]
    fig_time = px.line(agg_time, x="Period", y="Demand",
                       template="plotly_dark", color_discrete_sequence=["#5bc8f5"])
    fig_time.update_layout(height=300, margin=dict(t=10, b=10),
                           plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                           font_color="#c8ddf0",
                           xaxis=dict(gridcolor="#1e3a5f"),
                           yaxis=dict(gridcolor="#1e3a5f", title="Demand"))
    fig_time.update_traces(line_width=2)
    st.plotly_chart(fig_time, key="analytics_time")

    col_l, col_r = st.columns(2)

    # ── Sales by category ─────────────────────────────────────────────────
    with col_l:
        section_header("Sales by Category")
        cat_agg = (df_view.groupby("Category")["Demand"]
                   .sum().reset_index().sort_values("Demand", ascending=True))
        fig_cat = px.bar(cat_agg, x="Demand", y="Category", orientation="h",
                         template="plotly_dark",
                         color="Demand", color_continuous_scale="Blues")
        fig_cat.update_layout(height=320, margin=dict(t=10, b=10),
                              plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                              font_color="#c8ddf0", yaxis_title="",
                              xaxis=dict(gridcolor="#1e3a5f"),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_cat, key="analytics_cat")

    # ── Weekend vs weekday ────────────────────────────────────────────────
    with col_r:
        section_header("Weekend vs Weekday Demand")
        if "is_weekend" in df_view.columns:
            wkd = df_view.copy()
        else:
            wkd = df_view.copy()
            wkd["is_weekend"] = (wkd["Date"].dt.dayofweek >= 5).astype(int)
        wkd_agg = wkd.groupby("is_weekend")["Demand"].mean().reset_index()
        wkd_agg["Day Type"] = wkd_agg["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        fig_wkd = px.bar(wkd_agg, x="Day Type", y="Demand", template="plotly_dark",
                         color="Day Type",
                         color_discrete_sequence=["#3a8fcc", "#f39c12"])
        fig_wkd.update_layout(height=320, margin=dict(t=10, b=10),
                              plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                              font_color="#c8ddf0", showlegend=False,
                              yaxis=dict(gridcolor="#1e3a5f", title="Avg Daily Demand"))
        st.plotly_chart(fig_wkd, key="analytics_wkd")

    col_a, col_b = st.columns(2)

    # ── Seasonal demand ───────────────────────────────────────────────────
    with col_a:
        section_header("Demand by Season")
        if "Seasonality" in df_view.columns:
            seas = df_view.groupby("Seasonality")["Demand"].mean().reset_index()
            fig_seas = px.bar(seas, x="Seasonality", y="Demand", template="plotly_dark",
                              color="Seasonality",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_seas.update_layout(height=280, margin=dict(t=10, b=10),
                                   plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                   font_color="#c8ddf0", showlegend=False,
                                   yaxis=dict(gridcolor="#1e3a5f", title="Avg Daily Demand"))
            st.plotly_chart(fig_seas, key="analytics_seas")

    # ── Promotion impact ──────────────────────────────────────────────────
    with col_b:
        section_header("Promotion Impact on Demand")
        if "Promotion" in df_view.columns:
            promo = df_view.copy()
            promo["Promo Status"] = promo["Promotion"].map({0: "No Promotion", 1: "On Promotion"})
            promo_agg = promo.groupby("Promo Status")["Demand"].mean().reset_index()
            fig_promo = px.bar(promo_agg, x="Promo Status", y="Demand",
                               template="plotly_dark",
                               color="Promo Status",
                               color_discrete_sequence=["#7ecfa0", "#f39c12"])
            fig_promo.update_layout(height=280, margin=dict(t=10, b=10),
                                    plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                    font_color="#c8ddf0", showlegend=False,
                                    yaxis=dict(gridcolor="#1e3a5f", title="Avg Daily Demand"))
            st.plotly_chart(fig_promo, key="analytics_promo")

    # ── Weather impact ────────────────────────────────────────────────────
    section_header("Demand by Weather Condition")
    if "Weather Condition" in df_view.columns:
        weather_agg = df_view.groupby("Weather Condition")["Demand"].mean().reset_index()
        fig_w = px.bar(weather_agg, x="Weather Condition", y="Demand",
                       template="plotly_dark",
                       color="Weather Condition",
                       color_discrete_sequence=px.colors.qualitative.Set3)
        fig_w.update_layout(height=260, margin=dict(t=10, b=10),
                            plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                            font_color="#c8ddf0", showlegend=False,
                            yaxis=dict(gridcolor="#1e3a5f", title="Avg Daily Demand"))
        st.plotly_chart(fig_w, key="analytics_weather")

    # ── Branch comparison ─────────────────────────────────────────────────
    section_header("Branch Performance Comparison")
    branch_agg = (df_view.groupby("Store ID")["Demand"]
                  .sum().reset_index().sort_values("Demand", ascending=False))
    fig_br = px.bar(branch_agg, x="Store ID", y="Demand",
                    template="plotly_dark",
                    color="Demand", color_continuous_scale="Teal")
    fig_br.update_layout(height=260, margin=dict(t=10, b=10),
                         plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                         font_color="#c8ddf0",
                         xaxis=dict(title="Branch"),
                         yaxis=dict(gridcolor="#1e3a5f", title="Total Demand"),
                         coloraxis_showscale=False)
    st.plotly_chart(fig_br, key="analytics_branch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Demand Forecast":
    if needs_pipeline("Demand Forecast"):
        st.stop()

    st.markdown("""
    <div class='page-title-strip'>
        <h1> Demand Forecast</h1>
        <p>AI-generated demand predictions for the selected forecast horizon</p>
    </div>""", unsafe_allow_html=True)

    st.info(
        f"ℹ The forecast predicts total demand per product/branch over the next "
        f"**{horizon} days** using the best-trained demand model. "
        f"Predictions are estimates — actual results depend on real-world conditions."
    )

    if fc_view.empty:
        st.warning("No forecast data available. Adjust filters or retrain the model.")
        st.stop()

    demand_col = "Total Predicted Demand" if "Total Predicted Demand" in fc_view.columns else "Predicted Demand"

    # ── Forecast KPIs ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(f"{fc_view[demand_col].sum():,.0f}", f"Total {horizon}-Day Demand"), unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{fc_view[demand_col].mean():.1f}", "Avg Per Product"),             unsafe_allow_html=True)
    c3.markdown(kpi_card(str((fc_view["Stockout Risk"] == "HIGH").sum()),   "HIGH Risk"),    unsafe_allow_html=True)
    c4.markdown(kpi_card(str((fc_view["Stockout Risk"] == "MEDIUM").sum()), "MEDIUM Risk"),  unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns([3, 2])

    with col_l:
        section_header(f"Predicted Demand — Top 20 Products ({horizon} days)")
        top_fc = fc_view.sort_values(demand_col, ascending=False).head(20)
        fig_fc = px.bar(
            top_fc.sort_values(demand_col),
            x=demand_col, y="Product Name", orientation="h",
            color="Category",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_fc.update_layout(height=480, margin=dict(t=10, b=10),
                             plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                             font_color="#c8ddf0", yaxis_title="",
                             xaxis=dict(gridcolor="#1e3a5f", title="Predicted Demand (units)"),
                             legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_fc, key="forecast_bar")

    with col_r:
        section_header("Risk Distribution")
        if "Stockout Risk" in fc_view.columns:
            risk_counts = fc_view["Stockout Risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            colours = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60"}
            fig_risk = px.pie(risk_counts, values="Count", names="Risk",
                              template="plotly_dark",
                              color="Risk",
                              color_discrete_map=colours,
                              hole=0.45)
            fig_risk.update_layout(height=260, margin=dict(t=10, b=10),
                                   paper_bgcolor="#0d1f3c", font_color="#c8ddf0")
            fig_risk.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_risk, key="forecast_risk_pie")

        section_header("Category Forecast")
        if "Category" in fc_view.columns:
            cat_fc = fc_view.groupby("Category")[demand_col].sum().reset_index()
            fig_cf = px.bar(cat_fc, x="Category", y=demand_col,
                            template="plotly_dark",
                            color=demand_col, color_continuous_scale="Blues")
            fig_cf.update_layout(height=220, margin=dict(t=10, b=10),
                                 plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                 font_color="#c8ddf0", showlegend=False,
                                 yaxis=dict(gridcolor="#1e3a5f", title=""),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_cf, key="forecast_cat")

    # ── Daily forecast trend (if available) ───────────────────────────────
    if not df_daily.empty:
        section_header(f"Daily Demand Forecast Trend — Next {horizon} Days")
        filt_daily = df_daily.copy()
        if sel_store != "All Branches" and "Store ID" in filt_daily.columns:
            filt_daily = filt_daily[filt_daily["Store ID"] == sel_store]
        if sel_category != "All Categories" and "Category" in filt_daily.columns:
            filt_daily = filt_daily[filt_daily["Category"] == sel_category]
        if not filt_daily.empty:
            daily_trend = (filt_daily.groupby("Forecast Date")["Predicted Demand"]
                          .sum().reset_index())
            fig_daily = px.line(daily_trend, x="Forecast Date", y="Predicted Demand",
                                template="plotly_dark",
                                color_discrete_sequence=["#5bc8f5"],
                                markers=True)
            fig_daily.update_layout(height=280, margin=dict(t=10, b=10),
                                    plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                    font_color="#c8ddf0",
                                    xaxis=dict(gridcolor="#1e3a5f"),
                                    yaxis=dict(gridcolor="#1e3a5f", title="Predicted Demand"))
            st.plotly_chart(fig_daily, key="forecast_daily")

    # ── Full forecast table ───────────────────────────────────────────────
    section_header("Full Forecast Table")
    show_cols = [c for c in ["Store ID", "Product Name", "Category", "Region",
                             "Inventory Level", demand_col, "Safety Stock",
                             "Recommended Order", "Stockout Risk"]
                 if c in fc_view.columns]
    st.dataframe(
        fc_view.sort_values(demand_col, ascending=False)[show_cols],
        use_container_width=True,
        height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Inventory":
    if needs_pipeline("Inventory Intelligence"):
        st.stop()

    st.markdown("""
    <div class='page-title-strip'>
        <h1> Inventory Intelligence</h1>
        <p>Safety stock calculations, reorder points and stockout risk analysis</p>
    </div>""", unsafe_allow_html=True)

    if fc_view.empty:
        st.warning("No inventory data available. Adjust filters or retrain the model.")
        st.stop()

    demand_col = "Total Predicted Demand" if "Total Predicted Demand" in fc_view.columns else "Predicted Demand"

    # KPIs
    total_inv  = int(fc_view["Inventory Level"].sum()) if "Inventory Level" in fc_view.columns else 0
    n_high     = int((fc_view["Stockout Risk"] == "HIGH").sum()) if "Stockout Risk" in fc_view.columns else 0
    n_medium   = int((fc_view["Stockout Risk"] == "MEDIUM").sum()) if "Stockout Risk" in fc_view.columns else 0
    n_reorder  = int((fc_view["Recommended Order"] > 0).sum()) if "Recommended Order" in fc_view.columns else 0
    total_ord  = int(fc_view["Recommended Order"].sum()) if "Recommended Order" in fc_view.columns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_card(f"{total_inv:,}",  "Total Stock On Hand"),   unsafe_allow_html=True)
    k2.markdown(kpi_card(str(n_high),       "HIGH Risk Items",  "⚠️ Order immediately" if n_high else "✅"), unsafe_allow_html=True)
    k3.markdown(kpi_card(str(n_medium),     "MEDIUM Risk Items"), unsafe_allow_html=True)
    k4.markdown(kpi_card(str(n_reorder),    "Items to Reorder"),     unsafe_allow_html=True)
    k5.markdown(kpi_card(f"{total_ord:,}",  "Total Units to Order"), unsafe_allow_html=True)

    st.divider()
    st.caption(
        f" **Assumptions:** Safety Stock = {safety_pct*100:.0f}% of {horizon}-day forecast demand.  "
        "Reorder Point = Predicted Demand + Safety Stock.  "
        "Recommended Order = max(Reorder Point − Current Inventory, 0).  "
        "These are prototype heuristics; a production system would use variance-based safety stock."
    )
    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        section_header("Inventory vs Forecast Demand by Category")
        if "Category" in fc_view.columns and "Inventory Level" in fc_view.columns:
            cat_inv = fc_view.groupby("Category").agg(
                Inventory=("Inventory Level", "sum"),
                Forecast=(demand_col, "sum"),
            ).reset_index()
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Bar(name="On Hand", x=cat_inv["Category"],
                                     y=cat_inv["Inventory"],
                                     marker_color="#3a8fcc"))
            fig_inv.add_trace(go.Bar(name="Forecast Demand", x=cat_inv["Category"],
                                     y=cat_inv["Forecast"],
                                     marker_color="#f39c12"))
            fig_inv.update_layout(barmode="group", template="plotly_dark",
                                  height=320, margin=dict(t=10, b=10),
                                  plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                  font_color="#c8ddf0",
                                  yaxis=dict(gridcolor="#1e3a5f"),
                                  legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_inv, key="inv_cat_bar")

    with col_r:
        section_header("Risk Profile by Branch")
        if "Stockout Risk" in fc_view.columns and "Store ID" in fc_view.columns:
            risk_br = (fc_view.groupby(["Store ID", "Stockout Risk"])
                       .size().reset_index(name="Count"))
            colours = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60"}
            fig_risk_br = px.bar(risk_br, x="Store ID", y="Count",
                                 color="Stockout Risk",
                                 template="plotly_dark",
                                 color_discrete_map=colours,
                                 barmode="stack")
            fig_risk_br.update_layout(height=320, margin=dict(t=10, b=10),
                                      plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                      font_color="#c8ddf0",
                                      yaxis=dict(gridcolor="#1e3a5f"),
                                      legend=dict(font=dict(size=10)),
                                      xaxis_title="Branch")
            st.plotly_chart(fig_risk_br, key="inv_risk_branch")

    # ── Priority reorder table ─────────────────────────────────────────────
    section_header(" Top Reorder Priorities")
    if "Recommended Order" in fc_view.columns:
        from src.inventory import RISK_ORDER
        priority = fc_view[fc_view["Recommended Order"] > 0].copy()
        priority["_r"] = priority["Stockout Risk"].map(RISK_ORDER).fillna(99)
        priority = priority.sort_values(["_r", "Recommended Order"],
                                        ascending=[True, False]).drop(columns=["_r"])
        show_cols = [c for c in ["Store ID", "Product Name", "Category",
                                 "Inventory Level", demand_col,
                                 "Safety Stock", "Reorder Point",
                                 "Recommended Order", "Stockout Risk"]
                     if c in priority.columns]
        st.dataframe(priority[show_cols], use_container_width=True, height=420)

        # Natural-language messages for HIGH risk items
        high = priority[priority["Stockout Risk"] == "HIGH"]
        if not high.empty:
            section_header(" Urgent Reorder Messages")
            from src.inventory import generate_reorder_message
            for _, row in high.head(8).iterrows():
                st.markdown(
                    f"<div class='pipeline-step'>{generate_reorder_message(row, horizon)}</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Products":
    if needs_pipeline("Product Performance"):
        st.stop()

    st.markdown("""
    <div class='page-title-strip'>
        <h1> Product Performance</h1>
        <p>Best sellers, slow movers, revenue leaders and category breakdown</p>
    </div>""", unsafe_allow_html=True)

    if df_view.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # ── Product-level aggregation ─────────────────────────────────────────
    prod_agg = (
        df_view.groupby(["Product ID", "Product Name", "Category", "Subcategory"],
                        as_index=False)
        .agg(
            Total_Demand=("Demand", "sum"),
            Avg_Daily=("Demand", "mean"),
            Total_Revenue=("Price", lambda x: (x * df_view.loc[x.index, "Demand"]).sum())
            if "Price" in df_view.columns else ("Demand", "sum"),
        )
        .sort_values("Total_Demand", ascending=False)
    )
    if "Price" in df_view.columns:
        prod_agg["Total_Revenue"] = prod_agg.apply(
            lambda row: (df_view[df_view["Product ID"] == row["Product ID"]]["Price"]
                         * df_view[df_view["Product ID"] == row["Product ID"]]["Demand"]).sum(),
            axis=1,
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        [" Best Sellers", " Slow Movers", " Revenue Leaders", " Category View"]
    )

    with tab1:
        top_n = st.slider("Show top N products", 5, 20, 10, key="top_n_slider")
        top_df = prod_agg.head(top_n)
        fig_top = px.bar(top_df.sort_values("Total_Demand"),
                         x="Total_Demand", y="Product Name",
                         orientation="h", color="Category",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         labels={"Total_Demand": "Total Demand (units)"})
        fig_top.update_layout(height=top_n*38+80, margin=dict(t=10, b=10),
                              plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                              font_color="#c8ddf0", yaxis_title="",
                              xaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig_top, key="prod_top_bar")
        st.dataframe(top_df[["Product Name", "Category", "Subcategory",
                              "Total_Demand", "Avg_Daily"]].rename(columns={
                                  "Total_Demand": "Total Demand",
                                  "Avg_Daily":    "Avg Daily Demand",
                              }), use_container_width=True)

    with tab2:
        slow_df = prod_agg.tail(10).sort_values("Total_Demand", ascending=True)
        fig_slow = px.bar(slow_df,
                          x="Total_Demand", y="Product Name",
                          orientation="h", color="Category",
                          template="plotly_dark",
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          labels={"Total_Demand": "Total Demand (units)"})
        fig_slow.update_layout(height=420, margin=dict(t=10, b=10),
                               plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                               font_color="#c8ddf0", yaxis_title="",
                               xaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig_slow, key="prod_slow_bar")
        st.caption(" Consider running promotions or reducing stock orders for these slow-moving items.")
        st.dataframe(slow_df[["Product Name", "Category", "Total_Demand", "Avg_Daily"]].rename(columns={
            "Total_Demand": "Total Demand", "Avg_Daily": "Avg Daily",
        }), use_container_width=True)

    with tab3:
        if "Price" in df_view.columns:
            rev_df = prod_agg.sort_values("Total_Revenue", ascending=False).head(10)
            fig_rev = px.bar(rev_df.sort_values("Total_Revenue"),
                             x="Total_Revenue", y="Product Name",
                             orientation="h", color="Category",
                             template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Set1,
                             labels={"Total_Revenue": "Estimated Revenue (R)"})
            fig_rev.update_layout(height=420, margin=dict(t=10, b=10),
                                  plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                  font_color="#c8ddf0", yaxis_title="",
                                  xaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig_rev, key="prod_rev_bar")
            st.caption("_Revenue = Price × Demand. Actual revenue may differ due to promotions and discounts._")
        else:
            st.info("Price data not available.")

    with tab4:
        cat_detail = (df_view.groupby(["Category", "Product Name"])["Demand"]
                      .sum().reset_index().sort_values(["Category", "Demand"],
                                                       ascending=[True, False]))
        fig_cat_detail = px.bar(cat_detail, x="Demand", y="Product Name",
                                color="Category", facet_col="Category",
                                facet_col_wrap=3,
                                template="plotly_dark",
                                color_discrete_sequence=px.colors.qualitative.Set2)
        fig_cat_detail.update_layout(height=600, margin=dict(t=40, b=10),
                                     paper_bgcolor="#0d1f3c", font_color="#c8ddf0",
                                     showlegend=False)
        fig_cat_detail.update_yaxes(title="")
        st.plotly_chart(fig_cat_detail, key="prod_cat_detail")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == " AI Assistant":
    if needs_pipeline("AI Manager Assistant"):
        st.stop()

    st.markdown("""
    <div class='page-title-strip'>
        <h1> AI Manager Assistant</h1>
        <p>Ask questions about your store in plain language — powered by real data</p>
    </div>""", unsafe_allow_html=True)

    st.info(
        " This assistant answers using **actual data** from your cleaned dataset "
        "and model forecasts. It does not use an external AI service — all answers "
        "are derived from real calculations. Numbers are never invented."
    )

    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input_value" not in st.session_state:
        st.session_state.chat_input_value = ""

    # Initialise assistant
    try:
        from src.assistant import ManagerAssistant
        assistant = ManagerAssistant(
            df_proc if df_view.empty else df_view,
            fc_view,
            metrics_df=df_metrics if not df_metrics.empty else None,
        )
        assistant_ready = True
    except Exception as e:
        st.error(f"Could not initialise assistant: {e}")
        assistant_ready = False

    # ── Quick-question buttons ─────────────────────────────────────────────
    st.markdown("**Quick questions:**")
    qqs = [
        "Which products need reordering?",
        "Show top selling products",
        "Which items are at high stockout risk?",
        "How is the Beverages category performing?",
        "What are my sales trends?",
        "Compare Snacks and Beverages",
        "What is the impact of promotions?",
        "Show demand forecast summary",
    ]
    qq_cols = st.columns(4)
    for i, qq in enumerate(qqs):
        if qq_cols[i % 4].button(qq, key=f"assistant_qq_{i}"):
            st.session_state.chat_input_value = qq

    st.divider()

    # ── Speech recognition toggle ──────────────────────────────────────────
    from src.speech import get_status as speech_status, transcribe_from_microphone, SPEECH_ENABLED
    speech_col, text_col = st.columns([1, 5])
    with speech_col:
        use_speech = st.toggle("🎤 Voice", value=False, disabled=not SPEECH_ENABLED,
                               help=speech_status()["description"])
        if use_speech and SPEECH_ENABLED:
            if st.button(" Record", key="record_btn"):
                with st.spinner("Listening… speak now"):
                    result = transcribe_from_microphone()
                    if result["success"]:
                        st.session_state.chat_input_value = result["text"]
                        st.success(f"Heard: {result['text']}")
                    else:
                        st.error(result["error"])

    # ── Text input ─────────────────────────────────────────────────────────
    with text_col:
        user_input = st.text_input(
            "Your question",
            value=st.session_state.chat_input_value,
            placeholder="e.g. Which products should I reorder?",
            key="main_chat_input",
            label_visibility="collapsed",
        )

    send_col, clear_col = st.columns([1, 5])
    if send_col.button("Send ", key="send_btn", use_container_width=True):
        if user_input.strip() and assistant_ready:
            st.session_state.chat_history.append(("user", user_input.strip()))
            resp = assistant.ask(user_input.strip())
            st.session_state.chat_history.append(("ai", resp))
            st.session_state.chat_input_value = ""
    if clear_col.button("Clear chat", key="clear_btn"):
        st.session_state.chat_history = []
        st.session_state.chat_input_value = ""

    st.divider()

    # ── Chat history display ───────────────────────────────────────────────
    for role, content in reversed(st.session_state.chat_history):
        if role == "user":
            st.markdown(
                f"<div class='chat-user'>👤 {content}</div><div class='chat-clear'></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-ai'>🤖 {content['text']}</div><div class='chat-clear'></div>",
                unsafe_allow_html=True,
            )
            if content.get("table") is not None and not content["table"].empty:
                st.dataframe(content["table"].head(12), use_container_width=True)
            hint = content.get("chart_hint")
            if hint and not df_view.empty:
                _draw_assistant_chart(hint, df_view, fc_view)

    if not st.session_state.chat_history:
        st.markdown(
            "<div class='chat-ai'> Hello! I'm your SmartFuel AI Manager Assistant. "
            "Ask me anything about your store — inventory, demand, sales trends, "
            "or which products need restocking. Type a question above or click a quick-question button.</div>"
            "<div class='chat-clear'></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Data Quality":
    st.markdown("""
    <div class='page-title-strip'>
        <h1> Data Quality &amp; Preprocessing</h1>
        <p>Transparency report: how the raw dataset was cleaned and transformed</p>
    </div>""", unsafe_allow_html=True)

    # ── Pipeline diagram ───────────────────────────────────────────────────
    section_header("Data Pipeline Overview")
    steps = [
        (" Raw Dataset",                  "76,000 rows | Columns: Date, Store ID, Product ID, Category, Region, Demand, …"),
        (" Column Validation",             "Verify all required columns present"),
        (" Date Parsing",                 "Convert Date to datetime; remove unparseable rows"),
        (" Numeric Coercion",             "Convert Price, Demand, Inventory etc. to numeric; fill defaults"),
        (" Duplicate Removal",            "Remove exact duplicate (Date, Store, Product) rows"),
        (" Invalid Value Removal",        "Remove rows with Demand < 0, Inventory < 0, Price ≤ 0"),
        (" Filling-Station Product Filter","Map all 20 Product IDs to real convenience-store products;\nRemove rows with non-FS original categories (Electronics, Clothing, Toys, Furniture)\nvia product allowlist in data/product_mapping.csv"),
        (" SA Context Mapping",           "Rename Regions → SA provinces; rename Store IDs → SA branch names;\nStandardise weather and seasonality labels"),
        (" Feature Engineering",          "Lag features (1, 7, 14, 28 days); Rolling statistics (7, 14, 28 days);\nDate features; Business features (price ratio, discount %)"),
        (" Model Training",              "Chronological 80/20 train/test split;\nFour models trained and compared by MAE"),
        (" Forecast Generation",         f"Iterative {7}-day ahead predictions per store/product"),
        (" Dashboard & Reports",         "Streamlit application with all pages"),
    ]
    for i, (title, desc) in enumerate(steps):
        st.markdown(f"<div class='pipeline-step'><strong>{title}</strong><br><small>{desc}</small></div>", unsafe_allow_html=True)
        if i < len(steps) - 1:
            st.markdown("<div class='pipeline-arrow'>↓</div>", unsafe_allow_html=True)

    st.divider()

    # ── Live cleaning stats ───────────────────────────────────────────────
    section_header("Data Quality Statistics")
    if PIPELINE_READY:
        try:
            from src.data_cleaning import clean
            with st.spinner("Calculating data quality statistics…"):
                _, stats = clean(verbose=False)

            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(kpi_card(f"{stats.get('raw_rows', 0):,}",       "Raw Rows"),          unsafe_allow_html=True)
                st.markdown(kpi_card(f"{stats.get('processed_rows', 0):,}", "Processed Rows"),    unsafe_allow_html=True)
                st.markdown(kpi_card(f"{stats.get('total_removed', 0):,}",  "Total Rows Removed"),unsafe_allow_html=True)
            with s2:
                st.markdown(kpi_card(f"{stats.get('duplicates_removed', 0):,}",     "Duplicates Removed"),  unsafe_allow_html=True)
                st.markdown(kpi_card(f"{stats.get('invalid_records_removed', 0):,}","Invalid Records"),     unsafe_allow_html=True)
                st.markdown(kpi_card(f"{stats.get('missing_values_filled', 0):,}",  "Missing Values Filled"),unsafe_allow_html=True)
            with s3:
                st.markdown(kpi_card(str(stats.get('n_products', 0)),    "Products"),              unsafe_allow_html=True)
                st.markdown(kpi_card(str(stats.get('n_stores', 0)),      "Branches"),              unsafe_allow_html=True)
                st.markdown(kpi_card(str(stats.get('n_categories', 0)),  "FS Categories"),         unsafe_allow_html=True)

            st.markdown(f"""
            **Date range:** {stats.get('date_min', 'N/A')} → {stats.get('date_max', 'N/A')}
            ({stats.get('date_range_days', 0):,} days)
            """)
        except Exception as e:
            st.warning(f"Could not recalculate stats: {e}")
    else:
        st.info("Run `python train_model.py` to generate data quality statistics.")

    st.divider()

    # ── Product mapping table ─────────────────────────────────────────────
    section_header("Product Mapping — Filling-Station Category Assignment")
    st.caption(
        " **Academic note:** The original dataset uses generic Product IDs (P0001–P0020) "
        "with categories such as Electronics, Clothing, Toys and Furniture — none of which "
        "are sold inside a filling-station convenience store.  A transparent product-mapping "
        "layer was created to assign each Product ID a realistic filling-station product name "
        "and category.  The original raw CSV is preserved unchanged in data/raw/.  "
        "This mapping is a documented prototype assumption, not fabricated sales data."
    )
    if not df_mapping.empty:
        st.dataframe(df_mapping, use_container_width=True)
    else:
        st.info("Product mapping file not found at data/product_mapping.csv")

    # ── Sample processed data ─────────────────────────────────────────────
    section_header("Sample Processed Dataset")
    if PIPELINE_READY:
        st.dataframe(df_proc.head(20), use_container_width=True)
    else:
        st.info("Run the training pipeline to see the processed dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Model Performance":
    if needs_pipeline("Model Performance"):
        st.stop()

    st.markdown("""
    <div class='page-title-strip'>
        <h1> Model Performance</h1>
        <p>Evaluation of demand-forecasting models using a chronological holdout test set</p>
    </div>""", unsafe_allow_html=True)

    # ── Metric explanations ───────────────────────────────────────────────
    with st.expander(" What do the metrics mean?", expanded=False):
        st.markdown("""
        | Metric | Full Name | What it means |
        |--------|-----------|---------------|
        | **MAE** | Mean Absolute Error | Average absolute difference between predicted and actual demand. Lower is better. E.g. MAE = 14 means predictions are off by 14 units on average. |
        | **RMSE** | Root Mean Squared Error | Similar to MAE but penalises large errors more heavily. Lower is better. |
        | **R²** | Coefficient of Determination | How much of demand variation the model explains. Range 0–1. Higher is better. R² = 0.81 means 81% of variation is explained. |
        | **MAPE** | Mean Absolute Percentage Error | Average percentage error. Lower is better. Rows with near-zero actual demand are excluded to avoid division-by-zero distortion. |

        **Evaluation method:** Strict chronological 80/20 train/test split. The model is trained on
        the earliest 80% of dates and evaluated on the most recent 20%. This simulates real-world
        deployment where the model is asked to predict future demand it has never seen.
        """)

    if not df_metrics.empty:
        # ── Model comparison table ─────────────────────────────────────────
        section_header("Model Comparison")
        display_metrics = df_metrics.copy()
        if "MAPE" in display_metrics.columns:
            display_metrics["MAPE"] = display_metrics["MAPE"].round(2).astype(str) + " %"
        if "MAE" in display_metrics.columns:
            display_metrics["MAE"] = display_metrics["MAE"].round(4)
        if "RMSE" in display_metrics.columns:
            display_metrics["RMSE"] = display_metrics["RMSE"].round(4)
        if "R2" in display_metrics.columns:
            display_metrics["R2"] = display_metrics["R2"].round(4)
        st.dataframe(display_metrics, use_container_width=True)

        # Best model callout
        numeric_metrics = df_metrics.copy()
        best_row = numeric_metrics.sort_values("MAE").iloc[0]
        st.success(
            f" **Best model by MAE:** {best_row['Model']}  |  "
            f"MAE: {best_row['MAE']:.4f}  |  "
            f"RMSE: {best_row['RMSE']:.4f}  |  "
            f"R²: {best_row['R2']:.4f}"
        )

        # ── Metric bar charts ──────────────────────────────────────────────
        section_header("Visual Comparison")
        mc1, mc2 = st.columns(2)
        with mc1:
            fig_mae = px.bar(df_metrics, x="Model", y="MAE",
                             template="plotly_dark",
                             color="Model",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             title="MAE (lower is better)")
            fig_mae.update_layout(height=300, margin=dict(t=40, b=10),
                                  plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                  font_color="#c8ddf0", showlegend=False,
                                  yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig_mae, key="mp_mae")
        with mc2:
            fig_r2 = px.bar(df_metrics, x="Model", y="R2",
                            template="plotly_dark",
                            color="Model",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            title="R² (higher is better)")
            fig_r2.update_layout(height=300, margin=dict(t=40, b=10),
                                 plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                                 font_color="#c8ddf0", showlegend=False,
                                 yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig_r2, key="mp_r2")

    else:
        st.info("Model metrics not found. Run `python train_model.py`.")

    # ── Actual vs Predicted ───────────────────────────────────────────────
    if not df_preds.empty:
        section_header("Actual vs Predicted Demand — Test Set Sample")
        sample = df_preds.sample(min(500, len(df_preds)), random_state=42).sort_values("Date")
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=sample["Date"], y=sample["Demand"],
                                     mode="markers", name="Actual",
                                     marker=dict(color="#5bc8f5", size=3, opacity=0.6)))
        fig_avp.add_trace(go.Scatter(x=sample["Date"], y=sample["Predicted Demand"],
                                     mode="markers", name="Predicted",
                                     marker=dict(color="#f39c12", size=3, opacity=0.6)))
        fig_avp.update_layout(height=340, template="plotly_dark",
                              margin=dict(t=10, b=10),
                              plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                              font_color="#c8ddf0",
                              xaxis=dict(gridcolor="#1e3a5f"),
                              yaxis=dict(gridcolor="#1e3a5f", title="Demand (units)"),
                              legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_avp, key="mp_avp")

        # ── Residual distribution ──────────────────────────────────────────
        section_header("Prediction Error Distribution")
        errors = df_preds["Error"].dropna()
        fig_err = px.histogram(errors, nbins=60, template="plotly_dark",
                               color_discrete_sequence=["#3a8fcc"],
                               labels={"value": "Prediction Error (units)"})
        fig_err.update_layout(height=280, margin=dict(t=10, b=10),
                              plot_bgcolor="#0d1f3c", paper_bgcolor="#0d1f3c",
                              font_color="#c8ddf0",
                              yaxis=dict(gridcolor="#1e3a5f"),
                              xaxis=dict(gridcolor="#1e3a5f"),
                              showlegend=False)
        fig_err.add_vline(x=0, line_color="#f39c12", line_dash="dash")
        st.plotly_chart(fig_err, key="mp_err")
        st.caption(
            "A symmetric distribution centred near 0 indicates well-calibrated predictions. "
            "Heavy tails indicate occasional large errors — expected in volatile demand environments."
        )

    # ── Academic honesty note ─────────────────────────────────────────────
    st.divider()
    st.info(
        " **Academic integrity note:** All metrics reported above are calculated on the actual "
        "held-out test data. No results have been modified or overstated. "
        "If the MAPE exceeds the assignment target of 15–20%, this is reported honestly. "
        "Improvement strategies are noted in the README."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ About":
    st.markdown("""
    <div class='page-title-strip'>
        <h1>ℹ About AI Sales Demand Prediction</h1>
        <p>Project information, methodology and academic context</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ## SmartFuel AI
        ### Intelligent Filling Station Sales, Demand Forecasting and Inventory Management System

        **Project type:** Academic AI/ML prototype
        **Domain:** South African filling-station convenience-store retail
        **Technology stack:** Python · scikit-learn · Streamlit · Plotly · Pandas

        ---

        ### Purpose
        SmartFuel AI is designed to help a filling-station convenience-store manager:
        - Understand and analyse historical sales data
        - Predict future product demand using machine learning
        - Identify products at risk of running out of stock
        - Receive automated reorder recommendations
        - Identify best-selling and slow-moving products
        - Ask natural-language questions about the store's data

        ---

        ### AI/ML Pipeline
        1. **Data collection** — CSV sales dataset loaded from `data/raw/`
        2. **Data cleaning** — validation, deduplication, missing-value handling
        3. **Product filtering** — domain-specific mapping removes non-FS categories
        4. **Feature engineering** — lag features, rolling statistics, date features
        5. **Model training** — four models compared on a chronological holdout
        6. **Model selection** — best model chosen by MAE
        7. **Demand forecasting** — iterative multi-horizon predictions
        8. **Inventory intelligence** — safety stock, reorder points, risk levels
        9. **AI assistant** — data-driven NLP query layer

        ---

        ### Filling-Station Product Categories
        | Category | Examples |
        |----------|---------|
        | Beverages | Bottled Water, Soft Drinks, Energy Drinks, Sports Drinks |
        | Snacks | Chips, Biscuits, Chocolate, Chewing Gum, Crisps |
        | Food | Bread, Instant Meals, Noodles |
        | Dairy & Refrigerated | Milk, Yogurt, Cheese |
        | Automotive | Engine Oil, Coolant, Car Accessories |
        | Personal Care | Wet Wipes, Hand Sanitiser, Tissues |
        | Household & Convenience | Batteries, Lighters, Umbrellas |

        ---

        ### Limitations & Assumptions
        - The original dataset uses **generic product IDs** with non-FS categories.
          A transparent product-mapping layer was applied as a prototype assumption.
        - The original raw data is preserved at `data/raw/sales_data.csv`.
        - Demand forecasts are **estimates only** and should be validated before
          making major business decisions.
        - Safety stock uses a **heuristic 20% buffer** — not a statistically derived
          safety factor. A production system would use demand variance and Z-scores.
        - Revenue estimates use Price × Demand and do not account for all discounts.
        - The speech recognition feature requires an internet connection (Google API).
        - MAPE may exceed aspirational targets — this is reported honestly.

        ---

        ### Ethical Considerations
        - No personal customer data is collected or used.
        - All statistics are derived from the actual dataset — no results are fabricated.
        - The system clearly labels forecasts as estimates.
        - Prototype assumptions are explicitly documented throughout.

        ---

        ### How to Run
        ```bash
        # 1. Install dependencies
        pip install -r requirements.txt

        # 2. Train the model (run once)
        python train_model.py

        # 3. Launch the dashboard
        streamlit run app.py
        ```
        """)

    with col2:
        st.markdown("""
        ### Project Structure
        ```
        SmartFuel_AI/
        ├── app.py
        ├── train_model.py
        ├── forecast.py
        ├── requirements.txt
        ├── README.md
        ├── run_project.bat
        │
        ├── data/
        │   ├── raw/
        │   │   └── sales_data.csv
        │   ├── processed/
        │   │   └── filling_station_sales.csv
        │   └── product_mapping.csv
        │
        ├── models/
        │   ├── model_bundle.joblib
        │   ├── metrics.csv
        │   ├── forecast.csv
        │   ├── forecast_daily.csv
        │   └── test_predictions.csv
        │
        ├── src/
        │   ├── data_cleaning.py
        │   ├── feature_engineering.py
        │   ├── model_training.py
        │   ├── forecasting.py
        │   ├── inventory.py
        │   ├── assistant.py
        │   └── speech.py
        │
        └── docs/
        ```

        ---

        ### Models Used
        | Model | Purpose |
        |-------|---------|
        | Naive Lag-1 | Baseline benchmark |
        | Ridge Regression | Linear demand model |
        | Random Forest | Non-linear ensemble |
        | HistGradientBoosting | Best performer (expected) |

        ---

        ### Currency
        South African Rand (R) is used for all monetary values.

        ---

        ### Academic Context
        This project was developed as an academic prototype demonstrating the end-to-end
        AI/ML development lifecycle for a business intelligence application.

        Results describe this dataset and holdout period only. Business success
        targets require real-world deployment and measurement.
        """)


# ── end of app.py ──────────────────────────────────────────────────────────

