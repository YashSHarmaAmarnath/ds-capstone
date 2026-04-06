import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import (
    load_and_clean,
    get_kpis,
    sales_trend,
    top_products_chart,
    country_bar_chart,
    country_map,
    build_rfm,
    rfm_segment_chart,
    rfm_scatter,
    get_at_risk,
    product_velocity,
)
from utils import *

st.set_page_config(page_title="Retail Analytics", page_icon="📊", layout="wide")

#  Sidebar 
# with st.sidebar:
#     st.title("📊 Retail Analytics")
#     uploaded = st.file_uploader("Upload CSV", type=["csv"])
uploaded = "./Online_retail.csv"
if not uploaded:
    st.title("📊 Retail Analytics Dashboard")
    st.info("Upload your retail CSV from the sidebar to get started.")
    st.caption("Required columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country")
    st.stop()

#  Load data 
@st.cache_data(show_spinner="Loading data...")
def cached_load(file):
    return load_and_clean(file)

try:
    df, warnings = cached_load(uploaded)
except ValueError as e:
    st.error(str(e))
    st.stop()

for w in warnings:
    st.warning(w)

#  Sidebar filters 
with st.sidebar:
    st.subheader("Filters")
    countries = sorted(df["Country"].unique())
    sel_countries = st.multiselect("Country", countries, default=countries)
    years = sorted(df["Year"].unique())
    sel_years = st.multiselect("Year", years, default=years)

if sel_countries:
    df = df[df["Country"].isin(sel_countries)]
if sel_years:
    df = df[df["Year"].isin(sel_years)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

kpis = get_kpis(df)
rfm  = build_rfm(df)

#  Page header 
d0 = kpis["date_range"][0].strftime("%b %Y")
d1 = kpis["date_range"][1].strftime("%b %Y")
st.title("📊 Retail Analytics Dashboard")
st.caption(f"Data range: {d0} → {d1}")

#  KPI metrics 
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Revenue",   f"£{kpis['total_revenue']:,.0f}")
c2.metric("🧾 Orders",    f"{kpis['total_orders']:,}")
c3.metric("👥 Customers", f"{kpis['total_customers']:,}")
c4.metric("📦 Products",  f"{kpis['total_products']:,}")
c5.metric("🛒 Avg Order", f"£{kpis['avg_order_value']:,.2f}")

st.divider()

#  Tabs 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📅 Sales Trends",
    "🏆 Top Products",
    "🌍 By Country",
    "👥 RFM Segments",
    "⚠️ At-Risk",
    "📦 Product Velocity",
])

#  Tab 1: Sales Trends 
with tab1:
    st.subheader("Sales Trends")
    freq = st.radio("Granularity", ["Monthly", "Weekly"], horizontal=True)
    st.plotly_chart(sales_trend(df, freq), use_container_width=True)

    st.subheader("Revenue by Weekday")
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wd = df.groupby("Weekday")["TotalPrice"].sum().reindex(weekday_order).reset_index()
    fig_wd = go.Figure(go.Bar(x=wd["Weekday"], y=wd["TotalPrice"]))
    fig_wd.update_layout(xaxis_title="Weekday", yaxis_title="Revenue (£)")
    st.plotly_chart(fig_wd, use_container_width=True)

#  Tab 2: Top Products 
with tab2:
    st.subheader("Best-Selling Products")
    c1, c2 = st.columns(2)
    metric = c1.radio("Metric", ["Quantity", "Revenue"], horizontal=True)
    top_n  = c2.slider("Top N", 5, 20, 10)
    st.plotly_chart(top_products_chart(df, top_n, metric), use_container_width=True)

#  Tab 3: Country 
with tab3:
    st.subheader("Sales by Country")
    st.plotly_chart(country_map(df), use_container_width=True)
    st.plotly_chart(country_bar_chart(df, 15), use_container_width=True)

#  Tab 4: RFM Segments 
with tab4:
    st.subheader("RFM Customer Segmentation")

    seg_counts = rfm["Segment"].value_counts()
    cols = st.columns(len(seg_counts))
    for col, (seg, cnt) in zip(cols, seg_counts.items()):
        col.metric(seg, cnt)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(rfm_segment_chart(rfm), use_container_width=True)
    with c2:
        st.plotly_chart(rfm_scatter(rfm), use_container_width=True)

    with st.expander("View RFM Table"):
        st.dataframe(
            rfm.sort_values("RFM_Score", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

#  Tab 5: At-Risk 
with tab5:
    st.subheader("⚠️ At-Risk Customers")
    at_risk = get_at_risk(rfm, df)

    if at_risk.empty:
        st.success("No at-risk customers found with current filters.")
    else:
        st.warning(
            f"{len(at_risk)} customers flagged as at-risk — "
            "they bought frequently before but haven't purchased recently."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("At-Risk Customers",  len(at_risk))
        c2.metric("Avg Recency (days)", f"{at_risk['Recency'].mean():.0f}")
        c3.metric("Revenue at Risk",    f"£{at_risk['Monetary'].sum():,.0f}")

        st.dataframe(at_risk, use_container_width=True)

        fig_ar = go.Figure(go.Histogram(x=at_risk["Recency"], nbinsx=20))
        fig_ar.update_layout(title="Recency Distribution", xaxis_title="Days since last purchase")
        st.plotly_chart(fig_ar, use_container_width=True)

#  Tab 6: Product Velocity 
with tab6:
    st.subheader("Fast vs Slow Moving Products")
    vel_n = st.slider("Top / Bottom N", 5, 20, 10)
    st.plotly_chart(product_velocity(df, vel_n), use_container_width=True)

    st.subheader("Revenue per Product (Top 20)")
    rev_prod = (
        df.groupby("Description")["TotalPrice"].sum()
        .sort_values(ascending=False).head(20).reset_index()
    )
    rev_prod.columns = ["Product", "Revenue"]
    fig_rev = go.Figure(go.Bar(
        x=rev_prod["Product"].str[:35], y=rev_prod["Revenue"]
    ))
    fig_rev.update_layout(xaxis_tickangle=-45, yaxis_title="Revenue (£)")
    st.plotly_chart(fig_rev, use_container_width=True)
