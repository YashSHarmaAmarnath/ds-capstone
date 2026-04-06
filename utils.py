import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Country"
]


# 
# DATA LOADING & CLEANING
# 

def load_and_clean(file) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    df = pd.read_csv(file)

    stray = [c for c in df.columns if c.startswith("Unnamed")]
    if stray:
        df.drop(columns=stray, inplace=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    before = len(df)
    df = df[df["CustomerID"].notnull()]
    dropped = before - len(df)
    if dropped:
        warnings.append(f"Dropped {dropped:,} rows with missing CustomerID.")

    df["CustomerID"] = df["CustomerID"].astype(int)

    neg = (df["Quantity"] <= 0).sum()
    if neg:
        warnings.append(f"Removed {neg:,} rows with non-positive Quantity.")
    df = df[df["Quantity"] > 0]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Year"]       = df["InvoiceDate"].dt.year
    df["Month"]      = df["InvoiceDate"].dt.month
    df["Week"]       = df["InvoiceDate"].dt.isocalendar().week.astype(int)
    df["Weekday"]    = df["InvoiceDate"].dt.day_name()
    df["YearMonth"]  = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["YearWeek"]   = df["InvoiceDate"].dt.strftime("%Y-W%V")

    return df, warnings


# 
# KPI SUMMARY
# 

def get_kpis(df: pd.DataFrame) -> dict:
    return {
        "total_revenue":   df["TotalPrice"].sum(),
        "total_orders":    df["InvoiceNo"].nunique(),
        "total_customers": df["CustomerID"].nunique(),
        "total_products":  df["Description"].nunique(),
        "avg_order_value": df.groupby("InvoiceNo")["TotalPrice"].sum().mean(),
        "date_range": (df["InvoiceDate"].min(), df["InvoiceDate"].max()),
    }


# 
# SALES TRENDS
# 

def sales_trend(df: pd.DataFrame, freq: str = "Monthly") -> go.Figure:
    col = "YearMonth" if freq == "Monthly" else "YearWeek"
    trend = df.groupby(col)["TotalPrice"].sum().reset_index()
    trend.columns = ["Period", "Revenue"]

    fig = go.Figure(go.Scatter(
        x=trend["Period"], y=trend["Revenue"],
        mode="lines+markers",
        hovertemplate="<b>%{x}</b><br>Revenue: £%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title=f"{freq} Revenue Trend",
        xaxis_title="Period",
        yaxis_title="Revenue (£)",
        xaxis=dict(tickangle=-45),
    )
    return fig


# 
# TOP PRODUCTS
# 

def top_products_chart(df: pd.DataFrame, n: int = 10, metric: str = "Quantity") -> go.Figure:
    col = "Quantity" if metric == "Quantity" else "TotalPrice"
    top = (
        df.groupby("Description")[col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top.columns = ["Product", "Value"]
    top["Short"] = top["Product"].str[:40]

    label = "Units Sold" if metric == "Quantity" else "Revenue (£)"
    fig = go.Figure(go.Bar( 
        x=top["Value"][::-1],
        y=top["Short"][::-1],
        orientation="h",
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Top {n} Products by {label}",
        xaxis_title=label,
        margin=dict(l=220)
    )
    return fig


# 
# SALES BY COUNTRY
# 

def country_bar_chart(df: pd.DataFrame, n: int = 10) -> go.Figure:
    cs = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    cs.columns = ["Country", "Revenue"]

    fig = go.Figure(go.Bar(
        x=cs["Revenue"][::-1],
        y=cs["Country"][::-1],
        orientation="h",
        hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Top {n} Countries by Revenue",
        xaxis_title="Revenue (£)",
        margin=dict(l=150)
    )
    return fig


def country_map(df: pd.DataFrame) -> go.Figure:
    cm = df.groupby("Country", as_index=False)["TotalPrice"].sum()
    fig = px.choropleth(
        cm,
        locations="Country",
        locationmode="country names",
        color="TotalPrice",
        color_continuous_scale=[
            "#9fb2ff", "#f88282", "#ff4d4d", "#cc0000", "#800000"
        ],
        title="Global Revenue Distribution",
        labels={"TotalPrice": "Revenue (£)"}
    )

    fig.update_layout(
        title="Global Revenue Distribution",
        autosize=True,          # ← add this
        # width=700,            # ← remove or comment out any hardcoded width
        height=500,             # keep or adjust height as desired
        margin=dict(l=0, r=0, t=40, b=0),   # ← zero out margins
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="equirectangular",
            # remove any fixed width/height on the geo subplot too
        )
    )
    return fig


# 
# RFM SEGMENTATION
# 

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max()

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate",  lambda x: (snapshot - x.max()).days),
        Frequency = ("InvoiceNo",    "nunique"),
        Monetary  = ("TotalPrice",   "sum")
    ).reset_index()

    rfm["R_score"] = pd.qcut(rfm["Recency"],   4, labels=[4, 3, 2, 1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    rfm["M_score"] = pd.qcut(rfm["Monetary"],  4, labels=[1, 2, 3, 4])
    rfm["RFM_Score"] = rfm[["R_score", "F_score", "M_score"]].astype(int).sum(axis=1)

    def label(row):
        s = row["RFM_Score"]
        r = int(row["R_score"])
        f = int(row["F_score"])
        if s >= 10:           return "🏆 Champions"
        if s >= 8:            return "💛 Loyal"
        if r >= 3 and f <= 2: return "🌱 Promising"
        if r <= 2 and f >= 3: return "⚠️ At Risk"
        if s <= 4:            return "😴 Lost"
        return "🔄 Needs Attention"

    rfm["Segment"] = rfm.apply(label, axis=1)
    return rfm


def rfm_segment_chart(rfm: pd.DataFrame) -> go.Figure:
    counts = rfm["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]

    fig = go.Figure(go.Pie(
        labels=counts["Segment"],
        values=counts["Count"],
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>%{value} customers<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(title="Customer Segments")
    return fig


def rfm_scatter(rfm: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        rfm, x="Recency", y="Monetary",
        color="Segment",
        size="Frequency",
        hover_data={"CustomerID": True, "RFM_Score": True},
        title="RFM Scatter — Recency vs Monetary",
        labels={"Recency": "Recency (days)", "Monetary": "Monetary (£)"}
    )
    return fig


# 
# AT-RISK CUSTOMERS
# 

def get_at_risk(rfm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    at_risk_ids = rfm[rfm["Segment"] == "⚠️ At Risk"]["CustomerID"]
    details = rfm[rfm["CustomerID"].isin(at_risk_ids)][
        ["CustomerID", "Recency", "Frequency", "Monetary", "RFM_Score"]
    ].sort_values("Recency", ascending=False)

    last_country = (
        df.sort_values("InvoiceDate")
        .groupby("CustomerID")["Country"]
        .last()
        .reset_index()
    )
    details = details.merge(last_country, on="CustomerID", how="left")
    details["Monetary"] = details["Monetary"].round(2)
    return details.reset_index(drop=True)


# 
# FAST vs SLOW MOVING PRODUCTS
# 

def product_velocity(df: pd.DataFrame, n: int = 10) -> go.Figure:
    pv = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    pv.columns = ["Product", "UnitsSold"]
    pv["Short"] = pv["Product"].str[:35]

    fast = pv.head(n).copy()
    slow = pv.tail(n).copy()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"🚀 Fast-Moving (Top {n})", f"🐢 Slow-Moving (Bottom {n})"))

    fig.add_trace(go.Bar(
        x=fast["UnitsSold"], y=fast["Short"],
        orientation="h", name="Fast",
        marker_color="green"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=slow["UnitsSold"][::-1], y=slow["Short"][::-1],
        orientation="h", name="Slow",
        marker_color="red"
    ), row=1, col=2)

    fig.update_layout(
        title_text=f"Product Velocity — Top & Bottom {n}",
        showlegend=False,
        height=420,
        margin=dict(l=220, r=80)
    )
    return fig