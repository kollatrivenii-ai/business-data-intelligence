"""
Business Data Intelligence - simple base project
Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Business Data Intelligence", page_icon="📊", layout="wide")

REQUIRED_COLUMNS = {"date", "region", "category", "product", "quantity", "revenue", "cost"}


# ---------------------------------------------------------------- data layer
@st.cache_data
def generate_sample_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Create realistic-looking sales data so the app works out of the box."""
    rng = np.random.default_rng(seed)
    catalog = {
        "Electronics": {"Laptop": 55000, "Phone": 30000, "Headphones": 3000, "Monitor": 12000},
        "Furniture": {"Chair": 6000, "Desk": 14000, "Bookshelf": 8000},
        "Stationery": {"Notebook": 150, "Pen Set": 400, "Planner": 600},
        "Clothing": {"T-Shirt": 700, "Jeans": 1800, "Jacket": 4500},
    }
    rows = []
    dates = pd.date_range("2024-01-01", "2026-10-31")
    for _ in range(n_rows):
        category = rng.choice(list(catalog))
        product = rng.choice(list(catalog[category]))
        price = catalog[category][product]
        qty = int(rng.integers(1, 8))
        date = rng.choice(dates)
        growth = 1 + (date - dates[0]).days / 1500          # gentle upward trend
        revenue = price * qty * growth * rng.uniform(0.9, 1.1)
        cost = revenue * rng.uniform(0.60, 0.85)
        rows.append([date, rng.choice(["North", "South", "East", "West"]),
                     category, product, qty, round(revenue, 2), round(cost, 2)])
    return pd.DataFrame(rows, columns=["date", "region", "category", "product",
                                       "quantity", "revenue", "cost"])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"Your file is missing columns: {', '.join(sorted(missing))}")
        st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["profit"] = df["revenue"] - df["cost"]
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def forecast_revenue(monthly: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    """Very simple linear-trend forecast (replace with Prophet/ARIMA later)."""
    if len(monthly) < 3:
        return pd.DataFrame()
    x = np.arange(len(monthly))
    slope, intercept = np.polyfit(x, monthly["revenue"], 1)
    future_x = np.arange(len(monthly), len(monthly) + periods)
    future_dates = pd.date_range(monthly["month"].iloc[-1], periods=periods + 1, freq="MS")[1:]
    return pd.DataFrame({"month": future_dates, "revenue": slope * future_x + intercept})


# ------------------------------------------------------------------ sidebar
st.title("📊 Business Data Intelligence")
st.caption("Upload your sales data or explore the built-in sample dataset.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload CSV", type="csv")
    st.caption("Columns needed: " + ", ".join(sorted(REQUIRED_COLUMNS)))

data = prepare(pd.read_csv(upload) if upload else generate_sample_data())

with st.sidebar:
    st.header("Filters")
    min_d, max_d = data["date"].min().date(), data["date"].max().date()
    date_range = st.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    regions = st.multiselect("Region", sorted(data["region"].unique()), default=sorted(data["region"].unique()))
    categories = st.multiselect("Category", sorted(data["category"].unique()), default=sorted(data["category"].unique()))

if len(date_range) != 2:
    st.info("Select both a start and end date.")
    st.stop()

df = data[
    data["date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    & data["region"].isin(regions)
    & data["category"].isin(categories)
]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# --------------------------------------------------------------------- KPIs
revenue, profit, orders = df["revenue"].sum(), df["profit"].sum(), len(df)
margin = profit / revenue * 100 if revenue else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Revenue", f"₹{revenue:,.0f}")
c2.metric("Profit", f"₹{profit:,.0f}")
c3.metric("Profit margin", f"{margin:.1f}%")
c4.metric("Orders", f"{orders:,}")
c5.metric("Avg order value", f"₹{revenue / orders:,.0f}")

st.divider()

# ------------------------------------------------------------------- charts
tab1, tab2, tab3 = st.tabs(["Trends & Forecast", "Breakdowns", "Raw data"])

with tab1:
    monthly = df.groupby("month", as_index=False)[["revenue", "profit"]].sum()
    fig = px.line(monthly, x="month", y=["revenue", "profit"], markers=True,
                  title="Monthly revenue and profit")
    fc = forecast_revenue(monthly)
    if not fc.empty:
        fig.add_scatter(x=fc["month"], y=fc["revenue"], mode="lines+markers",
                        name="revenue forecast", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Forecast is a simple linear trend over monthly revenue.")

with tab2:
    left, right = st.columns(2)
    by_cat = df.groupby("category", as_index=False)["revenue"].sum()
    left.plotly_chart(px.pie(by_cat, names="category", values="revenue", hole=0.45,
                             title="Revenue by category"), use_container_width=True)

    by_region = df.groupby("region", as_index=False)[["revenue", "profit"]].sum()
    right.plotly_chart(px.bar(by_region, x="region", y=["revenue", "profit"], barmode="group",
                              title="Revenue and profit by region"), use_container_width=True)

    top = df.groupby("product", as_index=False)["revenue"].sum().nlargest(10, "revenue")
    st.plotly_chart(px.bar(top, x="revenue", y="product", orientation="h",
                           title="Top 10 products by revenue")
                    .update_yaxes(categoryorder="total ascending"), use_container_width=True)

with tab3:
    st.dataframe(df.drop(columns="month"), use_container_width=True)
    st.download_button("Download filtered data (CSV)",
                       df.drop(columns="month").to_csv(index=False).encode("utf-8"),
                       "filtered_data.csv", "text/csv")
