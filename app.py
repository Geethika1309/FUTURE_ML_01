import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.20);
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. HEADER
# =========================================================

st.title("📈 Sales Intelligence Dashboard")

st.caption(
    "AI-Powered Sales & Demand Forecasting | Future Interns ML Project"
)

st.write(
    "Explore business performance, analyze regional sales, "
    "forecast future revenue, and discover actionable insights."
)

st.divider()


# =========================================================
# 3. LOAD DATASET
# =========================================================

st.sidebar.title("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload sales dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read CSV: {error}")
        st.stop()

else:
    data_folder = Path("data")

    preferred_files = [
        data_folder / "Sample_Superstore_Forecasting_Practice.csv",
        data_folder / "Sample - Superstore.csv",
        data_folder / "Sample_Superstore.csv"
    ]

    dataset_path = next(
        (path for path in preferred_files if path.exists()),
        None
    )

    if dataset_path is None and data_folder.exists():
        csv_files = sorted(data_folder.glob("*.csv"))

        if csv_files:
            dataset_path = csv_files[0]

    if dataset_path is None:
        st.error(
            "Dataset not found. Upload a CSV using the sidebar "
            "or place your dataset inside the data folder."
        )
        st.stop()

    try:
        df = pd.read_csv(dataset_path)
    except Exception as error:
        st.error(f"Unable to load dataset: {error}")
        st.stop()


# =========================================================
# 4. CLEAN DATA
# =========================================================

df.columns = df.columns.astype(str).str.strip()

required_columns = ["Order Date", "Sales"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.write("Columns found:", df.columns.tolist())

    st.info(
        "The dataset needs an Order Date column and a Sales column."
    )

    st.stop()


original_rows = len(df)

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

df["Sales"] = pd.to_numeric(
    df["Sales"],
    errors="coerce"
)

df = df.dropna(
    subset=["Order Date", "Sales"]
)

df = df.drop_duplicates()

df = df.sort_values(
    "Order Date"
).reset_index(drop=True)

if df.empty:
    st.error("No valid sales records remain after cleaning.")
    st.stop()


# =========================================================
# 5. DATASET INFORMATION
# =========================================================

with st.expander("📁 Dataset Information"):

    c1, c2, c3 = st.columns(3)

    c1.metric("Valid Records", f"{len(df):,}")
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Removed Records", f"{original_rows - len(df):,}")

    st.write("Available columns:")
    st.write(df.columns.tolist())

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 6. SIDEBAR FILTERS
# =========================================================

st.sidebar.divider()

st.sidebar.subheader("🔍 Filter Sales")

minimum_date = df["Order Date"].min().date()
maximum_date = df["Order Date"].max().date()

start_date = st.sidebar.date_input(
    "Start date",
    value=minimum_date,
    min_value=minimum_date,
    max_value=maximum_date
)

end_date = st.sidebar.date_input(
    "End date",
    value=maximum_date,
    min_value=minimum_date,
    max_value=maximum_date
)

if start_date > end_date:
    st.error("Start date cannot be later than end date.")
    st.stop()


# Region filter

if "Region" in df.columns:

    region_values = (
        df["Region"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    region_values = sorted(region_values)

    selected_region = st.sidebar.selectbox(
        "🌍 Select Region",
        ["All Regions"] + region_values
    )

else:

    selected_region = "All Regions"

    st.sidebar.warning(
        "No Region column found. Regional filtering is unavailable."
    )


# Category filter

if "Category" in df.columns:

    category_values = sorted(
        df["Category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "🛍️ Select Categories",
        category_values,
        default=category_values
    )

else:

    selected_categories = None


# Forecast settings

st.sidebar.divider()

forecast_days = st.sidebar.select_slider(
    "🔮 Forecast horizon",
    options=[7, 15, 30, 45, 60],
    value=30,
    format_func=lambda x: f"{x} days"
)

model_choice = st.sidebar.selectbox(
    "🧠 Forecasting model",
    ["Linear Regression", "Random Forest"]
)


# =========================================================
# 7. APPLY GLOBAL FILTERS
# =========================================================

filtered_df = df[
    (df["Order Date"].dt.date >= start_date)
    & (df["Order Date"].dt.date <= end_date)
].copy()

if selected_region != "All Regions":
    filtered_df = filtered_df[
        filtered_df["Region"].astype(str) == selected_region
    ]

if selected_categories is not None:
    filtered_df = filtered_df[
        filtered_df["Category"].astype(str).isin(
            selected_categories
        )
    ]


# Region-specific history for forecasting.
# The forecast uses all available historical dates up to the
# latest date in the dataset, not just the selected chart range.

forecast_history_df = df.copy()

if selected_region != "All Regions":
    forecast_history_df = forecast_history_df[
        forecast_history_df["Region"].astype(str) == selected_region
    ]

if selected_categories is not None:
    forecast_history_df = forecast_history_df[
        forecast_history_df["Category"].astype(str).isin(
            selected_categories
        )
    ]


# =========================================================
# 8. DAILY SALES AGGREGATION
# =========================================================

def create_daily_sales(data):

    if data.empty:
        return pd.Series(dtype=float)

    daily = (
        data.groupby(
            data["Order Date"].dt.normalize()
        )["Sales"]
        .sum()
        .sort_index()
    )

    calendar = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(calendar)

    # This is a demonstration assumption. A missing date may
    # mean no sales OR incomplete records.
    daily = daily.fillna(0)

    daily.index.name = "Order Date"

    return daily


daily_sales = create_daily_sales(
    forecast_history_df
)

if len(daily_sales) < 45:

    st.error(
        "At least 45 calendar days of historical sales are "
        "required for this forecasting demonstration. "
        "Choose a dataset with more history or broaden your filters."
    )

    st.stop()


# =========================================================
# 9. FEATURE ENGINEERING
# =========================================================

def create_features(series):

    result = pd.DataFrame(
        {"Sales": series}
    )

    result["Year"] = result.index.year
    result["Month"] = result.index.month
    result["Day"] = result.index.day
    result["DayOfWeek"] = result.index.dayofweek
    result["DayOfYear"] = result.index.dayofyear

    result["Lag1"] = result["Sales"].shift(1)
    result["Lag7"] = result["Sales"].shift(7)

    result["RollingMean7"] = (
        result["Sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    return result.dropna()


FEATURES = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "Lag1",
    "Lag7",
    "RollingMean7"
]

model_data = create_features(daily_sales)

if len(model_data) < 30:
    st.error("Not enough usable observations after feature engineering.")
    st.stop()


# =========================================================
# 10. TRAIN AND EVALUATE MODELS
# =========================================================

X = model_data[FEATURES]
y = model_data["Sales"]

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

if len(y_test) < 2:
    st.error("Not enough test observations for evaluation.")
    st.stop()


def make_model(name):

    if name == "Linear Regression":
        return LinearRegression()

    return RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )


def evaluate_model(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }


linear_model = make_model("Linear Regression")
linear_model.fit(X_train, y_train)

linear_predictions = np.maximum(
    0,
    linear_model.predict(X_test)
)


rf_model = make_model("Random Forest")
rf_model.fit(X_train, y_train)

rf_predictions = np.maximum(
    0,
    rf_model.predict(X_test)
)


# Naive baseline: predict today's sales using yesterday's actual sales.

baseline_predictions = (
    model_data["Sales"]
    .shift(1)
    .loc[y_test.index]
    .to_numpy()
)

baseline_predictions = np.maximum(
    0,
    baseline_predictions
)


linear_metrics = evaluate_model(
    y_test,
    linear_predictions
)

rf_metrics = evaluate_model(
    y_test,
    rf_predictions
)

baseline_metrics = evaluate_model(
    y_test,
    baseline_predictions
)


# =========================================================
# 11. FORECAST FUTURE SALES
# =========================================================

selected_model = make_model(model_choice)

selected_model.fit(X, y)

history = daily_sales.copy()

future_dates = pd.date_range(
    start=history.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)

future_predictions = []

for future_date in future_dates:

    row = pd.DataFrame(
        [[
            future_date.year,
            future_date.month,
            future_date.day,
            future_date.dayofweek,
            future_date.dayofyear,
            history.iloc[-1],
            history.iloc[-7],
            history.iloc[-7:].mean()
        ]],
        columns=FEATURES
    )

    prediction = float(
        selected_model.predict(row)[0]
    )

    prediction = max(0, prediction)

    future_predictions.append(prediction)

    history.loc[future_date] = prediction


forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": np.round(
        future_predictions,
        2
    )
})


forecast_total = forecast_df["Predicted Sales"].sum()

forecast_average = forecast_df["Predicted Sales"].mean()

historical_revenue = forecast_history_df["Sales"].sum()

average_daily_sales = daily_sales.mean()

selected_metrics = (
    linear_metrics
    if model_choice == "Linear Regression"
    else rf_metrics
)


# =========================================================
# 12. EXECUTIVE DASHBOARD
# =========================================================

st.header("📊 Executive Overview")

if selected_region == "All Regions":
    region_label = "All Regions"
else:
    region_label = selected_region

st.caption(
    f"Region: {region_label} | "
    f"Chart dates: {start_date} to {end_date} | "
    f"Forecast: next {forecast_days} days"
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Historical Revenue",
    f"{historical_revenue:,.2f}"
)

k2.metric(
    "Average Daily Sales",
    f"{average_daily_sales:,.2f}"
)

k3.metric(
    f"{forecast_days}-Day Forecast",
    f"{forecast_total:,.2f}"
)

k4.metric(
    "Test MAE",
    f"{selected_metrics['MAE']:,.2f}"
)

st.divider()


# =========================================================
# 13. DASHBOARD TABS
# =========================================================

overview_tab, explorer_tab, forecast_tab, model_tab, insights_tab = st.tabs([
    "🏠 Overview",
    "🔎 Sales Explorer",
    "🔮 Forecast Studio",
    "🧠 Model Lab",
    "💡 Business Insights"
])


# =========================================================
# 14. OVERVIEW TAB
# =========================================================

with overview_tab:

    st.subheader(
        f"Sales Trend — {region_label}"
    )

    overview_data = filtered_df.copy()

    if overview_data.empty:

        st.warning(
            "No records match your current filters."
        )

    else:

        overview_daily = (
            overview_data.groupby(
                overview_data["Order Date"].dt.normalize()
            )["Sales"]
            .sum()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            overview_daily.index,
            overview_daily.values,
            linewidth=2
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Sales Revenue")
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.subheader("Monthly Sales")

        monthly = overview_data.copy()

        monthly["Month"] = (
            monthly["Order Date"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_sales = (
            monthly.groupby("Month")["Sales"]
            .sum()
        )

        st.bar_chart(
            monthly_sales,
            use_container_width=True
        )

    st.subheader("Future Sales Forecast")

    fig, ax = plt.subplots(figsize=(12, 4))

    recent_history = daily_sales.tail(60)

    ax.plot(
        recent_history.index,
        recent_history.values,
        label="Historical Sales"
    )

    ax.plot(
        forecast_df["Date"],
        forecast_df["Predicted Sales"],
        linestyle="--",
        marker="o",
        markersize=3,
        label="Predicted Sales"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Revenue")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# =========================================================
# 15. SALES EXPLORER TAB
# =========================================================

with explorer_tab:

    st.subheader(
        f"Explore Sales — {region_label}"
    )

    if filtered_df.empty:

        st.warning(
            "No records match your selected filters."
        )

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Revenue",
            f"{filtered_df['Sales'].sum():,.2f}"
        )

        c2.metric(
            "Records",
            f"{len(filtered_df):,}"
        )

        c3.metric(
            "Average Transaction",
            f"{filtered_df['Sales'].mean():,.2f}"
        )

        period = st.radio(
            "Choose time grouping",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True
        )

        chart_source = filtered_df.copy()

        if period == "Daily":

            chart_data = (
                chart_source.groupby(
                    chart_source["Order Date"].dt.normalize()
                )["Sales"]
                .sum()
            )

        elif period == "Weekly":

            chart_data = (
                chart_source.groupby(
                    chart_source["Order Date"].dt.to_period("W")
                )["Sales"]
                .sum()
            )

            chart_data.index = chart_data.index.astype(str)

        else:

            chart_data = (
                chart_source.groupby(
                    chart_source["Order Date"].dt.to_period("M")
                )["Sales"]
                .sum()
            )

            chart_data.index = chart_data.index.astype(str)

        st.subheader(f"{period} Sales")

        st.line_chart(
            chart_data,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        if "Category" in chart_source.columns:

            with col1:

                st.subheader("Revenue by Category")

                category_sales = (
                    chart_source.groupby("Category")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                )

                st.bar_chart(category_sales)

        if "Region" in chart_source.columns:

            with col2:

                st.subheader("Revenue by Region")

                region_sales = (
                    chart_source.groupby("Region")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                )

                st.bar_chart(region_sales)

        product_column = next(
            (
                column for column in [
                    "Product Name",
                    "Product"
                ]
                if column in chart_source.columns
            ),
            None
        )

        if product_column is not None:

            st.subheader("🏆 Top 10 Products")

            product_sales = (
                chart_source.groupby(product_column)["Sales"]
                .sum()
                .nlargest(10)
                .sort_values()
            )

            st.bar_chart(product_sales)

        with st.expander("View Sales Records"):

            st.dataframe(
                chart_source,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Filtered Dataset",
                data=chart_source.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="filtered_sales.csv",
                mime="text/csv"
            )


# =========================================================
# 16. FORECAST STUDIO TAB
# =========================================================

with forecast_tab:

    st.subheader(
        f"Forecast — {region_label}"
    )

    st.write(
        f"Selected model: **{model_choice}**"
    )

    st.metric(
        f"Predicted Revenue for {forecast_days} Days",
        f"{forecast_total:,.2f}"
    )

    st.metric(
        "Average Predicted Daily Revenue",
        f"{forecast_average:,.2f}"
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    recent_history = daily_sales.tail(60)

    ax.plot(
        recent_history.index,
        recent_history.values,
        label="Historical Sales"
    )

    ax.plot(
        forecast_df["Date"],
        forecast_df["Predicted Sales"],
        linestyle="--",
        marker="o",
        markersize=3,
        label="Forecast"
    )

    ax.axvline(
        daily_sales.index[-1],
        linestyle=":",
        label="Forecast Start"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Revenue")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 Download Forecast CSV",
        data=forecast_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv"
    )


# =========================================================
# 17. MODEL LAB TAB
# =========================================================

with model_tab:

    st.subheader(
        f"Model Evaluation — {region_label}"
    )

    metrics_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Naive Baseline"
        ],

        "MAE": [
            linear_metrics["MAE"],
            rf_metrics["MAE"],
            baseline_metrics["MAE"]
        ],

        "RMSE": [
            linear_metrics["RMSE"],
            rf_metrics["RMSE"],
            baseline_metrics["RMSE"]
        ],

        "R²": [
            linear_metrics["R2"],
            rf_metrics["R2"],
            baseline_metrics["R2"]
        ]
    })

    st.dataframe(
        metrics_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    best_model = metrics_df.loc[
        metrics_df["MAE"].idxmin(),
        "Model"
    ]

    st.success(
        f"Lowest test-set MAE: {best_model}"
    )

    st.caption(
        "A lower MAE indicates smaller average absolute errors. "
        "Compare this metric with the baseline before choosing a model."
    )

    st.subheader("Actual vs Predicted Sales")

    comparison_df = pd.DataFrame(
        {
            "Actual Sales": y_test.to_numpy(),
            "Linear Regression": linear_predictions,
            "Random Forest": rf_predictions,
            "Naive Baseline": baseline_predictions
        },
        index=y_test.index
    )

    st.line_chart(
        comparison_df,
        use_container_width=True
    )

    st.markdown("""
    **MAE:** Average absolute prediction error.

    **RMSE:** Error measure that penalizes large mistakes more heavily.

    **R²:** Measures how much variation in the test data is explained
    by the model. It can be negative.

    **Naive baseline:** Uses the previous day's actual sales as
    a simple reference prediction.
    """)


# =========================================================
# 18. BUSINESS INSIGHTS TAB
# =========================================================

with insights_tab:

    st.subheader(
        f"Business Insights — {region_label}"
    )

    st.metric(
        "Historical Revenue",
        f"{historical_revenue:,.2f}"
    )

    st.metric(
        f"Forecast Revenue ({forecast_days} Days)",
        f"{forecast_total:,.2f}"
    )

    st.metric(
        "Average Daily Forecast",
        f"{forecast_average:,.2f}"
    )

    highest_day = forecast_df.loc[
        forecast_df["Predicted Sales"].idxmax()
    ]

    lowest_day = forecast_df.loc[
        forecast_df["Predicted Sales"].idxmin()
    ]

    left, right = st.columns(2)

    with left:

        st.markdown("### 📈 Highest Forecast")

        st.metric(
            "Predicted Sales",
            f"{highest_day['Predicted Sales']:,.2f}"
        )

        st.write(
            f"Date: {highest_day['Date'].date()}"
        )

    with right:

        st.markdown("### 📉 Lowest Forecast")

        st.metric(
            "Predicted Sales",
            f"{lowest_day['Predicted Sales']:,.2f}"
        )

        st.write(
            f"Date: {lowest_day['Date'].date()}"
        )

    st.divider()

    if not filtered_df.empty and "Category" in filtered_df.columns:

        category_sales = (
            filtered_df.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if not category_sales.empty:

            st.markdown("### 🛍️ Top Category")

            st.write(
                f"{category_sales.index[0]} is the highest-revenue "
                f"category in the selected view, generating "
                f"{category_sales.iloc[0]:,.2f} in sales."
            )

    if not filtered_df.empty and "Region" in filtered_df.columns:

        region_sales = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if not region_sales.empty:

            st.markdown("### 🌍 Regional Comparison")

            st.bar_chart(region_sales)

            st.write(
                f"Highest-revenue region in the selected view: "
                f"**{region_sales.idxmax()}**."
            )

    st.markdown("### 🏢 Business Recommendations")

    st.markdown("""
    - **Revenue planning:** Use forecast revenue as an input
      to financial planning.
    - **Staffing:** Investigate predicted busy periods.
    - **Marketing:** Review regional and category sales trends
      before planning campaigns.
    - **Inventory:** Combine demand forecasts with product-level
      quantities, stock levels, and supplier lead times.
    - **Monitoring:** Compare actual future sales with predictions
      to understand model performance.
    """)

    st.warning(
        "These predictions estimate sales revenue, not units sold. "
        "The model does not explicitly account for promotions, "
        "holidays, stock availability, or price changes. "
        "Treat forecasts as planning estimates, not guarantees."
    )


# =========================================================
# 19. FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Python, Streamlit, Pandas, NumPy, "
    "Matplotlib, and Scikit-learn."
)

st.caption(
    "Future Interns | Machine Learning Task 1: "
    "Sales & Demand Forecasting"
)