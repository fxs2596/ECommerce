# app.py
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px # Import Plotly Express
import datetime # Needed for date calculations
import numpy as np # Import numpy as np - REQUIRED for np.inf, np.nan

# --- Database Connection and Data Loading ---
@st.cache_data
def load_data():
    # Connect inside the cache function
    # Use absolute path or path relative to the script if db is not in the same directory
    conn = sqlite3.connect("mytest.db")
    df_orders = pd.read_sql("SELECT * FROM orders", conn)
    df_customers = pd.read_sql("SELECT * FROM customers", conn) # Corrected variable name
    conn.close()

    # Convert order_date and create month column ONCE here
    # Ensure order_date column exists and is not null for conversion
    if 'order_date' in df_orders.columns and df_orders['order_date'].notna().all():
        df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
        df_orders['month'] = df_orders['order_date'].dt.to_period("M").astype(str)
    else:
        st.error("Order date column is missing or contains missing/invalid data.")
        df_orders['month'] = 'Unknown' # Handle error case


    return df_orders, df_customers

# Load the original data (cached)
df_orders_original, df_customers_original = load_data()


# --- Prepare Data for ML Churn Prediction ---
print("\n--- Preparing Data for ML Churn Prediction ---")

# 1. Establish the Snapshot Point (Latest date in the order data)
if 'order_date' in df_orders_original.columns and not df_orders_original.empty:
    snapshot_date = df_orders_original['order_date'].max()
    print(f"\nSnapshot date for churn analysis: {snapshot_date.strftime('%Y-%m-%d')}")
else:
    # If essential data is missing, cannot prepare ML data. Handle gracefully.
    st.error("Order date column missing or data is empty. Cannot prepare ML data.")
    snapshot_date = None # Set to None if data is missing
    df_ml = pd.DataFrame() # Ensure df_ml is empty if data is missing


# Check if snapshot_date was successfully determined before proceeding with ML prep
if snapshot_date is not None and 'customer_id' in df_orders_original.columns and 'order_date' in df_orders_original.columns and 'amount' in df_orders_original.columns and 'order_id' in df_orders_original.columns:

    # 2. Define the Churn Cutoff (Snapshot date - 180 days)
    churn_window_days = 180
    churn_cutoff_date = snapshot_date - pd.Timedelta(days=churn_window_days)
    print(f"Churn cutoff date ({churn_window_days} days inactivity): {churn_cutoff_date.strftime('%Y-%m-%d')}")


    # 3. Identify Each Customer's Key Dates & Metrics
    # Group by customer to get aggregated metrics
    customer_data = df_orders_original.groupby('customer_id').agg(
        FirstOrderDate=('order_date', 'min'),
        LastOrderDate=('order_date', 'max'),
        Frequency=('order_id', 'count'), # Total number of orders
        Monetary=('amount', 'sum')
    ).reset_index()

    print(f"\nAggregated key data for {len(customer_data)} customers.")


    # 4. Calculate Core ML Features
    print("\nCalculating ML features per customer...")
    # Recency (in days from the snapshot date)
    customer_data['Recency'] = (snapshot_date - customer_data['LastOrderDate']).dt.days

    # Days Since First Order (Tenure in days from the snapshot date)
    customer_data['DaysSinceFirstOrder'] = (snapshot_date - customer_data['FirstOrderDate']).dt.days

    # Average Order Value
    # Avoid division by zero if Frequency is 0 (shouldn't happen with groupby, but safe)
    customer_data['AverageOrderValue'] = customer_data['Monetary'] / customer_data['Frequency']
    # Handle potential NaN/Inf if Frequency is somehow 0
    customer_data['AverageOrderValue'] = customer_data['AverageOrderValue'].replace([np.inf, -np.inf], np.nan).fillna(0) # Replace Inf with NaN, then fill NaN with 0


    # 5. Create the Churn Label (Y)
    print(f"Assigning churn label based on inactivity > {churn_window_days} days...")
    # Churn = 1 if LastOrderDate < ChurnCutoffDate, else 0
    customer_data['churn'] = (customer_data['LastOrderDate'] < churn_cutoff_date).astype(int)

    # --- Handle customers whose *first* order is after the churn cutoff? ---
    # These customers haven't had a chance to "churn" according to the definition.
    # A common approach is to exclude them from the training data if their tenure is < churn_window_days.
    # For simplicity in the first pass, we label all customers based on the cutoff.
    # If a customer's FirstOrderDate is after the ChurnCutoffDate, their Recency will be < 180,
    # and they will naturally be labeled as 'not churned'. This is acceptable.


    # 6. Assemble the Dataset (Select Final Columns)
    print("\nAssembling final ML dataset (customer-level)...")
    # Ensure 'id' and 'name' from df_customers_original are merged for the final ML df
    df_ml = customer_data.merge(df_customers_original[['id', 'name']], left_on='customer_id', right_on='id', how='left')

    # Define the final columns for the ML dataset
    ml_features = ['Recency', 'Frequency', 'Monetary', 'DaysSinceFirstOrder', 'AverageOrderValue']
    ml_columns = ['customer_id', 'name'] + ml_features + ['churn']

    df_ml = df_ml[ml_columns] # Select and order columns

    print(f"\nML Dataset created with shape: {df_ml.shape}")
    print("ML Dataset columns:")
    print(df_ml.columns.tolist())

    print("\nML Dataset Head:")
    # Display head in Streamlit (optional, but good for verification)
    # st.write("ML Dataset Head:")
    # st.dataframe(df_ml.head())

    print("\nDistribution of Churn Label in ML Dataset:")
    churn_distribution = df_ml['churn'].value_counts(normalize=True) * 100
    print(churn_distribution)
    # st.write("Churn Label Distribution:")
    # st.dataframe(churn_distribution) # Display distribution in Streamlit

    # --- Save the ML Dataset to CSV ---
    output_csv_filename = 'customer_churn_ml_dataset.csv'
    try:
        df_ml.to_csv(output_csv_filename, index=False)
        print(f"\nML dataset saved successfully to '{output_csv_filename}'")
        # st.success(f"ML dataset saved successfully to '{output_csv_filename}'") # Show success message in Streamlit
    except Exception as e:
         print(f"\nError saving ML dataset to CSV: {e}")
         # st.error(f"Error saving ML dataset to CSV: {e}")


else:
    # If essential data was missing, df_ml was set to empty earlier
    print("\nML Data Preparation Skipped due to missing essential data.")
    # Ensure df_ml is explicitly defined as empty if data was missing
    if 'df_ml' not in locals():
         df_ml = pd.DataFrame() # Define df_ml as empty if data was missing


print("\nML Data Preparation Complete (including CSV export if successful).")


# --- Dashboard Title and Filters ---
st.title("🛍️ E-Commerce Analytics Dashboard")
st.markdown("Interactive insights on customer orders and revenue.")

st.sidebar.header("Date Filter")
# df_orders_display calculation is here
# df_orders_original is already loaded

df_orders_display = df_orders_original.copy()

if 'order_date' in df_orders_original.columns:
    min_date = df_orders_original['order_date'].min().date()
    max_date = df_orders_original['order_date'].max().date()

    start_date = st.sidebar.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
        df_orders_display = pd.DataFrame(columns=df_orders_original.columns)
    else:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)

        df_orders_display = df_orders_original[
            (df_orders_original['order_date'] >= start_datetime) &
            (df_orders_original['order_date'] <= end_datetime)
        ].copy()
else:
    st.warning("Order date column not found, date filter disabled.")
    df_orders_display = df_orders_original.copy()


# --- Tabs for Organization ---
tab1, tab2, tab3 = st.tabs(["Overview", "Trends", "Customers"])

# --- Tab 1: Overview ---
with tab1:
    st.header("Dashboard Overview")

    total_revenue = df_orders_original['amount'].sum()
    total_orders = len(df_orders_original)
    unique_customers = df_orders_original['customer_id'].nunique()
    average_order_value = total_revenue / total_orders if total_orders > 0 else 0
    orders_per_customer = total_orders / unique_customers if unique_customers > 0 else 0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    kpi_col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    kpi_col2.metric("📦 Total Orders", total_orders)
    kpi_col3.metric("👥 Unique Customers", unique_customers)

    kpi_col4.metric("🛒 Avg. Order Value", f"${average_order_value:,.2f}")
    kpi_col5.metric("📈 Orders/Customer", f"{orders_per_customer:.2f}")


    st.subheader("Summary for Selected Date Range")
    filtered_revenue = df_orders_display['amount'].sum()
    filtered_orders = len(df_orders_display)
    filtered_unique_customers = df_orders_display['customer_id'].nunique()
    filtered_average_order_value = filtered_revenue / filtered_orders if filtered_orders > 0 else 0

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Filtered Revenue", f"${filtered_revenue:,.2f}")
    summary_col2.metric("Filtered Orders", filtered_orders)
    summary_col3.metric("Filtered Customers", filtered_unique_customers)
    summary_col4.metric("Filtered AOV", f"${filtered_average_order_value:,.2f}")


# --- Tab 2: Trends ---
with tab2:
    st.header("Sales Trends Analysis")

    st.subheader("📈 Monthly Revenue Trend (Filtered Date Range)")
    monthly = df_orders_display.groupby('month')['amount'].sum().reset_index()

    if not monthly.empty:
        fig_monthly = px.line(
            monthly,
            x='month',
            y='amount',
            markers=True,
            title="Monthly Revenue (Filtered by Date Range)"
        )
        fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_monthly, use_container_width=True)

        monthly_revenue_sql_query = """
SELECT
    strftime('%Y-%m', order_date) as month, -- Assuming order_date format allows this
    SUM(amount) as total_revenue
FROM
    orders
"""
        if 'order_date' in df_orders_original.columns and start_date != df_orders_original['order_date'].min().date() or end_date != df_orders_original['order_date'].max().date():
             monthly_revenue_sql_query += f"\nWHERE order_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')"

        monthly_revenue_sql_query += """
GROUP BY
    1
ORDER BY
    1;
"""
        with st.expander("See the SQL query for Monthly Revenue"):
            st.code(monthly_revenue_sql_query, language="sql")
            st.markdown(f"*(Conceptual query, inspired by analysis in [repo](https://github.com/fxs2596/ECommerce))*")

    else:
        st.warning("No data available for the selected date range to show monthly trend.")

    st.subheader("📉 Daily Revenue Trend (Filtered Date Range)")
    daily = df_orders_display.groupby(df_orders_display['order_date'].dt.date)['amount'].sum().reset_index()
    daily['order_date'] = pd.to_datetime(daily['order_date'])

    if not daily.empty:
        fig_daily = px.line(
            daily,
            x='order_date',
            y='amount',
            title="Daily Revenue (Filtered by Date Range)"
        )
        fig_daily.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_daily, use_container_width=True)

        daily_revenue_sql_query = """
SELECT
    DATE(order_date) as order_day,
    SUM(amount) as total_revenue
FROM
    orders
"""
        if 'order_date' in df_orders_original.columns and start_date != df_orders_original['order_date'].min().date() or end_date != df_orders_original['order_date'].max().date():
             daily_revenue_sql_query += f"\nWHERE order_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')"

        daily_revenue_sql_query += """
GROUP BY
    1
ORDER BY
    1;
"""
        with st.expander("See the SQL query for Daily Revenue"):
            st.code(daily_revenue_sql_query, language="sql")
            st.markdown(f"*(Conceptual query, inspired by analysis in [repo](https://github.com/fxs2596/ECommerce))*")

    else:
        st.warning("No data available for the selected date range to show daily trend.")


# --- Tab 3: Customers ---
with tab3:
    st.header("Customer Insights")
    st.subheader("📊 RFM Analysis & Segmentation")

    # --- RFM Analysis Calculation (on FULL data) ---
    # This is the main IF block for RFM calculation validity
    if 'order_date' in df_orders_original.columns and 'customer_id' in df_orders_original.columns and 'amount' in df_orders_original.columns:
        latest_date_for_rfm = df_orders_original['order_date'].max()

        # Calculate raw RFM metrics
        rfm_df = df_orders_original.groupby('customer_id').agg(
            Recency=('order_date', lambda date: (latest_date_for_rfm - date.max()).days),
            Frequency=('order_id', 'count'), # Assuming order_id counts as a distinct order
            Monetary=('amount', 'sum')
        ).reset_index()

        # Merge with customer names
        rfm_df = rfm_df.merge(df_customers_original[['id', 'name']], left_on='customer_id', right_on='id', how='left')
        rfm_df = rfm_df[['customer_id', 'name', 'Recency', 'Frequency', 'Monetary']] # Select and reorder columns


        # --- RFM Scoring (e.g., Quartiles) ---
        # Define scoring function (Reverse for Recency, Standard for Frequency/Monetary)
        def score_rfm(df):
             if df.empty:
                 return pd.DataFrame(columns=['customer_id', 'name', 'Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score'])

             # Use rank(method='first') to handle ties before qcut
             # Recency: Lower is better, so reverse rank/labels
             # Handle cases where all Recency values might be the same (qcut needs unique bin edges)
             try:
                 df['R_rank'] = df['Recency'].rank(method='first', ascending=False)
                 # Ensure labels cover all possible ranks or handle fewer unique ranks
                 # Use unique ranks if less than 4 bins are possible
                 num_unique_ranks_r = df['R_rank'].nunique()
                 q_labels_r = [1, 2, 3, 4][:min(4, num_unique_ranks_r)]
                 if q_labels_r: # Only attempt qcut if there are labels possible
                     # Use 'duplicates='drop' to handle bins with identical values
                     df['R_score'] = pd.qcut(df['R_rank'], q=min(4, num_unique_ranks_r), labels=q_labels_r, duplicates='drop').astype(int)
                 else:
                     df['R_score'] = 1 # Default score if no variance

             except Exception as e:
                 # st.warning(f"Error scoring Recency: {e}. Assigning default score.") # Avoid warning storm if expected
                 df['R_score'] = 1 # Assign lowest score if qcut fails (e.g., all same recency)


             try:
                 # Frequency: Higher is better
                 df['F_rank'] = df['Frequency'].rank(method='first', ascending=True)
                 num_unique_ranks_f = df['F_rank'].nunique()
                 q_labels_f = [1, 2, 3, 4][:min(4, num_unique_ranks_f)]
                 if q_labels_f:
                      df['F_score'] = pd.qcut(df['F_rank'], q=min(4, num_unique_ranks_f), labels=q_labels_f, duplicates='drop').astype(int) # Labels 1-4
                 else:
                     df['F_score'] = 1

             except Exception as e:
                  # st.warning(f"Error scoring Frequency: {e}. Assigning default score.")
                  df['F_score'] = 1 # Assign lowest score if qcut fails

             try:
                 # Monetary: Higher is better
                 df['M_rank'] = df['Monetary'].rank(method='first', ascending=True)
                 num_unique_ranks_m = df['M_rank'].nunique()
                 q_labels_m = [1, 2, 3, 4][:min(4, num_unique_ranks_m)]
                 if q_labels_m:
                     df['M_score'] = pd.qcut(df['M_rank'], q=min(4, num_unique_ranks_m), labels=q_labels_m, duplicates='drop').astype(int) # Labels 1-4
                 else:
                     df['M_score'] = 1

             except Exception as e:
                  # st.warning(f"Error scoring Monetary: {e}. Assigning default score.")
                  df['M_score'] = 1 # Assign lowest score if qcut fails


             return df.drop(columns=['R_rank', 'F_rank', 'M_rank']) # Drop rank columns


        # Apply scoring
        rfm_scored_df = score_rfm(rfm_df.copy()) # Work on a copy


        # --- RFM Segmentation ---
        # Function to map scores to segment (simplified based on R and F)
        def simple_segment(row):
            # Handle cases where scores might be missing if qcut failed
            if pd.isna(row['R_score']) or pd.isna(row['F_score']):
                 return 'Unknown'

            # Use the integer scores for comparison
            r_score = int(row['R_score'])
            f_score = int(row['F_score'])

            if r_score >= 3 and f_score >= 3: return 'Champions/Loyal'
            if r_score <= 2 and f_score <= 2: return 'At Risk/Lost' # Low R & F
            if r_score <= 2 and f_score >= 3: return 'Needs Attention' # At Risk based on Recency, but high F
            if r_score >= 3 and f_score <= 2: return 'New/Promising' # High R, Low F - but recent order
            return 'Others' # Catch-all based on R and F

        if not rfm_scored_df.empty:
            rfm_scored_df['RFM_Segment'] = rfm_scored_df.apply(simple_segment, axis=1)
            st.write("RFM metrics and scores calculated per customer.")

            # Add RFM Score String for display/hover
            # Ensure scores are treated as integers before concatenating
            rfm_scored_df['RFM_Score_String'] = rfm_scored_df['R_score'].astype(str) + rfm_scored_df['F_score'].astype(str) + rfm_scored_df['M_score'].astype(str)


            # --- Display RFM Data ---
            st.write("Customer RFM Data Sample (with Scores and Segments):")
            # Add a filter for segment
            selected_segment = st.selectbox("Filter RFM by Segment", options=["All"] + sorted(rfm_scored_df['RFM_Segment'].unique()))
            if selected_segment != "All":
                display_rfm_df = rfm_scored_df[rfm_scored_df['RFM_Segment'] == selected_segment].copy() # Use copy after filter
            else:
                 display_rfm_df = rfm_scored_df.copy()

            st.dataframe(display_rfm_df[['customer_id', 'name', 'Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_Segment', 'RFM_Score_String']].head(20), use_container_width=True) # Displaying head, maybe more rows


            # --- Segment Distribution Chart ---
            st.subheader("Customer Segment Distribution")
            segment_counts = rfm_scored_df['RFM_Segment'].value_counts().reset_index() # Use the full data for segment counts
            segment_counts.columns = ['Segment', 'Customer Count']

            if not segment_counts.empty:
                fig_segments = px.bar(
                    segment_counts,
                    x='Segment',
                    y='Customer Count',
                    title="Distribution of Customers by RFM Segment"
                )
                st.plotly_chart(fig_segments, use_container_width=True)


            # --- RFM Scatter Plot ---
            st.subheader("RFM Scatter Plot")
            st.write("Frequency vs Monetary (Color by Recency, Size by Monetary, Hover for Name/Segment)")
            # Use the full rfm_scored_df for the scatter plot usually, as it shows the overall customer landscape
            if not rfm_scored_df.empty:
                fig_rfm_scatter = px.scatter(
                    rfm_scored_df, # Use full data for scatter
                    x='Frequency',
                    y='Monetary',
                    hover_name='name',
                    hover_data=['Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_Segment', 'RFM_Score_String'], # Add scores to hover
                    size='Monetary',
                    color='Recency',
                    color_continuous_scale=px.colors.sequential.Viridis_r, # Reverse color scale (lower Recency = greener/brighter)
                    title="Customer RFM (Frequency vs Monetary, Colored by Recency)"
                )
                st.plotly_chart(fig_rfm_scatter, use_container_width=True)

            # Add conceptual SQL for RFM calculation
            rfm_sql_query = """
-- Conceptual SQL for RFM Calculation, Scoring, and Segmentation
-- Steps:
-- 1. Calculate Recency, Frequency, Monetary per customer (Aggregations).
-- 2. Rank customers based on R, F, M (Window Functions like RANK() or NTILE()).
-- 3. Assign R, F, M scores based on ranks/percentiles (e.g., Case Statements on NTILE results).
-- 4. Combine scores into RFM Score String (e.g., R_score || F_score || M_score).
-- 5. Assign Segment based on RFM Score String or Score combinations (Case Statements).

-- Example Start (Calculating R, F, M):
WITH CustomerOrders AS (
    SELECT
        customer_id,
        order_date,
        amount
    FROM
        orders
    -- Add WHERE clause here if RFM should be calculated for a date range
    -- WHERE order_date BETWEEN DATE('start_date') AND DATE('end_date')
),
CustomerAgg AS (
    SELECT
        customer_id,
        MAX(order_date) as LastOrderDate,
        COUNT(order_id) as Frequency,
        SUM(amount) as Monetary
    FROM
        CustomerOrders
    GROUP BY
        customer_id
),
RFM_Metrics AS (
    SELECT
        customer_id,
        (SELECT MAX(order_date) FROM CustomerOrders) - LastOrderDate AS Recency_TimeDelta, -- Calculate timedelta first
        Frequency,
        Monetary
    FROM
        CustomerAgg
),
RFM_Metrics_Days AS ( -- Convert timedelta to days (syntax varies by DB)
    SELECT
        customer_id,
        CAST(Recency_TimeDelta AS INTEGER) AS Recency_Days, -- Example cast to days
        Frequency,
        Monetary
    FROM RFM_Metrics
    WHERE Recency_TimeDelta IS NOT NULL -- Exclude if Recency calculation failed
)
-- Scoring (Conceptual using NTILE):
-- WITH RFM_Scores AS (
--    SELECT
--        customer_id, Recency_Days, Frequency, Monetary,
--        NTILE(4) OVER (ORDER BY Recency_Days DESC) as R_score, -- Higher days = Lower R_score
--        NTILE(4) OVER (ORDER BY Frequency ASC) as F_score,   -- Higher frequency = Higher F_score
--        NTILE(4) OVER (ORDER BY Monetary ASC) as M_score     -- Higher monetary = Higher M_score
--    FROM RFM_Metrics_Days
-- )
-- Segmentation & Final Select:
-- SELECT
--    c.name, rs.*, -- Include name and RFM data
--    (R_score || '' || F_score || '' || M_score) as RFM_Score_String,
--    CASE -- Example Segmentation based on scores
--        WHEN R_score >= 4 AND F_score >= 4 THEN 'Champions'
--        WHEN R_score <= 2 AND F_score <= 2 THEN 'At Risk/Lost'
--        -- Add more cases for other segments...
--        ELSE 'Other Segment'
--    END as RFM_Segment
-- FROM RFM_Scores rs
-- JOIN customers c ON rs.customer_id = c.id
-- ORDER BY RFM_Score_String DESC -- Example order
;
"""

            # Ensure this expander is indented correctly within the 'if not rfm_scored_df.empty:' block
            with st.expander("See the SQL query for RFM Calculation & Scoring (Conceptual)"):
                st.code(rfm_sql_query, language="sql")
                st.markdown(f"*(Conceptual query, inspired by analysis in [repo](https://github.com/fxs2596/ECommerce). Note: RFM Calculation, Scoring, and Segmentation logic varies significantly by SQL dialect and desired method.)*")


        # NO else needed for scatter plot if it's okay for it just not to appear when rfm_scored_df is empty.

    # This ELSE block aligns with the initial RFM calculation validity check IF
    else:
        st.warning("Cannot perform RFM analysis. Ensure 'order_date', 'customer_id', 'amount' columns exist in your data.")


    # --- Top Customers by Spend (Filtered - Moved to Customer Tab) ---
    st.subheader("🏆 Top Customers (Filtered Date Range)")
    num_top_customers = st.slider("Show top N customers", 5, 50, 10, key="top_cust_slider")
    # Merge and group using the filtered DataFrame
    merged = df_orders_display.merge(df_customers_original, left_on='customer_id', right_on='id', how='left')

    if not merged.empty:
        top_customers = merged.groupby('name')['amount'].sum().sort_values(ascending=False).head(num_top_customers)

        if not top_customers.empty:
            fig_top_customers = px.bar(
                top_customers,
                y=top_customers.index,
                x='amount',
                orientation='h',
                title=f"Top {num_top_customers} Customers by Spend (Filtered by Date Range)"
            )
            # Ensure update_layout is correctly indented within the if block
            fig_top_customers.update_layout(xaxis_title="Total Spend ($)", yaxis_title="Customer Name", yaxis={'categoryorder':'total ascending'})

            st.plotly_chart(fig_top_customers, use_container_width=True)

             # Add conceptual SQL for Top Customers
            top_customers_sql_query = f"""
SELECT
    c.name,
    SUM(o.amount) as total_spend
FROM
    orders o
JOIN
    customers c ON o.customer_id = c.id
"""
            if 'order_date' in df_orders_original.columns and start_date != df_orders_original['order_date'].min().date() or end_date != df_orders_original['order_date'].max().date():
                 top_customers_sql_query += f"\nWHERE o.order_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')"

            top_customers_sql_query += f"""
GROUP BY
    c.name
ORDER BY
    total_spend DESC
LIMIT {num_top_customers};
"""
            with st.expander("See the SQL query for Top Customers"):
                st.code(top_customers_sql_query, language="sql")
                st.markdown(f"*(Conceptual query, inspired by analysis in [repo](https://github.com/fxs2596/ECommerce))*")


        else:
            st.warning(f"No customer data available for the selected date range to show top {num_top_customers} customers.")
    else:
         st.warning("No order/customer data available for the selected date range.")


    # --- Latest Orders (Filtered - Moved to Customer Tab) ---
    st.subheader("🧾 Latest Orders (Filtered Date Range)")
    num_latest_orders = st.slider("Show latest N orders", 5, 100, 10, key="latest_orders_slider")
    if not df_orders_display.empty:
        st.dataframe(
            df_orders_display.sort_values(by='order_date', ascending=False)
            [['order_id', 'customer_id', 'amount', 'order_date']]
            .head(num_latest_orders),
            use_container_width=True
        )
        # Add conceptual SQL for Latest Orders
        latest_orders_sql_query = f"""
SELECT
    order_id,
    customer_id,
    amount,
    order_date
FROM
    orders
"""
        if 'order_date' in df_orders_original.columns and start_date != df_orders_original['order_date'].min().date() or end_date != df_orders_original['order_date'].max().date():
             latest_orders_sql_query += f"\nWHERE order_date BETWEEN DATE('{start_date}') AND DATE('{end_date}')"

        latest_orders_sql_query += f"""
ORDER BY
    order_date DESC
LIMIT {num_latest_orders};
"""
        with st.expander("See the SQL query for Latest Orders"):
            st.code(latest_orders_sql_query, language="sql")
            st.markdown(f"*(Conceptual query, inspired by analysis in [repo](https://github.com/fxs2596/ECommerce))*")

    else:
        st.warning("No order data available for the selected date range.")


# --- End of Script ---
# Note: No conn.close() at the end is correct for Streamlit's execution model with caching.