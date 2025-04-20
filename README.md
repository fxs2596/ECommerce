# E-Commerce SQL Analytics Engine 🛍️

## Project Description
This project simulates an e-commerce database and provides an interactive dashboard for analyzing customer and order data from a SQLite database. It combines SQL analytics with Python data processing and visualization, and includes a step to prepare data specifically for Machine Learning churn prediction.

The project demonstrates an end-to-end workflow, from data creation and database management to interactive data visualization and foundational steps for ML modeling.

## Features

### Database & Data Creation
-   Uses a local `mytest.db` SQLite database.
-   Includes a process to create and populate the database with sample e-commerce customer and order data (details documented in the data creation notebook).

### Interactive Dashboard (Built with Streamlit)
-   Connects to the `mytest.db` SQLite database to fetch data.
-   Displays **Key Performance Indicators (KPIs)** for overall and filtered sales (Total Revenue, Total Orders, Unique Customers, Average Order Value, Orders per Customer).
-   Shows **Interactive Sales Trends** (monthly and daily revenue) using Plotly charts, with data filtered by date range.
-   Provides a **Date Range Filter** in the sidebar for dynamic data exploration.
-   Uses **Tabs** to organize the dashboard content into logical sections (Overview, Trends, Customers).
-   Performs **Advanced Customer Analysis** including:
    -   **RFM (Recency, Frequency, Monetary) Analysis & Segmentation:** Calculates RFM metrics, assigns scores (1-4), and segments customers into categories (e.g., 'Champions/Loyal', 'At Risk/Lost').
    -   **RFM Visualization:** Displays customer segment distribution and an interactive RFM scatter plot (Frequency vs Monetary, colored by Recency).
    -   **Top Customers:** Shows a table and chart of top customers by spend within the selected date range.
    -   **Latest Orders:** Displays a table of the most recent orders within the selected date range.
-   Includes **Conceptual SQL Queries** behind key charts and analyses, linking the dashboard's output back to the underlying database interaction logic.

### ML Data Preparation
-   Upon running the Streamlit application (`app.py`), a CSV file named `customer_churn_ml_dataset.csv` is automatically generated.
-   This file contains customer-level data with engineered features (RFM metrics, Customer Tenure, Average Order Value) and a binary `churn` label (defined as inactivity for the last 180 days relative to the latest order date).
-   This dataset is ready to be used as input for training a Machine Learning model.

## Technologies Used
-   Python
-   Streamlit
-   SQLite3
-   Pandas
-   Matplotlib (for internal use, main plots use Plotly)
-   Plotly / Plotly Express
-   NumPy
-   Faker (Used in the data creation process - documented in notebook)

## Database Schema

**Customers**
-   `id`: Primary key
-   `name`: Customer name
-   `email`: Customer email
-   `signup_date`: Date they signed up

**Orders**
-   `order_id`: Primary key
-   `customer_id`: Foreign key to Customers
-   `amount`: Order amount in dollars
-   `order_date`: Date of the order

*(Note: This schema reflects the data used in the project. Additional columns like location or social media source were discussed but are not included in the current `mytest.db` based on the provided code and schema).*

## Sample Analysis Queries
*(These queries represent examples of analysis that can be performed on the database, conceptually linked to the dashboard features).*

**🔹 Top Customers by Spend**
```sql
SELECT
    c.name,
    SUM(o.amount) AS total_spent
FROM
    customers c
JOIN
    orders o ON c.id = o.customer_id
GROUP BY
    c.id, c.name -- Group by id and name for robustness
ORDER BY
    total_spent DESC
LIMIT 5;
