# E-Commerce SQL Analytics Engine 🛍️

This project simulates an e-commerce database with customers and orders. It includes SQL queries to generate business insights like top customers, revenue trends, and order behaviors.

## 🗃️ Tech Stack
- SQLite (local file-based SQL database)
- Python (for visualizations)
- Faker (to generate realistic seed data)
- Pandas + Matplotlib (for charts)

## 📐 Database Schema

- **Customers**
  - `id`: Primary key
  - `name`: Customer name
  - `email`: Customer email
  - `signup_date`: Date they signed up

- **Orders**
  - `order_id`: Primary key
  - `customer_id`: Foreign key to Customers
  - `amount`: Order amount in dollars
  - `order_date`: Date of the order

## 🧪 Sample Analysis Queries

### 🔹 Top 5 Customers by Spend
```sql
SELECT name, SUM(amount) AS total_spent
FROM customers
JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.id
ORDER BY total_spent DESC
LIMIT 5;
