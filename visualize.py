import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("mytest.db")  # update if you used a different name

query = """
SELECT
  strftime('%Y-%m', order_date) AS month,
  SUM(amount) AS revenue
FROM orders
GROUP BY month
ORDER BY month;
"""

df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 5))
plt.plot(df['month'], df['revenue'], marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_revenue.png")
plt.show()
