import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Business Sales Performance Analytics
**Data Science & Analytics Internship – Task 1 (2026)**  
*Organization: Future Interns*  
*Role: Business Data Analyst*

---

## Executive Overview & Business Objectives
In this project, we analyze sales transactions for a multi-category retail enterprise to answer critical strategic business questions:
1. **Which products generate the most revenue?**
2. **How do sales change over time?**
3. **Which categories or regions are most profitable?**
4. **Where should the business focus to grow faster?**

This notebook walks through data preparation, exploratory data analysis (EDA), KPI engineering, and strategic recommendations."""))

# Setup and Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("Environment configured successfully!")"""))

# Loading cleaned data
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Quality Inspection
Loading the cleaned dataset produced by our preprocessing pipeline."""))

cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('../data/processed/superstore_cleaned.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
df.head(3)"""))

# Executive KPIs
cells.append(nbf.v4.new_markdown_cell("""## 2. High-Level Executive Key Performance Indicators (KPIs)
Calculating headline metrics for executive stakeholders."""))

cells.append(nbf.v4.new_code_cell("""total_revenue = df['Sales'].sum()
total_profit = df['Profit'].sum()
overall_margin = (total_profit / total_revenue) * 100
total_orders = df['Order ID'].nunique()
unique_customers = df['Customer ID'].nunique()
aov = total_revenue / total_orders

print('=' * 45)
print('          EXECUTIVE SUMMARY METRICS          ')
print('=' * 45)
print(f'Total Revenue:           ${total_revenue:,.2f}')
print(f'Total Net Profit:        ${total_profit:,.2f}')
print(f'Overall Profit Margin:   {overall_margin:.2f}%')
print(f'Total Unique Orders:     {total_orders:,}')
print(f'Total Unique Customers:  {unique_customers:,}')
print(f'Average Order Value:     ${aov:.2f}')
print('=' * 45)"""))

# Sales over time
cells.append(nbf.v4.new_markdown_cell("""## 3. Business Question 1: How Do Sales Change Over Time?
Aggregating sales and profit by month to detect seasonality, growth velocity, and cyclic dips."""))

cells.append(nbf.v4.new_code_cell("""monthly = df.groupby('Order_Year_Month').agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum')
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 5))
x = range(len(monthly))

ax1.plot(x, monthly['Sales'], color='#0284C7', marker='o', linewidth=2.5, label='Sales Revenue')
ax1.set_ylabel('Monthly Sales ($)', color='#0284C7', fontweight='bold')
ax1.set_xticks(x[::4])
ax1.set_xticklabels(monthly['Order_Year_Month'].iloc[::4], rotation=45)

ax2 = ax1.twinx()
ax2.plot(x, monthly['Profit'], color='#10B981', marker='s', linewidth=2, linestyle='--', label='Net Profit')
ax2.set_ylabel('Net Profit ($)', color='#10B981', fontweight='bold')
ax2.grid(False)

plt.title('Monthly Sales Revenue & Profit Performance (2011 - 2014)', fontsize=14, fontweight='bold')
plt.show()"""))

# Top and Bottom Products
cells.append(nbf.v4.new_markdown_cell("""## 4. Business Question 2: Which Products Generate the Most Revenue?
Evaluating top revenue generators alongside margin-bleeding products."""))

cells.append(nbf.v4.new_code_cell("""prod_summary = df.groupby('Product Name').agg(
    Category=('Category', 'first'),
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum'),
    Quantity=('Quantity', 'sum')
).reset_index()

top_10_revenue = prod_summary.sort_values('Sales', ascending=False).head(10)
bottom_10_profit = prod_summary.sort_values('Profit', ascending=True).head(10)

print("--- TOP 5 REVENUE GENERATING PRODUCTS ---")
print(top_10_revenue[['Product Name', 'Category', 'Sales', 'Profit']].head(5).to_string(index=False))

print("\n--- TOP 5 MARGIN-BLEEDING PRODUCTS (NET LOSS) ---")
print(bottom_10_profit[['Product Name', 'Category', 'Sales', 'Profit']].head(5).to_string(index=False))"""))

# Category & Regional Performance
cells.append(nbf.v4.new_markdown_cell("""## 5. Business Question 3: Which Categories or Regions Are Most Profitable?
Inspecting profit margins across categories and geographic regions."""))

cells.append(nbf.v4.new_code_cell("""cat_perf = df.groupby('Category').agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum')
).reset_index()
cat_perf['Margin_Pct'] = (cat_perf['Profit'] / cat_perf['Sales']) * 100

reg_perf = df.groupby('Region').agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum')
).reset_index()
reg_perf['Margin_Pct'] = (reg_perf['Profit'] / reg_perf['Sales']) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=cat_perf, x='Category', y='Margin_Pct', ax=axes[0], hue='Category', palette=['#0284C7', '#6366F1', '#10B981'], legend=False)
axes[0].set_title('Profit Margin (%) by Category', fontweight='bold')
axes[0].set_ylabel('Margin (%)')

sns.barplot(data=reg_perf, x='Region', y='Margin_Pct', ax=axes[1], hue='Region', palette=['#0284C7', '#6366F1', '#10B981', '#F59E0B'], legend=False)
axes[1].set_title('Profit Margin (%) by Region', fontweight='bold')
axes[1].set_ylabel('Margin (%)')

plt.tight_layout()
plt.show()"""))

# Discount Impact
cells.append(nbf.v4.new_markdown_cell("""## 6. Deep Dive: Impact of Discounts on Profitability
Evaluating why certain areas and categories suffer severe losses."""))

cells.append(nbf.v4.new_code_cell("""disc_perf = df.groupby('Discount_Band').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Order_Count=('Order ID', 'count')
).reset_index()
disc_perf['Realized_Margin'] = (disc_perf['Total_Profit'] / disc_perf['Total_Sales']) * 100

print(disc_perf[['Discount_Band', 'Total_Sales', 'Total_Profit', 'Realized_Margin', 'Order_Count']].to_string(index=False))"""))

# Strategic Recommendations
cells.append(nbf.v4.new_markdown_cell("""## 7. Business Question 4: Where Should the Business Focus to Grow Faster?
### Strategic Action Plan & Consultant Recommendations:
1. **Cap Discount Policies at 20%**: Analysis proves transactions discounted >20% yield deep negative margins (-30% to -60%). Implement hard discount limits for sales reps.
2. **Eliminate Loss-Making Subcategories (Tables)**: Restructure the supply chain for Tables (-$17.7K net loss) or bundle them with high-margin Chairs & Technology.
3. **Scale Technology & Copiers**: Technology boasts the highest margin (17.4%). Increase stock and B2B marketing for Copiers and Phones.
4. **Target B2B Corporate Accounts in West & East**: West region delivers 14.9% margin; scale corporate sales forces in California, New York, and Washington."""))

nb.cells = cells

os.makedirs('notebooks', exist_ok=True)
notebook_path = os.path.join('notebooks', 'Business_Sales_Performance_Analysis.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created successfully at: {notebook_path}")
