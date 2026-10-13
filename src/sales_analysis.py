"""
Business Sales Performance Analytics - Comprehensive EDA & Visualizations
Future Interns - Task 1
Author: Data Analytics Team
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

COLORS = {
    'primary': '#8B5CF6',      # Electric Violet
    'secondary': '#06B6D4',    # Cyber Cyan
    'sunset': '#FF6B6B',       # Radiant Sunset Coral
    'accent': '#FBBF24',       # Cyber Gold / Amber
    'danger': '#FF3366',       # Flame Crimson
    'success': '#10E7A0',      # Neon Mint / Emerald
    'neutral_dark': '#0F172A', # Dark Slate
    'neutral_light': '#F8FAFC' # Light Alabaster
}

def run_sales_analysis(data_path: str, visuals_dir: str, json_export_path: str):
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_export_path), exist_ok=True)

    print(f"Reading cleaned sales data from: {data_path}")
    df = pd.read_csv(data_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # ----------------------------------------------------
    # 1. Executive Key Performance Indicators (KPIs)
    # ----------------------------------------------------
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    profit_margin = (total_profit / total_sales) * 100
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()
    total_units = df["Quantity"].sum()
    aov = total_sales / total_orders
    avg_discount = df["Discount"].mean() * 100

    kpis = {
        "total_revenue": round(total_sales, 2),
        "total_profit": round(total_profit, 2),
        "profit_margin_pct": round(profit_margin, 2),
        "total_orders": int(total_orders),
        "total_customers": int(total_customers),
        "total_units_sold": int(total_units),
        "average_order_value": round(aov, 2),
        "average_discount_pct": round(avg_discount, 2)
    }

    print("\n--- EXECUTIVE SUMMARY KPIs ---")
    print(f"Total Revenue:       ${kpis['total_revenue']:,.2f}")
    print(f"Total Profit:        ${kpis['total_profit']:,.2f}")
    print(f"Profit Margin:       {kpis['profit_margin_pct']}%")
    print(f"Total Orders:        {kpis['total_orders']:,}")
    print(f"Total Customers:     {kpis['total_customers']:,}")
    print(f"Avg Order Value:     ${kpis['average_order_value']:,.2f}")

    # ----------------------------------------------------
    # 2. Time Trend Analysis (Monthly & Yearly)
    # ----------------------------------------------------
    monthly_trend = df.groupby("Order_Year_Month").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order ID", "nunique")
    ).reset_index().sort_values("Order_Year_Month")

    yearly_trend = df.groupby("Order_Year").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order ID", "nunique")
    ).reset_index().sort_values("Order_Year")

    yearly_trend["YoY_Sales_Growth"] = yearly_trend["Sales"].pct_change() * 100
    yearly_trend["YoY_Profit_Growth"] = yearly_trend["Profit"].pct_change() * 100
    yearly_trend["Profit_Margin"] = (yearly_trend["Profit"] / yearly_trend["Sales"]) * 100

    # VISUAL 1: Monthly Sales & Profit Trend Line Chart
    fig, ax1 = plt.subplots(figsize=(14, 6))
    x_indices = range(len(monthly_trend))
    
    line1 = ax1.plot(x_indices, monthly_trend["Sales"] / 1000, color=COLORS['sunset'], marker='o', linewidth=2.5, label='Revenue ($K)')
    ax1.set_ylabel('Revenue ($ in Thousands)', color=COLORS['sunset'], fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=COLORS['sunset'])
    ax1.set_xticks(x_indices[::4])
    ax1.set_xticklabels(monthly_trend["Order_Year_Month"].iloc[::4], rotation=45, ha='right', fontsize=10)

    ax2 = ax1.twinx()
    line2 = ax2.plot(x_indices, monthly_trend["Profit"] / 1000, color=COLORS['accent'], marker='s', linewidth=2.5, linestyle='--', label='Profit ($K)')
    ax2.set_ylabel('Profit ($ in Thousands)', color=COLORS['accent'], fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax2.grid(False) # avoid clutter

    plt.title('Business Revenue & Profit Trajectory Over Time (2011 - 2014)', fontsize=15, fontweight='bold', pad=15)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True)
    
    fig.tight_layout()
    chart1_path = os.path.join(visuals_dir, "1_monthly_sales_profit_trend.png")
    fig.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved: {chart1_path}")

    # ----------------------------------------------------
    # 3. Category & Sub-Category Performance
    # ----------------------------------------------------
    cat_summary = df.groupby("Category").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    ).reset_index()
    cat_summary["Profit_Margin"] = (cat_summary["Profit"] / cat_summary["Sales"]) * 100
    cat_summary = cat_summary.sort_values("Sales", ascending=False)

    subcat_summary = df.groupby(["Category", "Sub-Category"]).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Avg_Discount=("Discount", "mean")
    ).reset_index()
    subcat_summary["Profit_Margin"] = (subcat_summary["Profit"] / subcat_summary["Sales"]) * 100
    subcat_summary = subcat_summary.sort_values("Profit", ascending=True)

    # VISUAL 2: Sub-Category Profitability (Highlighting Tables & Losses)
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = [COLORS['danger'] if p < 0 else COLORS['secondary'] for p in subcat_summary["Profit"]]
    
    bars = ax.barh(subcat_summary["Sub-Category"], subcat_summary["Profit"] / 1000, color=bar_colors, edgecolor='none', height=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Total Net Profit ($ in Thousands)', fontsize=12, fontweight='bold')
    ax.set_title('Sub-Category Profitability: Identifying Growth Engines vs. Loss Leaders', fontsize=14, fontweight='bold', pad=15)
    
    # Add values on bars
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (0.8 if w >= 0 else -3.2)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"${w:.1f}K", 
                va='center', fontsize=9, fontweight='bold', color=COLORS['neutral_dark'])

    fig.tight_layout()
    chart2_path = os.path.join(visuals_dir, "2_category_performance.png")
    fig.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved: {chart2_path}")

    # ----------------------------------------------------
    # 4. Regional Breakdown & State Outliers
    # ----------------------------------------------------
    region_summary = df.groupby("Region").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order ID", "nunique"),
        Avg_Discount=("Discount", "mean")
    ).reset_index()
    region_summary["Profit_Margin"] = (region_summary["Profit"] / region_summary["Sales"]) * 100
    region_summary = region_summary.sort_values("Sales", ascending=False)

    # VISUAL 3: Regional Sales vs Profit Comparison
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(region_summary))
    width = 0.35

    rects1 = ax1.bar(x - width/2, region_summary["Sales"]/1000, width, label='Revenue ($K)', color=COLORS['primary'])
    rects2 = ax1.bar(x + width/2, region_summary["Profit"]/1000, width, label='Profit ($K)', color=COLORS['secondary'])

    ax1.set_ylabel('Amount ($ in Thousands)', fontsize=12, fontweight='bold')
    ax1.set_title('Regional Sales & Profit Comparison: Central Region Margin Lag', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(region_summary["Region"], fontsize=11, fontweight='bold')
    ax1.legend(frameon=True)

    # Annotate margin percentage
    for i, row in region_summary.iterrows():
        margin = row["Profit_Margin"]
        ax1.text(i, max(row["Sales"]/1000, row["Profit"]/1000) + 15, f"Margin: {margin:.1f}%", 
                 ha='center', fontsize=10, fontweight='bold', color=COLORS['neutral_dark'])

    fig.tight_layout()
    chart3_path = os.path.join(visuals_dir, "3_regional_breakdown.png")
    fig.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved: {chart3_path}")

    # ----------------------------------------------------
    # 5. Top & Bottom Products (Pareto & Margin Bleeders)
    # ----------------------------------------------------
    prod_summary = df.groupby("Product Name").agg(
        Category=("Category", "first"),
        SubCategory=("Sub-Category", "first"),
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    ).reset_index()

    top_revenue_prods = prod_summary.sort_values("Sales", ascending=False).head(10)
    top_profit_prods = prod_summary.sort_values("Profit", ascending=False).head(10)
    bottom_profit_prods = prod_summary.sort_values("Profit", ascending=True).head(10)

    # VISUAL 4: Top 10 Revenue Products vs. Top 10 Profit Bleeders
    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(16, 7))

    # Top Revenue
    top_10_disp = top_revenue_prods.iloc[::-1]
    short_names_top = [name[:30] + '...' if len(name) > 30 else name for name in top_10_disp["Product Name"]]
    ax_top.barh(short_names_top, top_10_disp["Sales"]/1000, color=COLORS['primary'], height=0.65)
    ax_top.set_xlabel('Revenue ($ in Thousands)', fontsize=11, fontweight='bold')
    ax_top.set_title('Top 10 Revenue-Generating Products', fontsize=13, fontweight='bold')

    # Bottom Profit (Bleeders)
    bot_10_disp = bottom_profit_prods.iloc[::-1]
    short_names_bot = [name[:30] + '...' if len(name) > 30 else name for name in bot_10_disp["Product Name"]]
    ax_bot.barh(short_names_bot, bot_10_disp["Profit"]/1000, color=COLORS['danger'], height=0.65)
    ax_bot.set_xlabel('Net Loss ($ in Thousands)', fontsize=11, fontweight='bold')
    ax_bot.set_title('Top 10 Margin-Bleeding Products (Action Required)', fontsize=13, fontweight='bold')

    fig.tight_layout()
    chart4_path = os.path.join(visuals_dir, "4_top_bottom_products.png")
    fig.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"Saved: {chart4_path}")

    # ----------------------------------------------------
    # 6. Discount Impact Analysis
    # ----------------------------------------------------
    discount_analysis = df.groupby("Discount_Band").agg(
        Order_Count=("Order ID", "count"),
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Avg_Profit_Margin=("Profit_Margin_Pct", "mean")
    ).reset_index()

    # Sort discount bands logically
    band_order = ["0% (No Discount)", "1% - 10%", "11% - 20%", "21% - 40%", "> 40% (Deep Discount)"]
    discount_analysis["Band_Order"] = discount_analysis["Discount_Band"].map(lambda x: band_order.index(x) if x in band_order else 99)
    discount_analysis = discount_analysis.sort_values("Band_Order")
    discount_analysis["Realized_Margin"] = (discount_analysis["Total_Profit"] / discount_analysis["Total_Sales"]) * 100

    # VISUAL 5: Discount vs Profit Margin
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors_disc = [COLORS['success'] if m > 0 else COLORS['danger'] for m in discount_analysis["Realized_Margin"]]
    bars_disc = ax.bar(discount_analysis["Discount_Band"], discount_analysis["Realized_Margin"], color=bar_colors_disc, width=0.55)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Realized Profit Margin (%)', fontsize=12, fontweight='bold')
    ax.set_title('Discount Destruction: Severe Negative Margins Beyond 20% Discount', fontsize=14, fontweight='bold', pad=15)
    
    for bar in bars_disc:
        h = bar.get_height()
        y_pos = h + (1.5 if h >= 0 else -4.0)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{h:.1f}%", ha='center', fontsize=10, fontweight='bold')

    fig.tight_layout()
    chart5_path = os.path.join(visuals_dir, "5_discount_impact_analysis.png")
    fig.savefig(chart5_path, dpi=300)
    plt.close()
    print(f"Saved: {chart5_path}")

    # ----------------------------------------------------
    # 7. Customer Segment Performance
    # ----------------------------------------------------
    segment_summary = df.groupby("Segment").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Customers=("Customer ID", "nunique"),
        Orders=("Order ID", "nunique")
    ).reset_index()
    segment_summary["Profit_Margin"] = (segment_summary["Profit"] / segment_summary["Sales"]) * 100
    segment_summary["AOV"] = segment_summary["Sales"] / segment_summary["Orders"]

    # VISUAL 6: Segment Breakdown
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))

    ax_pie.pie(segment_summary["Sales"], labels=segment_summary["Segment"], autopct='%1.1f%%',
               colors=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], startangle=140,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax_pie.set_title('Revenue Share by Customer Segment', fontsize=13, fontweight='bold')

    bars_seg = ax_bar.bar(segment_summary["Segment"], segment_summary["Profit_Margin"], color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], width=0.5)
    ax_bar.set_ylabel('Profit Margin (%)', fontsize=11, fontweight='bold')
    ax_bar.set_title('Profit Margin by Customer Segment', fontsize=13, fontweight='bold')
    for b in bars_seg:
        ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4, f"{b.get_height():.1f}%", ha='center', fontweight='bold')

    fig.tight_layout()
    chart6_path = os.path.join(visuals_dir, "6_customer_segment_kpis.png")
    fig.savefig(chart6_path, dpi=300)
    plt.close()
    print(f"Saved: {chart6_path}")

    # ----------------------------------------------------
    # 8. Export Aggregated Analytics Data for Interactive Dashboard
    # ----------------------------------------------------
    dashboard_payload = {
        "kpis": kpis,
        "monthly_trend": monthly_trend.to_dict(orient="records"),
        "yearly_trend": yearly_trend.to_dict(orient="records"),
        "category_summary": cat_summary.to_dict(orient="records"),
        "subcategory_summary": subcat_summary.sort_values("Sales", ascending=False).to_dict(orient="records"),
        "region_summary": region_summary.to_dict(orient="records"),
        "segment_summary": segment_summary.to_dict(orient="records"),
        "discount_analysis": discount_analysis.drop(columns=["Band_Order"], errors='ignore').to_dict(orient="records"),
        "top_10_products": top_revenue_prods.to_dict(orient="records"),
        "bottom_10_products": bottom_profit_prods.to_dict(orient="records"),
        "state_summary": df.groupby(["State", "Region"]).agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique")
        ).reset_index().sort_values("Sales", ascending=False).head(20).to_dict(orient="records")
    }

    with open(json_export_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_payload, f, indent=2)
    print(f"Dashboard data exported to: {json_export_path}")

    print("\nAnalysis and visualization generation completed successfully!")

if __name__ == "__main__":
    cleaned_file = os.path.join("data", "processed", "superstore_cleaned.csv")
    visuals_folder = "visuals"
    json_export = os.path.join("dashboard", "sales_data.json")
    run_sales_analysis(cleaned_file, visuals_folder, json_export)
