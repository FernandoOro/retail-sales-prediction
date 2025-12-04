import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def generate_eda_figures(df, output_dir='results/figures'):
    """
    Generates all EDA figures and saves them to the output directory.
    
    Args:
        df (pd.DataFrame): The dataframe containing sales data.
        output_dir (str): Directory to save the figures.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Generating figures in {output_dir}...")

    # 1. Distribution of Numerical Variables
    print("1/7 - Generating eda_distribucion_numericas.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Numerical Variables', fontsize=16, fontweight='bold', y=1.00)

    # Histogram Units_Sold
    axes[0, 0].hist(df['Units_Sold'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Units_Sold', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Units Sold')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['Units_Sold'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Units_Sold"].mean():.2f}')
    axes[0, 0].axvline(df['Units_Sold'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Units_Sold"].median():.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram Unit_Price
    axes[0, 1].hist(df['Unit_Price'], bins=15, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of Unit_Price', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Unit Price ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['Unit_Price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["Unit_Price"].mean():.2f}')
    axes[0, 1].axvline(df['Unit_Price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["Unit_Price"].median():.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Boxplot Units_Sold
    axes[1, 0].boxplot(df['Units_Sold'], vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='black'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))
    axes[1, 0].set_title('Boxplot of Units_Sold', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Units Sold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Boxplot Unit_Price
    axes[1, 1].boxplot(df['Unit_Price'], vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightcoral', color='black'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))
    axes[1, 1].set_title('Boxplot of Unit_Price', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Unit Price ($)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_distribucion_numericas.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation Matrix
    print("2/7 - Generating eda_correlacion.png...")
    df_temp = df.copy()
    df_temp['Total_Revenue'] = df_temp['Units_Sold'] * df_temp['Unit_Price']
    corr_data = df_temp[['Units_Sold', 'Unit_Price', 'Total_Revenue']].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_correlacion.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Price vs Sales
    print("3/7 - Generating eda_precio_vs_ventas.png...")
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = df['Category'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

    for i, category in enumerate(categories):
        df_cat = df[df['Category'] == category]
        ax.scatter(df_cat['Unit_Price'], df_cat['Units_Sold'], 
                   s=150, alpha=0.6, c=colors[i % len(colors)], edgecolors='black', linewidth=1.5,
                   label=category)

    z = np.polyfit(df['Unit_Price'], df['Units_Sold'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Unit_Price'].min(), df['Unit_Price'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='General Trend')

    ax.set_xlabel('Unit Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Units Sold', fontsize=12, fontweight='bold')
    ax.set_title('Price vs Sales Volume by Category', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_precio_vs_ventas.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Sales by Store
    print("4/7 - Generating eda_ventas_tienda.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sales Analysis by Store', fontsize=16, fontweight='bold')

    ventas_tienda = df.groupby('Store')['Units_Sold'].sum().sort_values(ascending=False)
    colors_tienda = ['#2ecc71' if x == ventas_tienda.max() else '#e74c3c' if x == ventas_tienda.min() else '#3498db' 
                     for x in ventas_tienda.values]

    axes[0].bar(ventas_tienda.index.astype(str), ventas_tienda.values, color=colors_tienda, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Store', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Total Units Sold', fontsize=11, fontweight='bold')
    axes[0].set_title('Total Volume by Store', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    promedio_tienda = df.groupby('Store')['Units_Sold'].mean().sort_values(ascending=False)
    colors_prom = ['#2ecc71' if x == promedio_tienda.max() else '#e74c3c' if x == promedio_tienda.min() else '#3498db' 
                   for x in promedio_tienda.values]

    axes[1].bar(promedio_tienda.index.astype(str), promedio_tienda.values, color=colors_prom, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Store', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Avg Units per Transaction', fontsize=11, fontweight='bold')
    axes[1].set_title('Average per Transaction', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_ventas_tienda.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Sales by Category
    print("5/7 - Generating eda_ventas_categoria.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sales Analysis by Product Category', fontsize=16, fontweight='bold')

    ventas_cat = df.groupby('Category')['Units_Sold'].sum().sort_values(ascending=False)
    colors_cat = ['#9b59b6', '#e67e22', '#1abc9c']

    axes[0].bar(range(len(ventas_cat)), ventas_cat.values, color=colors_cat, edgecolor='black', linewidth=1.5)
    axes[0].set_xticks(range(len(ventas_cat)))
    axes[0].set_xticklabels(ventas_cat.index, rotation=15, ha='right')
    axes[0].set_ylabel('Total Units Sold', fontsize=11, fontweight='bold')
    axes[0].set_title('Total Volume by Category', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    promedio_cat = df.groupby('Category')['Units_Sold'].mean().sort_values(ascending=False)
    axes[1].bar(range(len(promedio_cat)), promedio_cat.values, color=colors_cat, edgecolor='black', linewidth=1.5)
    axes[1].set_xticks(range(len(promedio_cat)))
    axes[1].set_xticklabels(promedio_cat.index, rotation=15, ha='right')
    axes[1].set_ylabel('Avg Units per Transaction', fontsize=11, fontweight='bold')
    axes[1].set_title('Average per Transaction', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_ventas_categoria.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Outliers
    print("6/7 - Generating eda_outliers.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Outlier Detection (IQR Method)', fontsize=16, fontweight='bold')

    def detect_outliers(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    lower_units, upper_units = detect_outliers(df['Units_Sold'])
    axes[0].boxplot(df['Units_Sold'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='black', linewidth=2),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor='red', markersize=10, linestyle='none'))
    axes[0].axhline(lower_units, color='orange', linestyle='--', linewidth=2, label=f'Lower: {lower_units:.2f}')
    axes[0].axhline(upper_units, color='orange', linestyle='--', linewidth=2, label=f'Upper: {upper_units:.2f}')
    axes[0].set_ylabel('Units Sold', fontsize=11, fontweight='bold')
    axes[0].set_title('Units_Sold Outliers', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    lower_price, upper_price = detect_outliers(df['Unit_Price'])
    axes[1].boxplot(df['Unit_Price'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightcoral', color='black', linewidth=2),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor='red', markersize=10, linestyle='none'))
    axes[1].axhline(upper_price, color='orange', linestyle='--', linewidth=2, label=f'Upper: ${upper_price:.2f}')
    axes[1].set_ylabel('Unit Price ($)', fontsize=11, fontweight='bold')
    axes[1].set_title('Unit_Price Outliers', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Time Series
    print("7/7 - Generating eda_serie_temporal.png...")
    ventas_diarias = df.groupby('Date')['Units_Sold'].sum().reset_index()
    ventas_diarias = ventas_diarias.sort_values('Date')

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ventas_diarias['Date'], ventas_diarias['Units_Sold'], 
            marker='o', linewidth=2.5, markersize=8, color='#2c3e50', 
            markerfacecolor='#e74c3c', markeredgecolor='black', markeredgewidth=1.5,
            label='Daily Sales')

    ventas_diarias['MA3'] = ventas_diarias['Units_Sold'].rolling(window=3, center=True).mean()
    ax.plot(ventas_diarias['Date'], ventas_diarias['MA3'], 
            linewidth=2, color='#27ae60', linestyle='--', alpha=0.7,
            label='3-Day Moving Average')

    promedio_general = ventas_diarias['Units_Sold'].mean()
    ax.axhline(promedio_general, color='red', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Average: {promedio_general:.1f}')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Units Sold', fontsize=12, fontweight='bold')
    ax.set_title('Sales Time Series - January 2024', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_serie_temporal.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All figures generated successfully.")

if __name__ == "__main__":
    # Test execution
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_preprocessing import load_data, handle_duplicates
    
    df = load_data('../data/raw/sales_data.xlsx')
    df = handle_duplicates(df)
    generate_eda_figures(df)
